"""Export SemCEB sf1000 pure-text corpora and filter queries into SemStats format.

Matches references/repos/SemCEB/src/semceb/data/loader.py exactly:
- products: shuffle file_row_number with pandas random_state=42, take first 1000 rows.
- reviews: filter r.asin IN (products.parent_asin) ordered by file_row_number.

Outputs under experiments/semstats-semceb-sf1000/data/:
- products/corpus/documents.jsonl  (full text concatenation for SemStats catalog/index/embedding)
- products/corpus/records.jsonl    (raw structured columns per row for per-query field-scoped judging)
- products/queries/single.json     (30 pure-text filter queries + gpt-5.4-nano sf1000 ground truth)
- reviews/corpus/documents.jsonl
- reviews/corpus/records.jsonl
- reviews/queries/single.json      (19 pure-text filter queries + gpt-5.4-nano sf1000 ground truth)
- manifest.json
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import duckdb
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_DIR.parent.parent
SEMCEB_DATA_DIR = REPO_ROOT / "references" / "datasets" / "semceb" / "amazon-reviews"
SEMCEB_REPO_DIR = REPO_ROOT / "references" / "repos" / "SemCEB"
SEMSTATS_REPO_DIR = REPO_ROOT / "references" / "repos" / "SemanticCardinalityEstimation"
OUT_DIR = EXP_DIR / "data"
PREDICATES_PATH = EXP_DIR / "configs" / "semceb_filter_predicates.json"

SCALE_FACTOR = 1000
EXPECTED_PRODUCTS_QUERIES = 30
EXPECTED_REVIEWS_QUERIES = 19
PRODUCTS_DATASET = "amazon-reviews/products_filtered_with_embeddings"
REVIEWS_DATASET = "amazon-reviews/reviews_filtered_with_embeddings"
IMAGE_COL = "main_image_local"

PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_.]+)\}")


def strip_alias_placeholders(raw_filter: str) -> str:
    """Normalize {alias.col} -> {col} to match SemCEB ground-truth cache keys."""
    return re.sub(r"\{[a-zA-Z0-9_]+\.([a-zA-Z0-9_]+)\}", r"{\1}", raw_filter)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_head(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_scoped_predicates(path: Path = PREDICATES_PATH) -> dict[str, dict]:
    """Load the inspectable, individually authored query adaptations."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != 1:
        raise ValueError(f"Unsupported predicate mapping schema in {path}")
    out = {}
    for row in payload["queries"]:
        case_id = row["case_id"]
        if case_id in out:
            raise ValueError(f"Duplicate predicate mapping for {case_id}")
        out[case_id] = row
    return out


def scoped_predicate(case_id: str, raw_filter: str, columns: list[str], mappings: dict) -> str:
    """Require exact source and field matches; never silently fall back to regex rewriting."""
    if case_id not in mappings:
        raise ValueError(f"Missing field-scoped predicate for {case_id}")
    mapping = mappings[case_id]
    source_sha256 = hashlib.sha256(raw_filter.encode("utf-8")).hexdigest()
    if mapping["source_filter_sha256"] != source_sha256:
        raise ValueError(f"Source filter changed for {case_id}; review its field-scoped predicate")
    if mapping["columns"] != list(dict.fromkeys(columns)):
        raise ValueError(f"Field scope mismatch for {case_id}")
    pred = mapping["predicate"]
    if not pred.strip() or "{" in pred or "}" in pred:
        raise ValueError(f"Empty predicate or unresolved template for {case_id}")
    if any(f"({col})" not in pred for col in mapping["columns"]):
        raise ValueError(f"Predicate does not state every source field for {case_id}")
    return pred


def fmt_val(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def load_sf1000_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    products_parquet = SEMCEB_DATA_DIR / "products_filtered_with_embeddings.parquet"
    reviews_parquet = SEMCEB_DATA_DIR / "reviews_filtered_with_embeddings.parquet"

    with duckdb.connect() as con:
        product_rows = con.execute(
            """
            SELECT file_row_number
            FROM read_parquet(?, file_row_number=true)
            ORDER BY file_row_number
            """,
            [str(products_parquet)],
        ).df()

    product_rows = (
        product_rows.sample(frac=1.0, replace=False, random_state=42)
        .reset_index(drop=True)
        .head(SCALE_FACTOR)
        .reset_index(drop=True)
    )
    product_rows["sample_order"] = range(len(product_rows))

    with duckdb.connect() as con:
        con.register("selected_product_rows", product_rows)
        products_df = con.execute(
            """
            SELECT
                p.parent_asin,
                p.product_title,
                p.subtitle,
                p.author,
                p.features_json,
                p.description_json,
                p.details_json
            FROM read_parquet(?, file_row_number=true) AS p
            JOIN selected_product_rows AS s
              ON p.file_row_number = s.file_row_number
            ORDER BY s.sample_order
            """,
            [str(products_parquet)],
        ).df().reset_index(drop=True)

    selected_asins = pd.DataFrame(
        {"asin": products_df["parent_asin"].dropna().unique()}
    )

    with duckdb.connect() as con:
        con.register("selected_asins", selected_asins)
        reviews_df = con.execute(
            """
            SELECT
                r.asin,
                r.user_id,
                r.rating,
                r.verified_purchase,
                r.helpful_vote,
                r.review_title,
                r.review_text
            FROM read_parquet(?, file_row_number=true) AS r
            WHERE r.asin IN (SELECT asin FROM selected_asins)
            ORDER BY r.file_row_number
            """,
            [str(reviews_parquet)],
        ).df().reset_index(drop=True)

    return products_df, reviews_df


def write_products_corpus(df: pd.DataFrame, docs_path: Path, records_path: Path) -> None:
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ("parent_asin", "product_title", "subtitle", "author", "features_json", "description_json", "details_json")
    with open(docs_path, "w", encoding="utf-8") as fd, open(records_path, "w", encoding="utf-8") as fr:
        for i, row in df.iterrows():
            parts = []
            for label, col in (
                ("Title", "product_title"),
                ("Subtitle", "subtitle"),
                ("Author", "author"),
                ("Features", "features_json"),
                ("Description", "description_json"),
                ("Details", "details_json"),
            ):
                val = fmt_val(row[col])
                if val:
                    parts.append(f"{label}: {val}")
            doc_id = f"prod_{i:04d}_{fmt_val(row['parent_asin'])}"
            fd.write(json.dumps({"doc_id": doc_id, "text": "\n".join(parts)}, ensure_ascii=False) + "\n")
            rec = {"doc_id": doc_id, "fields": {c: fmt_val(row[c]) for c in cols}}
            fr.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_reviews_corpus(df: pd.DataFrame, docs_path: Path, records_path: Path) -> None:
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ("asin", "user_id", "rating", "verified_purchase", "helpful_vote", "review_title", "review_text")
    with open(docs_path, "w", encoding="utf-8") as fd, open(records_path, "w", encoding="utf-8") as fr:
        for i, row in df.iterrows():
            parts = []
            for label, col in (
                ("Review Title", "review_title"),
                ("Rating", "rating"),
                ("Verified Purchase", "verified_purchase"),
                ("Helpful Votes", "helpful_vote"),
                ("Review Text", "review_text"),
            ):
                val = fmt_val(row[col])
                if val:
                    parts.append(f"{label}: {val}")
            doc_id = f"rev_{i:05d}_{fmt_val(row['asin'])}"
            fd.write(json.dumps({"doc_id": doc_id, "text": "\n".join(parts)}, ensure_ascii=False) + "\n")
            rec = {"doc_id": doc_id, "fields": {c: fmt_val(row[c]) for c in cols}}
            fr.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_ground_truth_index() -> dict[tuple[str, str], dict]:
    gt_path = SEMCEB_REPO_DIR / "benchmark_queries" / f"ground_truth_cache_gpt-5.4-nano_sf{SCALE_FACTOR}.json"
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        m_ds = re.search(r"datasets='([^']+)'", k)
        m_q = re.search(r"query='(.*)'$", k, re.DOTALL)
        if not (m_ds and m_q):
            continue
        ds_name = m_ds.group(1).split(":")[-1].strip()
        out[(ds_name, m_q.group(1))] = v
    return out


def export_queries(
    gt_index: dict[tuple[str, str], dict], predicate_path: Path = PREDICATES_PATH
) -> tuple[list[dict], list[dict]]:
    q_path = SEMCEB_REPO_DIR / "benchmark_queries" / "queries.jsonl"
    mappings = load_scoped_predicates(predicate_path)
    products_qs, reviews_qs = [], []
    with open(q_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            if "filter" not in q or len(q.get("datasets", [])) != 1:
                continue
            raw_filter = q["filter"]
            norm_filter = strip_alias_placeholders(raw_filter)
            cols = [c.split(".")[-1] for c in PLACEHOLDER_RE.findall(raw_filter)]
            if IMAGE_COL in cols:
                continue
            ds = re.split(r"\s+as\s+", q["datasets"][0], flags=re.IGNORECASE)[0].strip()
            gt = gt_index.get((ds, norm_filter)) or gt_index.get((ds, raw_filter))
            if gt is None:
                raise KeyError(f"Missing sf{SCALE_FACTOR} ground truth for Q{q['id']:03d} ({ds})")
            case_id = f"Q{q['id']:03d}"
            pred = scoped_predicate(case_id, raw_filter, cols, mappings)
            rec = {
                "case_id": case_id,
                "query_id": int(q["id"]),
                "type": q["type"],
                "category": q.get("category", []),
                "columns": cols,
                "source_filter": raw_filter,
                "raw_filter": norm_filter,
                "predicate_source": "configs/semceb_filter_predicates.json",
                "query": pred,
                "predicates": [pred],
                "ground_truth_cardinality": int(gt["cardinality"]),
                "ground_truth_selectivity": float(gt["selectivity"]),
            }
            if ds == PRODUCTS_DATASET:
                products_qs.append(rec)
            elif ds == REVIEWS_DATASET:
                reviews_qs.append(rec)

    if len(products_qs) != EXPECTED_PRODUCTS_QUERIES:
        raise AssertionError(f"Expected {EXPECTED_PRODUCTS_QUERIES} products queries, got {len(products_qs)}")
    if len(reviews_qs) != EXPECTED_REVIEWS_QUERIES:
        raise AssertionError(f"Expected {EXPECTED_REVIEWS_QUERIES} reviews queries, got {len(reviews_qs)}")
    used = {q["case_id"] for q in products_qs + reviews_qs}
    if used != set(mappings):
        raise ValueError(f"Unused predicate mappings: {sorted(set(mappings) - used)}")
    return products_qs, reviews_qs


def main() -> None:
    products_df, reviews_df = load_sf1000_tables()
    prod_corpus = OUT_DIR / "products" / "corpus" / "documents.jsonl"
    prod_records = OUT_DIR / "products" / "corpus" / "records.jsonl"
    rev_corpus = OUT_DIR / "reviews" / "corpus" / "documents.jsonl"
    rev_records = OUT_DIR / "reviews" / "corpus" / "records.jsonl"
    write_products_corpus(products_df, prod_corpus, prod_records)
    write_reviews_corpus(reviews_df, rev_corpus, rev_records)

    gt_index = load_ground_truth_index()
    prod_qs, rev_qs = export_queries(gt_index)

    prod_q_path = OUT_DIR / "products" / "queries" / "single.json"
    rev_q_path = OUT_DIR / "reviews" / "queries" / "single.json"
    prod_q_path.parent.mkdir(parents=True, exist_ok=True)
    rev_q_path.parent.mkdir(parents=True, exist_ok=True)
    prod_q_path.write_text(json.dumps(prod_qs, ensure_ascii=False, indent=2), encoding="utf-8")
    rev_q_path.write_text(json.dumps(rev_qs, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "scale_factor": SCALE_FACTOR,
        "semceb_commit": git_head(SEMCEB_REPO_DIR),
        "semstats_commit": git_head(SEMSTATS_REPO_DIR),
        "predicate_mapping_sha256": sha256_file(PREDICATES_PATH),
        "products": {
            "rows": len(products_df),
            "queries": len(prod_qs),
            "corpus_sha256": sha256_file(prod_corpus),
            "records_sha256": sha256_file(prod_records),
            "queries_sha256": sha256_file(prod_q_path),
        },
        "reviews": {
            "rows": len(reviews_df),
            "queries": len(rev_qs),
            "corpus_sha256": sha256_file(rev_corpus),
            "records_sha256": sha256_file(rev_records),
            "queries_sha256": sha256_file(rev_q_path),
        },
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
