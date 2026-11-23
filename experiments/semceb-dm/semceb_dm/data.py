"""Reuse the pinned official loader and parser; keep evaluation truth separate."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "fec6bde7d1b23fc751f8701872fafda777a63942"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_query(repo, query_id):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError("SemCEB checkout differs from validated revision")
    sys.path.insert(0, str(repo / "src"))
    from semceb.queries.query_specification import QuerySpecification
    rows = [json.loads(line) for line in (repo / "benchmark_queries/queries.jsonl").read_text().splitlines() if line.strip()]
    raw = next(row for row in rows if row["id"] == query_id)
    query = QuerySpecification.from_dict(raw)
    if len(query.datasets) != 2:
        raise ValueError("First prototype supports two-input joins only")
    if "image" in query.filter.lower():
        raise ValueError("Image queries unsupported by this text-only prototype")
    return query


def load_tables(repo, data_dir, query, sf):
    if sf <= 0:
        raise ValueError("scale_factor must be positive")
    from semceb.data.loader import DataLoader
    loader = DataLoader()
    loader.folderpath_datasets_data = data_dir.parent
    tables = loader.load([d.table_ref for d in query.datasets], sf)
    # Stable, compact row identities for traces; source order remains unchanged.
    ids = {}
    for name, table in tables.items():
        cols = ["parent_asin"] if "products_" in name else ["asin", "user_id", "parent_asin", "timestamp_ms"]
        ids[name] = table[cols].astype(str).to_dict("records")
    return tables, ids


def render(query, rows):
    def substitute(match):
        alias, col = match.group(1).split(".", 1)
        if alias not in rows:
            raise ValueError(f"Unknown query alias {alias}")
        return str(rows[alias][col])
    return re.sub(r"\{([^{}]+)\}", substitute, query.filter)


def reference_cardinality(repo, query, sf):
    """Evaluator-only access: full key comparison, not query-id or substring matching."""
    from semceb.queries.template_parser import QueryTemplatePartType
    aliases = [d.alias for d in query.datasets]
    if len(aliases) != 2:
        raise ValueError("Expected two referenced aliases")
    sides = dict(zip(aliases, ["left", "right"]))
    formatted = "".join(part.value if part.type == QueryTemplatePartType.TEXT else "{" + part.value.column_name + ":" + sides[part.value.dataset_ref] + "}" for part in query.filter_parsed.parts)
    import tomllib
    conf = tomllib.loads((repo / "config.toml").read_text())["general"]["ground_truth"]
    fields = {"model_name": conf["model_name"], "system_prompt": conf["system_prompt"], "query_type": "join", "datasets": ", ".join(f"{d.alias}:{d.table_ref}" for d in query.datasets), "scale_factor": sf, "query": formatted}
    key = " | ".join(f"{k}='{v}'" for k, v in fields.items())
    path = repo / f"benchmark_queries/ground_truth_cache_{conf['model_name']}_sf{sf}.json"
    if not path.exists():
        return None
    item = json.loads(path.read_text()).get(key)
    return None if item is None else {**item, "model": conf["model_name"], "cache_sha256": sha256(path)}
