"""Field-scoped direct LLM judge on saved SemStats sampling plans.

Reuses the exact plans_single.json produced by SemStats (same strata, sampled doc_ids,
and extrapolation weights), but formats each (query, doc) prompt using ONLY the exact
columns referenced by that SemCEB query's raw_filter template (preventing field leakage).
"""
import argparse
import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

EXP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "references" / "repos" / "SemanticCardinalityEstimation" / "src"))

from data import sha256_file, write_json  # noqa: E402
from judge import ATTEMPTS, REPAIR_MESSAGE, parse_binary  # noqa: E402
from llm import LLM  # noqa: E402

SYSTEM_PROMPT = (
    "You will be shown a claim or filter condition together with the exact record fields "
    "referenced by the query.\n"
    "Determine whether the claim is true for the given fields.\n"
    "Base your judgment strictly on the provided fields.\n"
    "Answer with only 0 or 1."
)


def load_records(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["doc_id"]] = rec["fields"]
    return out


def format_scoped_prompt(raw_filter: str, fields: dict[str, str]) -> str:
    return raw_filter.format(**fields) + "\n\nAnswer with only 0 or 1."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to SemStats YAML config")
    ap.add_argument("--plans", required=True, help="Path to saved plans_single.json from eval_mlp")
    ap.add_argument("--queries", required=True, help="Path to data/<table>/queries/single.json")
    ap.add_argument("--records", required=True, help="Path to data/<table>/corpus/records.jsonl")
    ap.add_argument("--out", required=True, help="Output directory for estimates_single.json")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plans_path = Path(args.plans)
    plans = json.loads(plans_path.read_text(encoding="utf-8"))["rows"]
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    records = load_records(Path(args.records))

    if len(plans) != len(queries):
        raise ValueError(f"Plan count {len(plans)} != query count {len(queries)}")
    for p, q in zip(plans, queries):
        if p["query"] != q["query"]:
            raise ValueError(f"Plan/query mismatch at {q['case_id']}")

    lc = cfg["llm"]
    cache_path = out_dir / "llm_scoped_cache.jsonl"
    llm = LLM(
        lc["base_url"],
        lc["model"],
        cache_path=cache_path,
        workers=lc.get("workers", 16),
        timeout=600,
        extra_body=lc.get("extra_body"),
    )

    by_doc = defaultdict(set)
    for qi, p in enumerate(plans):
        for g in p["groups"]:
            for doc_id in g["docs"]:
                by_doc[doc_id].add(qi)

    groups = [[(doc_id, qi) for qi in sorted(qis)] for doc_id, qis in sorted(by_doc.items())]
    total_unique = sum(len(g) for g in groups)
    total_draws = sum(len(g["docs"]) for p in plans for g in p["groups"])
    print(
        f"[scoped_llm] {len(queries)} queries, {total_draws} draws, "
        f"{total_unique} unique judgements, {len(groups)} distinct docs",
        flush=True,
    )

    t0 = time.time()
    lock = threading.Lock()
    stats = {"done": 0, "format_repairs": 0}
    audit_path = out_dir / "format_repairs.jsonl"

    def judge_one(item):
        doc_id, qi = item
        q = queries[qi]
        fields = records[doc_id]
        prompt = format_scoped_prompt(q["raw_filter"], fields)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        replies = []
        v = None
        for attempt in range(ATTEMPTS):
            raw = llm.chat(messages, temperature=0.0, max_tokens=1 if attempt == 0 else 8)[0]
            replies.append(raw)
            v = parse_binary(raw)
            if v is not None:
                break
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": REPAIR_MESSAGE},
            ]
        if v is None:
            raise ValueError(f"No 0/1 verdict in {ATTEMPTS} attempts ({q['case_id']}, {doc_id}): {replies!r}")
        with lock:
            stats["done"] += 1
            if attempt > 0:
                stats["format_repairs"] += 1
                with open(audit_path, "a", encoding="utf-8") as af:
                    af.write(
                        json.dumps(
                            {"case_id": q["case_id"], "doc_id": doc_id, "replies": replies, "verdict": v},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if stats["done"] % 200 == 0:
                print(f"  judged {stats['done']}/{total_unique}  {time.time() - t0:.0f}s", flush=True)
        return qi, doc_id, v

    verdicts = {}
    for res in llm.map_groups(groups, judge_one):
        for qi, doc_id, v in res:
            verdicts[(qi, doc_id)] = v

    rows = []
    for qi, (q, p) in enumerate(zip(queries, plans)):
        est = 0.0
        sampled = []
        for g in p["groups"]:
            docs = g["docs"]
            ws = g["w"]
            contribs = [verdicts[(qi, d)] for d in docs]
            est += float(np.dot(ws, contribs))
            sampled.extend(docs)
        rows.append({
            "case_id": q["case_id"],
            "query": q["query"],
            "predicates": list(q["predicates"]),
            "columns": list(q["columns"]),
            "estimate": float(est),
            "cand_size": p["cand_size"],
            "sample_size": len(sampled),
            "n_unique": len(set(sampled)),
            "n_strata": len(p["groups"]),
            "nodes": p["nodes"],
            "fallback": p["fallback"],
            "fallback_nodes": p["fallback_nodes"],
        })

    meta = {
        "judge": "scoped_llm",
        "llm": llm.model,
        "plans_path": str(plans_path),
        "plans_sha256": sha256_file(plans_path),
        "total_draws": total_draws,
        "total_unique_judgements": total_unique,
        "format_repairs": stats["format_repairs"],
        "llm_new_calls": llm.calls,
        "llm_cache_hits": llm.hits,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    write_json(out_dir / "verdicts_single.json", [[qi, d, int(v)] for (qi, d), v in sorted(verdicts.items())])
    write_json(out_dir / "estimates_single.json", {"meta": meta, "rows": rows}, indent=1)
    write_json(out_dir / "meta.json", meta, indent=1)
    print(f"[scoped_llm] done in {meta['elapsed_seconds']}s ({total_unique} unique judgements)", flush=True)


if __name__ == "__main__":
    main()
