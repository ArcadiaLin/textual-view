"""Run from this project using python -m semceb_dm.runner."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv
from .client import DecisionClient, digest
from .data import load_query, load_tables, reference_cardinality, sha256
from .estimators import pair_sample, state_proxy, q_error

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=int, default=63)
    parser.add_argument("--scale-factor", type=int, default=10)
    parser.add_argument("--method", choices=["pair-sample", "state-proxy"], default="pair-sample")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-requests", type=int, default=2)
    parser.add_argument("--state-config", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "references/datasets/semceb/amazon-reviews")
    parser.add_argument("--repo", type=Path, default=ROOT / "references/repos/SemCEB")
    parser.add_argument("--output", type=Path, required=True, help="New run directory; refuses overwrite")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and reference coverage without model calls")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("samples must be positive")
    if args.max_requests < 0:
        parser.error("max-requests cannot be negative")
    args.output.mkdir(parents=True, exist_ok=False)
    load_dotenv(ROOT / ".env", override=False)
    query = load_query(args.repo, args.query)
    started = time.perf_counter()
    tables, identities = load_tables(args.repo, args.data_dir, query, args.scale_factor)
    preparation_seconds = time.perf_counter() - started
    state = json.loads(args.state_config.read_text()) if args.state_config else None
    if args.method == "state-proxy" and state is None:
        parser.error("state-proxy requires --state-config")
    estimated_requests = min(args.samples, len(tables[query.datasets[0].table_ref]) * len(tables[query.datasets[1].table_ref]))
    if state:
        estimated_requests = sum(len(tables[d.table_ref]) for d in query.datasets) + len(state["left"]["states"]) * len(state["right"]["states"])
    manifest = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "query": query.to_dict(), "state_config": state, "row_counts": {k: len(v) for k, v in tables.items()}, "row_identity_hashes": {k: digest(v) for k, v in identities.items()}, "preparation_seconds": preparation_seconds, "cold_requests_required": estimated_requests, "semceb_commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(), "data_files": [], "source_sha256": {str(f.relative_to(Path(__file__).parent.parent)): sha256(f) for f in Path(__file__).parent.glob('*.py')}, "lock_sha256": sha256(Path(__file__).parent.parent / 'uv.lock')}
    for f in sorted(args.data_dir.glob("*.parquet")):
        manifest["data_files"].append({"path": str(f), "bytes": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns})
    manifest["decision_model"] = os.environ.get("GEMINI_DECISION_MODEL", "gemini-decision-1")
    manifest["repository_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    manifest["semceb_source_sha256"] = {name: sha256(args.repo / name) for name in ["src/semceb/data/loader.py", "src/semceb/queries/query_specification.py", "src/semceb/queries/template_parser.py", "benchmark_queries/queries.jsonl", "config.toml"]}
    manifest["data_provenance_note"] = "File size/mtime and ordered slice identity hashes; not full content checksums."
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        truth = reference_cardinality(args.repo, query, args.scale_factor)
        result = {"status": "dry-run", "row_counts": manifest["row_counts"], "cold_requests_required": estimated_requests, "reference_available": truth is not None}
    else:
        client = DecisionClient(ROOT / "cache/semceb-dm/decisions", args.output / "requests.jsonl", args.max_requests)
        start = time.perf_counter()
        try:
            if args.method == "pair-sample":
                result = pair_sample(query, tables, identities, client, args.samples, args.seed, args.output / "samples.jsonl")
            else:
                result = state_proxy(query, tables, client, state, args.output / "states.jsonl")
            result["online_seconds"] = time.perf_counter() - start
            result["api"] = client.stats
            # Evaluate only after the estimator returns; truth never enters model prompts.
            truth = reference_cardinality(args.repo, query, args.scale_factor)
            result["reference"] = truth
            result["integer_estimate"] = max(0, min(result["population"], int(result["soft_estimate"] + .5)))
            if truth is not None:
                result["q_error_raw"] = q_error(result["soft_estimate"], truth["cardinality"])
                result["q_error_integer"] = q_error(result["integer_estimate"], truth["cardinality"])
            result["status"] = "exploratory-smoke"
        except Exception:
            (args.output / "failure.json").write_text(json.dumps({"api": client.stats, "online_seconds": time.perf_counter()-start}, indent=2))
            raise
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
