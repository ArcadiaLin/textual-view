#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_semstats.sh products run1 30
#   bash scripts/run_semstats.sh reviews  run1 60

TABLE="${1:?Usage: $0 <products|reviews> [run_name] [label_minutes]}"
RUN_NAME="${2:-run1}"
LABEL_MINUTES="${3:-30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
SEMSTATS_RUN="${REPO_ROOT}/references/repos/SemanticCardinalityEstimation/src/run.py"

CONFIG="${EXP_DIR}/configs/${TABLE}_sf1000.yaml"
QUERIES="${EXP_DIR}/data/${TABLE}/queries/single.json"
RECORDS="${EXP_DIR}/data/${TABLE}/corpus/records.jsonl"
WS_RUN="${EXP_DIR}/workspace/${TABLE}_sf1000/runs/${RUN_NAME}"

export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

cd "${EXP_DIR}"

if [[ ! -d "${EXP_DIR}/models/all-MiniLM-L6-v2" ]]; then
  echo "ERROR: Missing local Top2Vec model at ${EXP_DIR}/models/all-MiniLM-L6-v2" >&2
  exit 1
fi

# Freeze & verify run provenance manifest before reusing or creating artifacts
mkdir -p "${WS_RUN}"
uv run python - "${EXP_DIR}" "${CONFIG}" "${WS_RUN}/run_manifest.json" "${TABLE}" "${RUN_NAME}" "${LABEL_MINUTES}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path

exp_dir, cfg_path, manifest_path, table, run_name, label_minutes = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5], float(sys.argv[6])
)

def sha256(p: Path):
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

current = {
    "table": table,
    "run_name": run_name,
    "config_sha256": sha256(cfg_path),
    "uv_lock_sha256": sha256(exp_dir / "uv.lock"),
    "data_manifest_sha256": sha256(exp_dir / "data" / "manifest.json"),
    "top2vec_manifest_sha256": sha256(exp_dir / "models" / "all-MiniLM-L6-v2.manifest.json"),
}
if manifest_path.exists():
    prev = json.loads(manifest_path.read_text(encoding="utf-8"))
    for k in ("config_sha256", "uv_lock_sha256", "data_manifest_sha256", "top2vec_manifest_sha256"):
        if prev.get(k) != current.get(k):
            raise SystemExit(
                f"ERROR: {manifest_path} mismatch on {k}: previous={prev.get(k)} != current={current.get(k)}. "
                "Use a new --run name instead of reusing stale artifacts."
            )
    print(f"[manifest] Verified matching frozen run manifest: {manifest_path}")
else:
    current["label_minutes_initial"] = label_minutes
    manifest_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    (manifest_path.parent / "config_snapshot.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[manifest] Frozen new run manifest: {manifest_path}")
PY

run_stage() {
  echo ">>> [$TABLE/$RUN_NAME] $*"
  uv run python "${SEMSTATS_RUN}" --config "${CONFIG}" --run "${RUN_NAME}" "$@"
}

if [[ ! -f "${EXP_DIR}/workspace/${TABLE}_sf1000/doc_embeddings.npy" ]]; then
  run_stage embed
fi
if [[ ! -f "${WS_RUN}/catalog/semantic_catalog.json" ]]; then
  run_stage build-catalog
fi
if [[ ! -f "${WS_RUN}/index/index.json" ]]; then
  run_stage build-index
fi
if [[ ! -f "${WS_RUN}/query_gen/queries.json" ]]; then
  run_stage gen-queries
fi
if [[ ! -f "${WS_RUN}/snapshots/s1/data_single.json" ]]; then
  rm -rf "${WS_RUN}/snapshots/s1"
  run_stage label --minutes "${LABEL_MINUTES}"
  run_stage snapshot --name s1
fi
if [[ ! -f "${WS_RUN}/models/judge/meta.json" ]]; then
  rm -rf "${WS_RUN}/models/judge"
  run_stage train --snapshot s1 --name judge --device cuda
fi
if [[ ! -f "${WS_RUN}/estimates/eval_mlp/avg/estimates_single.json" ]]; then
  rm -rf "${WS_RUN}/estimates/eval_mlp"
  run_stage estimate --model judge --name eval_mlp --sets single
fi
if [[ ! -f "${WS_RUN}/estimates/eval_llm_scoped/estimates_single.json" ]]; then
  rm -rf "${WS_RUN}/estimates/eval_llm_scoped"
  echo ">>> [$TABLE/$RUN_NAME] scoped LLM judge on eval_mlp plans_single.json"
  uv run python "${EXP_DIR}/scripts/run_scoped_llm_judge.py" \
    --config "${CONFIG}" \
    --plans "${WS_RUN}/estimates/eval_mlp/plans_single.json" \
    --queries "${QUERIES}" \
    --records "${RECORDS}" \
    --out "${WS_RUN}/estimates/eval_llm_scoped"
fi

echo ">>> Evaluating MLPJudge against SemCEB sf1000 ground truth"
uv run python "${EXP_DIR}/scripts/evaluate_semceb.py" \
  --estimates "${WS_RUN}/estimates/eval_mlp/avg/estimates_single.json" \
  --queries "${QUERIES}" \
  --out "${WS_RUN}/estimates/eval_mlp/semceb_eval"

echo ">>> Evaluating field-scoped LLM Judge (same plans) against SemCEB sf1000 ground truth"
uv run python "${EXP_DIR}/scripts/evaluate_semceb.py" \
  --estimates "${WS_RUN}/estimates/eval_llm_scoped/estimates_single.json" \
  --queries "${QUERIES}" \
  --out "${WS_RUN}/estimates/eval_llm_scoped/semceb_eval"
