# SemStats Reproduction on SemCEB (`sf1000`, Pure-Text Filters)

This experiment directory adapts the unmodified SemStats repository
(`references/repos/SemanticCardinalityEstimation`, commit `0ee9952115d1a3f40eece0df28d8b4b182a9da2b`)
to SemCEB (`references/repos/SemCEB`, commit `fec6bde7d1b23fc751f8701872fafda777a63942`)
on the `sf1000` pure-text `AI_FILTER` slices:

- **`products` (`sf1000`)**: 1,000 rows, 30 pure-text filter queries.
- **`reviews` (`sf1000`)**: 10,335 rows, 19 pure-text filter queries.

> Image/multimodal filters (13 queries) and semantic joins (40 queries) are outside
> the released SemStats codebase and are not included here.

## Model Endpoints & Pinned Weights

- **Chat LLM**: `http://localhost:30000/v1`, model `qwen3.8-27b` (`infra/sglang-qwen38-27b`), with
  `chat_template_kwargs: {"enable_thinking": false}` so 1-token binary verdicts do not emit reasoning tokens.
- **Embedding**: `http://localhost:8002/v1`, model `qwen3-embedding-8b` (`infra/vllm-qwen3-embed-8b`).
- **Top2Vec local model**: `sentence-transformers/all-MiniLM-L6-v2` pinned to revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, stored at `./models/all-MiniLM-L6-v2` (note: upstream
  `catalog.py` resolves `model_path` relative to CWD, so commands must run from this experiment directory).
- **Index mode**: `multi` (multi-label teacher + per-node XGBoost).

## Evaluation & Methodological Notes

- **Nominal vs. actual sampling budget**: `sample.budget_ratio: 0.01` is only a *nominal* budget
  (10 on `products`, 103 on `reviews`), because upstream `sample.py` enforces `MIN_SHARE = 6` per
  selected stratum. `evaluate_semceb.py` reports actual `sample_size` (draws) and `n_unique`
  (distinct judgements) per query and in aggregate.
- **Field scope**:
  - `eval_mlp` (`MLPJudge`) operates on whole-record document embeddings (`documents.jsonl`), which is
    inherent to SemStats's single-corpus design.
  - `eval_llm_scoped` (`scripts/run_scoped_llm_judge.py`) reuses the exact `plans_single.json` from
    `eval_mlp`, but feeds **only the columns referenced by each SemCEB query's `raw_filter`** from
    `records.jsonl`, avoiding field leakage (e.g., Q009 sees only `features_json`, not `details_json`).
- **q-error definition**: `evaluate_semceb.py` reports **unsigned q-error clamped at 1.0**:
  `max(max(1, est), max(1, true)) / min(max(1, est), max(1, true))` (always `>= 1.0`, safe for
  `true = 0` such as Q056). Note that SemCEB's native reporting uses *signed* q-error (negative for
  underestimation).
- **Reproducibility & resume safety**: `scripts/run_semstats.sh` freezes a `run_manifest.json`
  (config, `uv.lock`, data manifest, Top2Vec manifest SHA256) and aborts if a resumed run has
  mismatched hashes; it also checks final output files (not just directory existence) and clears
  incomplete estimate/snapshot directories before retrying.

## Quick Start (After Review)

```bash
cd experiments/semstats-semceb-sf1000

# 1. Install dependencies & prepare data/model
uv sync
uv run python scripts/prepare_semceb_sf1000.py
uv run python scripts/download_top2vec_model.py

# 2. Run products (30 min label budget) and/or reviews (60 min label budget)
bash scripts/run_semstats.sh products run1 30
bash scripts/run_semstats.sh reviews run1 60
```
