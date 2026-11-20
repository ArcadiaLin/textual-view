"""
Setwise & Pointwise Re-ranking primitives and evaluation utilities for Decision Models
(Local AnyJev Tacit-9B & Google Gemini Decision 1 API) on the CLERC legal dataset.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import bm25s

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent.parent
COOKBOOK_CACHE_FILE = (
    PROJECT_ROOT
    / "references/repos/typesafe-public-examples/cookbooks/rerank_typesafe/rerank_typesafe_cache.json"
)
SCRATCH_DIR = EXP_DIR / "notebooks" / "_scratch"

NOUL_INSTRUCTIONS = (
    "The query excerpt comes from a US federal court opinion and was written "
    "immediately around a citation to a precedent; the citation itself has been "
    "removed. Could the candidate passage be from that cited precedent — does it "
    "establish the specific legal proposition the query excerpt invokes at its "
    "citation point?"
)
NOUL_TRUE_DESC = (
    "The candidate passage states or establishes the specific rule, standard, "
    "holding, or fact pattern that the query excerpt attributes to its removed "
    "citation."
)
NOUL_FALSE_DESC = (
    "The candidate passage is merely on a similar topic or doctrine; it does not "
    "supply the specific proposition the query excerpt relies on."
)

CHOICE_INSTRUCTIONS = (
    "The query excerpt comes from a US federal court opinion and was written "
    "immediately around a citation to a precedent; the citation itself has been "
    "removed. Which candidate passage is from that cited precedent and establishes "
    "the specific legal proposition the query excerpt invokes at its citation point?"
)


def load_env(env_path: Path | None = None) -> None:
    """Load environment variables from project root `.env` and `tmps/.env`."""
    paths = [env_path] if env_path else [PROJECT_ROOT / ".env", PROJECT_ROOT / "tmps/.env"]
    for p in paths:
        if p and p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_clerc_slice() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, list[str]]]:
    """Load the exact 40 queries and 3,565 passages from `rerank_typesafe` and build BM25 top-30 shortlists."""
    with open(COOKBOOK_CACHE_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    ds = raw["build_slice|170|40|seed=0"]
    corpus: dict[str, str] = ds["corpus"]
    queries = {q["qid"]: q["query"] for q in ds["queries"]}
    golds = {q["qid"]: q["gold"] for q in ds["queries"]}

    cids = list(corpus)
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([corpus[c] for c in cids], stopwords="en"))
    qids = list(queries)
    idxs, _ = retriever.retrieve(
        bm25s.tokenize([queries[q] for q in qids], stopwords="en"), k=30
    )
    candidates = {q: [cids[i] for i in idxs[row]] for row, q in enumerate(qids)}
    return corpus, queries, golds, candidates


class DiskCache:
    """Deterministic JSON disk cache for decision model API calls."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    @staticmethod
    def make_key(namespace: str, payload: Any) -> str:
        raw = json.dumps([namespace, payload], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class AnyJevBackend:
    """
    Local AnyJev Gateway (`POST http://127.0.0.1:8100/v1/decide`).
    Uses direct connection (`ProxyHandler({})`) so localhost never goes through HTTP proxy.
    """

    def __init__(self, endpoint: str | None = None, cache: DiskCache | None = None):
        self.endpoint = endpoint or os.environ.get("ANYJEV_ENDPOINT", "http://127.0.0.1:8100/v1/decide")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.cache = cache
        self.reset_stats()

    def reset_stats(self) -> None:
        self.calls = 0
        self.decisions = 0
        self.total_latency_s = 0.0
        self.cache_hits = 0

    def _post(self, payload: dict[str, Any], n_decisions: int = 1) -> dict[str, Any]:
        if self.cache is not None:
            ck = self.cache.make_key(f"anyjev:{self.endpoint}", payload)
            cached = self.cache.get(ck)
            if cached is not None:
                self.calls += 1
                self.decisions += n_decisions
                self.cache_hits += 1
                self.total_latency_s += float(cached.get("_latency_s", 0.0))
                return cached["body"]

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with self._opener.open(req, timeout=180.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        lat = time.perf_counter() - t0
        self.total_latency_s += lat
        self.calls += 1
        self.decisions += n_decisions
        if self.cache is not None:
            self.cache.set(ck, {"body": body, "_latency_s": lat})
        return body

    def score_pointwise_batch(self, query: str, cids: list[str], corpus: dict[str, str]) -> dict[str, float]:
        """Batch all 30 pointwise `yes_no` decisions into 1 gateway request."""
        question = (
            f"{NOUL_INSTRUCTIONS}\n"
            f"- Yes: {NOUL_TRUE_DESC}\n"
            f"- No: {NOUL_FALSE_DESC}"
        )
        items = [
            {
                "state": f"Query Excerpt:\n{query}\n\nCandidate Passage:\n{corpus[c]}",
                "question": question,
                "kind": "yes_no",
            }
            for c in cids
        ]
        res = self._post({"items": items}, n_decisions=len(items))
        decisions = res["decisions"]
        return {c: float(d["probs"]["Yes"]) for c, d in zip(cids, decisions)}

    def _build_choice_item(
        self, query: str, cids: list[str], corpus: dict[str, str], mode: str = "tagged"
    ) -> tuple[dict[str, Any], list[str]]:
        assert len(cids) <= 26, "AnyJev letter readout supports at most 26 options per Choice"
        if mode == "tagged":
            tags = [f"P{i+1:02d}" for i in range(len(cids))]
            tagged_block = "\n\n".join(f"[{t}] {corpus[c]}" for t, c in zip(tags, cids))
            state = f"Query Excerpt:\n{query}\n\nCandidate Passages:\n{tagged_block}"
            item = {
                "state": state,
                "question": CHOICE_INSTRUCTIONS,
                "options": tags,
                "kind": "choice",
            }
            return item, tags
        else:
            options = [f"[{c}] {corpus[c]}" for c in cids]
            state = f"Query Excerpt:\n{query}"
            item = {
                "state": state,
                "question": CHOICE_INSTRUCTIONS,
                "options": options,
                "kind": "choice",
            }
            return item, options

    def choice_setwise(
        self, query: str, cids: list[str], corpus: dict[str, str], mode: str = "tagged"
    ) -> dict[str, float]:
        item, keys = self._build_choice_item(query, cids, corpus, mode=mode)
        res = self._post(item, n_decisions=1)
        probs = res["probs"]
        return {c: float(probs.get(k, 0.0)) for k, c in zip(keys, cids)}

    def choice_setwise_batch(
        self, query: str, groups: list[list[str]], corpus: dict[str, str], mode: str = "tagged"
    ) -> list[dict[str, float]]:
        """Evaluate multiple Setwise Choice groups in a single batched gateway call."""
        items = []
        group_keys = []
        for g in groups:
            it, keys = self._build_choice_item(query, g, corpus, mode=mode)
            items.append(it)
            group_keys.append(keys)
        res = self._post({"items": items}, n_decisions=len(items))
        out = []
        for g, keys, d in zip(groups, group_keys, res["decisions"]):
            probs = d["probs"]
            out.append({c: float(probs.get(k, 0.0)) for k, c in zip(keys, g)})
        return out


class GeminiDecisionBackend:
    """
    Google Gemini Decision 1 API (`POST https://generativelanguage.googleapis.com/v1beta/decisions`).
    Explicitly routes through HTTP/HTTPS proxy on port 10809 (`http://127.0.0.1:10809`).
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str | None = None,
        proxy_url: str | None = None,
        cache: DiskCache | None = None,
    ):
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("X-goog-api-key")
        )
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY / X-goog-api-key in environment or .env")
        self.endpoint = endpoint or os.environ.get(
            "GEMINI_DECISION_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/decisions"
        )
        self.model = model or os.environ.get("GEMINI_DECISION_MODEL", "gemini-decision-1")
        self.proxy_url = proxy_url or os.environ.get("GEMINI_PROXY", "http://127.0.0.1:10809")
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": self.proxy_url, "https": self.proxy_url})
        )
        self.cache = cache
        self.reset_stats()

    def reset_stats(self) -> None:
        self.calls = 0
        self.decisions = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.cache_hits = 0

    def _post(self, payload: dict[str, Any], n_decisions: int = 1, retries: int = 5) -> dict[str, Any]:
        if self.cache is not None:
            ck = self.cache.make_key(f"gemini:{self.model}", payload)
            cached = self.cache.get(ck)
            if cached is not None:
                body = cached["body"]
                self.calls += 1
                self.decisions += n_decisions
                self.cache_hits += 1
                self.total_tokens += int((body.get("usage") or {}).get("total_input_tokens", 0))
                self.total_latency_s += float(cached.get("_latency_s", 0.0))
                return body

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
            "Connection": "close",
        }
        last_err: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
            t0 = time.perf_counter()
            try:
                with self._opener.open(req, timeout=120.0) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                lat = time.perf_counter() - t0
                self.total_latency_s += lat
                self.calls += 1
                self.decisions += n_decisions
                self.total_tokens += int((body.get("usage") or {}).get("total_input_tokens", 0))
                if self.cache is not None:
                    self.cache.set(ck, {"body": body, "_latency_s": lat})
                return body
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"Gemini Decisions API failed after {retries} attempts: {last_err}")

    def score_pointwise_batch(
        self, query: str, cids: list[str], corpus: dict[str, str], max_workers: int = 4
    ) -> dict[str, float]:
        """Cookbook baseline: 1 Noul API request per (query, candidate) pair."""
        question = {
            "type": "noul",
            "instructions": NOUL_INSTRUCTIONS,
            "criteria": {
                "true_description": NOUL_TRUE_DESC,
                "false_description": NOUL_FALSE_DESC,
            },
        }

        def _one(cid: str) -> tuple[str, float]:
            inp = f"query_excerpt: {query}\n\ncandidate_passage: {corpus[cid]}"
            body = self._post(
                {
                    "model": self.model,
                    "input": inp,
                    "questions": {"is_cited_source": question},
                },
                n_decisions=1,
            )
            return cid, float(body["answers"]["is_cited_source"]["noul"])

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pairs = list(pool.map(_one, cids))
        return dict(pairs)

    def score_pointwise_fanout_one_call(
        self, query: str, cids: list[str], corpus: dict[str, str]
    ) -> dict[str, float]:
        """
        Speculative fan-out: 1 API call with 30 Noul questions,
        each embedding one candidate passage in its instructions.
        """
        questions = {
            c: {
                "type": "noul",
                "instructions": f"{NOUL_INSTRUCTIONS}\n\nCandidate passage:\n{corpus[c]}",
                "criteria": {
                    "true_description": NOUL_TRUE_DESC,
                    "false_description": NOUL_FALSE_DESC,
                },
            }
            for c in cids
        }
        body = self._post(
            {
                "model": self.model,
                "input": f"Query Excerpt:\n{query}",
                "questions": questions,
            },
            n_decisions=len(cids),
        )
        return {c: float(body["answers"][c]["noul"]) for c in cids}

    def choice_setwise(
        self, query: str, cids: list[str], corpus: dict[str, str], mode: str = "criteria"
    ) -> dict[str, float]:
        """
        Evaluate a subset of candidates (2..255) with one Choice question.
        mode="criteria": passage text is placed in criteria[cid].
        mode="tagged": passages are tagged in `input`, and criteria[cid] references the tag.
        """
        if mode == "tagged":
            tagged_block = "\n\n".join(f"[{c}] {corpus[c]}" for c in cids)
            inp = f"Query Excerpt:\n{query}\n\nCandidate Passages:\n{tagged_block}"
            criteria = {c: f"Passage [{c}]" for c in cids}
        else:
            inp = f"Query Excerpt:\n{query}"
            criteria = {c: corpus[c] for c in cids}

        body = self._post(
            {
                "model": self.model,
                "input": inp,
                "questions": {
                    "best_passage": {
                        "type": "choice",
                        "instructions": CHOICE_INSTRUCTIONS,
                        "criteria": criteria,
                    }
                },
            },
            n_decisions=1,
        )
        probs = body["answers"]["best_passage"]["probabilities"]
        return {c: float(probs.get(c, 0.0)) for c in cids}


# ---------------------------------------------------------------------------
# Re-ranking Algorithms
# ---------------------------------------------------------------------------

def rerank_pointwise(backend, query: str, cids: list[str], corpus: dict[str, str]) -> list[str]:
    scores = backend.score_pointwise_batch(query, cids, corpus)
    return sorted(cids, key=lambda c: (-scores[c], cids.index(c)))


def rerank_pointwise_fanout(
    backend: GeminiDecisionBackend, query: str, cids: list[str], corpus: dict[str, str]
) -> list[str]:
    scores = backend.score_pointwise_fanout_one_call(query, cids, corpus)
    return sorted(cids, key=lambda c: (-scores[c], cids.index(c)))


def rerank_oneshot_choice(
    backend, query: str, cids: list[str], corpus: dict[str, str], mode: str = "criteria"
) -> list[str]:
    probs = backend.choice_setwise(query, cids, corpus, mode=mode)
    return sorted(cids, key=lambda c: (-probs[c], cids.index(c)))


def rerank_setwise_tournament(
    backend,
    query: str,
    cids: list[str],
    corpus: dict[str, str],
    group_size: int = 5,
    top_per_group: int = 2,
    mode: str = "tagged",
) -> list[str]:
    """
    2-stage Setwise Tournament:
    Stage 1: Split 30 candidates into groups of `group_size` (e.g. 6 groups of 5).
             Select top `top_per_group` per group (e.g. top 2 -> 12 finalists).
             On AnyJev, all 6 groups run in a single batched gateway call!
    Stage 2: Run 1 final Setwise Choice over the finalists.
    """
    groups = [cids[i : i + group_size] for i in range(0, len(cids), group_size)]
    stage1_probs: dict[str, float] = {}
    finalists: list[str] = []

    if hasattr(backend, "choice_setwise_batch"):
        multi_groups = [g for g in groups if len(g) > 1]
        batch_probs = backend.choice_setwise_batch(query, multi_groups, corpus, mode=mode) if multi_groups else []
        it = iter(batch_probs)
        for g in groups:
            if len(g) == 1:
                stage1_probs[g[0]] = 1.0
                finalists.append(g[0])
            else:
                p = next(it)
                stage1_probs.update(p)
                ordered_g = sorted(g, key=lambda c: (-p[c], g.index(c)))
                finalists.extend(ordered_g[:top_per_group])
    else:
        for g in groups:
            if len(g) == 1:
                stage1_probs[g[0]] = 1.0
                finalists.append(g[0])
            else:
                p = backend.choice_setwise(query, g, corpus, mode=mode)
                stage1_probs.update(p)
                ordered_g = sorted(g, key=lambda c: (-p[c], g.index(c)))
                finalists.extend(ordered_g[:top_per_group])

    final_probs = backend.choice_setwise(query, finalists, corpus, mode=mode)
    ordered_finalists = sorted(finalists, key=lambda c: (-final_probs[c], finalists.index(c)))
    finalist_set = set(ordered_finalists)
    rest = sorted(
        [c for c in cids if c not in finalist_set],
        key=lambda c: (-stage1_probs[c], cids.index(c)),
    )
    return ordered_finalists + rest


def rerank_coarse_to_fine(
    backend,
    query: str,
    cids: list[str],
    corpus: dict[str, str],
    top_m: int = 10,
    mode: str = "tagged",
    use_fanout: bool = False,
) -> list[str]:
    """
    Coarse-to-Fine Cascade (inspired by `infinite_cardinality_choice`):
    Stage 1: Pointwise Noul scores all 30 candidates and keeps top `top_m` (e.g., 10).
    Stage 2: Setwise Choice over the top `top_m` candidates directly compares the hardest contenders.
    """
    if use_fanout and hasattr(backend, "score_pointwise_fanout_one_call"):
        pw_scores = backend.score_pointwise_fanout_one_call(query, cids, corpus)
    else:
        pw_scores = backend.score_pointwise_batch(query, cids, corpus)
    pw_ordered = sorted(cids, key=lambda c: (-pw_scores[c], cids.index(c)))
    top_subset = pw_ordered[:top_m]
    rest = pw_ordered[top_m:]
    choice_probs = backend.choice_setwise(query, top_subset, corpus, mode=mode)
    top_reordered = sorted(top_subset, key=lambda c: (-choice_probs[c], top_subset.index(c)))
    return top_reordered + rest


def rerank_setwise_sliding_window(
    backend,
    query: str,
    cids: list[str],
    corpus: dict[str, str],
    window_size: int = 5,
    step: int = 3,
    passes: int = 1,
    mode: str = "tagged",
) -> list[str]:
    """Setwise Sliding Window (`listwise.likelihood` from Zhuang et al., 2024)."""
    arr = list(cids)
    n = len(arr)
    for _ in range(passes):
        end = n
        while end > 1:
            start = max(0, end - window_size)
            window = arr[start:end]
            if len(window) > 1:
                p = backend.choice_setwise(query, window, corpus, mode=mode)
                arr[start:end] = sorted(window, key=lambda c: (-p[c], window.index(c)))
            if start == 0:
                break
            end -= step
    return arr


def rerank_setwise_heap(
    backend,
    query: str,
    cids: list[str],
    corpus: dict[str, str],
    num_child: int = 3,
    top_k: int = 10,
    mode: str = "tagged",
) -> list[str]:
    """Setwise Max-Heap Sort (Zhuang et al., 2024 §3.2) extracting top-k."""
    arr = list(cids)
    n = len(arr)

    def heapify(size: int, root: int) -> None:
        children = [num_child * root + j + 1 for j in range(num_child) if num_child * root + j + 1 < size]
        if not children:
            return
        subset_indices = [root] + children
        subset_cids = [arr[idx] for idx in subset_indices]
        probs = backend.choice_setwise(query, subset_cids, corpus, mode=mode)
        best_cid = max(subset_cids, key=lambda c: probs[c])
        best_pos = subset_indices[subset_cids.index(best_cid)]
        if best_pos != root:
            arr[root], arr[best_pos] = arr[best_pos], arr[root]
            heapify(size, best_pos)

    last_internal = (n - 2) // num_child
    for i in range(last_internal, -1, -1):
        heapify(n, i)

    ranked_top: list[str] = []
    size = n
    for _ in range(min(top_k, n)):
        ranked_top.append(arr[0])
        size -= 1
        if size <= 0:
            break
        arr[0] = arr[size]
        if size > 1:
            heapify(size, 0)

    seen = set(ranked_top)
    rest = [c for c in arr[:size] if c not in seen]
    return ranked_top + rest


def evaluate_rankings(
    rankings: dict[str, list[str]], golds: dict[str, str], qids: list[str]
) -> dict[str, Any]:
    ranks = [rankings[q].index(golds[q]) + 1 for q in qids]
    n = len(qids)
    import math
    ndcg10 = sum((1.0 / math.log2(r + 1)) if r <= 10 else 0.0 for r in ranks) / n
    return {
        "n_queries": n,
        "top1": sum(r == 1 for r in ranks) / n,
        "top5": sum(r <= 5 for r in ranks) / n,
        "top10": sum(r <= 10 for r in ranks) / n,
        "mrr": sum(1.0 / r for r in ranks) / n,
        "ndcg10": ndcg10,
        "mean_rank": sum(ranks) / n,
        "ranks": {q: r for q, r in zip(qids, ranks)},
    }
