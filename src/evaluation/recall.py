"""Recall against exact ground truth — the only place a recall number is computed.

``k-recall@k = |approx_k ∩ true_k| / k``, averaged over queries (the ANN-Benchmarks
definition). The miss rate ``1 - recall`` is what an approximate index trades for speed or
memory; it is a dial, not a bug, and it is unknown until you measure it.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.indexes import FloatArray, Index
from src.indexes.exact import ExactIndex


def per_query_recall(pred_ids: NDArray, gt_ids: NDArray, k: int) -> NDArray[np.float64]:
    """Recall of each query: share of its k true neighbours found among the k returned.

    Padding ids (-1) never match, because ground-truth ids are non-negative.
    """
    pred = np.asarray(pred_ids)[:, :k]
    true = np.asarray(gt_ids)[:, :k]
    if pred.shape[1] < k or true.shape[1] < k:
        raise ValueError(f"need at least k={k} ids per query")
    hits = (pred[:, :, None] == true[:, None, :]).any(axis=2).sum(axis=1)
    return hits / k


def recall_at_k(pred_ids: NDArray, gt_ids: NDArray, k: int) -> float:
    """Mean k-recall@k over queries."""
    return float(per_query_recall(pred_ids, gt_ids, k).mean())


@dataclass
class RecallEstimate:
    """Recall measured on a sample of queries, with a normal-approximation 95% interval."""

    mean: float
    ci_low: float
    ci_high: float
    n_queries: int
    k: int


def recall_on_sample(
    index: Index,
    xb: FloatArray,
    xq: FloatArray,
    k: int = 10,
    n_sample: int = 200,
    seed: int = 0,
) -> RecallEstimate:
    """Recall-test any index on a random sample of queries against brute force.

    The habit notebook 06 argues for: before trusting an index (or a vector database), run
    exact search on a few hundred queries and measure what it actually returns. ``xb`` must be
    the vectors the index was built on, in the same order (ids are row numbers).
    """
    rng = np.random.default_rng(seed)
    take = min(n_sample, xq.shape[0])
    sample = xq[np.sort(rng.choice(xq.shape[0], size=take, replace=False))]
    exact = ExactIndex()
    exact.add(xb)
    _, true_ids = exact.search(sample, k)
    _, pred_ids = index.search(sample, k)
    per_query = per_query_recall(pred_ids, true_ids, k)
    mean = float(per_query.mean())
    half = 1.96 * float(per_query.std(ddof=1)) / math.sqrt(take) if take > 1 else 0.0
    return RecallEstimate(
        mean=mean,
        ci_low=max(0.0, mean - half),
        ci_high=min(1.0, mean + half),
        n_queries=take,
        k=k,
    )
