"""Speed in two currencies that never share an axis (CLAUDE.md §8).

1. **Work** — vectors scored per query (``SearchStats.work``). Implementation-independent:
   every from-scratch claim and every test uses it. Work speedup = ``N / work``.
2. **Wall-clock** — median per-query latency after a warm-up, with every engine pinned to the
   same thread count. Hardware-specific, so it is reported as speedup vs exact on the same
   engine, and only for compiled or vectorised code: a pure-Python search loop measures the
   interpreter, not the algorithm.
"""

import os
import platform
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from src.evaluation.recall import recall_at_k
from src.indexes import FloatArray, IdArray, Index
from src.reference.compare import library_versions, set_threads


@contextmanager
def pinned_threads(n: int) -> Generator[None]:
    """Limit BLAS (numpy/scikit-learn) and faiss to ``n`` threads inside the block."""
    with threadpool_limits(limits=n):
        set_threads(n)
        try:
            yield
        finally:
            set_threads(os.cpu_count() or 1)


def median_latency_ms(
    index: Index, xq: FloatArray, k: int, warmup: int = 3, max_queries: int | None = None
) -> float:
    """Median wall-clock time of single-query searches, in milliseconds."""
    queries = xq if max_queries is None else xq[:max_queries]
    for i in range(min(warmup, queries.shape[0])):
        index.search(queries[i : i + 1], k)
    samples = np.empty(queries.shape[0])
    for i in range(queries.shape[0]):
        start = time.perf_counter()
        index.search(queries[i : i + 1], k)
        samples[i] = time.perf_counter() - start
    return float(np.median(samples) * 1e3)


def evaluate(
    index: Index,
    xq: FloatArray,
    gt_ids: IdArray,
    k: int,
    *,
    timed: bool = False,
    warmup: int = 3,
    max_queries: int | None = None,
) -> dict[str, float]:
    """Recall and work on all queries; optionally the median single-query latency too."""
    _, ids = index.search(xq, k)
    stats = index.last_stats
    out = {
        "recall": recall_at_k(ids, gt_ids, k),
        "ndis": float(np.nanmean(stats.ndis)) if not np.isnan(stats.ndis).all() else np.nan,
        "n_adc": float(np.nanmean(stats.n_adc)) if not np.isnan(stats.n_adc).all() else np.nan,
    }
    out["work"] = out["ndis"] + out["n_adc"]
    out["latency_ms"] = (
        median_latency_ms(index, xq, k, warmup=warmup, max_queries=max_queries)
        if timed
        else np.nan
    )
    out["qps"] = 1e3 / out["latency_ms"] if timed else np.nan
    return out


def timed_build(index: Index, xb: FloatArray) -> float:
    """Train + add, returning wall-clock seconds."""
    start = time.perf_counter()
    index.train(xb)
    index.add(xb)
    return time.perf_counter() - start


def machine_info(threads: int) -> dict[str, Any]:
    """The line every timing table carries: where and how it was measured."""
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        names = [
            ln.split(":", 1)[1].strip()
            for ln in cpuinfo.read_text().splitlines()
            if ln.startswith("model name")
        ]
        cpu = names[0] if names else cpu
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "threads": threads,
        "python": platform.python_version(),
        "numpy": np.__version__,
        **library_versions(),
    }
