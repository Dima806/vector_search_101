"""Exact k-nearest-neighbour search: compare the query with every stored vector.

Perfectly correct — it defines the ground truth every other index is scored against — and
perfectly unscalable: each query costs ``N`` distance evaluations, so query time grows in
lockstep with the database.
"""

import numpy as np

from src.evaluation.memory import exact_bytes
from src.indexes import FloatArray, IdArray, SearchStats, as_float32, select_topk
from src.metrics.distance import Metric, normalise, pairwise, sq_norms


class ExactIndex:
    """Brute force: all distances, then the k smallest (``argpartition``, not a full sort)."""

    name = "exact"

    def __init__(self, metric: Metric = "l2", block_mb: float = 256.0) -> None:
        self.metric: Metric = metric
        self.block_mb = block_mb
        self._xb: FloatArray = np.zeros((0, 0), dtype=np.float32)
        self._norms: FloatArray = np.zeros(0, dtype=np.float32)
        self.last_stats = SearchStats.zeros(0)

    @property
    def ntotal(self) -> int:
        return int(self._xb.shape[0])

    @property
    def d(self) -> int:
        return int(self._xb.shape[1])

    @property
    def xb(self) -> FloatArray:
        """The stored vectors (normalised if the metric is cosine)."""
        return self._xb

    def train(self, xb: FloatArray) -> None:
        """Nothing to learn: exact search has no parameters."""

    def add(self, xb: FloatArray) -> None:
        x = as_float32(xb)
        if self.metric == "cosine":
            x = normalise(x)
        self._xb = x.copy() if self.ntotal == 0 else np.vstack([self._xb, x])
        self._norms = sq_norms(self._xb)

    def search(self, xq: FloatArray, k: int) -> tuple[FloatArray, IdArray]:
        """Distances to all N vectors, in query blocks of at most ``block_mb`` megabytes."""
        q = as_float32(xq)
        if self.metric == "cosine":
            q = normalise(q)
        nq, n = q.shape[0], self.ntotal
        rows = max(1, int(self.block_mb * 2**20 / (4 * max(n, 1))))
        out_d = np.empty((nq, k), dtype=np.float32)
        out_i = np.empty((nq, k), dtype=np.int64)
        for start in range(0, nq, rows):
            stop = min(start + rows, nq)
            block = pairwise(q[start:stop], self._xb, self.metric, xb_norms=self._norms)
            out_d[start:stop], out_i[start:stop] = select_topk(block, k)
        self.last_stats = SearchStats(ndis=np.full(nq, float(n)), n_adc=np.zeros(nq))
        return out_d, out_i

    @property
    def nbytes(self) -> int:
        return exact_bytes(self.ntotal, self.d if self.ntotal else 0)
