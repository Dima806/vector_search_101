"""IVF + PQ: cluster to skip comparisons, compress to save memory — both levers at once.

The coarse quantizer is IVF's k-means. Inside each list, vectors are stored as PQ codes of
their *residual* ``r = x - c(x)`` (the offset from their centroid; faiss ``by_residual``),
which is smaller and easier to quantize than the vector itself. At query time the ADC table
is built for ``q - c_list`` of every probed list. Even at ``nprobe = nlist`` recall stays
below 1.0: that remaining gap is PQ's quantization error.
"""

import numpy as np

from src.evaluation.memory import ivfpq_bytes
from src.indexes import FloatArray, IdArray, QueryTrace, SearchStats, as_float32, select_topk
from src.indexes.ivf import assign_to_nearest, train_kmeans, training_sample
from src.indexes.pq import ProductQuantizer
from src.metrics.distance import sq_l2, sq_norms


class IVFPQIndex:
    """IVF coarse quantizer + PQ codes of residuals, scanned with per-list ADC tables."""

    name = "ivfpq"

    def __init__(
        self,
        nlist: int,
        n_subvectors: int,
        nbits: int = 8,
        nprobe: int = 1,
        train_points_per_centroid: int = 64,
        pq_train_size: int = 20_000,
        kmeans_max_iter: int = 20,
        seed: int = 0,
    ) -> None:
        self.nlist = nlist
        self.n_subvectors = n_subvectors
        self.nbits = nbits
        self.nprobe = nprobe
        self.train_points_per_centroid = train_points_per_centroid
        self.pq_train_size = pq_train_size
        self.kmeans_max_iter = kmeans_max_iter
        self.seed = seed
        self.centroids: FloatArray | None = None
        self._centroid_norms: FloatArray = np.zeros(0, dtype=np.float32)
        self.pq: ProductQuantizer | None = None
        self.list_ids: list[IdArray] = []
        self.list_codes: list[np.ndarray] = []  # each (n_j, n_subvectors) uint8
        self._ntotal = 0
        self._d = 0
        self.last_stats = SearchStats.zeros(0)

    @property
    def ntotal(self) -> int:
        return self._ntotal

    def train(self, xb: FloatArray) -> None:
        """k-means for the coarse centroids, then PQ codebooks on the training residuals."""
        x = as_float32(xb)
        self._d = x.shape[1]
        coarse_sample = training_sample(x, self.train_points_per_centroid * self.nlist, self.seed)
        self.centroids = train_kmeans(coarse_sample, self.nlist, self.kmeans_max_iter, self.seed)
        self._centroid_norms = sq_norms(self.centroids)
        pq_sample = training_sample(x, self.pq_train_size, self.seed + 1)
        residuals = pq_sample - self.centroids[assign_to_nearest(pq_sample, self.centroids)]
        self.pq = ProductQuantizer(
            self._d, self.n_subvectors, self.nbits, self.kmeans_max_iter, self.seed
        ).train(residuals)
        self.list_ids = [np.zeros(0, dtype=np.int64) for _ in range(self.nlist)]
        self.list_codes = [
            np.zeros((0, self.n_subvectors), dtype=np.uint8) for _ in range(self.nlist)
        ]

    def add(self, xb: FloatArray) -> None:
        if self.centroids is None or self.pq is None:
            raise RuntimeError("train() the IVFPQ index before add()")
        x = as_float32(xb)
        labels = assign_to_nearest(x, self.centroids)
        codes = self.pq.encode(x - self.centroids[labels])
        new_ids = np.arange(self._ntotal, self._ntotal + x.shape[0], dtype=np.int64)
        order = np.argsort(labels, kind="stable")
        bounds = np.searchsorted(labels[order], np.arange(self.nlist + 1))
        for j in range(self.nlist):
            rows = order[bounds[j] : bounds[j + 1]]
            if rows.size:
                self.list_ids[j] = np.concatenate([self.list_ids[j], new_ids[rows]])
                self.list_codes[j] = np.vstack([self.list_codes[j], codes[rows]])
        self._ntotal += x.shape[0]

    def probe_order(self, xq: FloatArray) -> np.ndarray:
        """The ``nprobe`` nearest lists per query, nearest first."""
        assert self.centroids is not None
        coarse = sq_l2(xq, self.centroids, xb_norms=self._centroid_norms)
        return select_topk(coarse, min(self.nprobe, self.nlist))[1]

    def _scan(self, q: FloatArray, lists: np.ndarray) -> tuple[FloatArray, IdArray]:
        """ADC distances from one query to every code in the probed lists."""
        assert self.centroids is not None and self.pq is not None
        lists = np.array([j for j in lists if self.list_ids[j].size], dtype=np.int64)
        if lists.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
        tables = self.pq.distance_tables(q[None, :] - self.centroids[lists])  # one per list
        dists = [self.pq.adc(tables[t], self.list_codes[j]) for t, j in enumerate(lists)]
        return np.concatenate(dists), np.concatenate([self.list_ids[j] for j in lists])

    def search(self, xq: FloatArray, k: int) -> tuple[FloatArray, IdArray]:
        q = as_float32(xq)
        probes = self.probe_order(q)
        nq = q.shape[0]
        out_d = np.full((nq, k), np.inf, dtype=np.float32)
        out_i = np.full((nq, k), -1, dtype=np.int64)
        n_adc = np.empty(nq)
        for i in range(nq):
            dists, ids = self._scan(q[i], probes[i])
            top_d, pos = select_topk(dists.reshape(1, -1), k)
            valid = pos[0] >= 0
            out_d[i] = top_d[0]
            out_i[i, valid] = ids[pos[0][valid]]
            n_adc[i] = dists.size
        self.last_stats = SearchStats(ndis=np.full(nq, float(self.nlist)), n_adc=n_adc)
        return out_d, out_i

    def trace(self, q: FloatArray, k: int) -> QueryTrace:
        qq = as_float32(q)
        lists = self.probe_order(qq)[0]
        dists, ids = self._scan(qq[0], lists)
        top_d, pos = select_topk(dists.reshape(1, -1), k)
        top_i = np.where(pos[0] >= 0, ids[np.maximum(pos[0], 0)] if ids.size else -1, -1)
        return QueryTrace(
            index=self.name,
            ids=[int(v) for v in top_i],
            dists=top_d[0].tolist(),
            ndis=int(self.nlist),
            probed_lists=lists.tolist(),
            list_sizes=[int(self.list_ids[j].size) for j in lists],
            n_candidates=int(dists.size),
        )

    @property
    def nbytes(self) -> int:
        return ivfpq_bytes(self._ntotal, self._d, self.nlist, self.n_subvectors, self.nbits)
