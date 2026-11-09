"""Inverted file (IVF-Flat): cluster the vectors, then search only the clusters near the query.

Train: k-means learns ``nlist`` centroids (the *coarse quantizer*). Add: every vector goes into
the inverted list of its nearest centroid. Search: rank the centroids by distance to the query,
scan the ``nprobe`` nearest lists exactly, keep the top k. One knob slides from fast and
approximate (``nprobe = 1``) to exact (``nprobe = nlist``, every list scanned). Misses happen
when a true neighbour sits just across a cluster boundary, in a list that was not probed.
"""

import numpy as np
from sklearn.cluster import KMeans

from src.evaluation.memory import ivf_flat_bytes
from src.indexes import (
    FloatArray,
    IdArray,
    QueryTrace,
    SearchStats,
    as_float32,
    select_topk,
    topk_with_ids,
)
from src.metrics.distance import sq_l2, sq_l2_to_one, sq_norms


def train_kmeans(x: FloatArray, n_clusters: int, max_iter: int, seed: int) -> FloatArray:
    """k-means centroids (scikit-learn, one seeded random initialisation) as float32.

    Random initialisation, as faiss does: k-means++ seeding costs more than the Lloyd
    iterations themselves at 256 clusters and barely changes the final inertia.
    """
    km = KMeans(
        n_clusters=n_clusters, init="random", n_init=1, max_iter=max_iter, random_state=seed
    )
    km.fit(x)
    return np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)


def assign_to_nearest(x: FloatArray, centroids: FloatArray, block: int = 8192) -> np.ndarray:
    """Index of the nearest centroid for every row of ``x``."""
    c_norms = sq_norms(centroids)
    labels = np.empty(x.shape[0], dtype=np.int64)
    for start in range(0, x.shape[0], block):
        dists = sq_l2(x[start : start + block], centroids, xb_norms=c_norms)
        labels[start : start + block] = dists.argmin(axis=1)
    return labels


def training_sample(x: FloatArray, size: int, seed: int) -> FloatArray:
    """A seeded random subset of at most ``size`` rows."""
    if x.shape[0] <= size:
        return x
    rng = np.random.default_rng(seed)
    return x[np.sort(rng.choice(x.shape[0], size=size, replace=False))]


class IVFIndex:
    """IVF-Flat from scratch: k-means coarse quantizer + exact scan of the probed lists."""

    name = "ivf"

    def __init__(
        self,
        nlist: int,
        nprobe: int = 1,
        train_points_per_centroid: int = 64,
        kmeans_max_iter: int = 20,
        seed: int = 0,
    ) -> None:
        self.nlist = nlist
        self.nprobe = nprobe
        self.train_points_per_centroid = train_points_per_centroid
        self.kmeans_max_iter = kmeans_max_iter
        self.seed = seed
        self.centroids: FloatArray | None = None
        self._centroid_norms: FloatArray = np.zeros(0, dtype=np.float32)
        self.list_ids: list[IdArray] = []
        self.list_vecs: list[FloatArray] = []
        self.list_norms: list[FloatArray] = []
        self._ntotal = 0
        self._d = 0
        self.last_stats = SearchStats.zeros(0)

    @property
    def ntotal(self) -> int:
        return self._ntotal

    @property
    def list_sizes(self) -> np.ndarray:
        return np.array([ids.size for ids in self.list_ids], dtype=np.int64)

    def train(self, xb: FloatArray) -> None:
        """Learn the coarse centroids on a sample of ``train_points_per_centroid * nlist``."""
        x = as_float32(xb)
        sample = training_sample(x, self.train_points_per_centroid * self.nlist, self.seed)
        self.centroids = train_kmeans(sample, self.nlist, self.kmeans_max_iter, self.seed)
        self._centroid_norms = sq_norms(self.centroids)
        self._d = x.shape[1]
        self.list_ids = [np.zeros(0, dtype=np.int64) for _ in range(self.nlist)]
        self.list_vecs = [np.zeros((0, self._d), dtype=np.float32) for _ in range(self.nlist)]
        self.list_norms = [np.zeros(0, dtype=np.float32) for _ in range(self.nlist)]

    def add(self, xb: FloatArray) -> None:
        """Put each vector into the inverted list of its nearest centroid."""
        if self.centroids is None:
            raise RuntimeError("train() the IVF index before add()")
        x = as_float32(xb)
        labels = assign_to_nearest(x, self.centroids)
        new_ids = np.arange(self._ntotal, self._ntotal + x.shape[0], dtype=np.int64)
        order = np.argsort(labels, kind="stable")
        bounds = np.searchsorted(labels[order], np.arange(self.nlist + 1))
        for j in range(self.nlist):
            rows = order[bounds[j] : bounds[j + 1]]
            if rows.size == 0:
                continue
            vecs = x[rows]
            self.list_ids[j] = np.concatenate([self.list_ids[j], new_ids[rows]])
            self.list_vecs[j] = np.ascontiguousarray(np.vstack([self.list_vecs[j], vecs]))
            self.list_norms[j] = np.concatenate([self.list_norms[j], sq_norms(vecs)])
        self._ntotal += x.shape[0]

    def probe_order(self, xq: FloatArray) -> np.ndarray:
        """The ``nprobe`` nearest lists for each query, nearest first, shape ``(nq, nprobe)``."""
        assert self.centroids is not None
        coarse = sq_l2(xq, self.centroids, xb_norms=self._centroid_norms)
        _, order = select_topk(coarse, min(self.nprobe, self.nlist))
        return order

    def _scan(self, q: FloatArray, lists: np.ndarray) -> tuple[FloatArray, IdArray]:
        """Exact distances from one query to every vector in the given lists."""
        q_norm = float(q @ q)
        dists = [
            sq_l2_to_one(q, self.list_vecs[j], self.list_norms[j], q_norm)
            for j in lists
            if self.list_ids[j].size
        ]
        ids = [self.list_ids[j] for j in lists if self.list_ids[j].size]
        if not dists:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
        return np.concatenate(dists), np.concatenate(ids)

    def search(self, xq: FloatArray, k: int) -> tuple[FloatArray, IdArray]:
        q = as_float32(xq)
        probes = self.probe_order(q)
        nq = q.shape[0]
        out_d = np.empty((nq, k), dtype=np.float32)
        out_i = np.empty((nq, k), dtype=np.int64)
        ndis = np.empty(nq)
        for i in range(nq):
            dists, ids = self._scan(q[i], probes[i])
            out_d[i], out_i[i] = topk_with_ids(dists, ids, k)
            ndis[i] = self.nlist + dists.size  # coarse distances + list scan
        self.last_stats = SearchStats(ndis=ndis, n_adc=np.zeros(nq))
        return out_d, out_i

    def trace(self, q: FloatArray, k: int) -> QueryTrace:
        """Which lists one query probed (nearest first), their sizes, and what it returned."""
        qq = as_float32(q)
        lists = self.probe_order(qq)[0]
        dists, ids = self._scan(qq[0], lists)
        top_d, top_i = topk_with_ids(dists, ids, k)
        return QueryTrace(
            index=self.name,
            ids=top_i.tolist(),
            dists=top_d.tolist(),
            ndis=int(self.nlist + dists.size),
            probed_lists=lists.tolist(),
            list_sizes=[int(self.list_ids[j].size) for j in lists],
            n_candidates=int(dists.size),
        )

    @property
    def nbytes(self) -> int:
        return ivf_flat_bytes(self._ntotal, self._d, self.nlist)
