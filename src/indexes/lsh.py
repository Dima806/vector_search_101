"""Locality-sensitive hashing for dense vectors: random hyperplanes + banding.

``hashing_101`` builds LSH for *sets* (MinHash signatures, Jaccard similarity). The banding
idea carries over unchanged; only the hash family changes. Here each bit is the side of a
random hyperplane a vector falls on, ``bit = h.x > 0`` (Charikar 2002): two vectors at angle
``theta`` agree on a bit with probability ``1 - theta/pi``. Group ``b * r`` bits into ``b``
bands of ``r`` bits; two vectors become candidates if any band matches exactly, which happens
with probability ``1 - (1 - p^r)^b`` — the same S-curve as in hashing_101. Candidates are
re-ranked with exact distances, so LSH keeps the raw vectors in memory.

Hyperplanes pass through the origin, so vectors are centred first (sentence embeddings share a
common direction; uncentred, most hyperplanes would put every vector on the same side).
"""

import numpy as np

from src.evaluation.memory import lsh_bytes
from src.indexes import FloatArray, IdArray, QueryTrace, SearchStats, as_float32, topk_with_ids
from src.metrics.distance import sq_l2_to_one, sq_norms


def candidate_probability(theta: np.ndarray | float, bands: int, rows: int) -> np.ndarray:
    """Chance a pair at angle ``theta`` (radians) shares at least one band: the LSH S-curve."""
    p = 1.0 - np.asarray(theta, dtype=np.float64) / np.pi
    return 1.0 - (1.0 - p**rows) ** bands


class LSHIndex:
    """Random-hyperplane LSH with ``bands`` x ``rows`` bits, exact re-ranking of candidates."""

    name = "lsh"

    def __init__(self, bands: int, rows: int, seed: int = 0, center: bool = True) -> None:
        if not 1 <= rows <= 62:
            raise ValueError("rows must be in 1..62 (a band key is packed into an int64)")
        self.bands = bands
        self.rows = rows
        self.seed = seed
        self.center = center
        self.mean: FloatArray | None = None
        self.hyperplanes: FloatArray | None = None  # (bands * rows, d)
        self._xb: FloatArray = np.zeros((0, 0), dtype=np.float32)
        self._norms: FloatArray = np.zeros(0, dtype=np.float32)
        self._keys = np.zeros((0, bands), dtype=np.int64)
        self.buckets: list[dict[int, IdArray]] = []
        self.last_stats = SearchStats.zeros(0)

    @property
    def ntotal(self) -> int:
        return int(self._xb.shape[0])

    def train(self, xb: FloatArray) -> None:
        """Learn the centring offset and draw the hyperplanes (seeded, N(0, I))."""
        x = as_float32(xb)
        d = x.shape[1]
        mean = x.mean(axis=0) if self.center else np.zeros(d)
        self.mean = mean.astype(np.float32)
        rng = np.random.default_rng(self.seed)
        self.hyperplanes = rng.standard_normal((self.bands * self.rows, d)).astype(np.float32)

    def band_keys(self, x: FloatArray) -> np.ndarray:
        """One integer key per band: that band's ``rows`` sign bits packed, shape ``(n, bands)``."""
        assert self.hyperplanes is not None and self.mean is not None
        bits = ((as_float32(x) - self.mean) @ self.hyperplanes.T) > 0
        weights = np.left_shift(np.int64(1), np.arange(self.rows, dtype=np.int64))
        return bits.reshape(-1, self.bands, self.rows).astype(np.int64) @ weights

    def add(self, xb: FloatArray) -> None:
        x = as_float32(xb)
        if self.hyperplanes is None:
            self.train(x)
        self._xb = x.copy() if self.ntotal == 0 else np.vstack([self._xb, x])
        self._norms = sq_norms(self._xb)
        self._keys = np.vstack([self._keys, self.band_keys(x)])
        self.buckets = []
        for band in range(self.bands):  # key -> ids, built by sorting once per band
            keys = self._keys[:, band]
            order = np.argsort(keys, kind="stable")
            uniq, starts = np.unique(keys[order], return_index=True)
            ends = np.append(starts[1:], order.size)
            self.buckets.append(
                {
                    int(u): order[s:e].astype(np.int64)
                    for u, s, e in zip(uniq, starts, ends, strict=True)
                }
            )

    def candidates(self, q_keys: np.ndarray) -> tuple[IdArray, list[int]]:
        """Union of the query's buckets across bands, and the size of each bucket hit."""
        hits = [
            self.buckets[b].get(int(key), np.zeros(0, np.int64)) for b, key in enumerate(q_keys)
        ]
        sizes = [int(h.size) for h in hits]
        return np.unique(np.concatenate(hits)), sizes

    def search(self, xq: FloatArray, k: int) -> tuple[FloatArray, IdArray]:
        q = as_float32(xq)
        keys = self.band_keys(q)
        nq = q.shape[0]
        out_d = np.empty((nq, k), dtype=np.float32)
        out_i = np.empty((nq, k), dtype=np.int64)
        ndis = np.empty(nq)
        for i in range(nq):
            cand, _ = self.candidates(keys[i])
            dists = sq_l2_to_one(q[i], self._xb[cand], self._norms[cand])
            out_d[i], out_i[i] = topk_with_ids(dists, cand, k)
            ndis[i] = cand.size
        self.last_stats = SearchStats(ndis=ndis, n_adc=np.zeros(nq))
        return out_d, out_i

    def trace(self, q: FloatArray, k: int) -> QueryTrace:
        qq = as_float32(q)
        cand, sizes = self.candidates(self.band_keys(qq)[0])
        dists = sq_l2_to_one(qq[0], self._xb[cand], self._norms[cand])
        top_d, top_i = topk_with_ids(dists, cand, k)
        return QueryTrace(
            index=self.name,
            ids=top_i.tolist(),
            dists=top_d.tolist(),
            ndis=int(cand.size),
            bucket_sizes=sizes,
            n_candidates=int(cand.size),
        )

    @property
    def nbytes(self) -> int:
        d = self._xb.shape[1] if self.ntotal else 0
        return lsh_bytes(self.ntotal, d, self.bands, self.rows)
