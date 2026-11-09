"""Product quantization: compress each vector into a few bytes, and search on the bytes.

Split every ``d``-dim vector into ``n_subvectors`` pieces of ``d_sub = d / n_subvectors`` dims.
In each sub-space, k-means learns a codebook of ``2^nbits`` (= 256) codewords; a vector is
stored as the index of the nearest codeword in each sub-space — one ``uint8`` per sub-vector.
A 384-d float32 vector (1,536 bytes) becomes 48 bytes with 48 sub-vectors: 32x smaller.

Search uses *asymmetric distance computation* (ADC): the query stays exact. For each sub-space
build a table ``T[j, c] = ||q_j - C_j[c]||^2`` of shape ``(n_subvectors, 256)``; then the
distance to a stored vector is a sum of table lookups, ``sum_j T[j, code[j]]``
(``T[np.arange(n_subvectors), codes].sum(axis=1)``). The database is never decoded to floats.
The price is recall: distances are now approximate (Jegou, Douze & Schmid 2011).
"""

import numpy as np

from src.evaluation.memory import pq_bytes
from src.indexes import FloatArray, IdArray, SearchStats, as_float32, select_topk
from src.indexes.ivf import assign_to_nearest, train_kmeans, training_sample
from src.metrics.distance import sq_l2


class ProductQuantizer:
    """The PQ codec: per-sub-space codebooks, encode / decode, ADC distance tables."""

    def __init__(
        self, d: int, n_subvectors: int, nbits: int = 8, kmeans_max_iter: int = 20, seed: int = 0
    ) -> None:
        if d % n_subvectors:
            raise ValueError(f"d={d} is not divisible by n_subvectors={n_subvectors}")
        if not 1 <= nbits <= 8:
            raise ValueError("nbits must be in 1..8 (codes are stored as uint8)")
        self.d = d
        self.n_subvectors = n_subvectors
        self.d_sub = d // n_subvectors
        self.nbits = nbits
        self.ksub = 2**nbits
        self.kmeans_max_iter = kmeans_max_iter
        self.seed = seed
        self.codebooks: FloatArray = np.zeros((n_subvectors, self.ksub, self.d_sub), np.float32)

    def _sub(self, x: FloatArray, j: int) -> FloatArray:
        return np.ascontiguousarray(x[:, j * self.d_sub : (j + 1) * self.d_sub])

    def train(self, x: FloatArray) -> "ProductQuantizer":
        """One k-means with ``2^nbits`` centroids per sub-space."""
        x = as_float32(x)
        if x.shape[0] < self.ksub:
            raise ValueError(f"need at least {self.ksub} training vectors, got {x.shape[0]}")
        for j in range(self.n_subvectors):
            self.codebooks[j] = train_kmeans(
                self._sub(x, j), self.ksub, self.kmeans_max_iter, self.seed + j
            )
        return self

    def encode(self, x: FloatArray) -> np.ndarray:
        """Nearest codeword per sub-vector: codes of shape ``(n, n_subvectors)``, uint8."""
        x = as_float32(x)
        codes = np.empty((x.shape[0], self.n_subvectors), dtype=np.uint8)
        for j in range(self.n_subvectors):
            codes[:, j] = assign_to_nearest(self._sub(x, j), self.codebooks[j])
        return codes

    def decode(self, codes: np.ndarray) -> FloatArray:
        """Reconstruct vectors from codes (concatenated codewords), shape ``(n, d)``."""
        parts = [self.codebooks[j][codes[:, j]] for j in range(self.n_subvectors)]
        return np.ascontiguousarray(np.hstack(parts), dtype=np.float32)

    def distance_tables(self, xq: FloatArray) -> FloatArray:
        """ADC tables ``T[i, j, c] = ||q_i,j - C_j[c]||^2``, shape ``(nq, n_subvectors, 2^nbits)``."""
        q = as_float32(xq)
        tables = np.empty((q.shape[0], self.n_subvectors, self.ksub), dtype=np.float32)
        for j in range(self.n_subvectors):
            tables[:, j, :] = sq_l2(self._sub(q, j), self.codebooks[j])
        return tables

    def adc(self, table: FloatArray, codes: np.ndarray) -> FloatArray:
        """Approximate distances from one query's ``table`` to every code: sum of lookups."""
        return table[np.arange(self.n_subvectors), codes].sum(axis=1, dtype=np.float32)


class PQIndex:
    """Flat PQ: every vector stored as a code, every query scans all codes with ADC."""

    name = "pq"

    def __init__(
        self,
        n_subvectors: int,
        nbits: int = 8,
        train_size: int = 20_000,
        kmeans_max_iter: int = 20,
        seed: int = 0,
    ) -> None:
        self.n_subvectors = n_subvectors
        self.nbits = nbits
        self.train_size = train_size
        self.kmeans_max_iter = kmeans_max_iter
        self.seed = seed
        self.pq: ProductQuantizer | None = None
        self._codes_t = np.zeros((n_subvectors, 0), dtype=np.uint8)  # (n_subvectors, N)
        self._d = 0
        self.last_stats = SearchStats.zeros(0)

    @property
    def ntotal(self) -> int:
        return int(self._codes_t.shape[1])

    @property
    def codes(self) -> np.ndarray:
        """Stored codes, shape ``(N, n_subvectors)``, uint8."""
        return np.ascontiguousarray(self._codes_t.T)

    def train(self, xb: FloatArray) -> None:
        x = as_float32(xb)
        self._d = x.shape[1]
        sample = training_sample(x, self.train_size, self.seed)
        self.pq = ProductQuantizer(
            self._d, self.n_subvectors, self.nbits, self.kmeans_max_iter, self.seed
        ).train(sample)

    def add(self, xb: FloatArray) -> None:
        if self.pq is None:
            raise RuntimeError("train() the PQ index before add()")
        codes = self.pq.encode(as_float32(xb))
        self._codes_t = np.ascontiguousarray(np.hstack([self._codes_t, codes.T]))

    def reconstruct(self) -> FloatArray:
        """All stored vectors decoded back to float32 (for distortion plots, not for search)."""
        assert self.pq is not None
        return self.pq.decode(self.codes)

    def search(
        self, xq: FloatArray, k: int, block_mb: float = 128.0
    ) -> tuple[FloatArray, IdArray]:
        """ADC over all codes, in query blocks of at most ``block_mb`` megabytes."""
        assert self.pq is not None
        q = as_float32(xq)
        nq, n = q.shape[0], self.ntotal
        tables = self.pq.distance_tables(q)
        rows = max(1, int(block_mb * 2**20 / (4 * max(n, 1))))
        out_d = np.empty((nq, k), dtype=np.float32)
        out_i = np.empty((nq, k), dtype=np.int64)
        for start in range(0, nq, rows):
            stop = min(start + rows, nq)
            acc = np.zeros((stop - start, n), dtype=np.float32)
            for j in range(self.n_subvectors):  # sum_j T[j, code_j], vectorised over queries
                # np.take with intp indices is ~3x faster than fancy indexing with uint8 codes;
                # the cast is per sub-space and temporary — storage stays one byte per code.
                codes_j = self._codes_t[j].astype(np.intp)
                acc += np.take(tables[start:stop, j, :], codes_j, axis=1)
            out_d[start:stop], out_i[start:stop] = select_topk(acc, k)
        self.last_stats = SearchStats(ndis=np.zeros(nq), n_adc=np.full(nq, float(n)))
        return out_d, out_i

    @property
    def nbytes(self) -> int:
        return pq_bytes(self.ntotal, self._d, self.n_subvectors, self.nbits)
