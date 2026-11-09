"""faiss-cpu and hnswlib wrapped behind the project's Index protocol.

These are the *only* place the compiled libraries are imported. They serve two purposes:
validating the from-scratch indexes (same parameters -> similar recall and work) and supplying
honest wall-clock numbers, since pure-Python search loops measure the interpreter, not the
algorithm (CLAUDE.md §8).

Work counters: faiss reports distance evaluations through global stats objects
(``faiss.cvar.indexIVF_stats`` / ``faiss.cvar.hnsw_stats``). The IVF wrapper adds the ``nlist``
coarse distances so it uses the same convention as ``src.indexes.ivf``. hnswlib does not expose
counters, so its work is NaN.
"""

import tempfile
from pathlib import Path
from typing import Any

import faiss
import hnswlib  # ty: ignore[import-not-found]  (compiled, no stubs)
import numpy as np

from src.indexes import FloatArray, IdArray, SearchStats, as_float32

_CVAR: Any = faiss.cvar  # global work counters (indexIVF_stats, hnsw_stats)


def set_threads(n: int) -> None:
    """Pin faiss's OpenMP pool (hnswlib threads are set per index, numpy via threadpoolctl)."""
    faiss.omp_set_num_threads(n)


def library_versions() -> dict[str, str]:
    """Versions of the reference engines, for the machine line under every timing table."""
    from importlib.metadata import version  # noqa: PLC0415

    return {"faiss": faiss.__version__, "hnswlib": version("hnswlib")}


def faiss_nbytes(index: faiss.Index) -> int:
    """Serialised size of a faiss index — its in-memory payload plus a small header."""
    return int(faiss.serialize_index(index).nbytes)


class _FaissIndex:
    """Shared plumbing: lazy construction once ``d`` is known, train/add/search, size."""

    engine = "faiss"
    name = "faiss"

    def __init__(self) -> None:
        self.index: Any = None  # SWIG objects: attributes exist only at runtime
        self.last_stats = SearchStats.zeros(0)

    def _build(self, d: int) -> faiss.Index:
        raise NotImplementedError

    def _ensure(self, d: int) -> faiss.Index:
        if self.index is None:
            self.index = self._build(d)
        return self.index

    def train(self, xb: FloatArray) -> None:
        x = as_float32(xb)
        index = self._ensure(x.shape[1])
        if not index.is_trained:
            index.train(x)

    def add(self, xb: FloatArray) -> None:
        x = as_float32(xb)
        index = self._ensure(x.shape[1])
        if not index.is_trained:
            index.train(x)
        index.add(x)

    def _stats(self, nq: int) -> SearchStats:
        return SearchStats.unknown(nq)

    def _reset(self) -> None:
        """Clear faiss's global work counters before a search."""

    def search(self, xq: FloatArray, k: int) -> tuple[FloatArray, IdArray]:
        assert self.index is not None
        q = as_float32(xq)
        self._reset()
        dists, ids = self.index.search(q, k)
        self.last_stats = self._stats(q.shape[0])
        return dists.astype(np.float32), ids.astype(np.int64)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    @property
    def nbytes(self) -> int:
        return faiss_nbytes(self.index) if self.index is not None else 0


class FaissFlat(_FaissIndex):
    """``IndexFlatL2``: exact search with BLAS."""

    name = "exact"

    def _build(self, d: int) -> faiss.Index:
        return faiss.IndexFlatL2(d)

    def _stats(self, nq: int) -> SearchStats:
        return SearchStats(ndis=np.full(nq, float(self.ntotal)), n_adc=np.zeros(nq))


class FaissIVF(_FaissIndex):
    """``IndexIVFFlat`` with the same nlist / nprobe meaning as the scratch IVF."""

    name = "ivf"

    def __init__(self, nlist: int, nprobe: int = 1, kmeans_max_iter: int = 20, seed: int = 0):
        super().__init__()
        self.nlist = nlist
        self._nprobe = nprobe
        self.kmeans_max_iter = kmeans_max_iter
        self.seed = seed

    def _build(self, d: int) -> faiss.Index:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_L2)
        index.cp.niter = self.kmeans_max_iter
        index.cp.seed = self.seed
        index.nprobe = self._nprobe
        self._quantizer = quantizer  # keep the coarse quantizer alive alongside the index
        return index

    @property
    def nprobe(self) -> int:
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self._nprobe = value
        if self.index is not None:
            self.index.nprobe = value

    def _reset(self) -> None:
        _CVAR.indexIVF_stats.reset()

    def _stats(self, nq: int) -> SearchStats:
        scanned = _CVAR.indexIVF_stats.ndis / max(nq, 1)
        return SearchStats(ndis=np.full(nq, scanned + self.nlist), n_adc=np.zeros(nq))


class FaissPQ(_FaissIndex):
    """``IndexPQ``: flat product quantization, ``n_subvectors`` x ``nbits``."""

    name = "pq"

    def __init__(self, n_subvectors: int, nbits: int = 8):
        super().__init__()
        self.n_subvectors = n_subvectors
        self.nbits = nbits

    def _build(self, d: int) -> faiss.Index:
        return faiss.IndexPQ(d, self.n_subvectors, self.nbits, faiss.METRIC_L2)

    def _stats(self, nq: int) -> SearchStats:
        return SearchStats(ndis=np.zeros(nq), n_adc=np.full(nq, float(self.ntotal)))


class FaissIVFPQ(_FaissIndex):
    """``IndexIVFPQ``: coarse quantizer + PQ on residuals."""

    name = "ivfpq"

    def __init__(
        self, nlist: int, n_subvectors: int, nbits: int = 8, nprobe: int = 1, seed: int = 0
    ):
        super().__init__()
        self.nlist = nlist
        self.n_subvectors = n_subvectors
        self.nbits = nbits
        self._nprobe = nprobe
        self.seed = seed

    def _build(self, d: int) -> faiss.Index:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.n_subvectors, self.nbits)
        index.cp.seed = self.seed
        index.nprobe = self._nprobe
        self._quantizer = quantizer
        return index

    @property
    def nprobe(self) -> int:
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self._nprobe = value
        if self.index is not None:
            self.index.nprobe = value

    def _reset(self) -> None:
        _CVAR.indexIVF_stats.reset()

    def _stats(self, nq: int) -> SearchStats:
        scanned = _CVAR.indexIVF_stats.ndis / max(nq, 1)
        return SearchStats(ndis=np.full(nq, float(self.nlist)), n_adc=np.full(nq, scanned))


class FaissHNSW(_FaissIndex):
    """``IndexHNSWFlat``: faiss's HNSW, used for its distance counter (``hnsw_stats.ndis``)."""

    name = "hnsw"

    def __init__(self, m: int = 16, ef_construction: int = 64, ef_search: int = 64):
        super().__init__()
        self.m = m
        self.ef_construction = ef_construction
        self._ef_search = ef_search

    def _build(self, d: int) -> faiss.Index:
        index = faiss.IndexHNSWFlat(d, self.m)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self._ef_search
        return index

    @property
    def ef_search(self) -> int:
        return self._ef_search

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        self._ef_search = value
        if self.index is not None:
            self.index.hnsw.efSearch = value

    def _reset(self) -> None:
        _CVAR.hnsw_stats.reset()

    def _stats(self, nq: int) -> SearchStats:
        stats = _CVAR.hnsw_stats
        return SearchStats(
            ndis=np.full(nq, stats.ndis / max(nq, 1)),
            n_adc=np.zeros(nq),
            nhops=np.full(nq, stats.nhops / max(nq, 1)),
        )


class HnswlibHNSW:
    """hnswlib's HNSW (``space="l2"``, squared distances) — the production reference."""

    engine = "hnswlib"
    name = "hnsw"

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 64,
        seed: int = 0,
        threads: int = 1,
    ) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self._ef_search = ef_search
        self.seed = seed
        self.threads = threads
        self.index: hnswlib.Index | None = None
        self._d = 0
        self.last_stats = SearchStats.zeros(0)

    def train(self, xb: FloatArray) -> None:
        """Nothing to learn."""

    def add(self, xb: FloatArray) -> None:
        x = as_float32(xb)
        start = self.ntotal
        if self.index is None:
            self._d = x.shape[1]
            self.index = hnswlib.Index(space="l2", dim=self._d)
            self.index.init_index(
                max_elements=x.shape[0],
                M=self.m,
                ef_construction=self.ef_construction,
                random_seed=self.seed,
            )
            self.index.set_num_threads(self.threads)
            self.index.set_ef(self._ef_search)
        else:
            self.index.resize_index(start + x.shape[0])
        self.index.add_items(x, np.arange(start, start + x.shape[0]))

    @property
    def ef_search(self) -> int:
        return self._ef_search

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        self._ef_search = value
        if self.index is not None:
            self.index.set_ef(value)

    def search(self, xq: FloatArray, k: int) -> tuple[FloatArray, IdArray]:
        assert self.index is not None
        q = as_float32(xq)
        if self._ef_search < k:  # hnswlib requires ef >= k
            self.index.set_ef(k)
        labels, dists = self.index.knn_query(q, k=k)
        self.index.set_ef(max(self._ef_search, k))
        self.last_stats = SearchStats.unknown(q.shape[0])
        return dists.astype(np.float32), labels.astype(np.int64)

    @property
    def ntotal(self) -> int:
        return int(self.index.get_current_count()) if self.index is not None else 0

    @property
    def nbytes(self) -> int:
        """Size of the saved index file — hnswlib's in-memory layout written out verbatim."""
        if self.index is None:
            return 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.bin"
            self.index.save_index(str(path))
            return path.stat().st_size


def make_reference(name: str, engine: str = "faiss", **params: Any) -> Any:
    """Reference index by name: exact / ivf / pq / ivfpq / hnsw on ``faiss`` or ``hnswlib``."""
    if engine == "hnswlib":
        if name != "hnsw":
            raise ValueError("hnswlib only provides 'hnsw'")
        return HnswlibHNSW(**params)
    classes: dict[str, type] = {
        "exact": FaissFlat,
        "ivf": FaissIVF,
        "pq": FaissPQ,
        "ivfpq": FaissIVFPQ,
        "hnsw": FaissHNSW,
    }
    if name not in classes:
        raise ValueError(f"unknown reference index {name!r}")
    return classes[name](**params)
