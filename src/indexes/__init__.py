"""The one interface every index implements (faiss-shaped), plus shared helpers.

Every index — from scratch here, or a faiss/hnswlib wrapper in ``src.reference`` — exposes::

    index.train(xb)                 # no-op for exact / hnsw / lsh
    index.add(xb)
    dists, ids = index.search(xq, k)   # (nq, k) float32 ascending, (nq, k) int64, padded inf/-1
    index.nbytes                    # analytic compact size (src.evaluation.memory)
    index.last_stats                # SearchStats of the last search: work per query

Build-time knobs are constructor arguments; search-time knobs (``nprobe``, ``ef_search``) are
plain mutable attributes, so a sweep builds once and searches many times.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IdArray = NDArray[np.int64]
StatArray = NDArray[np.float64]


@dataclass
class SearchStats:
    """Work done by the last search, per query.

    ``ndis`` counts full-precision distance evaluations, ``n_adc`` PQ code (table-lookup)
    evaluations; NaN means the engine does not report it (hnswlib). ``work`` is their sum:
    vectors scored per query, the implementation-independent speed currency.
    """

    ndis: StatArray
    n_adc: StatArray
    nhops: StatArray | None = None

    @property
    def work(self) -> StatArray:
        return self.ndis + self.n_adc

    @classmethod
    def zeros(cls, nq: int) -> "SearchStats":
        return cls(ndis=np.zeros(nq), n_adc=np.zeros(nq))

    @classmethod
    def unknown(cls, nq: int) -> "SearchStats":
        return cls(ndis=np.full(nq, np.nan), n_adc=np.full(nq, np.nan))


@dataclass
class QueryTrace:
    """What one query did inside an index — the raw material for the inspector figures.

    IVF/IVFPQ fill ``probed_lists``/``list_sizes``; HNSW fills ``entry_point``,
    ``layer_paths`` (layer -> nodes in visiting order) and ``hops`` ((layer, distance) per
    step); LSH fills ``bucket_sizes``. ``ids``/``dists`` are the returned neighbours.
    """

    index: str
    ids: list[int] = field(default_factory=list)
    dists: list[float] = field(default_factory=list)
    ndis: int = 0
    probed_lists: list[int] = field(default_factory=list)
    list_sizes: list[int] = field(default_factory=list)
    entry_point: int | None = None
    layer_paths: dict[int, list[int]] = field(default_factory=dict)
    hops: list[tuple[int, float]] = field(default_factory=list)
    bucket_sizes: list[int] = field(default_factory=list)
    n_candidates: int = 0


@runtime_checkable
class Index(Protocol):
    """The shared search interface (see module docstring)."""

    name: str
    last_stats: SearchStats

    def train(self, xb: FloatArray) -> None: ...

    def add(self, xb: FloatArray) -> None: ...

    def search(self, xq: FloatArray, k: int) -> tuple[FloatArray, IdArray]: ...

    @property
    def nbytes(self) -> int: ...

    @property
    def ntotal(self) -> int: ...


def as_float32(x: NDArray) -> FloatArray:
    """C-contiguous float32, 2-D — applied at every API boundary."""
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {arr.shape}")
    return arr


def select_topk(dists: FloatArray, k: int) -> tuple[FloatArray, IdArray]:
    """Row-wise k smallest of ``dists (nq, n)``: positions and values, sorted ascending.

    ``argpartition`` then a sort of just those k: O(n) per row, not O(n log n). Rows with
    fewer than k columns are padded with ``inf`` / ``-1`` (the faiss convention).
    """
    nq, n = dists.shape
    kk = min(k, n)
    out_d = np.full((nq, k), np.inf, dtype=np.float32)
    out_i = np.full((nq, k), -1, dtype=np.int64)
    if kk == 0:
        return out_d, out_i
    part = (
        np.argpartition(dists, kk - 1, axis=1)[:, :kk]
        if kk < n
        else np.tile(np.arange(n), (nq, 1))
    )
    vals = np.take_along_axis(dists, part, axis=1)
    order = np.argsort(vals, axis=1, kind="stable")
    out_i[:, :kk] = np.take_along_axis(part, order, axis=1)
    out_d[:, :kk] = np.take_along_axis(vals, order, axis=1)
    return out_d, out_i


def topk_with_ids(dists: FloatArray, ids: IdArray, k: int) -> tuple[FloatArray, IdArray]:
    """k smallest of a 1-D candidate list, mapped back to their ids, padded to length k."""
    top_d, pos = select_topk(dists.reshape(1, -1), k)
    top_i = np.where(pos[0] >= 0, ids[np.maximum(pos[0], 0)], -1).astype(np.int64)
    return top_d[0], top_i


def make_index(name: str, **params: Any) -> Index:
    """Build a from-scratch index by name: exact, ivf, pq, ivfpq, hnsw or lsh."""
    from src.indexes.exact import ExactIndex
    from src.indexes.hnsw import HNSWIndex
    from src.indexes.ivf import IVFIndex
    from src.indexes.ivfpq import IVFPQIndex
    from src.indexes.lsh import LSHIndex
    from src.indexes.pq import PQIndex

    registry: dict[str, type] = {
        "exact": ExactIndex,
        "ivf": IVFIndex,
        "pq": PQIndex,
        "ivfpq": IVFPQIndex,
        "hnsw": HNSWIndex,
        "lsh": LSHIndex,
    }
    if name not in registry:
        raise ValueError(f"unknown index {name!r}; choose from {sorted(registry)}")
    return registry[name](**params)
