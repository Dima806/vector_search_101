"""From-scratch indexes reproduce faiss / hnswlib: recall and work within tolerance.

"Speed matches" is asserted in *work* (vectors scored per query), never wall-clock: pure Python
cannot match C++ on time, and should not have to — the algorithm is what is being checked.
"""

import numpy as np
import pytest

from src.evaluation.recall import recall_at_k
from src.indexes.exact import ExactIndex
from src.indexes.hnsw import HNSWIndex
from src.indexes.ivf import IVFIndex
from src.indexes.ivfpq import IVFPQIndex
from src.indexes.pq import PQIndex
from src.reference.compare import (
    FaissFlat,
    FaissHNSW,
    FaissIVF,
    FaissIVFPQ,
    FaissPQ,
    HnswlibHNSW,
    make_reference,
)

K = 10
RECALL_TOL = 0.05


def built(index, data):
    index.train(data.base)
    index.add(data.base)
    return index


def recall_of(index, data) -> float:
    _, ids = index.search(data.queries, K)
    return recall_at_k(ids, data.gt_ids, K)


def test_exact_matches_faiss_flat(clustered) -> None:
    ours, ref = built(ExactIndex(), clustered), built(FaissFlat(), clustered)
    _, ids_ours = ours.search(clustered.queries, K)
    _, ids_ref = ref.search(clustered.queries, K)
    assert recall_at_k(ids_ours, ids_ref, K) >= 0.999  # float32 near-ties only
    assert np.all(ref.last_stats.ndis == clustered.n)


def test_ivf_matches_faiss_recall_and_work(clustered) -> None:
    ours = built(IVFIndex(nlist=32, seed=0), clustered)
    ref = built(FaissIVF(nlist=32, seed=0), clustered)
    for nprobe in (1, 2, 4, 8):
        ours.nprobe = ref.nprobe = nprobe
        assert abs(recall_of(ours, clustered) - recall_of(ref, clustered)) <= RECALL_TOL
        work_ratio = ours.last_stats.ndis.mean() / ref.last_stats.ndis.mean()
        assert 0.67 < work_ratio < 1.5


def test_pq_matches_faiss(clustered) -> None:
    ours = built(PQIndex(n_subvectors=8, seed=0), clustered)
    ref = built(FaissPQ(n_subvectors=8), clustered)
    assert abs(recall_of(ours, clustered) - recall_of(ref, clustered)) <= RECALL_TOL


def test_ivfpq_matches_faiss(clustered) -> None:
    ours = built(IVFPQIndex(nlist=16, n_subvectors=8, nprobe=4, seed=0), clustered)
    ref = built(FaissIVFPQ(nlist=16, n_subvectors=8, nprobe=4), clustered)
    assert abs(recall_of(ours, clustered) - recall_of(ref, clustered)) <= RECALL_TOL


@pytest.fixture(scope="module")
def scratch_hnsw(clustered) -> HNSWIndex:
    index = HNSWIndex(m=8, ef_construction=48, seed=0)
    index.add(clustered.base)
    return index


@pytest.mark.parametrize("ef", [16, 64])
def test_hnsw_matches_hnswlib_recall(clustered, scratch_hnsw: HNSWIndex, ef: int) -> None:
    scratch_hnsw.ef_search = ef
    ref = HnswlibHNSW(m=8, ef_construction=48, ef_search=ef, seed=0)
    ref.add(clustered.base)
    assert abs(recall_of(scratch_hnsw, clustered) - recall_of(ref, clustered)) <= RECALL_TOL
    assert np.isnan(ref.last_stats.ndis).all()  # hnswlib exposes no counters


def test_hnsw_work_matches_faiss_hnsw(clustered, scratch_hnsw: HNSWIndex) -> None:
    scratch_hnsw.ef_search = 32
    ref = built(FaissHNSW(m=8, ef_construction=48, ef_search=32), clustered)
    recall_of(scratch_hnsw, clustered)
    recall_of(ref, clustered)
    ratio = scratch_hnsw.last_stats.ndis.mean() / ref.last_stats.ndis.mean()
    assert 0.5 < ratio < 2.0


def test_analytic_bytes_match_faiss_serialised(clustered, scratch_hnsw: HNSWIndex) -> None:
    pairs = [
        (built(ExactIndex(), clustered), built(FaissFlat(), clustered)),
        (built(IVFIndex(nlist=16, seed=0), clustered), built(FaissIVF(nlist=16), clustered)),
        (
            built(PQIndex(n_subvectors=8, seed=0), clustered),
            built(FaissPQ(n_subvectors=8), clustered),
        ),
        (
            built(IVFPQIndex(nlist=16, n_subvectors=8, seed=0), clustered),
            built(FaissIVFPQ(nlist=16, n_subvectors=8), clustered),
        ),
    ]
    pairs.append((scratch_hnsw, built(FaissHNSW(m=8, ef_construction=48), clustered)))
    for ours, ref in pairs:
        assert abs(ours.nbytes / ref.nbytes - 1) < 0.10, ours.name


def test_hnswlib_size_and_ef_below_k(clustered) -> None:
    ref = HnswlibHNSW(m=8, ef_construction=32, ef_search=4)
    ref.add(clustered.base[:1000])
    ref.add(clustered.base[1000:])  # resize path
    assert ref.ntotal == clustered.n
    _, ids = ref.search(clustered.queries, K)  # ef < k is raised to k internally
    assert ids.shape == (clustered.queries.shape[0], K)
    assert ref.nbytes > clustered.n * clustered.d * 4


def test_make_reference_dispatch() -> None:
    assert isinstance(make_reference("ivf", nlist=4), FaissIVF)
    assert isinstance(make_reference("hnsw", engine="hnswlib", m=4), HnswlibHNSW)
    with pytest.raises(ValueError):
        make_reference("lsh")
    with pytest.raises(ValueError):
        make_reference("ivf", engine="hnswlib")
