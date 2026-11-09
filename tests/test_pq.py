"""PQ: bytes instead of floats, distances on the codes, and the recall it costs."""

import numpy as np
import pytest

from src.evaluation.memory import compression_ratio, exact_bytes, pq_bytes, pq_code_bytes
from src.evaluation.recall import recall_at_k
from src.indexes.ivfpq import IVFPQIndex
from src.indexes.pq import PQIndex, ProductQuantizer

K = 10


@pytest.fixture(scope="module")
def pq_index(clustered) -> PQIndex:
    index = PQIndex(n_subvectors=8, seed=0)
    index.train(clustered.base)
    index.add(clustered.base)
    return index


def test_codes_are_one_byte_per_subvector(pq_index: PQIndex, clustered) -> None:
    codes = pq_index.codes
    assert codes.dtype == np.uint8
    assert codes.shape == (clustered.n, 8)


def test_adc_equals_distance_to_reconstruction(pq_index: PQIndex, clustered) -> None:
    assert pq_index.pq is not None
    q = clustered.queries[:5]
    tables = pq_index.pq.distance_tables(q)
    recon = pq_index.reconstruct()
    for i in range(q.shape[0]):
        adc = pq_index.pq.adc(tables[i], pq_index.codes)
        direct = ((recon - q[i]) ** 2).sum(axis=1)
        np.testing.assert_allclose(adc, direct, rtol=1e-4, atol=1e-3)


def test_search_matches_adc_ranking(pq_index: PQIndex, clustered) -> None:
    assert pq_index.pq is not None
    dists, ids = pq_index.search(clustered.queries[:3], K)
    tables = pq_index.pq.distance_tables(clustered.queries[:3])
    for i in range(3):
        adc = pq_index.pq.adc(tables[i], pq_index.codes)
        np.testing.assert_array_equal(ids[i], np.argsort(adc, kind="stable")[:K])
    assert np.all(pq_index.last_stats.n_adc == clustered.n)


def test_compression_is_an_order_of_magnitude() -> None:
    n, d = 50_000, 384  # the configured real dataset
    assert pq_code_bytes(48) == 48
    assert exact_bytes(n, d) / (n * pq_code_bytes(48)) == 4 * d / 48
    assert compression_ratio(n, d, pq_bytes(n, d, 48)) >= 10


def test_more_subvectors_less_error_more_recall(clustered) -> None:
    errors, recalls = [], []
    for m in (2, 4, 8):
        index = PQIndex(n_subvectors=m, seed=0)
        index.train(clustered.base)
        index.add(clustered.base)
        errors.append(float(((index.reconstruct() - clustered.base) ** 2).sum(axis=1).mean()))
        _, ids = index.search(clustered.queries, K)
        recalls.append(recall_at_k(ids, clustered.gt_ids, K))
    assert errors[0] > errors[1] > errors[2]
    assert recalls[2] >= recalls[0]


def test_ivfpq_plateaus_below_one(clustered) -> None:
    index = IVFPQIndex(nlist=16, n_subvectors=4, seed=0)
    index.train(clustered.base)
    index.add(clustered.base)
    recalls = []
    for nprobe in (1, 4, 16):
        index.nprobe = nprobe
        _, ids = index.search(clustered.queries, K)
        recalls.append(recall_at_k(ids, clustered.gt_ids, K))
    assert recalls[-1] >= recalls[0]
    assert recalls[-1] < 0.999  # the remaining gap is PQ's quantization error
    assert np.all(index.last_stats.ndis == 16)
    trace = index.trace(clustered.queries[0], K)
    assert len(trace.probed_lists) == 16 and trace.n_candidates == clustered.n


def test_invalid_shapes_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ProductQuantizer(d=10, n_subvectors=3)
    with pytest.raises(ValueError, match="nbits"):
        ProductQuantizer(d=8, n_subvectors=2, nbits=9)
    with pytest.raises(ValueError, match="training vectors"):
        ProductQuantizer(d=8, n_subvectors=2).train(np.zeros((10, 8), dtype=np.float32))
