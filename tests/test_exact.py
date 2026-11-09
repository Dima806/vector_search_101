"""Exact kNN returns the true neighbours — it defines ground truth for everything else."""

import numpy as np
import pytest

from src.indexes import select_topk
from src.indexes.exact import ExactIndex
from src.metrics.distance import normalise, pairwise


def naive_knn(xb: np.ndarray, xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Textbook reference: explicit differences, norms, full argsort (float64)."""
    d = np.linalg.norm(xq[:, None, :].astype(np.float64) - xb[None, :, :], axis=2) ** 2
    ids = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(d, ids, axis=1), ids


def test_matches_naive_reference(rng: np.random.Generator) -> None:
    xb = rng.standard_normal((800, 24)).astype(np.float32)
    xq = rng.standard_normal((40, 24)).astype(np.float32)
    index = ExactIndex()
    index.add(xb)
    dists, ids = index.search(xq, 10)
    ref_d, ref_i = naive_knn(xb, xq, 10)
    np.testing.assert_array_equal(ids, ref_i)
    np.testing.assert_allclose(dists, ref_d, rtol=1e-4, atol=1e-4)


def test_every_base_vector_is_its_own_nearest_neighbour(rng: np.random.Generator) -> None:
    xb = rng.standard_normal((500, 16)).astype(np.float32)
    index = ExactIndex()
    index.add(xb)
    dists, ids = index.search(xb[:100], 1)
    np.testing.assert_array_equal(ids[:, 0], np.arange(100))
    assert np.all(dists[:, 0] < 1e-3)


def test_query_blocks_do_not_change_results(clustered) -> None:
    whole, tiny = ExactIndex(block_mb=256), ExactIndex(block_mb=0.001)
    whole.add(clustered.base)
    tiny.add(clustered.base)
    d1, i1 = whole.search(clustered.queries, 10)
    d2, i2 = tiny.search(clustered.queries, 10)
    np.testing.assert_array_equal(i1, i2)
    # BLAS picks different kernels for different block shapes: last-bit float32 differences
    np.testing.assert_allclose(d1, d2, rtol=1e-4, atol=1e-5)


def test_metrics_rank_identically_on_normalised_vectors(rng: np.random.Generator) -> None:
    xb = normalise(rng.standard_normal((600, 32)))
    xq = normalise(rng.standard_normal((30, 32)))
    results = []
    for metric in ("l2", "ip", "cosine"):
        index = ExactIndex(metric=metric)
        index.add(xb)
        results.append(index.search(xq, 10)[1])
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])


def test_work_is_n_per_query(clustered) -> None:
    index = ExactIndex()
    index.add(clustered.base)
    index.search(clustered.queries, 5)
    assert np.all(index.last_stats.ndis == clustered.n)
    assert index.nbytes == clustered.n * clustered.d * 4


def test_ground_truth_fixture_agrees(clustered) -> None:
    index = ExactIndex()
    index.add(clustered.base)
    _, ids = index.search(clustered.queries, 20)
    np.testing.assert_array_equal(ids, clustered.gt_ids)


def test_padding_when_k_exceeds_n(rng: np.random.Generator) -> None:
    index = ExactIndex()
    index.add(rng.standard_normal((3, 4)).astype(np.float32))
    dists, ids = index.search(rng.standard_normal((2, 4)), 5)
    assert np.all(ids[:, 3:] == -1)
    assert np.all(np.isinf(dists[:, 3:]))


def test_select_topk_sorted_and_correct(rng: np.random.Generator) -> None:
    d = rng.random((7, 50)).astype(np.float32)
    vals, pos = select_topk(d, 6)
    assert np.all(np.diff(vals, axis=1) >= 0)
    np.testing.assert_array_equal(pos, np.argsort(d, axis=1)[:, :6])


def test_unknown_metric_raises(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        pairwise(rng.random((2, 3)), rng.random((2, 3)), metric="manhattan")  # type: ignore[arg-type]
