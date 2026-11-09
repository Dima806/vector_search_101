"""HNSW: recall rises with ef_search, the graph is navigable, and search is sub-linear."""

import numpy as np
import pytest

from src.datasets.vectors import exact_ground_truth, make_synthetic
from src.evaluation.recall import recall_at_k
from src.indexes.hnsw import HNSWIndex

K = 10
M = 8


@pytest.fixture(scope="module")
def hnsw(clustered) -> HNSWIndex:
    index = HNSWIndex(m=M, ef_construction=48, ef_search=16, seed=0)
    index.add(clustered.base)
    return index


def test_recall_rises_with_ef_search(hnsw: HNSWIndex, clustered) -> None:
    recalls = []
    for ef in (10, 16, 32, 64, 128):
        hnsw.ef_search = ef
        _, ids = hnsw.search(clustered.queries, K)
        recalls.append(recall_at_k(ids, clustered.gt_ids, K))
    # Beam search is not strictly monotone per query; on average it is, within a hair.
    assert all(b >= a - 0.01 for a, b in zip(recalls, recalls[1:], strict=False))
    assert recalls[-1] >= 0.95
    assert recalls[-1] > recalls[0]


def test_link_caps_and_integrity(hnsw: HNSWIndex, clustered) -> None:
    n = clustered.n
    for node in range(n):
        layer0 = hnsw.neighbors(node, 0)
        assert layer0.size <= 2 * M
        assert node not in layer0
        assert np.all((layer0 >= 0) & (layer0 < n))
        for layer in range(1, int(hnsw.levels[node]) + 1):
            links = hnsw.neighbors(node, layer)
            assert links.size <= M
            assert node not in links
            # links on layer l only point to nodes that live on layer l
            assert np.all(hnsw.levels[links] >= layer)


def test_graph_is_navigable(hnsw: HNSWIndex) -> None:
    # Pruning can orphan a node in principle; on this data it should be (almost) never.
    assert hnsw.reachable_fraction() >= 0.99


def test_level_distribution_is_geometric(hnsw: HNSWIndex, clustered) -> None:
    above_zero = float((hnsw.levels > 0).mean())
    assert abs(above_zero - 1 / M) < 0.03
    assert hnsw.level_counts().sum() == clustered.n


def test_search_touches_a_shrinking_fraction() -> None:
    # One incremental build gives the whole ladder: the graph after n inserts is the graph
    # built on those n vectors.
    x = make_synthetic(
        6000 + 50, 16, n_clusters=30, intrinsic_dim=8, cluster_std=0.4, noise=0.05, seed=3
    )
    base, queries = x[50:], x[:50]
    index = HNSWIndex(m=8, ef_construction=24, ef_search=24, seed=0)
    fractions, recalls, start = [], [], 0
    for n in (750, 1500, 3000, 6000):
        index.add(base[start:n])
        start = n
        _, ids = index.search(queries, K)
        gt, _ = exact_ground_truth(base[:n], queries, K)
        fractions.append(float(index.last_stats.ndis.mean()) / n)
        recalls.append(recall_at_k(ids, gt, K))
    assert all(b < a for a, b in zip(fractions, fractions[1:], strict=False))
    assert fractions[-1] < 0.15
    assert min(recalls) > 0.8


def test_save_load_roundtrip(hnsw: HNSWIndex, clustered, tmp_path) -> None:
    path = tmp_path / "graph.npz"
    hnsw.ef_search = 32
    hnsw.save(path)
    loaded = HNSWIndex.load(path, ef_search=32)
    d1, i1 = hnsw.search(clustered.queries, K)
    d2, i2 = loaded.search(clustered.queries, K)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(d1, d2)
    assert loaded.nbytes == hnsw.nbytes


def test_trace_walks_every_layer(hnsw: HNSWIndex, clustered) -> None:
    hnsw.ef_search = 32
    trace = hnsw.trace(clustered.queries[0], K)
    _, ids = hnsw.search(clustered.queries[:1], K)
    assert trace.ids == ids[0].tolist()
    assert set(trace.layer_paths) == set(range(hnsw.max_level + 1))
    assert trace.entry_point == hnsw.entry_point
    # greedy descent never moves away from the query on the upper layers
    for layer in range(hnsw.max_level, 0, -1):
        dists = [d for lay, d in trace.hops if lay == layer]
        assert all(b < a for a, b in zip(dists, dists[1:], strict=False))


def test_m_below_two_rejected() -> None:
    with pytest.raises(ValueError):
        HNSWIndex(m=1)
