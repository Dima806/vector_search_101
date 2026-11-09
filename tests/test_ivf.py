"""IVF: recall rises to exactly 1.0 as nprobe grows, and the work grows with it."""

import numpy as np
import pytest

from src.evaluation.recall import recall_at_k
from src.indexes.ivf import IVFIndex

NLIST = 32
K = 10


@pytest.fixture(scope="module")
def ivf(clustered) -> IVFIndex:
    index = IVFIndex(nlist=NLIST, seed=0)
    index.train(clustered.base)
    index.add(clustered.base)
    return index


def sweep(index: IVFIndex, data, values: list[int]) -> tuple[list[float], list[float]]:
    recalls, work = [], []
    for nprobe in values:
        index.nprobe = nprobe
        _, ids = index.search(data.queries, K)
        recalls.append(recall_at_k(ids, data.gt_ids, K))
        work.append(float(index.last_stats.ndis.mean()))
    return recalls, work


def test_recall_non_decreasing_and_exact_at_full_probe(ivf: IVFIndex, clustered) -> None:
    # Probed sets are nested and each list is scanned identically, so this is a theorem:
    # no tolerance.
    recalls, _ = sweep(ivf, clustered, list(range(1, NLIST + 1)))
    assert all(b >= a for a, b in zip(recalls, recalls[1:], strict=False))
    assert recalls[-1] == 1.0
    assert recalls[0] < 1.0  # nprobe=1 does miss neighbours on this data


def test_work_grows_with_nprobe(ivf: IVFIndex, clustered) -> None:
    _, work = sweep(ivf, clustered, [1, 2, 4, 8, 16, 32])
    assert all(b >= a for a, b in zip(work, work[1:], strict=False))
    assert work[-1] == clustered.n + NLIST  # every list scanned + coarse distances


def test_every_vector_in_exactly_one_list(ivf: IVFIndex, clustered) -> None:
    ids = np.sort(np.concatenate(ivf.list_ids))
    np.testing.assert_array_equal(ids, np.arange(clustered.n))
    assert ivf.list_sizes.sum() == clustered.n


def test_same_seed_same_index(clustered) -> None:
    def build(seed: int) -> IVFIndex:
        index = IVFIndex(nlist=16, nprobe=3, seed=seed)
        index.train(clustered.base)
        index.add(clustered.base)
        return index

    a, b, c = build(1), build(1), build(2)
    assert a.centroids is not None and b.centroids is not None and c.centroids is not None
    np.testing.assert_array_equal(a.centroids, b.centroids)
    np.testing.assert_array_equal(
        a.search(clustered.queries, K)[1], b.search(clustered.queries, K)[1]
    )
    assert not np.array_equal(a.centroids, c.centroids)


def test_trace_matches_search(ivf: IVFIndex, clustered) -> None:
    ivf.nprobe = 4
    trace = ivf.trace(clustered.queries[0], K)
    _, ids = ivf.search(clustered.queries[:1], K)
    assert len(trace.probed_lists) == 4
    assert trace.ids == ids[0].tolist()
    assert trace.n_candidates == sum(trace.list_sizes)


def test_add_before_train_raises(clustered) -> None:
    with pytest.raises(RuntimeError):
        IVFIndex(nlist=4).add(clustered.base)


def test_incremental_add_keeps_ids(clustered) -> None:
    index = IVFIndex(nlist=8, nprobe=8, seed=0)
    index.train(clustered.base)
    index.add(clustered.base[:1000])
    index.add(clustered.base[1000:])
    _, ids = index.search(clustered.queries, K)
    assert recall_at_k(ids, clustered.gt_ids, K) == 1.0
