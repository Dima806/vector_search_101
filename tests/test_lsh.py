"""Random-hyperplane LSH: the measured collision rate follows the S-curve."""

import numpy as np

from src.evaluation.recall import recall_at_k
from src.indexes.lsh import LSHIndex, candidate_probability
from src.metrics.distance import sq_l2

K = 10


def pairs_at_angle(
    theta: float, n: int, d: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """n random unit-vector pairs separated by exactly ``theta`` radians."""
    u = rng.standard_normal((n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = rng.standard_normal((n, d))
    w -= (w * u).sum(axis=1, keepdims=True) * u
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    return u.astype(np.float32), (np.cos(theta) * u + np.sin(theta) * w).astype(np.float32)


def test_candidate_rate_tracks_s_curve(rng: np.random.Generator) -> None:
    bands, rows, d = 6, 4, 32
    index = LSHIndex(bands=bands, rows=rows, seed=0, center=False)
    index.train(np.zeros((1, d), dtype=np.float32))
    for theta in (0.3, 0.7, 1.1, 1.6):
        a, b = pairs_at_angle(theta, 3000, d, rng)
        collide = (index.band_keys(a) == index.band_keys(b)).any(axis=1).mean()
        assert abs(collide - candidate_probability(theta, bands, rows)) < 0.05


def test_more_bands_more_recall_more_work(clustered) -> None:
    recalls, work = [], []
    for bands in (1, 4, 16):
        index = LSHIndex(bands=bands, rows=8, seed=0)
        index.add(clustered.base)
        _, ids = index.search(clustered.queries, K)
        recalls.append(recall_at_k(ids, clustered.gt_ids, K))
        work.append(float(index.last_stats.ndis.mean()))
    assert recalls[0] < recalls[1] < recalls[2]
    assert work[0] < work[1] < work[2]


def test_candidates_are_reranked_exactly(clustered) -> None:
    index = LSHIndex(bands=8, rows=6, seed=0)
    index.add(clustered.base)
    dists, ids = index.search(clustered.queries[:5], K)
    for i in range(5):
        valid = ids[i] >= 0
        exact = sq_l2(clustered.queries[i : i + 1], clustered.base[ids[i][valid]])[0]
        np.testing.assert_allclose(dists[i][valid], exact, rtol=1e-4, atol=1e-4)
    trace = index.trace(clustered.queries[0], K)
    assert len(trace.bucket_sizes) == 8 and trace.ids == ids[0].tolist()
    assert index.nbytes > clustered.n * clustered.d * 4  # keeps the floats for re-ranking
