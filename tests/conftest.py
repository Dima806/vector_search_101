"""Shared fixtures: small seeded datasets with exact ground truth (offline, fast)."""

import numpy as np
import pytest

from src.datasets.vectors import VectorDataset, exact_ground_truth, make_synthetic

K = 10


def _dataset(name: str, n: int, d: int, n_queries: int, **kwargs: float) -> VectorDataset:
    x = make_synthetic(n + n_queries, d, seed=0, **kwargs)  # type: ignore[arg-type]
    base, queries = x[n_queries:], x[:n_queries]
    gt_ids, gt_dists = exact_ground_truth(base, queries, 20)
    return VectorDataset(name, base, queries, gt_ids, gt_dists, {"seed": 0})


@pytest.fixture(scope="session")
def clustered() -> VectorDataset:
    """3,000 x 32-d Gaussian mixture on an 8-d subspace, 100 held-out queries."""
    return _dataset(
        "clustered", 3000, 32, 100, n_clusters=20, intrinsic_dim=8, cluster_std=0.4, noise=0.05
    )


@pytest.fixture(scope="session")
def isotropic() -> VectorDataset:
    """2,000 x 16-d isotropic Gaussian, 50 queries — no structure to exploit."""
    return _dataset("isotropic", 2000, 16, 50, n_clusters=1, intrinsic_dim=16, cluster_std=1.0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)
