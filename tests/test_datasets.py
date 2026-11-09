"""Synthetic generators, ground truth, the dataset cache and the (offline) build pipeline."""

import numpy as np
import pytest

from src.config import RealDatasetConfig, Settings, SyntheticDatasetConfig, get_settings
from src.datasets import vectors
from src.datasets.vectors import (
    VectorDataset,
    build_synthetic,
    cached_names,
    exact_ground_truth,
    find_dataset_dir,
    load_dataset,
    make_synthetic,
    save_dataset,
    unique_rows,
)


def test_synthetic_is_seeded_and_shaped() -> None:
    a = make_synthetic(500, 12, n_clusters=5, intrinsic_dim=3, cluster_std=0.3, noise=0.0, seed=1)
    b = make_synthetic(500, 12, n_clusters=5, intrinsic_dim=3, cluster_std=0.3, noise=0.0, seed=1)
    assert a.shape == (500, 12) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    # with no noise the points live on a 3-d subspace
    assert np.linalg.matrix_rank(a.astype(np.float64), tol=1e-3) == 3
    with pytest.raises(ValueError):
        make_synthetic(10, 4, 1, 5, 1.0, 0.0, 0)


def test_isotropic_control_has_unit_variance() -> None:
    x = make_synthetic(
        20_000, 8, n_clusters=1, intrinsic_dim=8, cluster_std=1.0, noise=0.0, seed=0
    )
    np.testing.assert_allclose(x.var(axis=0), 1.0, atol=0.05)


def test_ground_truth_is_sorted_and_correct(clustered) -> None:
    ids, dists = exact_ground_truth(clustered.base, clustered.queries[:5], 7)
    assert np.all(np.diff(dists, axis=1) >= 0)
    brute = ((clustered.base[None] - clustered.queries[:5, None]) ** 2).sum(-1)
    np.testing.assert_array_equal(ids, np.argsort(brute, axis=1)[:, :7])


def test_unique_rows_keeps_first_occurrence() -> None:
    x = np.array([[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]], dtype=np.float32)
    np.testing.assert_array_equal(unique_rows(x), [0, 1, 3])


def test_cache_roundtrip(tmp_path) -> None:
    spec = SyntheticDatasetConfig(name="toy", n=300, d=8, n_clusters=3, intrinsic_dim=4)
    ds = build_synthetic(spec, n_queries=20, k_gt=5, seed=7)
    assert ds.queries.shape == (20, 8) and ds.gt_ids.shape == (20, 5)
    path = save_dataset(ds, tmp_path)
    assert path.name == "toy_n300_d8_s7"
    assert find_dataset_dir("toy", 7, tmp_path) == path
    assert cached_names(tmp_path) == ["toy"]
    back = load_dataset("toy", seed=7, root=tmp_path)
    np.testing.assert_array_equal(back.base, ds.base)
    np.testing.assert_array_equal(back.gt_ids, ds.gt_ids)
    assert back.meta["intrinsic_dim"] == 4 and back.texts is None
    with pytest.raises(FileNotFoundError, match="make data"):
        load_dataset("missing", seed=7, root=tmp_path)
    assert cached_names(tmp_path / "nowhere") == []


def test_real_pipeline_offline(tmp_path, monkeypatch) -> None:
    """The news build with the download and the encoder stubbed out."""
    rng = np.random.default_rng(0)
    sentences = [f"sentence number {i} about topic {i % 7}" for i in range(400)]
    fake = rng.standard_normal((len(sentences), 16)).astype(np.float32)
    fake[5] = fake[3]  # one duplicate vector must be dropped
    monkeypatch.setattr(vectors, "load_news_sentences", lambda *_: sentences)
    monkeypatch.setattr(vectors, "embed_texts", lambda texts, *_: fake[: len(texts)])
    cfg = RealDatasetConfig(name="news", max_n=300, min_chars=1, max_chars=100)
    ds = vectors.build_real(cfg, n_queries=50, k_gt=10, seed=0)
    assert ds.n == 300 and ds.queries.shape[0] == 50
    assert len(ds.texts or []) == 300 and len(ds.query_texts or []) == 50
    assert np.unique(np.vstack([ds.base, ds.queries]), axis=0).shape[0] == 350

    monkeypatch.setenv("VS101_DATA_DIR", str(tmp_path))
    settings = Settings(
        datasets={
            "n_queries": 50,
            "k_gt": 10,
            "real": cfg.model_dump(),
            "synthetic": [{"name": "toy", "n": 200, "d": 8, "intrinsic_dim": 4}],
        },
    )
    built = vectors.build_all(settings)
    assert len(built) == 2
    assert vectors.build_all(settings) == []  # cached: nothing rebuilt
    assert load_dataset("news", seed=settings.seed).texts is not None


def test_dataset_properties(clustered) -> None:
    assert isinstance(clustered, VectorDataset)
    assert clustered.key == f"clustered_n{clustered.n}_d{clustered.d}_s0"


def test_settings_load_yaml_and_env(monkeypatch) -> None:
    settings = get_settings()
    assert settings.eval.k == 10 and settings.datasets.real.name == "news"
    assert {s.name for s in settings.datasets.synthetic} == {"clustered", "isotropic"}
    monkeypatch.setenv("VS101_EVAL__K", "20")
    assert Settings().eval.k == 20
