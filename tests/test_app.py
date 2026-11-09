"""The Streamlit app runs headless: without data (helpful message) and on a tiny dataset."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.config import get_settings
from src.datasets.vectors import VectorDataset, exact_ground_truth, make_synthetic, save_dataset
from src.metrics.distance import normalise

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("VS101_DATA_DIR", str(tmp_path))
    st.cache_resource.clear()
    yield tmp_path
    st.cache_resource.clear()


def run_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_app_without_data_explains_what_to_do(data_dir: Path) -> None:
    at = run_app()
    assert not at.exception
    assert any("make data" in info.value for info in at.info)


def test_app_with_tiny_dataset(data_dir: Path) -> None:
    x = normalise(
        make_synthetic(650, 16, n_clusters=8, intrinsic_dim=6, cluster_std=0.4, noise=0.05, seed=0)
    )
    base, queries = x[50:], x[:50]
    gt_ids, gt_dists = exact_ground_truth(base, queries, 20)
    texts = [f"base sentence {i}" for i in range(base.shape[0])]
    q_texts = [f"query sentence {i}" for i in range(queries.shape[0])]
    meta = {"seed": get_settings().seed}
    save_dataset(VectorDataset("news", base, queries, gt_ids, gt_dists, meta, texts, q_texts))

    at = run_app()
    assert not at.exception
    assert len(at.tabs) == 4
    at.button[0].click().run()  # time exact search
    assert not at.exception
    for family in ("PQ", "IVFPQ", "HNSW"):
        at.radio[0].set_value(family).run()
        assert not at.exception, family
    at.radio[1].set_value(at.radio[1].options[1]).run()  # wall-clock triangle
    assert not at.exception
