"""Every figure function renders on small inputs and returns a figure (no display needed)."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

from src import visualisation as vis


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def sweep_frame() -> pd.DataFrame:
    rows = []
    for idx, params, recall, speed, bpv in [
        ("exact", {}, 1.0, 1.0, 1536.0),
        ("ivf", {"nlist": 64, "nprobe": 1}, 0.5, 40.0, 1550.0),
        ("ivf", {"nlist": 64, "nprobe": 8}, 0.9, 6.0, 1550.0),
        ("pq", {"n_subvectors": 8}, 0.3, 1.0, 16.0),
        ("pq", {"n_subvectors": 48}, 0.7, 1.0, 56.0),
        ("ivfpq", {"nlist": 64, "n_subvectors": 48, "nprobe": 4}, 0.6, 20.0, 64.0),
        ("hnsw", {"m": 16, "ef_search": 16}, 0.88, 50.0, 1680.0),
        ("hnsw", {"m": 16, "ef_search": 64}, 0.98, 20.0, 1680.0),
        ("hnsw", {"m": 8, "ef_search": 64}, 0.95, 25.0, 1610.0),
        ("lsh", {"bands": 16, "rows": 8}, 0.4, 15.0, 1680.0),
    ]:
        rows.append(
            {
                "index": idx,
                "params": json.dumps(params),
                "recall": recall,
                "work_speedup": speed,
                "speedup_vs_exact": speed,
                "bytes_per_vector": bpv,
                "nbytes": bpv * 1000,
            }
        )
    return pd.DataFrame(rows)


def test_number_formatting() -> None:
    assert (
        vis.human(12_500) == "12.5k" and vis.human(2_000_000) == "2M" and vis.human(999) == "999"
    )
    assert vis.human_bytes(1536) == "1.54 KB" and vis.human_bytes(12) == "12 B"
    assert (
        vis.human_ms(0.5) == "500 µs"
        and vis.human_ms(1500) == "1.5 s"
        and vis.human_ms(120_000) == "2 min"
    )


def test_notebook_01_figures() -> None:
    fig = vis.plot_scaling_wall(
        [1e4, 1e5], {"numpy": [1.0, 10.0], "faiss": [0.9, 9.0]}, 1e-4, 0.0, [1_000_000], 384
    )
    assert isinstance(fig, Figure)
    rng = np.random.default_rng(0)
    assert isinstance(vis.plot_needle(rng.random(1000) + 1, 10, "a query"), Figure)


def test_notebook_02_figures() -> None:
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((200, 2)).astype(np.float32)
    cents = rng.standard_normal((5, 2)).astype(np.float32)
    xx, yy = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    cell = rng.integers(0, 5, xx.shape)
    lists = rng.integers(0, 5, 200)
    true_ids = np.arange(10)
    panels = [{"nprobe": 1, "probed": [0], "found": [0, 1], "scanned": 40}]
    assert isinstance(
        vis.plot_ivf_toy(pts, cents, (xx, yy, cell), lists, pts[0], true_ids, panels), Figure
    )
    scratch = pd.DataFrame(
        {
            "dataset": ["news"] * 3,
            "nprobe": [1, 4, 16],
            "recall": [0.4, 0.7, 0.9],
            "work_speedup": [90, 30, 8],
            "n": 50000,
            "d": 384,
        }
    )
    compiled = scratch.assign(speedup_vs_exact=[100, 30, 9])
    assert isinstance(vis.plot_ivf_sweep(scratch, compiled, 10), Figure)
    assert isinstance(vis.plot_ivf_misses(rng.random(400) + 1, rng.random(400)), Figure)


def test_notebook_03_figures() -> None:
    rows = pd.DataFrame(
        {
            "label": ["Exact (float32)", "IVF-Flat (nlist=8)", "PQ 8 B code", "PQ 48 B code"],
            "bytes_per_vector": [1536, 1550, 16, 56],
            "recall": [1.0, np.nan, 0.3, 0.7],
        }
    )
    assert isinstance(vis.plot_pq_memory(rows, 10), Figure)
    rng = np.random.default_rng(0)
    panel = {
        "true": rng.random(100),
        "approx": rng.random(100),
        "is_top": np.arange(100) < 10,
        "spearman": 0.8,
        "kept": 5,
        "title": "8-byte codes",
    }
    assert isinstance(vis.plot_adc_distortion([panel, panel]), Figure)
    curve = pd.DataFrame({"nprobe": [1, 4, 16], "recall": [0.4, 0.6, 0.65]})
    mem = {"Exact (float32)": 7e7, "IVF-Flat": 7.1e7, "PQ (48 B)": 2.8e6, "IVFPQ (48 B)": 3.6e6}
    assert isinstance(
        vis.plot_ivfpq(curve.assign(recall=[0.5, 0.8, 1.0]), curve, 0.68, mem, 10), Figure
    )


def test_notebook_04_figures() -> None:
    hops = [(2, 2.0), (1, 1.5), (1, 1.2), (0, 1.1), (0, 1.3), (0, 0.9)]
    assert isinstance(
        vis.plot_hnsw_layers_walk(np.array([900, 90, 10]), 10, hops, 300, 1000), Figure
    )
    ladders = [{"label": "scratch", "n": [1000, 4000], "work": [500, 700], "color": "#009E73"}]
    assert isinstance(vis.plot_hnsw_sublinear(ladders), Figure)
    knobs = pd.DataFrame(
        {
            "m": [8, 8, 16, 16],
            "ef_search": [16, 64, 16, 64],
            "recall": [0.8, 0.95, 0.85, 0.98],
            "work": [300, 900, 400, 1100],
        }
    )
    hl = pd.DataFrame({"m": [16, 16], "ef_search": [16, 64], "recall": [0.85, 0.98]})
    builds = pd.DataFrame(
        {"m": [8, 16], "build_s": [10.0, 12.0], "bytes_per_vector": [1617.0, 1680.0]}
    )
    assert isinstance(vis.plot_hnsw_knobs(knobs, hl, builds, 1536.0, 10), Figure)


def test_notebook_05_and_06_figures() -> None:
    df = sweep_frame()
    assert isinstance(vis.plot_triangle(df, "work_speedup", "speed", "title", 10), Figure)
    pays = pd.DataFrame(
        {
            "n": [1e4, 1e5] * 3,
            "method": ["exact (faiss Flat)"] * 2 + ["IVF (faiss)"] * 2 + ["HNSW (hnswlib)"] * 2,
            "latency_ms": [0.2, 2.0, 0.02, 0.05, 0.02, 0.04],
            "build_s": [0, 0, 1, 5, 1, 6],
            "knob": ["-", "-", "nprobe=1", "nprobe=2", "ef=16", "ef=16"],
        }
    )
    assert isinstance(vis.plot_when_index_pays(pays, 0.95, 500_000), Figure)
    sizes = pd.DataFrame(
        {
            "n_sample": [25, 200],
            "median": [0.7, 0.71],
            "p2_5": [0.6, 0.68],
            "p97_5": [0.8, 0.74],
            "ci_low": [0.61, 0.675],
            "ci_high": [0.81, 0.745],
        }
    )
    assert isinstance(vis.plot_recall_sample_size(sizes, 0.71, "ivf"), Figure)


def test_plotly_figures() -> None:
    df = sweep_frame()
    assert isinstance(vis.triangle_plotly(df, "work_speedup", "speed", "speed"), go.Figure)
    assert isinstance(vis.triangle_plotly(df, "work_speedup", "speed", "memory"), go.Figure)
    assert isinstance(vis.scaling_plotly([1, 2], {"a": [1.0, 2.0]}), go.Figure)
