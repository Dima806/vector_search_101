"""Recall arithmetic, the Pareto frontier, sweeps and memory formulas."""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation import memory
from src.evaluation.recall import per_query_recall, recall_at_k, recall_on_sample
from src.evaluation.speed import (
    evaluate,
    machine_info,
    median_latency_ms,
    pinned_threads,
    timed_build,
)
from src.evaluation.triangle import (
    SWEEP_COLUMNS,
    dominance_report,
    frontier_2d,
    pareto_mask,
    sweep,
    to_frame,
)
from src.indexes import make_index
from src.indexes.exact import ExactIndex


def test_recall_identical_disjoint_partial() -> None:
    gt = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert recall_at_k(gt, gt, 4) == 1.0
    assert recall_at_k(gt + 100, gt, 4) == 0.0
    pred = np.array([[4, 3, 99, -1], [5, 0, 0, 0]])  # order inside top-k does not matter
    np.testing.assert_allclose(per_query_recall(pred, gt, 4), [0.5, 0.25])
    assert recall_at_k(pred, gt, 1) == 0.5  # recall@1: only query 2's top hit is correct


def test_recall_needs_k_columns() -> None:
    with pytest.raises(ValueError):
        recall_at_k(np.zeros((2, 3)), np.zeros((2, 5)), 5)


def test_recall_on_sample_exact_is_perfect(clustered) -> None:
    exact = ExactIndex()
    exact.add(clustered.base)
    est = recall_on_sample(exact, clustered.base, clustered.queries, k=10, n_sample=50)
    assert est.mean == 1.0 and est.n_queries == 50
    ivf = make_index("ivf", nlist=32, nprobe=1, seed=0)
    ivf.train(clustered.base)
    ivf.add(clustered.base)
    est = recall_on_sample(ivf, clustered.base, clustered.queries, k=10, n_sample=80)
    assert 0 <= est.ci_low <= est.mean <= est.ci_high <= 1


def test_pareto_mask_hand_made() -> None:
    df = pd.DataFrame(
        {
            "recall": [0.9, 0.8, 0.95, 0.7, 0.9],
            "speed": [10, 20, 5, 15, 9],
            "nbytes": [100, 100, 100, 100, 100],
        }
    )
    mask = pareto_mask(df, {"recall": "max", "speed": "max", "nbytes": "min"})
    # row 3 is beaten by row 1 (better on both); row 4 by row 0
    assert mask.tolist() == [True, True, True, False, False]
    front = frontier_2d(df, "speed", "recall")
    assert front["speed"].tolist() == [5, 10, 20]
    with pytest.raises(ValueError):
        pareto_mask(df, {"recall": "up"})


def test_dominance_report_names_dominated_family() -> None:
    df = pd.DataFrame(
        {
            "index": ["a", "a", "b", "c"],
            "recall": [0.9, 0.99, 0.95, 0.8],
            "qps": [100, 50, 80, 60],
            "nbytes": [10, 10, 10, 20],
        }
    )
    report = dominance_report(df, "qps")
    assert report["families_on_frontier"] == ["a", "b"]
    assert report["fully_dominated"] == ["c"]
    assert report["single_family_dominates"] is False


def test_sweep_rows_follow_schema(clustered) -> None:
    ivf = make_index("ivf", nlist=16, seed=0)
    build_s = timed_build(ivf, clustered.base)
    rows = sweep(
        ivf,
        "nprobe",
        [1, 4],
        clustered.queries,
        clustered.gt_ids,
        10,
        engine="scratch",
        dataset="clustered",
        build_params={"nlist": 16},
        build_s=build_s,
        seed=0,
        threads=1,
        timed=True,
        max_queries=10,
        exact_latency_ms=1.0,
    )
    df = to_frame(rows)
    assert list(df.columns) == SWEEP_COLUMNS
    assert json.loads(df["params"][1]) == {"nlist": 16, "nprobe": 4}
    assert df["recall"].is_monotonic_increasing
    assert (df["work_speedup"] > 1).all() and df["qps"].notna().all()


def test_evaluate_and_latency(clustered) -> None:
    exact = ExactIndex()
    exact.add(clustered.base)
    with pinned_threads(1):
        res = evaluate(exact, clustered.queries, clustered.gt_ids, 10, timed=True, max_queries=5)
        assert median_latency_ms(exact, clustered.queries, 10, warmup=1, max_queries=3) > 0
    assert res["recall"] == 1.0 and res["work"] == clustered.n and res["qps"] > 0
    info = machine_info(1)
    assert info["threads"] == 1 and info["logical_cpus"] >= 1


def test_memory_formulas() -> None:
    assert memory.exact_bytes(10, 4) == 160
    assert memory.ivf_flat_bytes(10, 4, 2) == 160 + 80 + 32
    assert memory.pq_bytes(10, 8, 4) == 40 + 256 * 8 * 4
    assert memory.pq_code_bytes(3, 4) == 2  # 12 bits -> 2 bytes
    assert memory.ivfpq_bytes(10, 8, 2, 4) == 10 * 12 + 64 + 256 * 32
    assert memory.hnsw_bytes(10, 4, 2, upper_slots=6) == 160 + 4 * (40 + 6) + 120
    assert memory.lsh_bytes(8, 4, 2, 4) == 8 + 128 + 128
    assert memory.compression_ratio(10, 4, 16) == 10.0
    assert memory.compression_ratio(10, 4, 0) == float("inf")
