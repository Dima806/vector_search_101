"""The recall-speed-memory triangle: parameter sweeps, the Pareto set, and who dominates whom.

Every sweep produces rows of one tidy schema (``SWEEP_COLUMNS``); notebooks 05/06 and the app
read that schema and nothing re-derives it. A setting is on the Pareto frontier if no other
setting is at least as good on every objective and strictly better on one — "no index
dominates" is computed here, not eyeballed from a chart.
"""

import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.speed import evaluate
from src.indexes import FloatArray, IdArray, Index

SWEEP_COLUMNS = [
    "index",
    "engine",
    "params",
    "dataset",
    "n",
    "d",
    "k",
    "recall",
    "ndis",
    "n_adc",
    "work",
    "work_speedup",
    "latency_ms",
    "qps",
    "speedup_vs_exact",
    "nbytes",
    "bytes_per_vector",
    "build_s",
    "threads",
    "seed",
]


def sweep(
    index: Index,
    param: str | None,
    values: Sequence[Any],
    xq: FloatArray,
    gt_ids: IdArray,
    k: int,
    *,
    engine: str,
    dataset: str,
    build_params: dict[str, Any],
    build_s: float,
    seed: int,
    threads: int,
    timed: bool = False,
    warmup: int = 3,
    max_queries: int | None = None,
    exact_latency_ms: float | None = None,
    on_value: Callable[[Any], None] | None = None,
) -> list[dict[str, Any]]:
    """Search one built index at each value of a search-time knob (``nprobe``, ``ef_search``).

    ``param=None`` evaluates the index once as built (``values`` is then ignored).
    """
    rows = []
    n = index.ntotal
    d = int(xq.shape[1])
    for value in values if param is not None else [None]:
        if param is not None:
            setattr(index, param, value)
        if on_value is not None:
            on_value(value)
        res = evaluate(index, xq, gt_ids, k, timed=timed, warmup=warmup, max_queries=max_queries)
        params = dict(build_params)
        if param is not None:
            params[param] = value
        latency = res["latency_ms"]
        rows.append(
            {
                "index": index.name,
                "engine": engine,
                "params": json.dumps(params, sort_keys=True),
                "dataset": dataset,
                "n": n,
                "d": d,
                "k": k,
                "recall": res["recall"],
                "ndis": res["ndis"],
                "n_adc": res["n_adc"],
                "work": res["work"],
                "work_speedup": n / res["work"]
                if res["work"] and np.isfinite(res["work"])
                else np.nan,
                "latency_ms": latency,
                "qps": res["qps"],
                "speedup_vs_exact": (
                    exact_latency_ms / latency
                    if exact_latency_ms is not None and np.isfinite(latency)
                    else np.nan
                ),
                "nbytes": index.nbytes,
                "bytes_per_vector": index.nbytes / max(n, 1),
                "build_s": build_s,
                "threads": threads,
                "seed": seed,
            }
        )
    return rows


def to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Sweep rows as a DataFrame with the canonical column order."""
    return pd.DataFrame(rows, columns=SWEEP_COLUMNS)


def pareto_mask(df: pd.DataFrame, objectives: dict[str, str]) -> np.ndarray:
    """Boolean mask of non-dominated rows. ``objectives`` maps column -> "max" or "min".

    Row a dominates row b if a is at least as good on every objective and strictly better on
    at least one. Rows with NaN in any objective are never on the frontier.
    """
    cols = list(objectives)
    vals = df[cols].to_numpy(dtype=float).copy()
    for j, col in enumerate(cols):
        if objectives[col] == "min":
            vals[:, j] = -vals[:, j]
        elif objectives[col] != "max":
            raise ValueError(f"objective for {col!r} must be 'max' or 'min'")
    valid = ~np.isnan(vals).any(axis=1)
    mask = np.zeros(len(df), dtype=bool)
    for i in np.flatnonzero(valid):
        others = vals[valid]
        dominated = ((others >= vals[i]).all(axis=1) & (others > vals[i]).any(axis=1)).any()
        mask[i] = not dominated
    return mask


def frontier_2d(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Rows on the 2-D frontier maximising both ``x`` and ``y``, sorted by ``x``."""
    mask = pareto_mask(df, {x: "max", y: "max"})
    return df[mask].sort_values(x)


def dominance_report(
    df: pd.DataFrame, speed_col: str, family_col: str = "index"
) -> dict[str, Any]:
    """Which index families own a piece of the 3-D (recall up, speed up, bytes down) frontier."""
    objectives = {"recall": "max", speed_col: "max", "nbytes": "min"}
    mask = pareto_mask(df, objectives)
    on_front = df[mask]
    counts = on_front[family_col].value_counts().to_dict()
    families = sorted(df[family_col].unique())
    return {
        "speed_axis": speed_col,
        "n_settings": len(df),
        "n_on_frontier": int(mask.sum()),
        "frontier_counts": {f: int(counts.get(f, 0)) for f in families},
        "families_on_frontier": sorted(counts),
        "fully_dominated": [f for f in families if counts.get(f, 0) == 0],
        "single_family_dominates": len(counts) == 1,
    }


def expand_params(df: pd.DataFrame) -> pd.DataFrame:
    """The sweep table with its ``params`` JSON spread into one column per parameter."""
    params = pd.DataFrame([json.loads(p) for p in df["params"]], index=df.index)
    return pd.concat([df, params.drop(columns=[c for c in params if c in df])], axis=1)
