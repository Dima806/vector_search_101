"""Every figure in the project. Matplotlib for the notebook PNGs, plotly for the app.

Functions return a figure and never show or save it — notebooks save with
``src.artifacts.save_figure``. One fixed colour per index family across the whole project
(Okabe-Ito, colour-blind safe); log axes for speed and bytes; recall axes say which k.
"""

import json
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullFormatter

INDEX_COLORS = {
    "exact": "#333333",
    "ivf": "#0072B2",
    "pq": "#E69F00",
    "ivfpq": "#CC79A7",
    "hnsw": "#009E73",
    "lsh": "#D55E00",
}
INDEX_MARKERS = {"exact": "*", "ivf": "o", "pq": "s", "ivfpq": "D", "hnsw": "^", "lsh": "v"}
INDEX_NAMES = {
    "exact": "Exact",
    "ivf": "IVF",
    "pq": "PQ",
    "ivfpq": "IVFPQ",
    "hnsw": "HNSW",
    "lsh": "LSH",
}
DATASET_COLORS = {"news": "#0072B2", "clustered": "#009E73", "isotropic": "#999999"}
ENGINE_COLORS = {"numpy": "#0072B2", "faiss": "#E69F00", "hnswlib": "#009E73"}

STYLE: dict[Any, Any] = {  # matplotlib types its keys as literals; ours are plain strings
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "legend.frameon": False,
    "figure.facecolor": "white",
}

P = ParamSpec("P")
R = TypeVar("R")


def styled(fn: Callable[P, R]) -> Callable[P, R]:
    """Run a figure function inside the project style (no global rcParams side effects)."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with plt.rc_context(STYLE):
            return fn(*args, **kwargs)

    return wrapper


def human(n: float) -> str:
    """1500 -> '1.5k', 2_000_000 -> '2M'."""
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= div:
            value = n / div
            if value == int(value) or value >= 100:
                return f"{value:.0f}{suffix}"
            return f"{value:.1f}{suffix}"
    return f"{n:.0f}"


def human_bytes(b: float) -> str:
    """Bytes as B / KB / MB / GB (decimal)."""
    for div, suffix in ((1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if b >= div:
            return f"{b / div:.3g} {suffix}"
    return f"{b:.0f} B"


def human_ms(ms: float) -> str:
    """Milliseconds as µs / ms / s / min."""
    if ms < 1:
        return f"{ms * 1e3:.0f} µs"
    if ms < 1e3:
        return f"{ms:.3g} ms"
    if ms < 6e4:
        return f"{ms / 1e3:.3g} s"
    return f"{ms / 6e4:.3g} min"


_HUMAN_TICKS = FuncFormatter(lambda v, _: human(v))


def _shade(color: str, factor: float) -> tuple[float, float, float]:
    """``factor`` < 1 darkens, > 1 lightens (towards white), 1 leaves the colour as is."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    out = rgb * factor if factor <= 1 else rgb + (1 - rgb) * (factor - 1)
    r, g, b = np.clip(out, 0, 1)
    return float(r), float(g), float(b)


def _family(name: str) -> str:
    """'IVF-Flat (nlist=256)' -> 'ivf', 'PQ 48 B code' -> 'pq', 'Exact (float32)' -> 'exact'."""
    return name.split()[0].split("-")[0].lower()


# =========================================================================== notebook 01
@styled
def plot_scaling_wall(
    sizes: Sequence[int],
    latency_ms: dict[str, Sequence[float]],
    slope_ms_per_vector: float,
    intercept_ms: float,
    extrapolate_to: Sequence[int],
    d: int,
    ram_gb: float = 8.0,
) -> Figure:
    """Exact search: per-query latency and memory as the database grows (log-log)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    sizes_arr = np.asarray(sizes, dtype=float)
    labels = {"numpy": "numpy brute force", "faiss": "faiss IndexFlatL2 (BLAS)"}
    for engine, values in latency_ms.items():
        ax1.plot(
            sizes_arr,
            values,
            marker="o" if engine == "numpy" else "s",
            color=ENGINE_COLORS[engine],
            label=f"{labels.get(engine, engine)} (measured)",
        )
    n_max = max(extrapolate_to)
    grid = np.geomspace(sizes_arr.max(), n_max, 50)
    ax1.plot(
        grid,
        intercept_ms + slope_ms_per_vector * grid,
        "--",
        color=ENGINE_COLORS["numpy"],
        alpha=0.7,
        label="linear fit, extrapolated",
    )
    ax1.axvspan(sizes_arr.max() * 1.15, n_max * 1.6, color="#999999", alpha=0.08)
    ax1.text(
        sizes_arr.max() * 1.3,
        0.02,
        "extrapolated (not measured)",
        color="#666666",
        fontsize=9.5,
        transform=ax1.get_xaxis_transform(),
    )
    for n in extrapolate_to:
        ms = intercept_ms + slope_ms_per_vector * n
        ax1.plot(n, ms, "o", mfc="white", mec=ENGINE_COLORS["numpy"], mew=1.8, zorder=5)
        ax1.annotate(
            f"{human(n)} vectors: {human_ms(ms)} / query",
            (n, ms),
            xytext=(-12, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9.5,
        )
    ax1.set(
        xscale="log",
        yscale="log",
        xlabel="vectors in the database (N)",
        ylabel="latency per query (ms, 1 thread)",
        title=f"Query time grows in lockstep with N ({d}-d)",
    )
    top_ms = intercept_ms + slope_ms_per_vector * n_max
    ax1.set_ylim(min(min(v) for v in latency_ms.values()) * 0.5, top_ms * 4)
    ax1.xaxis.set_major_formatter(_HUMAN_TICKS)
    ax1.set_xlim(sizes_arr.min() * 0.7, n_max * 1.6)
    ax1.legend(loc="upper left")

    mem_n = np.geomspace(sizes_arr.min(), n_max, 50)
    ax2.plot(mem_n, mem_n * d * 4 / 1e9, color="#333333", label="float32 vectors (N x d x 4 B)")
    ax2.axhline(ram_gb, color="#D55E00", ls=":", lw=1.8, label=f"{ram_gb:.0f} GB Codespace RAM")
    for n in [*sizes_arr[-1:], *extrapolate_to]:
        gb = n * d * 4 / 1e9
        ax2.plot(n, gb, "o", color="#333333")
        ax2.annotate(
            human_bytes(gb * 1e9),
            (n, gb),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            fontsize=9.5,
        )
    ax2.set(
        xscale="log",
        yscale="log",
        xlabel="vectors in the database (N)",
        ylabel="memory (GB)",
        title="…and so does the memory it scans",
    )
    ax2.xaxis.set_major_formatter(_HUMAN_TICKS)
    ax2.legend(loc="upper left")
    fig.tight_layout()
    return fig


@styled
def plot_needle(dists: np.ndarray, k: int, query_text: str | None = None) -> Figure:
    """All N distances from one query, with the k nearest marked: the needle in the haystack."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    sorted_d = np.sort(dists)
    kth = sorted_d[k - 1]
    ax.hist(dists, bins=150, color="#9fb7cc", edgecolor="none", log=True)
    ax.axvline(kth, color="#D55E00", lw=2)
    ax.axvspan(sorted_d[0] * 0.98, kth, color="#D55E00", alpha=0.12)
    share = k / dists.size
    ax.annotate(
        f"the {k} nearest neighbours\n= {share:.2%} of the database",
        (kth, ax.get_ylim()[1] * 0.3),
        xytext=(25, 0),
        textcoords="offset points",
        color="#D55E00",
        fontsize=10.5,
        va="center",
        arrowprops={"arrowstyle": "->", "color": "#D55E00"},
    )
    ax.set(xlabel="squared L2 distance from the query", ylabel="number of base vectors (log)")
    ax.set_title(
        f"To find {k} neighbours, exact search scores all {dists.size:,} vectors",
        pad=22 if query_text else 6,
    )
    if query_text:
        short = query_text if len(query_text) < 100 else query_text[:97] + "…"
        ax.text(
            0.5,
            1.01,
            f"query: {short}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
            style="italic",
        )
    fig.tight_layout()
    return fig


# =========================================================================== notebook 02
@styled
def plot_ivf_toy(
    points: np.ndarray,
    centroids: np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    point_lists: np.ndarray,
    query: np.ndarray,
    true_ids: np.ndarray,
    panels: Sequence[dict[str, Any]],
) -> Figure:
    """2-D IVF: Voronoi cells, the cells one query probes, true neighbours found vs missed."""
    xx, yy, cell = grid
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.6))
    axes = np.atleast_1d(axes)
    nlist = centroids.shape[0]
    for ax, panel in zip(axes, panels, strict=True):
        probed = np.asarray(panel["probed"])
        shade = np.isin(cell, probed).astype(float)
        ax.contourf(xx, yy, shade, levels=[-0.5, 0.5, 1.5], colors=["white", "#d6e9f5"])
        ax.contour(
            xx, yy, cell, levels=np.arange(nlist + 1) - 0.5, colors="#888888", linewidths=0.6
        )
        in_probed = np.isin(point_lists, probed)
        ax.scatter(*points[~in_probed].T, s=5, color="#cccccc", label="not scanned")
        ax.scatter(*points[in_probed].T, s=6, color="#5b7083", label="scanned (probed lists)")
        ax.scatter(*centroids.T, marker="x", s=45, color="black", lw=1.5, label="centroid")
        found = np.isin(true_ids, panel["found"])
        ax.scatter(
            *points[true_ids[found]].T,
            s=70,
            color="#009E73",
            edgecolor="black",
            lw=0.6,
            zorder=4,
            label="true neighbour, found",
        )
        ax.scatter(
            *points[true_ids[~found]].T,
            s=70,
            facecolor="none",
            edgecolor="#D55E00",
            lw=2,
            zorder=4,
            label="true neighbour, missed",
        )
        ax.scatter(
            *query,
            marker="*",
            s=320,
            color="#E69F00",
            edgecolor="black",
            lw=0.8,
            zorder=5,
            label="query",
        )
        ax.set(
            title=f"nprobe = {panel['nprobe']}: found {found.sum()} of {true_ids.size}, "
            f"scanned {panel['scanned']:,} of {points.shape[0]:,}",
            xticks=[],
            yticks=[],
        )
        ax.grid(False)
        span = float(np.sort(np.linalg.norm(centroids - query, axis=1))[2]) * 1.25
        ax.set_xlim(query[0] - span, query[0] + span)
        ax.set_ylim(query[1] - span, query[1] + span)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04), markerscale=1.2
    )
    fig.suptitle(
        "IVF searches only the cells nearest the query — neighbours across a border are missed",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


@styled
def plot_ivf_sweep(scratch: pd.DataFrame, compiled: pd.DataFrame, k: int) -> Figure:
    """Recall vs nprobe, recall vs work speedup (scratch), recall vs wall-clock (faiss)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, grp in scratch.groupby("dataset", sort=False):
        grp = grp.sort_values("nprobe")
        color = DATASET_COLORS.get(str(name), "#555555")
        label = f"{name} (N={human(grp['n'].iloc[0])}, d={grp['d'].iloc[0]})"
        axes[0].plot(grp["nprobe"], grp["recall"], marker="o", color=color, label=label)
        axes[1].plot(grp["work_speedup"], grp["recall"], marker="o", color=color, label=label)
    nlist = int(scratch["nprobe"].max())
    axes[0].set(
        xscale="log",
        xlabel=f"nprobe (lists scanned, of nlist = {nlist})",
        ylabel=f"recall@{k}",
        title="Recall is a dial: more lists, more recall",
    )
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].legend(loc="lower right")
    news = scratch[scratch["dataset"] == "news"].sort_values("nprobe")
    for _, row in news.iterrows():
        if row["nprobe"] in (1, 4, 16):  # below-left of each point = under the curve
            axes[1].annotate(
                f"nprobe={row['nprobe']:g}",
                (row["work_speedup"], row["recall"]),
                xytext=(-7, -7),
                textcoords="offset points",
                fontsize=9,
                ha="right",
                va="top",
                color=DATASET_COLORS["news"],
            )
    axes[1].set(
        xscale="log",
        xlabel="work speedup vs exact (N / vectors scored)",
        ylabel=f"recall@{k}",
        title="Scratch IVF: speed bought with recall",
    )
    for name, grp in compiled.groupby("dataset", sort=False):
        grp = grp.sort_values("nprobe")
        axes[2].plot(
            grp["speedup_vs_exact"],
            grp["recall"],
            marker="s",
            color=DATASET_COLORS.get(str(name), "#555555"),
            label=f"{name}",
        )
    axes[2].axvline(1, color="#333333", ls=":", lw=1.2)
    axes[2].text(
        1.08,
        0.3,
        "exact search\n(IndexFlatL2)",
        fontsize=9,
        transform=axes[2].get_xaxis_transform(),
    )
    axes[2].set(
        xscale="log",
        xlabel="wall-clock speedup vs exact (faiss, 1 thread)",
        ylabel=f"recall@{k}",
        title="faiss IVF: the same trade in real time",
    )
    axes[2].legend(loc="lower center")
    for ax in axes:
        ax.set_ylim(min(0.0, scratch["recall"].min() - 0.03), 1.03)
    fig.tight_layout()
    return fig


@styled
def plot_ivf_misses(margin: np.ndarray, recall: np.ndarray, n_bins: int = 8) -> Figure:
    """Per-query recall at nprobe=1 against how close the query sits to a cell border."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    edges = np.quantile(margin, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, margin, side="right") - 1, 0, n_bins - 1)
    means = np.array([recall[which == b].mean() for b in range(n_bins)])
    sems = np.array(
        [recall[which == b].std(ddof=1) / np.sqrt((which == b).sum()) for b in range(n_bins)]
    )
    centers = np.arange(n_bins)
    ax.bar(centers, means, yerr=1.96 * sems, color="#0072B2", alpha=0.85, capsize=3)
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [f"{edges[b]:.3f}–{edges[b + 1]:.3f}" for b in range(n_bins)], rotation=25, ha="right"
    )
    for c, m, se in zip(centers, means, sems, strict=True):
        ax.text(float(c), m + 1.96 * se + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=9.5)
    ax.set(
        ylim=(0, 1.1),
        xlabel="distance to 2nd-nearest centroid ÷ distance to nearest (1.0 = on the border)",
        ylabel="mean recall@10 at nprobe = 1",
        title="Misses concentrate on queries near a cluster border",
    )
    fig.tight_layout()
    return fig


# =========================================================================== notebook 03
@styled
def plot_pq_memory(rows: pd.DataFrame, k: int) -> Figure:
    """Bytes per vector for each layout, and recall against bytes per vector."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    bars = rows.sort_values("bytes_per_vector")
    colors = [INDEX_COLORS[_family(lab)] for lab in bars["label"]]
    ax1.barh(bars["label"], bars["bytes_per_vector"], color=colors)
    exact_b = float(rows.loc[rows["label"].str.startswith("Exact"), "bytes_per_vector"].iloc[0])
    for y, b in enumerate(bars["bytes_per_vector"]):
        txt = f"{b:,.0f} B" if b >= exact_b else f"{b:,.0f} B  ({exact_b / b:.0f}x smaller)"
        ax1.text(b * 1.08, y, txt, va="center", fontsize=9.5)
    ax1.set(
        xscale="log",
        xlabel="bytes per vector (log)",
        title="Memory: floats vs codes",
        xlim=(bars["bytes_per_vector"].min() * 0.6, exact_b * 30),
    )
    ax1.grid(axis="y", visible=False)

    pq = rows[rows["label"].str.startswith("PQ")].sort_values("bytes_per_vector")
    ax2.plot(
        pq["bytes_per_vector"],
        pq["recall"],
        marker="s",
        color=INDEX_COLORS["pq"],
        label="PQ (flat ADC scan)",
    )
    for _, row in pq.iterrows():
        ax2.annotate(
            row["label"].replace("PQ ", ""),
            (row["bytes_per_vector"], row["recall"]),
            xytext=(6, -13),
            textcoords="offset points",
            fontsize=9,
        )
    ax2.plot(
        exact_b,
        1.0,
        marker="*",
        ms=16,
        color=INDEX_COLORS["exact"],
        ls="none",
        label="exact (float32)",
    )
    ax2.set(
        xscale="log",
        xlabel="bytes per vector (log)",
        ylabel=f"recall@{k}",
        title="Fewer bytes, lower recall — the memory corner",
        ylim=(0, 1.05),
    )
    ax2.legend(loc="lower right")
    fig.tight_layout()
    return fig


@styled
def plot_adc_distortion(panels: Sequence[dict[str, Any]]) -> Figure:
    """True vs ADC distance for one query, one panel per code size."""
    fig, axes = plt.subplots(
        1, len(panels), figsize=(6 * len(panels), 5), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels, strict=True):
        true, approx, top = np.asarray(p["true"]), np.asarray(p["approx"]), np.asarray(p["is_top"])
        ax.scatter(
            true[~top], approx[~top], s=6, alpha=0.35, color="#8a8a8a", label="other vectors"
        )
        ax.scatter(
            true[top],
            approx[top],
            s=40,
            color="#D55E00",
            edgecolor="black",
            lw=0.5,
            label="true 10 nearest",
            zorder=3,
        )
        lo, hi = min(true.min(), approx.min()), max(true.max(), approx.max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="y = x")
        ax.set(xlabel="true squared distance", title=p["title"])
        ax.text(
            0.03,
            0.95,
            f"rank correlation ρ = {p['spearman']:.3f}\n"
            f"true top-10 kept in ADC top-10: {p['kept']}/10",
            transform=ax.transAxes,
            va="top",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "#cccccc"},
        )
    axes[0].set_ylabel("ADC distance (from the codes)")
    axes[-1].legend(loc="lower right")
    fig.suptitle(
        "PQ distances are approximate — fewer bytes, more scatter, more rank swaps",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


@styled
def plot_ivfpq(
    ivf: pd.DataFrame, ivfpq: pd.DataFrame, pq_recall: float, memory: dict[str, float], k: int
) -> Figure:
    """IVFPQ recall plateaus at PQ's ceiling; memory of the four layouts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax1.plot(
        ivf["nprobe"],
        ivf["recall"],
        marker="o",
        color=INDEX_COLORS["ivf"],
        label="IVF-Flat (exact distances)",
    )
    ax1.plot(
        ivfpq["nprobe"],
        ivfpq["recall"],
        marker="D",
        color=INDEX_COLORS["ivfpq"],
        label="IVFPQ (PQ codes of residuals)",
    )
    ax1.axhline(
        pq_recall,
        color=INDEX_COLORS["pq"],
        ls="--",
        lw=1.5,
        label=f"flat PQ, same code size ({pq_recall:.2f})",
    )
    top = ivfpq["nprobe"].max()
    gap_lo = float(ivfpq.loc[ivfpq["nprobe"] == top, "recall"].iloc[0])
    ax1.annotate(
        "",
        xy=(top, 1.0),
        xytext=(top, gap_lo),
        arrowprops={"arrowstyle": "<->", "color": "#555555"},
    )
    ax1.text(
        top * 0.8,
        (1 + gap_lo) / 2,
        "gap = PQ\nquantization\nerror",
        ha="right",
        va="center",
        fontsize=9.5,
        color="#555555",
    )
    ax1.set(
        xscale="log",
        xlabel="nprobe",
        ylabel=f"recall@{k}",
        ylim=(0, 1.05),
        title="Two levers: IVF skips, PQ compresses",
    )
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.legend(loc="lower right")
    labels = list(memory)
    values = [memory[lab] for lab in labels]
    ax2.barh(labels, values, color=[INDEX_COLORS[_family(lab)] for lab in labels])
    for y, v in enumerate(values):
        ax2.text(v * 1.08, y, human_bytes(v), va="center", fontsize=9.5)
    ax2.set(
        xscale="log",
        xlabel="index size (analytic, log)",
        title="Memory for the same vectors",
        xlim=(min(values) * 0.5, max(values) * 12),
    )
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: human_bytes(v)))
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# =========================================================================== notebook 04
@styled
def plot_hnsw_layers_walk(
    level_counts: np.ndarray, m: int, hops: Sequence[tuple[int, float]], ndis: int, n: int
) -> Figure:
    """Nodes per layer vs the geometric law, and one query's walk down the layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    levels = np.arange(level_counts.size)
    on_layer = np.cumsum(level_counts[::-1])[::-1]  # nodes present on layer l (top level >= l)
    expected = n * (1.0 / m) ** levels
    ax1.bar(
        levels, on_layer, color=INDEX_COLORS["hnsw"], alpha=0.85, label="nodes on layer (measured)"
    )
    ax1.plot(levels, expected, "o--", color="#333333", label=f"N / m^layer (m = {m})")
    for lev, cnt in zip(levels, on_layer, strict=True):  # inside the bar, clear of the line
        ax1.text(
            lev,
            cnt / 3.5,
            f"{cnt:,}",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )
    ax1.set(
        yscale="log",
        xlabel="layer",
        ylabel="nodes (log)",
        xticks=levels,
        title="A few express layers over a dense layer 0",
        ylim=(0.5, n * 4),
    )
    ax1.legend(loc="upper right")

    layer_colors = plt.get_cmap("viridis")(np.linspace(0.85, 0.15, max(level_counts.size, 2)))
    steps = np.arange(len(hops))
    dists = np.array([d for _, d in hops])
    layers = np.array([lay for lay, _ in hops])
    ax2.plot(steps, np.minimum.accumulate(dists), color="#333333", lw=1.4, label="best so far")
    for lay in sorted(set(layers.tolist()), reverse=True):
        sel = layers == lay
        kind = "greedy walk" if lay > 0 else "beam search"
        ax2.scatter(
            steps[sel],
            dists[sel],
            s=26,
            color=layer_colors[lay],
            zorder=3,
            label=f"layer {lay}: {kind}, {sel.sum()} nodes",
        )
    ax2.set(
        xlabel="step of the search",
        ylabel="distance to the query",
        title=f"One query: {ndis:,} of {n:,} vectors scored ({ndis / n:.1%})",
    )
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


@styled
def plot_hnsw_sublinear(ladders: Sequence[dict[str, Any]]) -> Figure:
    """Vectors scored per query vs N: exact is the diagonal, HNSW bends away from it."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    all_n = np.concatenate([np.asarray(lad["n"], dtype=float) for lad in ladders])
    grid = np.geomspace(all_n.min(), all_n.max(), 20)
    ax1.plot(grid, grid, "--", color=INDEX_COLORS["exact"], label="exact: all N vectors")
    for lad in ladders:
        n, work = np.asarray(lad["n"], float), np.asarray(lad["work"], float)
        ax1.plot(n, work, marker=lad.get("marker", "o"), color=lad["color"], label=lad["label"])
        ax2.plot(
            n, work / n, marker=lad.get("marker", "o"), color=lad["color"], label=lad["label"]
        )
        ax2.annotate(
            f"{work[-1] / n[-1]:.1%}",
            (n[-1], work[-1] / n[-1]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9.5,
            color=lad["color"],
        )
    ax1.set(
        xscale="log",
        yscale="log",
        xlabel="vectors in the database (N)",
        ylabel="vectors scored per query",
        title="Work per query: linear vs sub-linear",
    )
    ax2.set(
        xscale="log",
        yscale="log",
        xlabel="vectors in the database (N)",
        ylabel="share of the database scored",
        title="HNSW touches a shrinking fraction",
    )
    ax2.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.1%}" if v < 0.01 else f"{v:.0%}")
    )
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(_HUMAN_TICKS)
    ax1.legend(loc="upper left")
    fig.tight_layout()
    return fig


@styled
def plot_hnsw_knobs(
    scratch: pd.DataFrame,
    hnswlib_curve: pd.DataFrame,
    builds: pd.DataFrame,
    exact_bpv: float,
    k: int,
) -> Figure:
    """ef_search and m: recall vs ef, recall vs work, and what m costs in memory/build time."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    cmap = plt.get_cmap("Greens")
    ms = sorted(scratch["m"].unique())
    shades = {m: cmap(0.35 + 0.6 * i / max(len(ms) - 1, 1)) for i, m in enumerate(ms)}
    for m, grp in scratch.groupby("m"):
        grp = grp.sort_values("ef_search")
        axes[0].plot(
            grp["ef_search"], grp["recall"], marker="^", color=shades[m], label=f"scratch m={m}"
        )
        axes[1].plot(grp["work"], grp["recall"], marker="^", color=shades[m], label=f"m={m}")
    hl = hnswlib_curve.sort_values("ef_search")
    axes[0].plot(
        hl["ef_search"],
        hl["recall"],
        "k--",
        marker="x",
        lw=1.5,
        label=f"hnswlib m={int(hl['m'].iloc[0])}",
    )
    axes[0].set(
        xscale="log",
        xlabel="ef_search (beam width)",
        ylabel=f"recall@{k}",
        title="ef_search: wider beam, higher recall",
    )
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].set(
        xscale="log",
        xlabel="vectors scored per query (log)",
        ylabel=f"recall@{k}",
        title="…paid for in work per query",
    )
    axes[1].legend(loc="lower right", fontsize=9)
    lo = min(scratch["recall"].min(), hl["recall"].min())
    for ax in axes[:2]:
        ax.set_ylim(max(0.0, lo - 0.05), 1.01)
    b = builds.sort_values("m")
    xs = np.arange(len(b))
    overhead = b["bytes_per_vector"] - exact_bpv  # links + bookkeeping on top of the floats
    axes[2].bar(xs, overhead, color=[shades[m] for m in b["m"]])
    for x, (extra, total, secs) in enumerate(
        zip(overhead, b["bytes_per_vector"], b["build_s"], strict=True)
    ):
        axes[2].text(
            x,
            extra + overhead.max() * 0.03,
            f"+{extra:,.0f} B\n({total:,.0f} B total)\nbuild {secs:.0f} s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels([f"m={m}" for m in b["m"]])
    axes[2].set(
        ylabel=f"bytes per vector on top of the {exact_bpv:,.0f} B of floats",
        ylim=(0, overhead.max() * 1.45),
        title="m: more links, more memory",
    )
    axes[2].grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# =========================================================================== notebook 05
def _curve_label(row: pd.Series) -> str:
    params = json.loads(row["params"]) if isinstance(row["params"], str) else row["params"]
    name = INDEX_NAMES.get(row["index"], row["index"])
    if row["index"] == "hnsw" and "m" in params:
        return f"{name} m={params['m']}"
    if row["index"] == "lsh" and "rows" in params:
        return f"{name} r={params['rows']}"
    if row["index"] in ("ivf", "ivfpq") and "nlist" in params:
        extra = f", {params['n_subvectors']} B" if "n_subvectors" in params else ""
        return f"{name} nlist={params['nlist']}{extra}"
    return name


def _knob_value(params: str | dict) -> float:
    """The swept knob of a setting (nprobe, ef_search, code size or bands), for ordering curves."""
    p = json.loads(params) if isinstance(params, str) else params
    for key in ("nprobe", "ef_search", "n_subvectors", "bands"):
        if key in p:
            return float(p[key])
    return 0.0


def _build_label(row: pd.Series) -> str:
    """One build of an index (search knobs excluded), e.g. 'PQ 48 B codes', 'HNSW m=16'."""
    params = json.loads(row["params"]) if isinstance(row["params"], str) else row["params"]
    if row["index"] == "exact":
        return "Exact (float32)"
    if row["index"] == "pq":
        return f"PQ {params.get('n_subvectors')} B codes"
    if row["index"] == "lsh":
        return f"LSH {params.get('bands')} bands x {params.get('rows')} bits"
    return (
        _curve_label(row)
        if row["index"] != "ivfpq"
        else f"IVFPQ {params.get('n_subvectors')} B codes"
    )


@styled
def plot_triangle(
    df: pd.DataFrame, speed_col: str, speed_label: str, title: str, k: int
) -> Figure:
    """Left: recall vs speed for every setting, the recall-for-speed frontier as a grey band.
    Right: bytes per vector of every build, labelled with the best recall that build reaches.

    Search knobs do not change memory, so the right panel shows builds, not settings. The two
    panels have different owners (HNSW left, PQ right): that is the triangle in one picture.
    """
    from src.evaluation.triangle import pareto_mask  # noqa: PLC0415 - avoid a module cycle

    data = df.assign(_label=df.apply(_curve_label, axis=1)).reset_index(drop=True)
    builds = data.loc[data.groupby(["_label", "nbytes"], sort=False)["recall"].idxmax()].copy()
    builds["_build"] = builds.apply(_build_label, axis=1)
    builds = builds.sort_values("bytes_per_vector", ascending=False).reset_index(drop=True)
    mem_front = pareto_mask(builds, {"recall": "max", "bytes_per_vector": "min"})

    height = max(7.6, 0.42 * len(builds) + 2.6)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(17, height), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    linestyles = ["-", "--", ":"]
    shades = [1.0, 0.6, 1.45]  # same family, several curves: base, darker, lighter
    seen: dict[str, int] = {}
    for label, grp in data.groupby("_label", sort=False):
        fam = str(grp["index"].iloc[0])
        i = seen.get(fam, 0)
        seen[fam] = i + 1
        color = _shade(INDEX_COLORS[fam], shades[i % 3])
        # connect points in knob order: sub-0.1 ms timings are noisy enough to swap neighbours
        grp_s = grp.assign(_knob=grp["params"].map(_knob_value)).sort_values("_knob")
        ax1.plot(
            grp_s[speed_col],
            grp_s["recall"],
            ls=linestyles[i % 3],
            lw=2.6,
            marker=INDEX_MARKERS[fam],
            ms=16 if fam == "exact" else 7.5,
            color=color,
            label=label,
            zorder=3,
        )
    front = data[pareto_mask(data, {"recall": "max", speed_col: "max"})].sort_values(speed_col)
    ax1.plot(
        front[speed_col],
        front["recall"],
        color="black",
        lw=11,
        alpha=0.13,
        zorder=1,
        solid_capstyle="round",
        label="frontier: best recall at each speed",
    )
    ax1.set(
        xscale="log",
        xlabel=speed_label,
        ylabel=f"recall@{k}",
        ylim=(max(0.0, data["recall"].min() - 0.04), 1.03),
    )
    ax1.set_title("Recall vs speed (every setting)", fontsize=14)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax1.set_xlim(data[speed_col].min() / 1.3, data[speed_col].max() * 1.3)
    # below the panel, outside the data: timings move between runs, so no in-plot spot is safe
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=10.5)

    ys = np.arange(len(builds))
    colors = [INDEX_COLORS[str(f)] for f in builds["index"]]
    ax2.barh(ys, builds["bytes_per_vector"], color=colors, height=0.68)
    for y, (bpv, rec, on_front) in enumerate(
        zip(builds["bytes_per_vector"], builds["recall"], mem_front, strict=True)
    ):
        ax2.text(
            bpv * 1.1,
            y,
            f"{bpv:,.0f} B  ·  recall {rec:.2f}",
            va="center",
            fontsize=10.5,
            fontweight="bold" if on_front else "normal",
        )
    ax2.set_yticks(ys)
    ax2.set_yticklabels(builds["_build"], fontsize=10.5)
    ax2.set_xscale("log")
    ax2.set_xlim(builds["bytes_per_vector"].min() / 1.5, builds["bytes_per_vector"].max() * 18)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f} B"))
    ax2.set_xlabel("bytes per vector (analytic, log)")
    ax2.set_title("Memory per vector, and the best recall each build reaches", fontsize=14)
    ax2.grid(axis="y", visible=False)
    ax2.text(
        1.0,
        -0.13,
        "bold = best recall for its memory (frontier)",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


# =========================================================================== notebook 06
@styled
def plot_when_index_pays(df: pd.DataFrame, target_recall: float, reference_n: int) -> Figure:
    """Latency at a fixed recall target and build time, as N grows."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.9))
    styles = {
        "exact (faiss Flat)": (INDEX_COLORS["exact"], "*"),
        "IVF (faiss)": (INDEX_COLORS["ivf"], "o"),
        "HNSW (hnswlib)": (INDEX_COLORS["hnsw"], "^"),
    }
    for method, grp in df.groupby("method", sort=False):
        color, marker = styles.get(str(method), ("#555555", "o"))
        grp = grp.sort_values("n")
        ax1.plot(
            grp["n"],
            grp["latency_ms"],
            marker=marker,
            ms=9 if marker == "*" else 6,
            color=color,
            label=str(method),
        )
        if not str(method).startswith("exact"):
            ax2.plot(grp["n"], grp["build_s"], marker=marker, color=color, label=str(method))
            last = grp.iloc[-1]
            ax1.annotate(
                f"{last['knob']}",
                (last["n"], last["latency_ms"]),
                xytext=(6, -3),
                textcoords="offset points",
                fontsize=8.5,
                color=color,
            )
    ax1.set(
        xscale="log",
        yscale="log",
        xlabel="vectors in the database (N)",
        ylabel="latency per query (ms, 1 thread)",
        title=f"Query latency at recall@10 ≥ {target_recall:.2f}",
    )
    ax2.set(
        xscale="log",
        xlabel="vectors in the database (N)",
        ylabel="build time (s, 1 thread)",
        title="…and the build time you pay up front",
    )
    sizes = sorted(df["n"].unique())
    for ax in (ax1, ax2):
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(_HUMAN_TICKS)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


@styled
def plot_recall_sample_size(df: pd.DataFrame, true_recall: float, label: str) -> Figure:
    """How precisely n sampled queries pin down recall: spread of estimates vs the 95% CI."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xs = np.arange(len(df))
    ax.fill_between(
        xs,
        df["ci_low"],
        df["ci_high"],
        color="#0072B2",
        alpha=0.15,
        label="typical 95% interval from one sample",
    )
    ax.errorbar(
        xs,
        df["median"],
        yerr=[df["median"] - df["p2_5"], df["p97_5"] - df["median"]],
        fmt="o",
        color="#0072B2",
        capsize=4,
        label="middle 95% of repeated estimates",
    )
    ax.axhline(
        true_recall,
        color="#D55E00",
        ls="--",
        lw=1.5,
        label=f"recall on all queries ({true_recall:.3f})",
    )
    for x, half in zip(xs, (df["ci_high"] - df["ci_low"]) / 2, strict=True):
        ax.text(float(x), df["ci_high"].iloc[x] + 0.004, f"±{half:.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in df["n_sample"]])
    span = df["ci_high"].max() - df["ci_low"].min()
    ax.set_ylim(
        min(df["ci_low"].min(), df["p2_5"].min()) - 0.05 * span, df["ci_high"].max() + 0.12 * span
    )
    ax.set(
        xlabel="queries recall-tested against brute force",
        ylabel="estimated recall@10",
        title=f"A few hundred queries pin recall down ({label})",
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


# =========================================================================== app (plotly)
def triangle_plotly(df: pd.DataFrame, speed_col: str, speed_label: str, x_axis: str) -> go.Figure:
    """Interactive recall-vs-speed or recall-vs-memory chart for the Streamlit app."""
    fig = go.Figure()
    data = df.assign(_label=df.apply(_curve_label, axis=1))
    x_col = speed_col if x_axis == "speed" else "bytes_per_vector"
    for label, grp in data.groupby("_label", sort=False):
        fam = str(grp["index"].iloc[0])
        grp = grp.sort_values(x_col)
        fig.add_trace(
            go.Scatter(
                x=grp[x_col],
                y=grp["recall"],
                mode="lines+markers",
                name=str(label),
                line={"color": INDEX_COLORS[fam]},
                marker={"size": 12 if fam == "exact" else 8},
                text=grp["params"],
                hovertemplate="%{text}<br>x=%{x:.3g}<br>recall=%{y:.3f}",
            )
        )
    fig.update_layout(
        xaxis={"type": "log", "title": speed_label if x_axis == "speed" else "bytes per vector"},
        yaxis={"title": "recall@10", "range": [max(0, data["recall"].min() - 0.05), 1.02]},
        height=480,
        margin={"l": 40, "r": 10, "t": 30, "b": 40},
        legend={"orientation": "v"},
    )
    return fig


def scaling_plotly(sizes: Sequence[int], series: dict[str, Sequence[float]]) -> go.Figure:
    """Interactive latency-vs-N chart (log-log) for the app's scaling tab."""
    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(go.Scatter(x=list(sizes), y=list(values), mode="lines+markers", name=name))
    fig.update_layout(
        xaxis={"type": "log", "title": "vectors in the database (N)"},
        yaxis={"type": "log", "title": "latency per query (ms)"},
        height=420,
        margin={"l": 40, "r": 10, "t": 30, "b": 40},
    )
    return fig
