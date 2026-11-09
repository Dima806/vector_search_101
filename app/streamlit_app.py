"""Vector Search 101 — four views of the recall-speed-memory trade.

1. Scaling wall     — exact search's latency grows with N; tuned indexes stay nearly flat.
2. Index explorer   — move an index's knobs, watch recall, latency and memory move together.
3. The triangle     — every index on recall-vs-speed and recall-vs-memory axes.
4. Query inspector  — one query: the lists IVF probed, the path HNSW walked, hits and misses.

All algorithms live in ``src/``; this file only wires widgets to them. Speed numbers follow the
project's rule: wall-clock only for compiled engines (faiss, hnswlib), work for scratch code.
Run with ``make run`` after ``make data``.
"""

import json
import time

import pandas as pd
import streamlit as st

from src import visualisation as vis
from src.artifacts import RESULTS_DIR
from src.config import get_settings
from src.datasets.vectors import VectorDataset, cached_names, exact_ground_truth, load_dataset
from src.evaluation.speed import evaluate, median_latency_ms, pinned_threads
from src.indexes.exact import ExactIndex
from src.indexes.hnsw import HNSWIndex, load_or_build
from src.indexes.ivf import IVFIndex
from src.reference.compare import make_reference

S = get_settings()
K = S.eval.k
TIMED_QUERIES = 100

st.set_page_config(page_title="Vector Search 101", page_icon="🧭", layout="wide")


# ----------------------------------------------------------------------------- cached data
@st.cache_resource(show_spinner="Loading vectors…")
def dataset(name: str) -> VectorDataset:
    return load_dataset(name)


def results(slug: str) -> dict | None:
    path = RESULTS_DIR / f"{slug}.json"
    return json.loads(path.read_text()) if path.exists() else None


def max_nlist(n: int) -> int:
    """Largest nlist that still gives k-means ~39 training points per centroid."""
    return max(4, n // 39)


@st.cache_resource(show_spinner="Building the index (cached for this session)…")
def compiled_index(name: str, params: tuple[tuple[str, int], ...]) -> tuple[object, float]:
    ds = dataset("news")
    kwargs = dict(params)
    engine = "hnswlib" if name == "hnsw" else "faiss"
    index = make_reference(name, engine=engine, **kwargs)
    start = time.perf_counter()
    with pinned_threads(S.bench.threads):
        if name in ("pq", "ivfpq"):
            index.train(ds.base[: S.pq.train_size])
        index.add(ds.base)
    return index, time.perf_counter() - start


@st.cache_resource(show_spinner="Timing exact search…")
def exact_latency_ms() -> float:
    ds = dataset("news")
    flat = make_reference("exact")
    flat.add(ds.base)
    with pinned_threads(S.bench.threads):
        return median_latency_ms(flat, ds.queries, K, warmup=2, max_queries=TIMED_QUERIES)


@st.cache_resource(show_spinner="Building the scratch IVF index…")
def scratch_ivf() -> IVFIndex:
    ds = dataset("news")
    index = IVFIndex(nlist=min(S.ivf.nlist, max_nlist(ds.n)), seed=S.seed)
    index.train(ds.base)
    index.add(ds.base)
    return index


@st.cache_resource(show_spinner="Loading the scratch HNSW graph (built once by notebook 04)…")
def scratch_hnsw() -> tuple[HNSWIndex, int]:
    ds = dataset("news")
    n = min(S.hnsw.max_scratch_n, ds.n)
    graph, _, _ = load_or_build(
        ds.base[:n], ds.key, S.hnsw.m, S.hnsw.ef_construction, S.hnsw.ef_search, S.seed
    )
    return graph, n


# ----------------------------------------------------------------------------- tabs
def tab_scaling() -> None:
    st.subheader("Exact search touches every vector, so its cost grows with N")
    wall, pays = results("01_scaling_wall"), results("06_when_an_index_pays")
    left, right = st.columns(2)
    if wall:
        m = pd.DataFrame(wall["measured"])
        left.markdown(f"**Notebook 01** — random {wall['d']}-d vectors, 1 thread")
        left.plotly_chart(
            vis.scaling_plotly(
                m["n"], {"numpy exact": m["numpy_ms"], "faiss IndexFlatL2": m["faiss_ms"]}
            ),
            width="stretch",
        )
    if pays:
        df = pd.DataFrame(pays["rows"])
        right.markdown(
            f"**Notebook 06** — clustered vectors, every index tuned to recall@10 ≥ {pays['target_recall']}"
        )
        series = {m: g.sort_values("n")["latency_ms"] for m, g in df.groupby("method", sort=False)}
        right.plotly_chart(vis.scaling_plotly(sorted(df["n"].unique()), series), width="stretch")

    st.markdown("**Try it:** time exact search yourself on a prefix of the news vectors.")
    ds = dataset("news")
    sizes = sorted({max(1, ds.n // 8), max(1, ds.n // 4), max(1, ds.n // 2), ds.n})
    n = st.select_slider("database size N", options=sizes, value=sizes[-1])
    if st.button("Time exact search"):
        index = ExactIndex()
        index.add(ds.base[:n])
        with pinned_threads(S.bench.threads):
            ms = median_latency_ms(index, ds.queries, K, warmup=2, max_queries=50)
        st.session_state.setdefault("timings", {})[n] = ms
    timings = st.session_state.get("timings", {})
    if timings:
        table = pd.DataFrame(sorted(timings.items()), columns=["N", "ms per query"])
        table["µs per 1,000 vectors"] = 1e3 * table["ms per query"] / (table["N"] / 1e3)
        st.dataframe(table, hide_index=True)
        st.caption("Roughly constant cost per vector: that is what linear scaling means.")


def tab_explorer() -> None:
    st.subheader("Move the knobs: recall, speed and memory move together")
    ds = dataset("news")
    st.caption(
        f"news: {ds.n:,} x {ds.d} sentence embeddings, {ds.queries.shape[0]:,} held-out queries. "
        "Compiled engines (faiss, hnswlib), 1 thread; latency = median of single-query searches."
    )
    family = st.radio("index", ["IVF", "PQ", "IVFPQ", "HNSW"], horizontal=True)
    cols = st.columns(3)
    nlists = [v for v in (64, 128, 256, 512) if v <= max_nlist(ds.n)] or [max_nlist(ds.n)]
    subvecs = [v for v in S.pq.n_subvectors_sweep if ds.d % v == 0] or [1]
    default_nlist = nlists.index(S.ivf.nlist) if S.ivf.nlist in nlists else len(nlists) - 1
    if family == "IVF":
        nlist = cols[0].selectbox("nlist (build)", nlists, index=default_nlist)
        nprobe = cols[1].select_slider(
            "nprobe (search)",
            [p for p in (1, 2, 4, 8, 16, 32, 64, 128) if p <= nlist],
            8 if nlist >= 8 else 1,
        )
        name, build, search = "ivf", (("nlist", nlist),), ("nprobe", nprobe)
    elif family == "PQ":
        m = cols[0].selectbox("bytes per code (build)", subvecs, index=min(3, len(subvecs) - 1))
        name, build, search = "pq", (("n_subvectors", m),), None
    elif family == "IVFPQ":
        nlist = cols[0].selectbox("nlist (build)", nlists, index=default_nlist)
        m = cols[1].selectbox("bytes per code (build)", subvecs, index=min(3, len(subvecs) - 1))
        nprobe = cols[2].select_slider(
            "nprobe (search)",
            [p for p in (1, 2, 4, 8, 16, 32, 64, 128) if p <= nlist],
            8 if nlist >= 8 else 1,
        )
        name, build, search = "ivfpq", (("nlist", nlist), ("n_subvectors", m)), ("nprobe", nprobe)
    else:
        m = cols[0].selectbox("m — links per node (build)", [8, 16, 32], index=1)
        ef = cols[1].select_slider("ef_search (search)", [10, 16, 32, 64, 128, 256], 64)
        name, build, search = (
            "hnsw",
            (("m", m), ("ef_construction", S.hnsw.ef_construction)),
            ("ef_search", ef),
        )

    index, build_s = compiled_index(name, build)
    if search is not None:
        setattr(index, search[0], search[1])
    with pinned_threads(S.bench.threads):
        res = evaluate(
            index, ds.queries, ds.gt_ids, K, timed=True, warmup=2, max_queries=TIMED_QUERIES
        )
    exact_ms = exact_latency_ms()
    c = st.columns(4)
    c[0].metric(f"recall@{K}", f"{res['recall']:.3f}")
    c[1].metric(
        "latency / query",
        vis.human_ms(res["latency_ms"]),
        f"{exact_ms / res['latency_ms']:.1f}x vs exact",
    )
    c[2].metric(
        "memory / vector",
        f"{index.nbytes / ds.n:,.0f} B",
        f"{ds.d * 4 / (index.nbytes / ds.n):.1f}x smaller than floats",
        delta_color="off",
    )
    c[3].metric("build time", f"{build_s:.1f} s")


def tab_triangle() -> None:
    st.subheader("No index wins on all three axes")
    choice = st.radio(
        "measurement",
        ["from scratch — work (notebook 05)", "compiled — wall-clock (notebook 05)"],
        horizontal=True,
    )
    slug, col, label = (
        ("05_triangle_work", "work_speedup", "work speedup vs exact (N / vectors scored)")
        if choice.startswith("from scratch")
        else (
            "05_triangle_wallclock",
            "speedup_vs_exact",
            "wall-clock speedup vs exact (1 thread)",
        )
    )
    data = results(slug)
    if data is None:
        st.info("Run notebook 05 (`make notebooks`) to produce the sweep table.")
        return
    df = pd.DataFrame(data["rows"])
    left, right = st.columns(2)
    left.markdown("**Recall vs speed**")
    left.plotly_chart(vis.triangle_plotly(df, col, label, "speed"), width="stretch")
    right.markdown("**Recall vs memory**")
    right.plotly_chart(vis.triangle_plotly(df, col, label, "memory"), width="stretch")
    dom = data["dominance"]
    st.markdown(
        f"Families on the computed 3-D Pareto frontier: **{', '.join(vis.INDEX_NAMES[f] for f in dom['families_on_frontier'])}**. "
        f"Fully dominated: **{', '.join(vis.INDEX_NAMES[f] for f in dom['fully_dominated']) or 'none'}**."
    )


def neighbour_table(ds: VectorDataset, ids: list[int], true_ids: set[int]) -> pd.DataFrame:
    texts = ds.texts or [""] * ds.n
    return pd.DataFrame(
        {
            "hit": ["✅" if i in true_ids else "❌" for i in ids],
            "id": ids,
            "sentence": [texts[i] for i in ids],
        }
    )


def tab_inspector() -> None:
    st.subheader("Inside one query")
    ds = dataset("news")
    qi = st.number_input("held-out query #", 0, ds.queries.shape[0] - 1, 0)
    if ds.query_texts:
        st.markdown(f"> {ds.query_texts[qi]}")
    q = ds.queries[qi]
    left, right = st.columns(2)

    ivf = scratch_ivf()
    nprobe = left.select_slider(
        "IVF nprobe", [p for p in (1, 2, 4, 8, 16, 32, 64) if p <= ivf.nlist], 1
    )
    ivf.nprobe = nprobe
    tr = ivf.trace(q, K)
    true_ivf = set(ds.gt_ids[qi, :K].tolist())
    left.markdown(
        f"**IVF** (nlist={ivf.nlist}): probed lists {tr.probed_lists} with sizes {tr.list_sizes} — "
        f"scored {tr.ndis:,} of {ds.n:,} vectors, found **{len(true_ivf & set(tr.ids))}/{K}** true neighbours."
    )
    left.dataframe(neighbour_table(ds, tr.ids, true_ivf), hide_index=True)

    graph, n = scratch_hnsw()
    ef = right.select_slider("HNSW ef_search", [10, 16, 32, 64, 128], 32)
    graph.ef_search = ef
    tr_h = graph.trace(q, K)
    gt_h, _ = exact_ground_truth(ds.base[:n], q[None, :], K)
    true_h = set(gt_h[0].tolist())
    hops = {
        lay: len(path) - 1
        for lay, path in sorted(tr_h.layer_paths.items(), reverse=True)
        if lay > 0
    }
    right.markdown(
        f"**HNSW** (scratch, first {n:,} vectors, m={graph.m}): greedy hops per upper layer {hops}, "
        f"{len(tr_h.layer_paths[0])} beam expansions on layer 0 — scored {tr_h.ndis:,} of {n:,} vectors, "
        f"found **{len(true_h & set(tr_h.ids))}/{K}** true neighbours."
    )
    right.dataframe(neighbour_table(ds, tr_h.ids, true_h), hide_index=True)
    st.caption(
        "✅ = one of the query's true 10 nearest neighbours (exact search); ❌ = an approximate pick."
    )


def main() -> None:
    st.title("Vector Search 101")
    st.caption(
        "Exact nearest-neighbour search is perfect, and that is exactly why it does not scale."
    )
    if "news" not in cached_names():
        st.info(
            "No cached vectors found. Run `make data` first (embeds the corpus once, ~10 min on 2 CPUs)."
        )
        st.stop()
    tabs = st.tabs(["Scaling wall", "Index explorer", "The triangle", "Query inspector"])
    with tabs[0]:
        tab_scaling()
    with tabs[1]:
        tab_explorer()
    with tabs[2]:
        tab_triangle()
    with tabs[3]:
        tab_inspector()


main()
