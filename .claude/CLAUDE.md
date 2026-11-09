# Project: vector_search_101

> Scope source of truth: `.llm/PRD_vector_search_101.md` (gitignored, local only).
> This file is the operational contract: the PRD wins on *what*, this file wins on *how*.
> Update this file when either changes; keep §2 current.

---

## 1. Identity

**Approximate nearest-neighbour (ANN) search built from scratch**, CPU-only, for a data-science
audience. Deliverables: 6 notebooks + `src/` library + 1 Streamlit app + tests, supporting a Medium
article (2,200–2,500 words, 7–8 figures taken from the notebooks).

Build-up (each notebook depends on the one before; NB05 reuses NB04's cached graph, NB06 reads
NB03–05 JSON): `01 exact and why it fails` → `02 inverted file` → `03 product quantization` →
`04 HNSW from scratch` → `05 the triangle` → `06 decision framework`.

**Thesis.** Exact kNN compares the query with every stored vector: always right, cost linear in N.
So every fast search is **approximate**: it skips comparisons and accepts a measurable miss rate
(1 − recall). IVF skips by clustering, PQ shrinks by compressing, HNSW skips by walking a graph.
Every knob moves a point around the **recall–speed–memory triangle**; no index dominates. A vector
database is this machinery with the point picked for you.

**Showpiece:** notebook 04 — HNSW from scratch, validated against hnswlib, one query's walk traced
through the layers, touching a sub-linear fraction of the database.

**Lessons the code demonstrates** (measured, not just stated):
1. Exact = distance to every vector → perfectly correct, query cost ∝ N → dies at scale. (NB01)
2. Every fast search is approximate; miss rate = 1 − recall, and recall is a *knob*, not a bug. (NB02–04)
3. IVF: cluster, search the `nprobe` nearest clusters. PQ: vectors → bytes (memory). IVFPQ: both. (NB02–03)
4. HNSW: layered navigable graph, greedy walk in ~log N hops; knobs `m`, `ef_construction`,
   `ef_search`. What most vector DBs use. (NB04)
5. The triangle: HNSW owns recall-for-speed, PQ owns recall-for-memory, neither owns both; LSH is
   dominated on dense embeddings. (NB05)
6. Below ~500K vectors exact search is fine (`retrieval_arena`); above, pick by the binding
   constraint, and always recall-test against exact on a sample. (NB06)

**Related projects** (github.com/Dima806 — cross-link in prose, **never import**). The canonical
list with one-line descriptions is README "Related projects". Refresh with
`gh api "users/Dima806/repos?per_page=100" --paginate`; before claiming what a sibling covers, read
its README: `gh api repos/Dima806/<repo>/readme --jq .content | base64 -d`. Facts already checked:
- `retrieval_arena` — exact numpy search at 384-d: 100K ≈ 13 ms, 500K ≈ 60 ms per query (768 MB);
  HNSW earns its complexity only past ~500K vectors, tight SLOs or heavy updates.
- `embeddings_101` — same encoder as here (`all-MiniLM-L6-v2`, 384-d).
- `hashing_101` — **its LSH is MinHash + banding for Jaccard on sets**, not dense vectors. We port
  the banding (S-curve `1 − (1 − s^r)^b`) and swap the hash family for random hyperplanes (§6).
- `quantization_101` — *scalar* quantization of model weights (absmax / zero-point). It does **not**
  cover PQ or codebooks: PQ is "the same idea with a learned codebook per sub-vector".
- `distance_arena` — metric choice; on unit-normalised vectors L2, cosine and IP rank identically.
- `rag_101` — RAG's retrieve step is exactly this search.

---

## 2. Current state

**Built and green.** `src/` (all modules), `app/streamlit_app.py`, `config/settings.yaml`, 6
executed notebooks, 79 tests, `outputs/figures/` (15 PNGs, budget 19), `outputs/results/` (JSON
behind every figure and claim). `make ci` passes. README written (results table, notebooks,
quickstart, how it works, limitations, related projects). **Not built (deliberately):**
`.devcontainer/`, `.github/workflows/ci.yml`. Nothing committed yet.

Toolchain in `uv.lock`: Python 3.14, numpy 2.x, faiss-cpu 1.15.1 (abi3 wheel), hnswlib 0.8.0
(compiled from sdist by g++), torch 2.15.0+cpu (CPU index), sentence-transformers 6.x.

Measured on the 2-CPU Codespace (1 thread for every benchmark):
- `make data`: ~9 min the first time (embedding ~51k sentences dominates), then cached (207 MB).
- `make notebooks`: see the runtime table below; every notebook < 7 min.
- `make test`: 79 tests in ~30 s, offline.

| NB | runtime | PNGs |
|---|---|---|
| 01 | 35 s | 2 |
| 02 | 133 s | 3 |
| 03 | 215 s | 3 |
| 04 | 223 s | 3 |
| 05 | 313 s | 2 |
| 06 | 78 s | 2 |

**Keep this section current:** what is built, test count, notebook runtimes, PNG/JSON counts.

---

## 3. Stack and hard constraints

| Thing | Decision |
|---|---|
| Python | 3.11+ (`requires-python = ">=3.11"`, ruff `target-version = "py311"`) |
| Packages | `uv` only. Never `pip install`, never `python -m venv`. Deps via `uv add` / `uv remove` |
| Hardware | 2 CPU cores, 8 GB RAM, **no GPU**. Everything runs in a GitHub Codespace |
| Indexes | **From scratch in numpy** (+ scikit-learn k-means). faiss / hnswlib only in `src/reference/` (and tests) |
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2` (384-d), CPU, imported lazily in `src/datasets/vectors.py` only |
| dtypes | vectors `float32` C-contiguous · ids `int64` · PQ codes `uint8` · graph links `int32` |
| Memory | Peak process RSS < ~6 GB. Float sets capped at a few hundred thousand × ≤ 384-d (config); PQ shows fitting more |
| Notebook runtime | < 7 min each, whole notebook, measured by `make notebooks` / `make nb` |
| Licensing | 100% open source. No API keys, no hosted vector DB, no network except first-time model/corpus download |
| Quality gates | ruff (line-length 99) + `ty check src/` + pytest; `make lint` and `make test` zero errors |

If a change threatens a hard constraint (GPU dependency, memory, runtime budget), **stop and flag
it** rather than proceeding.

Runtime deps: the PRD list (`faiss-cpu`, `hnswlib`, `sentence-transformers`, `numpy`, `scipy`,
`scikit-learn`, `pandas`, `matplotlib`, `plotly`, `streamlit`, `pydantic`, `pydantic-settings`,
`pyyaml`) **plus** `torch` (declared so it can be pinned to the CPU wheel, §15.2) and `threadpoolctl`
(BLAS thread pinning, imported by `src/evaluation/speed.py`).
Dev deps: `pytest`, `pytest-cov`, `ruff`, `ty`, `jupyter`, `ipykernel`, `nbconvert`, `nbclient`.
A new dependency needs a reason in the commit message; prefer numpy / stdlib.

**Ruff rules that bite here** (`select = E,F,W,I,UP,N,B,A,SIM,PTH`):
- `N803`/`N806` — no capitalised args or locals: faiss's `D, I = index.search(xq, k)` becomes
  `dists, ids = ...`; `X` → `xb` / `xq`; HNSW `M` → `m`.
- `A001`/`A002` — don't shadow builtins: `id`, `hash`, `iter`, `input`, `type`, `max`, `min`, `filter`.
- `B006`/`B008`/`B007` — no mutable or call defaults, no unused loop variables. `PTH` — pathlib only.

**ty quirks we hit:** faiss SWIG objects have runtime-only attributes → `self.index: Any` and
`_CVAR: Any = faiss.cvar` in `src/reference/compare.py`; `import hnswlib  # ty: ignore[import-not-found]`
(compiled, no stubs); matplotlib wants floats, not numpy ints (`float(x)` at `ax.text` sites);
`np.savez(path, **arrays)` needs a line-level ignore. Never silence globally.

**Threads.** Every benchmark runs inside `pinned_threads(bench.threads)` (default 1): BLAS via
`threadpoolctl`, faiss via `faiss.omp_set_num_threads`, hnswlib via `index.set_num_threads`.

---

## 4. Commands

```
make setup         # first run: install uv if missing, uv sync --all-extras, register kernel "vector-search-101"
make sync          # uv sync --all-extras
make data          # uv run python -m src.datasets.vectors -> data/vectors/ (news, clustered, isotropic + exact GT)
make lint          # ruff format + ruff check --fix (src tests app) + ty check src/
make test          # pytest: offline, synthetic fixtures, no model download (~30 s)
make test-cov      # pytest --cov=src --cov-report=term-missing
make test-offline  # HF_HUB_OFFLINE=1, deselects @pytest.mark.network
make notebooks     # make data, then execute notebooks/0*.ipynb in place, timed; fails if any > 7 min
make nb NB=04      # execute one notebook in place, timed
make run           # streamlit on :8501
make lab           # jupyterlab on :8888
make ci            # sync + lint + test   <- must pass before any commit
make dev           # lint + test (fast loop)
make clean         # caches;  make reset = clean + rm .venv data/vectors data/indexes
```

Ad hoc, always through uv: `uv run pytest tests/test_ivf.py -k nprobe -x`,
`uv run python -m src.datasets.vectors --only news --force`. Bare `python` is the wrong interpreter.

---

## 5. Layout and module responsibilities

```
config/settings.yaml      every knob and seed (§7); nothing tunable is hard-coded elsewhere
src/
  config.py               pydantic-settings models + YAML source, cached get_settings(), env prefix VS101_
  artifacts.py            repo-root paths, data_dir() (VS101_DATA_DIR override), save_figure/save_json/load_json, params_key
  metrics/distance.py     sq_l2 / sq_l2_to_one / pairwise(l2|ip|cosine) / normalise / sq_norms
  indexes/
    __init__.py           Index protocol, SearchStats, QueryTrace, as_float32, select_topk, topk_with_ids, make_index
    exact.py              brute force + argpartition, query blocks of block_mb. Defines ground truth
    ivf.py                IVF-Flat + shared helpers: train_kmeans (random init), assign_to_nearest, training_sample
    pq.py                 ProductQuantizer codec (codebooks, encode/decode, ADC tables) + flat PQIndex
    ivfpq.py              IVF + PQ on residuals, per-list ADC tables
    hnsw.py               HNSWIndex (insert/search/trace/save/load, level_counts, reachable_fraction) + cache_path, load_or_build
    lsh.py                random-hyperplane LSH with banding (centred vectors), candidate_probability S-curve
  reference/compare.py    faiss (Flat/IVF/PQ/IVFPQ/HNSW) + hnswlib behind the Index protocol; set_threads, library_versions, make_reference
  datasets/vectors.py     `python -m src.datasets.vectors`: news + synthetic builders, exact GT, save/load/find/cached_names
  evaluation/
    recall.py             per_query_recall, recall_at_k (the only recall path), recall_on_sample (+ 95% CI)
    speed.py              pinned_threads, median_latency_ms, evaluate, timed_build, machine_info
    memory.py             analytic byte formulas for every index, compression_ratio
    triangle.py           sweep -> SWEEP_COLUMNS rows, to_frame, expand_params, pareto_mask, frontier_2d, dominance_report
  visualisation.py        every figure (matplotlib for PNGs, plotly for the app); colours, `styled` rc_context, human_* formatters
app/streamlit_app.py      4 tabs (§12); calls src/, contains no algorithm
notebooks/01..06          narrative only; import from src/; executed in place, outputs committed
tests/                    conftest + test_{exact,ivf,hnsw,vs_reference,pq,lsh,evaluation,imports,datasets,app,visualisation}.py
data/                     gitignored: vectors/ (datasets + GT), indexes/ (scratch HNSW cache), sklearn/ (20NG download)
outputs/figures/          committed PNGs, NN_<slug>.png (15 of a 19 budget)
outputs/results/          committed JSON, NN_<slug>.json — same slug as the figure it backs, plus NN_summary.json
```

Rules:
- **No algorithm lives in a notebook or the app.** If a notebook needs it, it goes in `src/`.
  (NB01 shows a 2-line numpy brute force as a *demo* and checks it equals `ExactIndex`.)
- `src/indexes/` imports numpy / scipy / scikit-learn only — `tests/test_imports.py` enforces it,
  and that faiss/hnswlib appear only under `src/reference/`.
- `src/` has no import-time side effects (no model loading, no file I/O, no seeding, no
  `matplotlib.use`). torch / sentence-transformers are imported inside `embed_texts` only.
- All paths resolve from the repo root (`src/config.py: REPO_ROOT`), never the cwd.

---

## 6. The algorithms, as implemented

Each bullet is also a test (§9).

**Distances.** Default metric is **squared L2** (what `IndexFlatL2` and hnswlib `"l2"` return),
computed as `‖q‖² + ‖x‖² − 2·q·x` in float32 and clipped at 0. Everything returns a distance
(smaller = closer): `ip` → `−q·x`, `cosine` → `1 − cos`. News embeddings are unit-norm, so L2,
cosine and IP rank identically; approximate indexes implement squared L2 only. BLAS uses different
kernels for different block shapes, so the same distance can differ in the last float32 bits
between code paths — compare ids exactly, distances with `rtol=1e-4`.

**Exact.** Distances in query blocks of ≤ `exact.block_mb`, `argpartition(k−1)` then sort those k.
Duplicate vectors are removed when datasets are built, so ties are measure-zero.

**k-means** (`train_kmeans`): scikit-learn `KMeans(init="random", n_init=1, max_iter=20, seeded)`.
Random init, as faiss does: k-means++ seeding cost more than the Lloyd iterations at 256 centroids
(0.57 s vs 0.22 s per sub-space) for the same inertia.

**IVF-Flat.** Train on `train_points_per_centroid × nlist` sampled vectors; each list holds ids +
vectors + norms. Search scans the `nprobe` nearest lists. `ndis = nlist + Σ|probed lists|`.
`nlist ≈ √N` (256 for the 50k news set, 128 for 20k; NB06 uses `2^round(log2 √N)`): the faiss
`4–16·√N` guideline is for N ≫ 1M and starves k-means of training points at our sizes.

**PQ.** `d % n_subvectors == 0`; 256 codewords per sub-space (`nbits = 8`, one `uint8` each);
codes stored transposed `(n_subvectors, N)`. ADC: tables `(nq, n_subvectors, 256)`, distance = sum
of lookups, accumulated per sub-space with `np.take(table_j, codes_j.astype(np.intp), axis=1)`
(~3× faster than fancy indexing with uint8). Never decode to search. Train ≥ 39 × 256 vectors.

**IVFPQ.** PQ trained on residuals `x − c(x)`; per query, ADC tables for all probed lists at once
(`distance_tables(q − centroids[lists])`), then `T[arange(m), codes_j].sum(1)` per list.
Recall plateaus below 1 at `nprobe = nlist` (PQ's error) — on news it lands *at* flat PQ's level
(0.67 vs 0.68), not above it.

**HNSW** (Malkov & Yashunin 2018, hnswlib's choices): level `⌊−ln(U)·mL⌋`, `mL = 1/ln m`;
insert picks `m` neighbours per layer with the heuristic (`getNeighborsByHeuristic2`: keep `c`
unless a kept `r` has `d(c, r) < d(c, base)`); caps `2m` on layer 0, `m` above, overflow re-pruned
with the same heuristic; the whole `ef_construction` result set seeds the next layer down. Search:
greedy descent to layer 1, beam `max(ef_search, k)` on layer 0. Layer-0 links live in an
`(N, 2m)` int32 array so a node's neighbours are scored in one numpy call (~2.3 ms per insert at
384-d, `m = 16`, `ef_construction = 64`). Cached as `.npz` via `load_or_build`.

**LSH.** `bands × rows` hyperplanes ~ N(0, I) (seeded); vectors are **centred** first (sentence
embeddings share a direction, so uncentred hyperplanes split them badly); band keys = packed sign
bits; one dict per band; candidates re-ranked exactly. `rows = 8` calibrated on news.

**Recall.** `k-recall@k = |approx_k ∩ true_k| / k`, mean over queries (`k = eval.k = 10`).

---

## 7. API conventions

```python
class Index(Protocol):
    name: str                          # "exact" | "ivf" | "pq" | "ivfpq" | "hnsw" | "lsh"
    last_stats: SearchStats            # per query: ndis (full distances), n_adc (code lookups), nhops; NaN = not reported
    def train(self, xb) -> None: ...   # no-op for exact / hnsw (lsh: centring + hyperplanes)
    def add(self, xb) -> None: ...
    def search(self, xq, k) -> tuple[FloatArray, IdArray]: ...   # (nq, k) ascending, padded inf / -1
    @property
    def nbytes(self) -> int: ...       # analytic compact size (§8)
    @property
    def ntotal(self) -> int: ...
```

- `xq` always 2-D. Build-time knobs are constructor args; search-time knobs (`nprobe`,
  `ef_search`) are mutable attributes — `triangle.sweep` sets them with `setattr`.
- `trace(q, k) -> QueryTrace` on IVF / IVFPQ / HNSW / LSH for one query (probed lists + sizes;
  entry point + per-layer paths + `(layer, distance)` hops; bucket sizes).
- `src/reference/compare.py` wrappers expose the same protocol; faiss IVF `ndis` adds the `nlist`
  coarse distances to match ours; hnswlib reports no counters (NaN); `nbytes` = faiss serialised
  size / hnswlib saved file size.
- Randomness: explicit `seed` everywhere (k-means `random_state`, HNSW levels, LSH planes, data).
- Typing: `FloatArray`, `IdArray`, `StatArray` aliases in `src/indexes/__init__.py`.

**Config.** `config/settings.yaml` → `src/config.py` (`YamlConfigSettingsSource`), `get_settings()`
cached, env override `VS101_<SECTION>__<KEY>`. Sections: `seed, datasets (n_queries, k_gt, real,
synthetic[]), eval, exact, scaling (NB01), ivf, pq, ivfpq, hnsw (incl. max_scratch_n, sweep_n,
ladder), lsh, triangle (NB05), decision (NB06), bench (threads, warmup, max_queries)`. Tests build
small inputs in code instead of reading YAML.

---

## 8. Measurement protocol

**Ground truth.** Exact top-`k_gt` (100) over the same base/query split; queries held out.
Notebooks on a prefix (`news.base[:20000]`) recompute GT on that prefix.

**Speed has two currencies — never on one axis:**
1. **Work** — `SearchStats.work` (vectors scored per query); work speedup `N / work`. Every
   from-scratch claim and every test.
2. **Wall-clock** — median of single-query latencies after warm-up, `pinned_threads(1)`, shown as
   speedup vs exact on the same engine (faiss `IndexFlatL2`). Compiled engines only (or
   numpy-vectorised vs numpy-vectorised, e.g. NB01's exact). Never asserted in tests.

Pure-Python HNSW at 20k vectors would lose to BLAS brute force on wall-clock — an interpreter
artefact. Hence: scratch indexes contribute recall + work; faiss/hnswlib supply wall-clock.

**Memory** (`src/evaluation/memory.py`, allocated slots, no Python overhead): exact `N·d·4`;
IVF-Flat `+ N·8 + nlist·d·4`; PQ `N·code + 256·d·4`; IVFPQ `N·(code + 8) + nlist·d·4 + 256·d·4`;
HNSW `N·d·4 + 4·(N·2m + Σ upper slots) + 12·N`; LSH `N·b·r/8 + b·N·8 + N·d·4`. Matches faiss
serialised sizes within 10% (test) — PQ/IVFPQ within 0.1%, scratch HNSW vs hnswlib file 0.3%.

**Sweep table** (`SWEEP_COLUMNS`): `index, engine, params (JSON), dataset, n, d, k, recall, ndis,
n_adc, work, work_speedup, latency_ms, qps, speedup_vs_exact, nbytes, bytes_per_vector, build_s,
threads, seed`. NB05/NB06/the app read it; `expand_params` spreads `params` into columns.

**Dominance.** `pareto_mask` + `dominance_report` compute the 3-D frontier (recall ↑, speed ↑,
bytes ↓). Because every build has its own memory, most settings are 3-D non-dominated, so the
*figure* (`plot_triangle`) shows the **2-D** frontiers instead: a grey band behind the
recall-vs-speed curves (owned by HNSW) and a bar per build sorted by bytes, labelled with its best
recall, frontier builds in bold (owned by PQ). The 3-D report goes in JSON/prose. Curves are
connected in knob order (sub-0.1 ms timings are noisy enough to swap neighbours) and the legend
sits below the panel, because timings move between runs and no in-plot spot stays free. Result on news: exact, HNSW, IVF, IVFPQ, PQ share the
3-D frontier; LSH is fully dominated.

---

## 9. Invariants the tests assert

| PRD §8 criterion | Proven by |
|---|---|
| Exact returns true neighbours; defines GT | `test_exact.py` (vs naive float64 reference, self-NN, blocks, metrics) |
| Exact query time linear in N | NB01: measured to 500k + linear fit (R² in `01_scaling_wall.json`) |
| IVF recall → 1.0 as nprobe ↑, speed ↓ | `test_ivf.py` (monotone with **no** tolerance, == 1.0 at nlist, ndis ↑) + NB02 |
| PQ memory ↓ ≥ 10× at small recall cost | `test_pq.py` + NB03 (48-byte codes: 27× with codebooks) |
| HNSW recall ↑ with ef_search; sub-linear | `test_hnsw.py` (ef sweep, caps, reachability ≥ 99%, ndis/N falls, levels ≈ 1/m) + NB04 |
| From-scratch ≈ faiss / hnswlib | `test_vs_reference.py` (recall ≤ 0.05, IVF work ratio 0.67–1.5, HNSW work ratio 0.5–2, bytes ±10%) |
| No index dominates; LSH on the frontier chart | `dominance_report` + NB05 |
| Recall always vs exact GT | `evaluation/recall.py` is the only recall path |
| CPU only | `test_imports.py::test_no_gpu_code`, torch CPU wheel |
| Notebooks < 7 min | `make notebooks` fails on overrun; runtimes in §2 |
| `make lint` + `make test` clean | `make ci` |

Also: `test_lsh.py` (measured collision rate tracks the S-curve within 0.05; more bands → recall
and work ↑; exact re-rank), `test_evaluation.py` (recall arithmetic, Pareto on a hand-made table,
sweep schema, memory formulas), `test_datasets.py` (generators, cache round trip, the news build
offline with the download and encoder monkeypatched), `test_app.py` (AppTest: no-data message;
tiny dataset in a tmp `VS101_DATA_DIR`, clicks through every tab), `test_visualisation.py` (every
figure function renders). Fixtures (`conftest.py`): seeded 3,000 × 32-d clustered set and
2,000 × 16-d isotropic set with exact GT. Never assert on wall-clock.

---

## 10. Datasets and caching

- **news** (real): 20 Newsgroups via `sklearn.datasets.fetch_20newsgroups(subset="all",
  remove=headers/footers/quotes)` → sentence split → 40–300 chars, ≥ 70% letters, case-insensitive
  dedupe (140,149 sentences available) → seeded sample of 51k → MiniLM (`normalize_embeddings=True`)
  → drop duplicate vectors → 1,000 queries + 50,000 base. Texts saved in `texts.json` (the app shows them).
- **clustered** (synthetic): 200k × 128, 200 Gaussian clusters on a 16-d subspace, noise 0.05.
- **isotropic** (control): 50k × 128 standard Gaussian — IVF needs most lists for any recall.
- NB01's scaling wall uses fresh random unit vectors at 384-d (content does not matter for exact cost).
- Cache: `data/vectors/<name>_n<N>_d<d>_s<seed>/{base,queries,gt_ids,gt_dists}.npy + meta.json`;
  `make data` skips cached names (`--force` rebuilds). HNSW graphs:
  `data/indexes/hnsw_<dataset-key>_n<N>_<params-hash>.npz` — NB04 writes the 20k graph, NB05 and
  the app read it.
- Extrapolations (1M / 10M / 100M in NB01) are drawn dashed and labelled "not measured".

---

## 11. Notebook conventions

- Narrate and orchestrate; logic from `src`. Setup cell: settings, `MACHINE = machine_info(...)`,
  and a 3-line `show(fig, slug)` helper (save PNG, close, display once).
- Fresh kernel, top to bottom, deterministic; each prints its runtime; `make nb NB=0N` times it.
- Open with question / claim / what you can do by the end; close with a **code cell** that
  renders "What this bought you, and what it cost" with `display(Markdown(f"..."))` from computed
  numbers, so prose cannot drift from output. Static markdown makes no numeric claims that are not
  printed above it.
- Every figure from `src/visualisation.py` → `outputs/figures/NN_slug.png` (150 dpi); the numbers
  behind it → `outputs/results/NN_slug.json`; plus `NN_summary.json` per notebook.
- **Image budget: ≤ 19 PNGs** (15 used: 2/3/3/3/2/2). Adding one means merging or dropping another.
- **Open every new or changed PNG** and check: no overlapping labels or legends, nothing clipped,
  labels off the data lines, human tick labels (`12.5k`, `1,536 B`, `10x`), units on axes. Past
  fixes: extrapolation labels colliding with titles, legends anchored outside the axes squeezing
  panels, white labels spilling out of bars, scientific minor ticks on log axes, rings and
  overlapping markers making the triangle unreadable (redesigned: frontier band + bars), legends
  over data that only appeared after a re-run moved the timings (put legends outside the axes).
- Executed in place (`--inplace`); committed notebooks carry outputs. Kernelspec `python3`.
- When prose and data disagree, fix the prose (it happened: "IVFPQ above flat PQ", "IVF close
  behind HNSW", "work barely moves" — all corrected to what the numbers say).

| NB | Must contain (as built) |
|---|---|
| 01 | 2-line brute force = `ExactIndex` = faiss = cached GT; needle-in-haystack histogram; scaling wall to 500k + memory; thesis |
| 02 | 2-D Voronoi toy (nprobe 1 vs 3); nprobe sweep on news / clustered / isotropic (work) + faiss wall-clock; misses vs border margin |
| 03 | PQ byte sweep (memory bars + recall); ADC vs true distance scatter (8 vs 48 B); IVFPQ plateau + memory; faiss check |
| 04 | heuristic source; incremental 20k build with size ladder; layers + one walk; sub-linear work (scratch + faiss to 200k); m / ef_search knobs; hnswlib check |
| 05 | every scratch index on 20k news (work) + compiled engines on 50k (wall-clock); speed panel with frontier band, memory panel as labelled bars per build; 3-D dominance report; cheapest setting at recall ≥ 0.9 |
| 06 | latency at recall ≥ 0.95 + build time vs N (clustered to 200k); decision table; knob→corner table; recall-vs-sample-size; vector-DB knob names; rag_101 |

---

## 12. Streamlit app

Four tabs: **Scaling wall** (NB01 + NB06 curves from JSON, plus live exact-search timing on a
news prefix), **Index explorer** (faiss IVF/PQ/IVFPQ or hnswlib HNSW on news; knobs → recall,
latency, bytes/vector, build time; builds cached per parameter set), **The triangle** (NB05 sweep
JSON, work or wall-clock, plotly), **Query inspector** (one held-out query: scratch IVF probed
lists and scratch HNSW walk on the cached 20k graph, neighbours marked hit/miss with their text).

- `@st.cache_resource` for datasets and indexes; first render ~25 s on real data, then interactive.
- No algorithm code; no torch import; works without `data/` (shows "run `make data`").
- Tested headless with `streamlit.testing.v1.AppTest` (`tests/test_app.py`).

---

## 13. Honesty guardrails

- Recall always against exact GT on the same split; never a library's self-report.
- Every number carries N, d, k, nq, seed, threads, engine. Speeds as speedup vs exact.
- If the data disagrees with the PRD's story, report it — don't tune seeds until it agrees.
  Examples kept as measured: LSH is fully dominated on dense embeddings; the PQ family cannot reach
  recall 0.9 at ≤ 96-byte codes; IVFPQ's plateau equals flat PQ's rather than beating it.
- Extrapolations dashed and labelled; memory labelled "analytic"; work vs wall-clock labelled.
- Cite where it does work: Malkov & Yashunin 2018; Jégou, Douze & Schmid 2011; Johnson, Douze &
  Jégou 2019 and Douze et al. 2024; Indyk & Motwani 1998; Charikar 2002; Aumüller et al. 2020.

---

## 14. Terminology

**base vectors** `xb` · **queries** `xq` (held out) · **ground truth (GT)** · **k-recall@k** ·
**work / ndis** vectors scored per query · **nlist / nprobe** · **inverted list** · **coarse
quantizer** · **sub-vector, codebook, code, ADC** · **residual** · **m** (HNSW links per node, 2m on
layer 0), **ef_construction, ef_search, entry point, layer** · **bands × rows** (LSH) · **frontier /
Pareto set** · **the triangle** = recall–speed–memory.
faiss uses `M` for both PQ's sub-quantizer count and HNSW's degree. Here: PQ → `n_subvectors`,
HNSW → `m`. Say "approximate" not "fuzzy", "recall" not "accuracy", "index" vs "vector database".

---

## 15. Scaffold fixes (PRD gaps) — all resolved

1. `[build-system]` hatchling + `packages = ["src"]` so `import src` works from notebooks/app/tests. ✔
2. `torch` declared + pinned to the pytorch-cpu index (uv ignores sources for transitive deps;
   verified: 15 nvidia-* packages otherwise). ✔
3. `pytest-cov` added. ✔
4. `make notebooks` uses `--inplace` and skips `*.nbconvert.ipynb`. ✔
5. Per-notebook timing (`NB_BUDGET = 420 s`) on top of nbconvert's per-cell timeout. ✔
6. `.gitignore`: `data/`, `*.nbconvert.ipynb`; `make reset` clears `data/indexes`; `clean` has `.ty_cache`. ✔
7. Skills live at `.claude/skills/<name>/SKILL.md` (gitignored here); the caveman rule is in §16.
8. hnswlib 0.8.0 is sdist-only → compiled by `uv sync` (g++ present). ✔ (worked first time)
9. "Speed matches faiss/hnswlib" asserted as work, not wall-clock. ✔
10. `src/indexes/lsh.py` + `tests/test_lsh.py` added. ✔
11. Additions vs PRD tree: `src/artifacts.py`, `outputs/results/`, `tests/conftest.py`, extra test files. ✔

**Decisions taken** (were open): corpus = 20 Newsgroups sentences via scikit-learn (keyless,
no extra dependency); k-means = scikit-learn `KMeans` with random init (fast enough at 200k);
sweep JSONs are committed so the app's triangle tab works without re-running notebooks.

---

## 16. Working agreements

- `make ci` passes before any commit. No "will fix in the next commit".
- Small commits, imperative subjects (`add ivf probe sweep`, `fix hnsw layer-0 cap`).
- Never commit `.llm/`, `.venv`, `data/`, Hugging Face caches, or secrets. This file is tracked.
- Never edit `uv.lock` by hand; `uv add` / `uv remove`.
- Teaching repo: the shortest readable implementation beats the clever one; HNSW reads like the
  paper's pseudocode. Comments explain *why* and name the formula or paper section.
- When unsure how an index behaves, write the test or measure it — don't assert it in prose.
- README numbers trace to `outputs/results/*.json`; when a re-run moves one, update README and §2.

### Token rules
- **Caveman mode: ON** for chat: terse, code-first, lead with the diff or command; no preamble, no
  restating the plan. It does **not** apply to deliverable prose — notebook markdown, README,
  docstrings and article text are full sentences for readers.
- `/compact` at notebook boundaries.
- Read the PRD section you need (`grep -n`), not the whole file.
