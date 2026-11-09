"""HNSW — hierarchical navigable small-world graph (Malkov & Yashunin 2018), from scratch.

Every vector is a node. Each node draws a top level ``l = floor(-ln(U) * mL)`` with
``mL = 1/ln(m)``, so a fraction ~1/m of nodes reach layer 1, ~1/m^2 layer 2, and so on: a few
long-range "express" layers on top of a dense layer 0 that contains everyone.

Insert (paper Alg. 1): greedy-walk down from the entry point to the node's level, then on each
layer from there to 0 run a beam search (``ef_construction`` wide), choose ``m`` neighbours with
the diversity heuristic (Alg. 4) and link both ways. A node may hold ``2m`` links on layer 0
and ``m`` above (hnswlib and faiss use the same caps); an overflowing list is re-pruned with
the same heuristic.

Search (Alg. 5): greedy descent through the upper layers (one best node per layer), then a
beam search of width ``max(ef_search, k)`` on layer 0. It reaches the query's neighbourhood
in roughly log N hops and scores a small fraction of the database — counted in ``ndis``.

The choices mirror hnswlib (``getNeighborsByHeuristic2``, no extendCandidates, no
keepPrunedConnections) so tests/test_vs_reference.py can compare recall like for like.
"""

import heapq
import math
from pathlib import Path

import numpy as np

from src.evaluation.memory import hnsw_bytes
from src.indexes import FloatArray, IdArray, QueryTrace, SearchStats, as_float32

Candidate = tuple[float, int]  # (distance to the query, node id)


class HNSWIndex:
    """Pure-Python HNSW with numpy distance batches. Build-time: m, ef_construction."""

    name = "hnsw"

    def __init__(
        self, m: int = 16, ef_construction: int = 64, ef_search: int = 64, seed: int = 0
    ) -> None:
        if m < 2:
            raise ValueError("m must be at least 2")
        self.m = m
        self.m0 = 2 * m
        self.ml = 1.0 / math.log(m)
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._xb: FloatArray = np.zeros((0, 0), dtype=np.float32)
        self._norms: FloatArray = np.zeros(0, dtype=np.float32)
        self.levels = np.zeros(0, dtype=np.int64)
        self._links0 = np.zeros((0, self.m0), dtype=np.int32)  # layer 0, padded with -1
        self._count0 = np.zeros(0, dtype=np.int32)
        self._upper: list[dict[int, np.ndarray]] = []  # _upper[l - 1]: node -> links on layer l
        self.entry_point = -1
        self.max_level = -1
        self._n = 0
        self.last_stats = SearchStats.zeros(0)

    # ------------------------------------------------------------------ basics
    @property
    def ntotal(self) -> int:
        return self._n

    @property
    def d(self) -> int:
        return int(self._xb.shape[1]) if self._xb.size else 0

    def train(self, xb: FloatArray) -> None:
        """Nothing to learn: the graph is built by insertion."""

    def neighbors(self, node: int, layer: int) -> np.ndarray:
        """Current out-links of ``node`` on ``layer``."""
        if layer == 0:
            return self._links0[node, : self._count0[node]]
        return self._upper[layer - 1].get(node, np.zeros(0, dtype=np.int32))

    def _dists(self, q: FloatArray, q_norm: float, ids: np.ndarray) -> FloatArray:
        """Squared L2 from ``q`` to the nodes ``ids``, in one numpy call."""
        d = self._norms[ids] - 2.0 * (self._xb[ids] @ q) + q_norm
        return np.maximum(d, 0.0)

    # ------------------------------------------------------------------ search primitives
    def _greedy(
        self, q: FloatArray, q_norm: float, ep: int, ep_d: float, layer: int
    ) -> tuple[int, float, int, list[Candidate]]:
        """Walk to the best neighbour until none is closer (ef = 1). Returns the path too."""
        ndis = 0
        path: list[Candidate] = [(ep_d, ep)]
        changed = True
        while changed:
            changed = False
            nbrs = self.neighbors(ep, layer)
            if nbrs.size == 0:
                break
            dists = self._dists(q, q_norm, nbrs)
            ndis += nbrs.size
            best = int(np.argmin(dists))
            if dists[best] < ep_d:
                ep, ep_d = int(nbrs[best]), float(dists[best])
                path.append((ep_d, ep))
                changed = True
        return ep, ep_d, ndis, path

    def _search_layer(
        self, q: FloatArray, q_norm: float, entry: list[Candidate], ef: int, layer: int
    ) -> tuple[list[Candidate], int, list[Candidate]]:
        """Beam search on one layer (paper Alg. 2).

        Keeps a min-heap of candidates to expand and a max-heap of the ``ef`` best found so
        far; stops when the nearest unexpanded candidate is farther than the worst result.
        Returns (results nearest-first, distances computed, nodes expanded in order).
        """
        visited = np.zeros(self._xb.shape[0], dtype=bool)
        for _, node in entry:
            visited[node] = True
        candidates = list(entry)
        heapq.heapify(candidates)
        results = [(-d, node) for d, node in entry]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)
        ndis = 0
        expanded: list[Candidate] = []
        while candidates:
            dist_c, c = heapq.heappop(candidates)
            if dist_c > -results[0][0]:
                break
            expanded.append((dist_c, c))
            nbrs = self.neighbors(c, layer)
            nbrs = nbrs[~visited[nbrs]]
            if nbrs.size == 0:
                continue
            visited[nbrs] = True
            dists = self._dists(q, q_norm, nbrs)
            ndis += nbrs.size
            worst = -results[0][0]
            for dist_n, node in zip(dists.tolist(), nbrs.tolist(), strict=True):
                if len(results) < ef or dist_n < worst:
                    heapq.heappush(candidates, (dist_n, node))
                    heapq.heappush(results, (-dist_n, node))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst = -results[0][0]
        found = sorted((-neg_d, node) for neg_d, node in results)
        return found, ndis, expanded

    # ------------------------------------------------------------------ construction
    def _select_neighbors(self, cand: np.ndarray, cand_d: np.ndarray, cap: int) -> np.ndarray:
        """Diversity heuristic (Alg. 4, as hnswlib's ``getNeighborsByHeuristic2``).

        Scan candidates nearest-first; keep ``c`` unless an already-kept ``r`` is closer to it
        than the base is (``d(c, r) < d(c, base)``). That skips near-duplicates of links we
        already have and keeps links pointing in different directions — what makes the graph
        navigable across clusters.
        """
        order = np.argsort(cand_d, kind="stable")
        cand, cand_d = cand[order], cand_d[order]
        if cand.size < cap:
            return cand
        vecs = self._xb[cand]
        norms = self._norms[cand]
        pair = norms[:, None] + norms[None, :] - 2.0 * (vecs @ vecs.T)
        kept: list[int] = []
        for a in range(cand.size):
            if kept and bool((pair[a, kept] < cand_d[a]).any()):
                continue
            kept.append(a)
            if len(kept) == cap:
                break
        return cand[kept]

    def _set_links(self, node: int, layer: int, ids: np.ndarray) -> None:
        if layer == 0:
            self._links0[node] = -1
            self._links0[node, : ids.size] = ids
            self._count0[node] = ids.size
        else:
            self._upper[layer - 1][node] = ids.astype(np.int32)

    def _link_back(self, node: int, new: int, layer: int) -> None:
        """Add ``new`` to ``node``'s list; if over the cap, re-prune with the heuristic."""
        cap = self.m0 if layer == 0 else self.m
        current = self.neighbors(node, layer)
        if current.size < cap:
            self._set_links(node, layer, np.append(current, new))
            return
        cand = np.append(current, new)
        cand_d = self._dists(self._xb[node], float(self._norms[node]), cand)
        self._set_links(node, layer, self._select_neighbors(cand, cand_d, cap))

    def _insert(self, i: int) -> None:
        q = self._xb[i]
        q_norm = float(self._norms[i])
        level = int(self.levels[i])
        while len(self._upper) < level:
            self._upper.append({})
        if self.entry_point < 0:
            self.entry_point, self.max_level = i, level
            self._n += 1
            return
        ep = self.entry_point
        ep_d = float(self._dists(q, q_norm, np.array([ep]))[0])
        for layer in range(self.max_level, level, -1):  # zoom in, one best node per layer
            ep, ep_d, _, _ = self._greedy(q, q_norm, ep, ep_d, layer)
        entry: list[Candidate] = [(ep_d, ep)]
        for layer in range(min(level, self.max_level), -1, -1):
            found, _, _ = self._search_layer(q, q_norm, entry, self.ef_construction, layer)
            cand = np.array([node for _, node in found], dtype=np.int64)
            cand_d = np.array([d for d, _ in found], dtype=np.float32)
            selected = self._select_neighbors(cand, cand_d, self.m)
            self._set_links(i, layer, selected)
            for node in selected.tolist():
                self._link_back(node, i, layer)
            entry = found
        if level > self.max_level:
            self.entry_point, self.max_level = i, level
        self._n += 1

    def add(self, xb: FloatArray) -> None:
        """Insert vectors one by one (incremental: ``add`` can be called repeatedly)."""
        x = as_float32(xb)
        n_new = x.shape[0]
        self._xb = x.copy() if self._n == 0 else np.vstack([self._xb, x])
        self._norms = np.einsum("ij,ij->i", self._xb, self._xb)
        u = 1.0 - self._rng.random(n_new)  # in (0, 1], so -ln(u) is finite
        self.levels = np.concatenate(
            [self.levels, np.floor(-np.log(u) * self.ml).astype(np.int64)]
        )
        self._links0 = np.vstack([self._links0, np.full((n_new, self.m0), -1, dtype=np.int32)])
        self._count0 = np.concatenate([self._count0, np.zeros(n_new, dtype=np.int32)])
        for i in range(self._n, self._n + n_new):
            self._insert(i)

    # ------------------------------------------------------------------ search
    def _search_one(
        self, q: FloatArray, k: int, ef: int
    ) -> tuple[list[Candidate], int, int, dict[int, list[Candidate]]]:
        q_norm = float(q @ q)
        ep = self.entry_point
        ep_d = float(self._dists(q, q_norm, np.array([ep]))[0])
        ndis, nhops = 1, 0
        paths: dict[int, list[Candidate]] = {}
        for layer in range(self.max_level, 0, -1):
            ep, ep_d, nd, path = self._greedy(q, q_norm, ep, ep_d, layer)
            ndis += nd
            nhops += len(path) - 1
            paths[layer] = path
        found, nd, expanded = self._search_layer(q, q_norm, [(ep_d, ep)], max(ef, k), 0)
        paths[0] = expanded
        return found[:k], ndis + nd, nhops + len(expanded), paths

    def search(self, xq: FloatArray, k: int) -> tuple[FloatArray, IdArray]:
        q = as_float32(xq)
        nq = q.shape[0]
        out_d = np.full((nq, k), np.inf, dtype=np.float32)
        out_i = np.full((nq, k), -1, dtype=np.int64)
        ndis = np.empty(nq)
        nhops = np.empty(nq)
        for i in range(nq):
            found, ndis[i], nhops[i], _ = self._search_one(q[i], k, self.ef_search)
            out_d[i, : len(found)] = [d for d, _ in found]
            out_i[i, : len(found)] = [node for _, node in found]
        self.last_stats = SearchStats(ndis=ndis, n_adc=np.zeros(nq), nhops=nhops)
        return out_d, out_i

    def trace(self, q: FloatArray, k: int) -> QueryTrace:
        """One query's walk: entry point, nodes visited per layer, distance at every hop."""
        qq = as_float32(q)[0]
        found, ndis, _, paths = self._search_one(qq, k, self.ef_search)
        hops = [(layer, d) for layer in sorted(paths, reverse=True) for d, _ in paths[layer]]
        return QueryTrace(
            index=self.name,
            ids=[node for _, node in found],
            dists=[d for d, _ in found],
            ndis=int(ndis),
            entry_point=self.entry_point,
            layer_paths={layer: [node for _, node in path] for layer, path in paths.items()},
            hops=hops,
        )

    # ------------------------------------------------------------------ diagnostics
    def level_counts(self) -> np.ndarray:
        """Number of nodes whose top level is 0, 1, 2, ..."""
        return np.bincount(self.levels[: self._n], minlength=self.max_level + 1)

    def reachable_fraction(self) -> float:
        """Share of nodes reachable from the entry point over layer-0 out-links (BFS)."""
        if self._n == 0:
            return 1.0
        seen = np.zeros(self._n, dtype=bool)
        seen[self.entry_point] = True
        frontier = np.array([self.entry_point])
        while frontier.size:
            nbrs = self._links0[frontier].ravel()
            nbrs = nbrs[nbrs >= 0]
            nbrs = np.unique(nbrs[~seen[nbrs]])
            seen[nbrs] = True
            frontier = nbrs
        return float(seen.mean())

    @property
    def nbytes(self) -> int:
        upper_slots = int(self.levels[: self._n].sum()) * self.m
        return hnsw_bytes(self._n, self.d, self.m, upper_slots)

    # ------------------------------------------------------------------ cache
    def save(self, path: Path) -> None:
        """Write the graph and vectors to an ``.npz`` (upper layers flattened to arrays)."""
        arrays: dict[str, np.ndarray] = {
            "xb": self._xb[: self._n],
            "levels": self.levels[: self._n],
            "links0": self._links0[: self._n],
            "count0": self._count0[: self._n],
            "meta": np.array(
                [self.m, self.ef_construction, self.seed, self.entry_point, self.max_level],
                dtype=np.int64,
            ),
        }
        for layer, links in enumerate(self._upper, start=1):
            nodes = np.array(sorted(links), dtype=np.int64)
            padded = np.full((nodes.size, self.m), -1, dtype=np.int32)
            for row, node in enumerate(nodes.tolist()):
                padded[row, : links[node].size] = links[node]
            arrays[f"nodes{layer}"] = nodes
            arrays[f"links{layer}"] = padded
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]

    @classmethod
    def load(cls, path: Path, ef_search: int = 64) -> "HNSWIndex":
        """Rebuild an index saved with :meth:`save`."""
        with np.load(path) as f:
            m, ef_c, seed, entry, max_level = (int(v) for v in f["meta"])
            index = cls(m=m, ef_construction=ef_c, ef_search=ef_search, seed=seed)
            index._xb = np.ascontiguousarray(f["xb"])
            index._norms = np.einsum("ij,ij->i", index._xb, index._xb)
            index.levels = f["levels"].copy()
            index._links0 = f["links0"].copy()
            index._count0 = f["count0"].copy()
            index.entry_point, index.max_level = entry, max_level
            index._n = int(index._xb.shape[0])
            layer = 1
            while f"nodes{layer}" in f:
                nodes, links = f[f"nodes{layer}"], f[f"links{layer}"]
                index._upper.append(
                    {int(n): row[row >= 0].copy() for n, row in zip(nodes, links, strict=True)}
                )
                layer += 1
        return index


def cache_path(dataset_key: str, n: int, m: int, ef_construction: int, seed: int) -> Path:
    """Where a scratch graph lives: the key covers every build parameter and the data."""
    from src.artifacts import indexes_dir, params_key  # noqa: PLC0415 - keep import light

    key = params_key(dataset=dataset_key, n=n, m=m, ef_construction=ef_construction, seed=seed)
    return indexes_dir() / f"hnsw_{dataset_key}_n{n}_{key}.npz"


def load_or_build(
    xb: FloatArray,
    dataset_key: str,
    m: int,
    ef_construction: int,
    ef_search: int,
    seed: int,
) -> tuple[HNSWIndex, float, bool]:
    """The cached scratch graph for ``xb`` if present, else build and cache it.

    Returns (index, build seconds — NaN when loaded from cache, whether it came from cache).
    """
    import time  # noqa: PLC0415

    path = cache_path(dataset_key, int(xb.shape[0]), m, ef_construction, seed)
    if path.exists():
        return HNSWIndex.load(path, ef_search=ef_search), float("nan"), True
    index = HNSWIndex(m=m, ef_construction=ef_construction, ef_search=ef_search, seed=seed)
    start = time.perf_counter()
    index.add(xb)
    build_s = time.perf_counter() - start
    index.save(path)
    return index, build_s, False
