"""Datasets with held-out queries and exact ground truth: ``python -m src.datasets.vectors``.

- **news** (real): sentences from the 20 Newsgroups corpus (fetched by scikit-learn, no key),
  embedded with ``all-MiniLM-L6-v2`` into unit-norm 384-d vectors — the structure semantic
  search actually sees. Duplicate vectors are dropped (the model is uncased, so texts that
  differ only in case embed identically), which keeps exact ties out of the ground truth.
- **synthetic**: Gaussian mixtures on a low-dimensional subspace with controllable size,
  dimension, cluster count and spread, plus an unstructured isotropic control.

Queries are a held-out set, disjoint from the base vectors (querying with a base vector would
find itself and inflate recall@1). Ground truth is the exact top ``k_gt`` from brute force.
Everything is cached under ``data/vectors/<name>_n<N>_d<d>_s<seed>/``; rebuilding is skipped
when the cache exists.
"""

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.artifacts import data_dir, vectors_dir
from src.config import RealDatasetConfig, Settings, SyntheticDatasetConfig, get_settings
from src.indexes import FloatArray, IdArray
from src.indexes.exact import ExactIndex

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


@dataclass
class VectorDataset:
    """Base vectors, held-out queries and their exact nearest neighbours."""

    name: str
    base: FloatArray
    queries: FloatArray
    gt_ids: IdArray
    gt_dists: FloatArray
    meta: dict[str, Any] = field(default_factory=dict)
    texts: list[str] | None = None
    query_texts: list[str] | None = None

    @property
    def n(self) -> int:
        return int(self.base.shape[0])

    @property
    def d(self) -> int:
        return int(self.base.shape[1])

    @property
    def key(self) -> str:
        return dataset_key(self.name, self.n, self.d, int(self.meta.get("seed", 0)))


def dataset_key(name: str, n: int, d: int, seed: int) -> str:
    """Cache directory name, e.g. ``news_n50000_d384_s42``."""
    return f"{name}_n{n}_d{d}_s{seed}"


# --------------------------------------------------------------------------- synthetic
def make_synthetic(
    n: int,
    d: int,
    n_clusters: int,
    intrinsic_dim: int,
    cluster_std: float,
    noise: float,
    seed: int,
    chunk: int = 50_000,
) -> FloatArray:
    """Gaussian mixture in ``intrinsic_dim`` dims, rotated into ``d`` dims, plus ``noise``.

    ``n_clusters=1, intrinsic_dim=d, cluster_std=1, noise=0`` is a plain isotropic Gaussian —
    the control where neither clusters nor low intrinsic dimension give an index anything to use.
    """
    if intrinsic_dim > d:
        raise ValueError("intrinsic_dim cannot exceed d")
    rng = np.random.default_rng(seed)
    centers = (
        rng.standard_normal((n_clusters, intrinsic_dim))
        if n_clusters > 1
        else np.zeros((1, intrinsic_dim))
    )
    basis, _ = np.linalg.qr(rng.standard_normal((d, intrinsic_dim)))  # orthonormal columns
    out = np.empty((n, d), dtype=np.float32)
    for start in range(0, n, chunk):
        size = min(chunk, n - start)
        labels = rng.integers(n_clusters, size=size)
        latent = centers[labels] + cluster_std * rng.standard_normal((size, intrinsic_dim))
        block = latent @ basis.T
        if noise:
            block += noise * rng.standard_normal((size, d))
        out[start : start + size] = block
    return out


# --------------------------------------------------------------------------- real
def load_news_sentences(min_chars: int, max_chars: int) -> list[str]:
    """Unique sentences from 20 Newsgroups (headers, signatures and quotes removed)."""
    from sklearn.datasets import fetch_20newsgroups  # noqa: PLC0415 - network fetch, lazy

    docs = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        data_home=str(data_dir() / "sklearn"),
    ).data
    seen: set[str] = set()
    sentences: list[str] = []
    for doc in docs:
        for part in _SENTENCE_SPLIT.split(doc):
            text = " ".join(part.split())
            if not min_chars <= len(text) <= max_chars:
                continue
            if sum(ch.isalpha() for ch in text) < 0.7 * len(text):
                continue  # skip tables, code, ascii art
            low = text.lower()
            if low not in seen:
                seen.add(low)
                sentences.append(text)
    return sentences


def embed_texts(texts: list[str], model_name: str, batch_size: int) -> FloatArray:
    """Unit-norm sentence embeddings on CPU. torch is imported only here."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - heavy, lazy

    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.ascontiguousarray(emb, dtype=np.float32)


def unique_rows(x: FloatArray) -> np.ndarray:
    """Indices of the first occurrence of every distinct row, in original order."""
    _, first = np.unique(x, axis=0, return_index=True)
    return np.sort(first)


# --------------------------------------------------------------------------- ground truth
def exact_ground_truth(
    base: FloatArray, queries: FloatArray, k_gt: int, block_mb: float = 256.0
) -> tuple[IdArray, FloatArray]:
    """The true ``k_gt`` nearest neighbours of every query, by brute force."""
    exact = ExactIndex(block_mb=block_mb)
    exact.add(base)
    dists, ids = exact.search(queries, k_gt)
    return ids, dists


# --------------------------------------------------------------------------- cache
def save_dataset(ds: VectorDataset, root: Path | None = None) -> Path:
    """Write a dataset to ``<root>/<key>/`` (.npy arrays + meta.json + optional texts.json)."""
    directory = (root or vectors_dir()) / ds.key
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "base.npy", ds.base)
    np.save(directory / "queries.npy", ds.queries)
    np.save(directory / "gt_ids.npy", ds.gt_ids)
    np.save(directory / "gt_dists.npy", ds.gt_dists)
    (directory / "meta.json").write_text(json.dumps(ds.meta, indent=2) + "\n")
    if ds.texts is not None:
        texts = {"base": ds.texts, "queries": ds.query_texts or []}
        (directory / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))
    return directory


def find_dataset_dir(name: str, seed: int | None = None, root: Path | None = None) -> Path | None:
    """The cached directory for dataset ``name`` (most recent if several), or None."""
    pattern = f"{name}_n*_s{seed}" if seed is not None else f"{name}_n*"
    matches = sorted((root or vectors_dir()).glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def load_dataset(name: str, seed: int | None = None, root: Path | None = None) -> VectorDataset:
    """Load a cached dataset; raises FileNotFoundError with the fix if it is missing."""
    if seed is None:
        seed = get_settings().seed
    directory = find_dataset_dir(name, seed, root)
    if directory is None:
        raise FileNotFoundError(f"dataset {name!r} (seed {seed}) not cached — run `make data`")
    texts = query_texts = None
    if (directory / "texts.json").exists():
        blob = json.loads((directory / "texts.json").read_text())
        texts, query_texts = blob["base"], blob["queries"]
    return VectorDataset(
        name=name,
        base=np.load(directory / "base.npy"),
        queries=np.load(directory / "queries.npy"),
        gt_ids=np.load(directory / "gt_ids.npy"),
        gt_dists=np.load(directory / "gt_dists.npy"),
        meta=json.loads((directory / "meta.json").read_text()),
        texts=texts,
        query_texts=query_texts,
    )


def cached_names(root: Path | None = None) -> list[str]:
    """Names of datasets present in the cache."""
    base = root or vectors_dir()
    if not base.exists():
        return []
    return sorted({p.name.split("_n")[0] for p in base.iterdir() if (p / "meta.json").exists()})


# --------------------------------------------------------------------------- builders
def build_synthetic(
    spec: SyntheticDatasetConfig, n_queries: int, k_gt: int, seed: int, block_mb: float = 256.0
) -> VectorDataset:
    """Generate ``spec.n + n_queries`` vectors, hold out the queries, compute exact GT."""
    x = make_synthetic(
        spec.n + n_queries,
        spec.d,
        spec.n_clusters,
        spec.intrinsic_dim,
        spec.cluster_std,
        spec.noise,
        seed,
    )
    base, queries = x[n_queries:], x[:n_queries]
    gt_ids, gt_dists = exact_ground_truth(base, queries, k_gt, block_mb)
    meta = {"source": "synthetic", "seed": seed, "k_gt": k_gt, **spec.model_dump()}
    return VectorDataset(spec.name, base, queries, gt_ids, gt_dists, meta)


def build_real(
    cfg: RealDatasetConfig, n_queries: int, k_gt: int, seed: int, block_mb: float = 256.0
) -> VectorDataset:
    """20 Newsgroups sentences -> MiniLM embeddings -> held-out queries + exact GT."""
    sentences = load_news_sentences(cfg.min_chars, cfg.max_chars)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sentences))
    want = min(len(sentences), int((cfg.max_n + n_queries) * 1.02))  # margin for duplicates
    texts = [sentences[i] for i in order[:want]]
    emb = embed_texts(texts, cfg.model, cfg.batch_size)
    keep = unique_rows(emb)[: cfg.max_n + n_queries]
    emb = emb[keep]
    texts = [texts[i] for i in keep]
    queries, base = emb[:n_queries], emb[n_queries:]
    gt_ids, gt_dists = exact_ground_truth(base, queries, k_gt, block_mb)
    meta = {
        "source": "20newsgroups",
        "model": cfg.model,
        "seed": seed,
        "k_gt": k_gt,
        "n_sentences_available": len(sentences),
        "normalised": True,
    }
    return VectorDataset(
        cfg.name, base, queries, gt_ids, gt_dists, meta, texts[n_queries:], texts[:n_queries]
    )


def build_all(settings: Settings, only: str | None = None, force: bool = False) -> list[str]:
    """Build every configured dataset that is not cached yet; returns the keys built."""
    ds_cfg = settings.datasets
    built = []
    jobs: list[tuple[str, Any]] = [(ds_cfg.real.name, ds_cfg.real)]
    jobs += [(spec.name, spec) for spec in ds_cfg.synthetic]
    for name, spec in jobs:
        if only and name != only:
            continue
        if not force and find_dataset_dir(name, settings.seed) is not None:
            print(f"✓ {name}: cached ({find_dataset_dir(name, settings.seed)})")
            continue
        start = time.perf_counter()
        if isinstance(spec, RealDatasetConfig):
            ds = build_real(
                spec, ds_cfg.n_queries, ds_cfg.k_gt, settings.seed, settings.exact.block_mb
            )
        else:
            ds = build_synthetic(
                spec, ds_cfg.n_queries, ds_cfg.k_gt, settings.seed, settings.exact.block_mb
            )
        ds.meta["build_seconds"] = round(time.perf_counter() - start, 1)
        path = save_dataset(ds)
        print(
            f"✓ {name}: built {ds.n:,} x {ds.d} (+{ds.queries.shape[0]} queries) in "
            f"{ds.meta['build_seconds']} s -> {path}"
        )
        built.append(ds.key)
    return built


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="build just this dataset name")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args(argv)
    build_all(get_settings(), only=args.only, force=args.force)


if __name__ == "__main__":
    main()
