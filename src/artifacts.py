"""Where things live on disk, and the two savers every notebook uses.

- ``save_figure(fig, "NN_slug")`` -> ``outputs/figures/NN_slug.png`` (150 dpi)
- ``save_json("NN_slug", data)``  -> ``outputs/results/NN_slug.json`` (numpy/pandas aware)

Paths resolve from the repo root, never the working directory (nbconvert runs notebooks from
``notebooks/``). The data directory can be redirected with ``VS101_DATA_DIR`` (used by tests).
"""

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import REPO_ROOT

FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
RESULTS_DIR = REPO_ROOT / "outputs" / "results"


def data_dir() -> Path:
    """Root of the gitignored cache (``data/``), overridable with ``VS101_DATA_DIR``."""
    override = os.environ.get("VS101_DATA_DIR")
    return Path(override) if override else REPO_ROOT / "data"


def vectors_dir() -> Path:
    """Cached datasets: base vectors, queries and exact ground truth."""
    return data_dir() / "vectors"


def indexes_dir() -> Path:
    """Cached index builds (the from-scratch HNSW graph is the main user)."""
    return data_dir() / "indexes"


def params_key(**params: Any) -> str:
    """A short, stable hash of build parameters, used in cache file names."""
    blob = json.dumps(to_jsonable(params), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def to_jsonable(obj: Any) -> Any:
    """Convert numpy / pandas / dataclass / Path values into plain JSON types (NaN -> None)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_figure(fig: Figure, slug: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure as ``outputs/figures/<slug>.png`` and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{slug}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def save_json(slug: str, data: Any) -> Path:
    """Save the numbers/text behind a figure or claim as ``outputs/results/<slug>.json``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{slug}.json"
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False) + "\n")
    return path


def load_json(slug: str) -> Any:
    """Read back ``outputs/results/<slug>.json``."""
    return json.loads((RESULTS_DIR / f"{slug}.json").read_text())
