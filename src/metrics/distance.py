"""Distances between query and base vectors, always returned as *distances* (smaller = closer).

- ``l2``     squared Euclidean ``||q - x||^2`` (what faiss IndexFlatL2 and hnswlib "l2" return)
- ``ip``     negative inner product ``-q.x``
- ``cosine`` ``1 - cos(q, x)``

Squared L2 is computed with the BLAS-friendly expansion ``||q||^2 + ||x||^2 - 2 q.x`` in
float32, clipped at zero (the expansion can go slightly negative through rounding). On
unit-normalised vectors ``||q - x||^2 = 2 - 2 q.x``, so all three metrics rank identically —
the reason the approximate indexes only implement squared L2 (see distance_arena for when the
choice of metric matters).
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

Metric = Literal["l2", "ip", "cosine"]
F32 = NDArray[np.float32]


def as_2d_float32(x: NDArray) -> F32:
    """C-contiguous float32 copy (or view) of ``x``, promoted to 2-D."""
    arr = np.ascontiguousarray(x, dtype=np.float32)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def sq_norms(x: NDArray) -> F32:
    """Row-wise squared L2 norms, shape ``(n,)``."""
    arr = as_2d_float32(x)
    return np.einsum("ij,ij->i", arr, arr)


def normalise(x: NDArray) -> F32:
    """Rows scaled to unit L2 norm (zero rows stay zero)."""
    arr = as_2d_float32(x)
    norms = np.sqrt(sq_norms(arr))
    norms[norms == 0] = 1.0
    return arr / norms[:, None]


def sq_l2(
    xq: NDArray,
    xb: NDArray,
    xb_norms: F32 | None = None,
    xq_norms: F32 | None = None,
) -> F32:
    """Squared L2 distances, shape ``(nq, nb)``: ``||q||^2 + ||x||^2 - 2 q.x``, clipped at 0."""
    q = as_2d_float32(xq)
    b = as_2d_float32(xb)
    qn = sq_norms(q) if xq_norms is None else xq_norms
    bn = sq_norms(b) if xb_norms is None else xb_norms
    dists = q @ b.T
    dists *= -2.0
    dists += qn[:, None]
    dists += bn[None, :]
    np.maximum(dists, 0.0, out=dists)
    return dists


def sq_l2_to_one(q: F32, xb: F32, xb_norms: F32, q_norm: float | None = None) -> F32:
    """Squared L2 from one query ``(d,)`` to every row of ``xb``, shape ``(n,)``."""
    qn = float(q @ q) if q_norm is None else q_norm
    dists = xb @ q
    dists *= -2.0
    dists += xb_norms
    dists += qn
    np.maximum(dists, 0.0, out=dists)
    return dists


def pairwise(xq: NDArray, xb: NDArray, metric: Metric = "l2", xb_norms: F32 | None = None) -> F32:
    """Distances from every query to every base vector under ``metric``, shape ``(nq, nb)``."""
    if metric == "l2":
        return sq_l2(xq, xb, xb_norms=xb_norms)
    if metric == "ip":
        return -(as_2d_float32(xq) @ as_2d_float32(xb).T)
    if metric == "cosine":
        return 1.0 - normalise(xq) @ normalise(xb).T
    raise ValueError(f"unknown metric {metric!r}; expected 'l2', 'ip' or 'cosine'")
