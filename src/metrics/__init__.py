"""Distance metrics: squared L2 (default), inner product and cosine, all as distances."""

from src.metrics.distance import (
    Metric,
    normalise,
    pairwise,
    sq_l2,
    sq_l2_to_one,
    sq_norms,
)

__all__ = ["Metric", "normalise", "pairwise", "sq_l2", "sq_l2_to_one", "sq_norms"]
