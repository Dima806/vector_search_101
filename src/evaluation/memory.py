"""Analytic index sizes: the compact layout a C/C++ index would hold in RAM.

These count what faiss and hnswlib actually allocate (vectors, codes, ids, centroids,
codebooks, graph-link slots) and deliberately ignore Python object overhead and process RSS,
so the from-scratch indexes and the compiled ones are measured in the same currency.
"""

import math

FLOAT_BYTES = 4  # float32
ID_BYTES = 8  # int64 ids stored next to codes in inverted lists
LINK_BYTES = 4  # int32 graph links


def exact_bytes(n: int, d: int) -> int:
    """Flat index: just the float32 vectors, ``N*d*4``."""
    return n * d * FLOAT_BYTES


def ivf_flat_bytes(n: int, d: int, nlist: int) -> int:
    """IVF-Flat: vectors + one id per vector + the coarse centroids."""
    return n * d * FLOAT_BYTES + n * ID_BYTES + nlist * d * FLOAT_BYTES


def pq_code_bytes(n_subvectors: int, nbits: int = 8) -> int:
    """Bytes per PQ code: ``n_subvectors * nbits`` bits, rounded up."""
    return math.ceil(n_subvectors * nbits / 8)


def pq_bytes(n: int, d: int, n_subvectors: int, nbits: int = 8) -> int:
    """PQ: one code per vector + the codebooks (``2^nbits`` codewords covering all ``d`` dims)."""
    return n * pq_code_bytes(n_subvectors, nbits) + (2**nbits) * d * FLOAT_BYTES


def ivfpq_bytes(n: int, d: int, nlist: int, n_subvectors: int, nbits: int = 8) -> int:
    """IVFPQ: code + id per vector, coarse centroids and the (shared) residual codebooks."""
    per_vector = pq_code_bytes(n_subvectors, nbits) + ID_BYTES
    return n * per_vector + nlist * d * FLOAT_BYTES + (2**nbits) * d * FLOAT_BYTES


def hnsw_bytes(n: int, d: int, m: int, upper_slots: int) -> int:
    """HNSW: it *keeps* the float vectors, plus allocated link slots and per-node bookkeeping.

    Layer 0 allocates ``2m`` slots per node; each upper layer a node lives on allocates ``m``
    (``upper_slots`` is that total). Bookkeeping is a level (4 B) and an offset/label (8 B).
    """
    return n * d * FLOAT_BYTES + LINK_BYTES * (n * 2 * m + upper_slots) + 12 * n


def lsh_bytes(n: int, d: int, bands: int, rows: int) -> int:
    """Banded LSH: signatures + one id per vector per band + the raw vectors it re-ranks."""
    return math.ceil(n * bands * rows / 8) + bands * n * ID_BYTES + n * d * FLOAT_BYTES


def compression_ratio(n: int, d: int, nbytes: int) -> float:
    """How many times smaller than the flat float32 index."""
    return exact_bytes(n, d) / nbytes if nbytes else math.inf
