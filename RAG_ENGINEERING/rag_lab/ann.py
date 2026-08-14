"""ANN indexing internals (notebook 24).

Real vector databases don't scan every vector — they use Approximate Nearest
Neighbor indexes that trade a little recall for a lot of speed. This module makes
the two dominant ideas concrete with tiny NumPy implementations:

- **IVF** (inverted file): cluster vectors, search only the nearest few clusters.
- **Scalar quantization**: store each dimension as one byte instead of a float32,
  cutting memory 4x with a small accuracy hit.

``recall_at_k`` measures what the approximation costs versus exact brute force.
An optional :func:`hnsw_search` uses ``hnswlib`` if installed.
"""

from __future__ import annotations

import numpy as np


def brute_force(matrix: np.ndarray, query: np.ndarray, k: int = 4) -> list[int]:
    """Exact top-k by dot product (the ground truth to compare against)."""
    scores = matrix @ query
    return np.argsort(scores)[::-1][:k].tolist()


class IVFIndex:
    """Inverted-file index: k-means centroids, probe the nearest ``nprobe`` lists."""

    def __init__(self, nlist: int = 8, nprobe: int = 2, iters: int = 10, seed: int = 0):
        self.nlist = nlist
        self.nprobe = nprobe
        self.iters = iters
        self.rng = np.random.default_rng(seed)
        self.centroids: np.ndarray | None = None
        self.lists: list[list[int]] = []
        self._matrix: np.ndarray | None = None

    def _kmeans(self, x: np.ndarray) -> np.ndarray:
        idx = self.rng.choice(len(x), size=min(self.nlist, len(x)), replace=False)
        centroids = x[idx].copy()
        for _ in range(self.iters):
            assign = np.argmax(x @ centroids.T, axis=1)
            for c in range(len(centroids)):
                members = x[assign == c]
                if len(members):
                    centroids[c] = members.mean(axis=0)
        return centroids

    def build(self, matrix: np.ndarray) -> "IVFIndex":
        self._matrix = matrix
        self.centroids = self._kmeans(matrix)
        assign = np.argmax(matrix @ self.centroids.T, axis=1)
        self.lists = [np.where(assign == c)[0].tolist() for c in range(len(self.centroids))]
        return self

    def search(self, query: np.ndarray, k: int = 4) -> list[int]:
        """Probe the ``nprobe`` nearest centroids, brute-force within those lists."""
        assert self.centroids is not None and self._matrix is not None
        near = np.argsort(self.centroids @ query)[::-1][: self.nprobe]
        cand = [i for c in near for i in self.lists[c]]
        if not cand:
            return []
        cand = np.array(cand)
        scores = self._matrix[cand] @ query
        return cand[np.argsort(scores)[::-1][:k]].tolist()


def quantize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalar-quantize a float matrix to uint8 per-dimension (returns codes+range)."""
    lo = matrix.min(axis=0)
    hi = matrix.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    codes = np.round((matrix - lo) / span * 255).astype(np.uint8)
    return codes, lo, span


def dequantize(codes: np.ndarray, lo: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Reconstruct approximate floats from uint8 codes."""
    return codes.astype(np.float32) / 255.0 * span + lo


def recall_at_k(approx: list[int], exact: list[int]) -> float:
    """Fraction of the exact top-k that the approximate search also found."""
    return len(set(approx) & set(exact)) / len(exact) if exact else 0.0


def hnsw_search(matrix: np.ndarray, query: np.ndarray, k: int = 4) -> list[int]:
    """Graph-based ANN via hnswlib (requires ``hnswlib``)."""
    import hnswlib  # lazy: only needed for this demo

    dim = matrix.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(matrix), ef_construction=100, M=16)
    index.add_items(matrix, np.arange(len(matrix)))
    index.set_ef(50)
    labels, _ = index.knn_query(query, k=k)
    return labels[0].tolist()
