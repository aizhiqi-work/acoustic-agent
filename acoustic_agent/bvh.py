from __future__ import annotations

from typing import Any

import numpy as np


def build_bvh(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    *,
    leaf_size: int = 4,
) -> dict[str, Any]:
    """Build a deterministic preorder BVH with stackless escape links."""
    lower = np.asarray(bounds_min, dtype=np.float64)
    upper = np.asarray(bounds_max, dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 2 or lower.shape[1] != 3:
        raise ValueError("BVH bounds must have shape [surface, 3]")
    if lower.shape[0] < 1:
        raise ValueError("BVH requires at least one surface")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("BVH bounds must be finite")
    if np.any(upper < lower):
        raise ValueError("BVH maximum bounds must not be below minimum bounds")

    max_leaf = max(1, int(leaf_size))
    centroids = 0.5 * (lower + upper)
    node_min: list[np.ndarray] = []
    node_max: list[np.ndarray] = []
    node_start: list[int] = []
    node_count: list[int] = []
    node_escape: list[int] = []
    primitives: list[int] = []
    max_depth = 0
    leaf_count = 0

    def append_node(indices: np.ndarray, depth: int) -> int:
        nonlocal max_depth, leaf_count
        node_index = len(node_min)
        node_min.append(np.min(lower[indices], axis=0))
        node_max.append(np.max(upper[indices], axis=0))
        node_start.append(-1)
        node_count.append(0)
        node_escape.append(-1)
        max_depth = max(max_depth, depth)

        if indices.size <= max_leaf:
            ordered = np.sort(indices, kind="stable")
            node_start[node_index] = len(primitives)
            node_count[node_index] = int(ordered.size)
            primitives.extend(int(value) for value in ordered)
            leaf_count += 1
        else:
            extent = np.ptp(centroids[indices], axis=0)
            axis = int(np.argmax(extent))
            order = np.argsort(centroids[indices, axis], kind="stable")
            sorted_indices = indices[order]
            split = int(sorted_indices.size // 2)
            append_node(sorted_indices[:split], depth + 1)
            append_node(sorted_indices[split:], depth + 1)

        node_escape[node_index] = len(node_min)
        return node_index

    append_node(np.arange(lower.shape[0], dtype=np.int64), 1)
    return {
        "bounds_min": np.asarray(node_min, dtype=np.float64),
        "bounds_max": np.asarray(node_max, dtype=np.float64),
        "start": np.asarray(node_start, dtype=np.int64),
        "count": np.asarray(node_count, dtype=np.int64),
        "escape": np.asarray(node_escape, dtype=np.int64),
        "primitives": np.asarray(primitives, dtype=np.int64),
        "node_count": len(node_min),
        "leaf_count": leaf_count,
        "max_depth": max_depth,
        "leaf_size": max_leaf,
    }
