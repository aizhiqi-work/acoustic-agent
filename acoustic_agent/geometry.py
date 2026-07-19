from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .materials import MaterialLibrary
from .models import Material, Room


def make_room(
    shape: str,
    *,
    size: Sequence[float] = (6.0, 4.0, 2.8),
    corners: Sequence[Sequence[float]] | None = None,
    materials: Mapping[str, str | Material] | None = None,
    material_library: MaterialLibrary | None = None,
    material_profile: str | Mapping[str, Any] | None = None,
    material_seed: int = 0,
    circle_segments: int = 32,
    room_id: str = "room",
    name: str | None = None,
) -> Room:
    """Create an extruded simplified indoor room.

    Supported shapes are ``rectangle``, ``triangle``, ``polygon``, ``circle``,
    ``l_shape``, ``t_shape``, ``trapezoid``, ``u_shape`` and ``fan_shape``.
    Sizes are in meters. Explicit XY corners override the built-in shape
    generator and put height in ``size[2]``.
    """
    sx, sy, height = _size3(size)
    key = shape.lower().replace("-", "_")
    if corners is not None:
        xy = tuple((float(p[0]), float(p[1])) for p in corners)
        geometry_model = "rectangular_room" if key in {"rectangle", "shoebox", "box"} and len(xy) == 4 else "extruded_polygon"
    elif key in {"rectangle", "shoebox", "box"}:
        xy = ((0.0, 0.0), (sx, 0.0), (sx, sy), (0.0, sy))
        geometry_model = "rectangular_room"
    elif key == "triangle":
        xy = ((0.0, 0.0), (sx, 0.0), (sx * 0.5, sy))
        geometry_model = "extruded_polygon"
    elif key == "circle":
        radius_x = sx * 0.5
        radius_y = sy * 0.5
        count = max(12, int(circle_segments))
        xy = tuple(
            (radius_x + math.cos(2.0 * math.pi * i / count) * radius_x,
             radius_y + math.sin(2.0 * math.pi * i / count) * radius_y)
            for i in range(count)
        )
        geometry_model = "extruded_polygon"
    elif key == "l_shape":
        notch_x = sx * 0.55
        notch_y = sy * 0.55
        xy = ((0.0, 0.0), (sx, 0.0), (sx, notch_y), (notch_x, notch_y), (notch_x, sy), (0.0, sy))
        geometry_model = "extruded_polygon"
    elif key == "t_shape":
        stem_w = sx * 0.34
        stem_x0 = (sx - stem_w) * 0.5
        head_h = sy * 0.38
        xy = (
            (0.0, 0.0), (sx, 0.0), (sx, head_h), (stem_x0 + stem_w, head_h),
            (stem_x0 + stem_w, sy), (stem_x0, sy), (stem_x0, head_h), (0.0, head_h),
        )
        geometry_model = "extruded_polygon"
    elif key == "trapezoid":
        top_w = sx * 0.62
        top_x = (sx - top_w) * 0.5
        xy = ((0.0, 0.0), (sx, 0.0), (top_x + top_w, sy), (top_x, sy))
        geometry_model = "extruded_polygon"
    elif key == "u_shape":
        gap_w = sx * 0.42
        gap_x = (sx - gap_w) * 0.5
        inner_y = sy * (1.0 - 0.48)
        xy = (
            (0.0, 0.0), (sx, 0.0), (sx, sy), (gap_x + gap_w, sy),
            (gap_x + gap_w, inner_y), (gap_x, inner_y), (gap_x, sy), (0.0, sy),
        )
        geometry_model = "extruded_polygon"
    elif key == "fan_shape":
        xy = _default_fan_corners(sx, sy)
        geometry_model = "extruded_polygon"
    elif key in {"polygon", "polyhedron", "polytope"}:
        xy = tuple((float(p[0]), float(p[1])) for p in _default_polygon_corners(sx, sy))
        geometry_model = "extruded_polygon"
    else:
        raise ValueError(f"unsupported room shape: {shape!r}")
    _validate_polygon(xy)
    if polygon_area(xy) < 0.0:
        xy = tuple(reversed(xy))
    library = material_library or MaterialLibrary.load()
    resolved = resolve_materials(
        materials or {},
        library,
        material_profile=material_profile,
        material_seed=material_seed,
    )
    return Room(
        id=room_id,
        name=name or key,
        corners=tuple((float(x), float(y)) for x, y in xy),
        height_m=float(height),
        materials=resolved,
        metadata={
            "shape": key,
            "geometry_model": geometry_model,
            "material_seed": int(material_seed),
            "material_profile": dict(material_profile) if isinstance(material_profile, Mapping) else material_profile or "auto",
            "material_selection": {
                surface: {
                    "semantic": material.semantic,
                    "material_id": material.id,
                    "material_name": material.name,
                    **dict(material.metadata),
                }
                for surface, material in resolved.items()
            },
        },
    )


def _default_polygon_corners(sx: float, sy: float) -> tuple[tuple[float, float], ...]:
    return (
        (0.0, 0.0),
        (sx * 0.72, 0.0),
        (sx, sy * 0.42),
        (sx * 0.62, sy),
        (sx * 0.12, sy * 0.86),
        (-sx * 0.02, sy * 0.26),
    )


def _default_fan_corners(sx: float, sy: float, *, angle_degrees: float = 90.0, inner_ratio: float = 0.28, segments: int = 24) -> tuple[tuple[float, float], ...]:
    half_angle = math.radians(angle_degrees) * 0.5
    outer = [
        (math.sin(-half_angle + (2.0 * half_angle * i / segments)), math.cos(-half_angle + (2.0 * half_angle * i / segments)))
        for i in range(segments + 1)
    ]
    inner = [
        (math.sin(half_angle - (2.0 * half_angle * i / segments)) * inner_ratio, math.cos(half_angle - (2.0 * half_angle * i / segments)) * inner_ratio)
        for i in range(segments + 1)
    ]
    return _normalize_corners(tuple(outer + inner), sx, sy, 0.02)


def _normalize_corners(corners: Sequence[Sequence[float]], sx: float, sy: float, pad_ratio: float = 0.0) -> tuple[tuple[float, float], ...]:
    pts = np.asarray(corners, dtype=float)
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    pad_x = sx * pad_ratio
    pad_y = sy * pad_ratio
    usable_x = max(sx - pad_x * 2.0, 0.1)
    usable_y = max(sy - pad_y * 2.0, 0.1)
    scaled = np.column_stack((
        pad_x + ((pts[:, 0] - min_xy[0]) / span[0]) * usable_x,
        pad_y + ((pts[:, 1] - min_xy[1]) / span[1]) * usable_y,
    ))
    return tuple((float(x), float(y)) for x, y in scaled)


def resolve_materials(
    raw: Mapping[str, str | Mapping[str, Any] | Material],
    library: MaterialLibrary,
    *,
    material_profile: str | Mapping[str, Any] | None = None,
    material_seed: int = 0,
) -> dict[str, Material]:
    out = library.sample_surface_set(material_profile, seed=int(material_seed), overrides=raw)
    for surface, spec in raw.items():
        if surface not in out:
            out[surface] = library.resolve(spec, default_semantic=str(surface), seed=int(material_seed))
    return out


def polygon_area(corners: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(corners, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_perimeter(corners: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(corners, dtype=float)
    return float(np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))


def polygon_bounds(corners: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    pts = np.asarray(corners, dtype=float)
    return float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))


def point_in_polygon(point: Sequence[float], corners: Sequence[Sequence[float]]) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    pts = list(corners)
    j = len(pts) - 1
    for i, pi in enumerate(pts):
        xi, yi = pi
        xj, yj = pts[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            denom = yj - yi
            if abs(denom) <= 1e-12:
                j = i
                continue
            x_at_y = (xj - xi) * (y - yi) / denom + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside or _point_on_boundary((x, y), corners)


def is_rectangle(corners: Sequence[Sequence[float]], tol: float = 1e-6) -> bool:
    if len(corners) != 4:
        return False
    x0, y0, x1, y1 = polygon_bounds(corners)
    expected = {(round(x0, 6), round(y0, 6)), (round(x1, 6), round(y0, 6)), (round(x1, 6), round(y1, 6)), (round(x0, 6), round(y1, 6))}
    got = {(round(float(x), 6), round(float(y), 6)) for x, y in corners}
    return expected == got and (x1 - x0) > tol and (y1 - y0) > tol


def _point_on_boundary(point: tuple[float, float], corners: Sequence[Sequence[float]], tol: float = 1e-8) -> bool:
    p = np.asarray(point, dtype=float)
    pts = np.asarray(corners, dtype=float)
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= tol:
            continue
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        if float(np.linalg.norm(a + t * ab - p)) <= tol:
            return True
    return False


def _validate_polygon(corners: Sequence[Sequence[float]]) -> None:
    pts = np.asarray(corners, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("room corners must be at least three finite XY points")
    if not np.all(np.isfinite(pts)):
        raise ValueError("room corners contain non-finite values")
    edges = np.roll(pts, -1, axis=0) - pts
    if np.any(np.linalg.norm(edges, axis=1) <= 1e-8):
        raise ValueError("room polygon contains a zero-length edge")
    if abs(polygon_area(corners)) <= 1e-8:
        raise ValueError("room polygon area is zero")
    count = len(pts)
    for first in range(count):
        a0 = pts[first]
        a1 = pts[(first + 1) % count]
        for second in range(first + 1, count):
            if second == first or second == (first + 1) % count or (second + 1) % count == first:
                continue
            b0 = pts[second]
            b1 = pts[(second + 1) % count]
            if _segments_intersect(a0, a1, b0, b1):
                raise ValueError("room polygon must not self-intersect")


def _segments_intersect(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, tol: float = 1e-10) -> bool:
    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    def on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
        return (
            abs(cross(end - start, point - start)) <= tol
            and min(start[0], end[0]) - tol <= point[0] <= max(start[0], end[0]) + tol
            and min(start[1], end[1]) - tol <= point[1] <= max(start[1], end[1]) + tol
        )

    o1 = cross(a1 - a0, b0 - a0)
    o2 = cross(a1 - a0, b1 - a0)
    o3 = cross(b1 - b0, a0 - b0)
    o4 = cross(b1 - b0, a1 - b0)
    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)):
        return True
    return (
        (abs(o1) <= tol and on_segment(b0, a0, a1))
        or (abs(o2) <= tol and on_segment(b1, a0, a1))
        or (abs(o3) <= tol and on_segment(a0, b0, b1))
        or (abs(o4) <= tol and on_segment(a1, b0, b1))
    )


def _size3(size: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(v) for v in size)
    if len(values) == 2:
        values = (values[0], values[1], 2.8)
    if len(values) != 3 or min(values) <= 0.0:
        raise ValueError("size must contain positive width/depth/height values")
    return values
