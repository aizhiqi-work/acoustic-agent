from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from functools import lru_cache
import heapq
from itertools import permutations
import math
from threading import RLock
import time
from typing import Any, Mapping, Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon

try:
    from numba import get_num_threads, get_thread_id, njit, prange
except Exception:  # pragma: no cover - numba is an optional accelerator at runtime
    njit = None

from .acoustics import (
    AIR_ABSORPTION_NP_PER_M,
    band_mean,
    multiply_bands,
    propagation_band_gains,
    steam_audio_pathing_deviation,
)
from .bvh import build_bvh
from .directivity import source_directivity, source_directivity_gain, source_forward
from .geometry import point_in_polygon
from .materials import MaterialLibrary, fallback_material, material_summary
from .models import FREQUENCY_BANDS, AcousticPath, Room, SimConfig
from .rir import render_impulses

_NUM_BANDS = len(FREQUENCY_BANDS)
_EPS = 1e-6
_SQRT_4PI = math.sqrt(4.0 * math.pi)
_SH_Y00 = 1.0 / _SQRT_4PI
_ENERGY_THRESHOLD = 1e-9
_HIT_OFFSET = 1e-2
_RT_VISUAL_RETAIN_LIMIT = 512
_RT_VISUAL_CANDIDATE_FACTOR = 32
_BVH_BOUNDS_EPS = 2.0e-6
_STATIC_SCENE_CACHE_LIMIT = 32
_WORKSPACE_CACHE_BYTES = 256 * 1024 * 1024
_PRECISION_CACHE_BYTES = 256 * 1024 * 1024
_STATIC_CACHE_LOCK = RLock()
_SCENE_SURFACE_CACHE: OrderedDict[tuple[Any, ...], tuple[Any, ...]] = OrderedDict()
_SCENE_ARRAY_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_RANDOM_WORKSPACE_CACHE: OrderedDict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()
_RANDOM_WORKSPACE_BYTES = 0
_PRECISION_ARRAY_CACHE: OrderedDict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, int]] = OrderedDict()
_PRECISION_ARRAY_BYTES = 0
_STATIC_CACHE_STATS = {
    "scene_hits": 0,
    "scene_misses": 0,
    "array_hits": 0,
    "array_misses": 0,
    "workspace_hits": 0,
    "workspace_misses": 0,
    "precision_hits": 0,
    "precision_misses": 0,
}


@dataclass(frozen=True)
class SteamRender:
    rir: np.ndarray
    band_rirs: dict[str, np.ndarray]
    ambisonic_rir: np.ndarray | None
    rt60_bands: dict[str, float]
    rt60_s: float
    direct: dict[str, Any]
    rir_rt60_bands: dict[str, float] = field(default_factory=dict)
    rir_rt60_s: float = 0.0
    steam_audio_rt60_bands: dict[str, float] = field(default_factory=dict)
    hybrid_rt60_bands: dict[str, float] = field(default_factory=dict)
    hybrid_rt60_s: float = 0.0
    paths: tuple[AcousticPath, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class _Surface:
    def __init__(self, kind: str, name: str, absorption: np.ndarray, scattering: np.ndarray, transmission: np.ndarray) -> None:
        self.kind = kind
        self.name = name
        self.absorption = absorption
        self.scattering = scattering
        self.transmission = transmission


class _WallSurface(_Surface):
    def __init__(
        self,
        a: np.ndarray,
        b: np.ndarray,
        height: float,
        name: str,
        absorption: np.ndarray,
        scattering: np.ndarray,
        transmission: np.ndarray,
        *,
        z_min: float = 0.0,
        z_max: float | None = None,
    ) -> None:
        super().__init__("wall", name, absorption, scattering, transmission)
        self.a = a
        self.b = b
        self.height = height
        self.z_min = max(0.0, float(z_min))
        self.z_max = min(float(height), float(height if z_max is None else z_max))
        seg = b - a
        normal_xy = np.asarray([seg[1], -seg[0]], dtype=float)
        n = float(np.linalg.norm(normal_xy))
        self.normal = np.asarray([normal_xy[0] / n, normal_xy[1] / n, 0.0], dtype=float) if n > _EPS else np.asarray([1.0, 0.0, 0.0])
        self._seg = seg

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> float:
        ray_xy = direction[:2]
        det = self._seg[0] * ray_xy[1] - self._seg[1] * ray_xy[0]
        if abs(det) <= 1e-12:
            return np.inf
        rel = origin[:2] - self.a
        t = (rel[0] * self._seg[1] - rel[1] * self._seg[0]) / det
        u = (rel[0] * ray_xy[1] - rel[1] * ray_xy[0]) / det
        if t <= _EPS or u < -1e-6 or u > 1.0 + 1e-6:
            return np.inf
        z = origin[2] + t * direction[2]
        if z < self.z_min - 1e-6 or z > self.z_max + 1e-6:
            return np.inf
        return float(t)

    def batch_intersect(self, origins: np.ndarray, dirs: np.ndarray, scene: "RoomRayScene") -> tuple[np.ndarray, np.ndarray]:
        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        sx, sy = self._seg
        ax, ay = self.a
        det = sx * dy - sy * dx
        relx, rely = ox - ax, oy - ay
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (relx * sy - rely * sx) / det
            u = (relx * dy - rely * dx) / det
        z = oz + t * dz
        valid = (np.abs(det) > 1e-12) & (t > _EPS) & (u >= -1e-6) & (u <= 1.0 + 1e-6) & (z >= self.z_min - 1e-6) & (z <= self.z_max + 1e-6)
        return np.where(valid, t, np.inf), np.tile(self.normal, (origins.shape[0], 1))


class _HorizontalSurface(_Surface):
    def __init__(self, z: float, up: bool, corners: Sequence[Sequence[float]], name: str, absorption: np.ndarray, scattering: np.ndarray, transmission: np.ndarray) -> None:
        super().__init__(name, name, absorption, scattering, transmission)
        self.z = float(z)
        self.corners = corners
        self.normal = np.asarray([0.0, 0.0, 1.0 if up else -1.0], dtype=float)

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> float:
        if abs(direction[2]) <= 1e-12:
            return np.inf
        t = (self.z - origin[2]) / direction[2]
        if t <= _EPS:
            return np.inf
        p = origin + t * direction
        return float(t) if point_in_polygon((p[0], p[1]), self.corners) else np.inf

    def batch_intersect(self, origins: np.ndarray, dirs: np.ndarray, scene: "RoomRayScene") -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.z - origins[:, 2]) / dirs[:, 2]
        p = origins + t[:, None] * dirs
        inside = _points_in_polygon_batch(p[:, :2], np.asarray(self.corners, dtype=float))
        valid = (np.abs(dirs[:, 2]) > 1e-12) & (t > _EPS) & inside
        return np.where(valid, t, np.inf), np.tile(self.normal, (origins.shape[0], 1))


class _BoxSurface(_Surface):
    def __init__(self, center: np.ndarray, size: np.ndarray, rotation_deg: float, z_center: float, name: str, absorption: np.ndarray, scattering: np.ndarray, transmission: np.ndarray) -> None:
        super().__init__("object", name, absorption, scattering, transmission)
        self.center = center.astype(float)
        self.size = np.maximum(size.astype(float), 1e-3)
        angle = math.radians(float(rotation_deg))
        self.axis_u = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
        self.axis_v = np.asarray([-math.sin(angle), math.cos(angle)], dtype=float)
        self.half_width = float(self.size[0] * 0.5)
        self.half_depth = float(self.size[1] * 0.5)
        half_height = float(self.size[2] * 0.5)
        self.z_min = float(max(0.0, z_center - half_height))
        self.z_max = float(max(self.z_min + 1e-3, z_center + half_height))
        self.normal = np.asarray([0.0, 0.0, 1.0], dtype=float)

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> float:
        t, _normal = self._hit(origin, direction)
        return float(t)

    def normal_at(self, origin: np.ndarray, direction: np.ndarray, _distance: float) -> np.ndarray:
        _t, normal = self._hit(origin, direction)
        return normal

    def batch_intersect(self, origins: np.ndarray, dirs: np.ndarray, scene: "RoomRayScene") -> tuple[np.ndarray, np.ndarray]:
        t_values = np.full(origins.shape[0], np.inf, dtype=float)
        normals = np.zeros((origins.shape[0], 3), dtype=float)
        for index in range(origins.shape[0]):
            t, normal = self._hit(origins[index], dirs[index])
            t_values[index] = t
            normals[index] = normal
        return t_values, normals

    def _hit(self, origin: np.ndarray, direction: np.ndarray) -> tuple[float, np.ndarray]:
        rel = origin[:2] - self.center
        ox = float(np.dot(rel, self.axis_u))
        oy = float(np.dot(rel, self.axis_v))
        oz = float(origin[2])
        dx = float(np.dot(direction[:2], self.axis_u))
        dy = float(np.dot(direction[:2], self.axis_v))
        dz = float(direction[2])
        hit = _box_hit_scalar(
            ox, oy, oz, dx, dy, dz,
            self.half_width, self.half_depth, self.z_min, self.z_max,
            self.axis_u[0], self.axis_u[1], self.axis_v[0], self.axis_v[1],
        )
        return hit[0], np.asarray(hit[1], dtype=float)


def _normalize_intersection_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in {"auto", "linear", "bvh"}:
        raise ValueError("intersection_backend must be auto, linear, or bvh")
    return backend


def _resolve_intersection_backend(value: str, surface_count: int, min_surfaces: int) -> str:
    requested = _normalize_intersection_backend(value)
    threshold = max(1, int(min_surfaces))
    return "bvh" if requested == "bvh" or (requested == "auto" and int(surface_count) >= threshold) else "linear"


def _surface_bounds(surface: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(surface, _WallSurface):
        lower = np.asarray([
            min(float(surface.a[0]), float(surface.b[0])),
            min(float(surface.a[1]), float(surface.b[1])),
            float(surface.z_min),
        ])
        upper = np.asarray([
            max(float(surface.a[0]), float(surface.b[0])),
            max(float(surface.a[1]), float(surface.b[1])),
            float(surface.z_max),
        ])
    elif isinstance(surface, _BoxSurface):
        extent = np.abs(surface.axis_u) * float(surface.half_width) + np.abs(surface.axis_v) * float(surface.half_depth)
        lower = np.asarray([
            float(surface.center[0] - extent[0]),
            float(surface.center[1] - extent[1]),
            float(surface.z_min),
        ])
        upper = np.asarray([
            float(surface.center[0] + extent[0]),
            float(surface.center[1] + extent[1]),
            float(surface.z_max),
        ])
    else:
        corners = np.asarray(surface.corners, dtype=np.float64)
        lower = np.asarray([float(np.min(corners[:, 0])), float(np.min(corners[:, 1])), float(surface.z)])
        upper = np.asarray([float(np.max(corners[:, 0])), float(np.max(corners[:, 1])), float(surface.z)])
    return lower - _BVH_BOUNDS_EPS, upper + _BVH_BOUNDS_EPS


def _ray_aabb_intersects(
    origin: np.ndarray,
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_distance: float,
) -> bool:
    enter = 0.0
    exit_ = float(max_distance) + _BVH_BOUNDS_EPS
    for axis in range(3):
        value = float(direction[axis])
        if abs(value) <= 1e-15:
            if float(origin[axis]) < float(lower[axis]) or float(origin[axis]) > float(upper[axis]):
                return False
            continue
        first = (float(lower[axis]) - float(origin[axis])) / value
        second = (float(upper[axis]) - float(origin[axis])) / value
        if first > second:
            first, second = second, first
        enter = max(enter, first)
        exit_ = min(exit_, second)
        if enter > exit_:
            return False
    return exit_ > _EPS


class RoomRayScene:
    def __init__(self, room: Room) -> None:
        self._set_room_state(room)
        corners = [np.asarray(c[:2], dtype=float) for c in room.corners]
        wall = room.materials.get("wall") or next(iter(room.materials.values()))
        floor = room.materials.get("floor", wall)
        ceiling = room.materials.get("ceiling", wall)
        self.surfaces: list[Any] = []
        self.surfaces.extend(_boundary_wall_surfaces(room, corners, wall))
        self.surfaces.append(_HorizontalSurface(0.0, True, room.corners, "floor", _band_array(floor, "absorption", 0.12), _band_array(floor, "scattering", 0.1), _band_array(floor, "transmission", 10.0 ** (-35.0 / 20.0))))
        self.surfaces.append(_HorizontalSurface(room.height_m, False, room.corners, "ceiling", _band_array(ceiling, "absorption", 0.1), _band_array(ceiling, "scattering", 0.1), _band_array(ceiling, "transmission", 10.0 ** (-30.0 / 20.0))))
        self.surfaces.extend(_object_box_surfaces(room, wall))
        self._batch_ready = False
        self._intersection_backend = "linear"
        self._bvh_arrays: Mapping[str, Any] | None = None

    def _set_room_state(self, room: Room) -> None:
        self.room = room
        multi_room = room.metadata.get("multi_room") if isinstance(room.metadata, Mapping) else None
        self.is_multi_room = bool(isinstance(multi_room, Mapping) and multi_room.get("enabled"))
        source_room_id = str(multi_room.get("source_room_id", "")) if isinstance(multi_room, Mapping) else ""
        receiver_room_id = str(multi_room.get("receiver_room_id", "")) if isinstance(multi_room, Mapping) else ""
        self.is_cross_room = bool(
            self.is_multi_room
            and source_room_id
            and receiver_room_id
            and source_room_id != receiver_room_id
        )

    def closest_hit(self, origin: np.ndarray, direction: np.ndarray) -> dict[str, Any]:
        best_t = np.inf
        best = None
        normal = None
        best_index = len(self.surfaces)
        candidates = self._bvh_candidate_indices(origin, direction, best_t) if self._intersection_backend == "bvh" else range(len(self.surfaces))
        for surface_index in candidates:
            surface = self.surfaces[surface_index]
            t = surface.intersect(origin, direction)
            if not np.isfinite(t):
                continue
            if t < best_t or (t == best_t and surface_index < best_index):
                best_t = t
                best = surface
                best_index = surface_index
                if hasattr(surface, "normal_at"):
                    normal = surface.normal_at(origin, direction, t)
                else:
                    normal = surface.normal.copy()
        if best is None:
            return {"valid": False, "distance": np.inf, "transmission": np.ones(_NUM_BANDS), "surface": None, "surface_index": -1}
        if float(np.dot(normal, direction)) > 0.0:
            normal = -normal
        return {"valid": True, "distance": best_t, "point": origin + best_t * direction, "normal": normal, "absorption": best.absorption, "scattering": best.scattering, "transmission": best.transmission, "surface": best.name, "surface_index": best_index}

    def any_hit(self, origin: np.ndarray, direction: np.ndarray, max_distance: float) -> bool:
        candidates = self._bvh_candidate_indices(origin, direction, max_distance) if self._intersection_backend == "bvh" else range(len(self.surfaces))
        for surface_index in candidates:
            surface = self.surfaces[surface_index]
            t = surface.intersect(origin, direction)
            if _EPS < t < max_distance - _EPS:
                return True
        return False

    def configure_intersection(self, backend: str, min_surfaces: int) -> str:
        resolved = _resolve_intersection_backend(backend, len(self.surfaces), min_surfaces)
        self._intersection_backend = resolved
        self._bvh_arrays = _scene_kernel_arrays(self) if resolved == "bvh" else None
        return resolved

    def _bvh_candidate_indices(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float,
    ) -> list[int]:
        arrays = self._bvh_arrays
        if arrays is None:
            return list(range(len(self.surfaces)))
        bounds_min = arrays["bvh_bounds_min"]
        bounds_max = arrays["bvh_bounds_max"]
        starts = arrays["bvh_start"]
        counts = arrays["bvh_count"]
        escapes = arrays["bvh_escape"]
        primitives = arrays["bvh_primitives"]
        candidates: list[int] = []
        node = 0
        while node < len(starts):
            if not _ray_aabb_intersects(origin, direction, bounds_min[node], bounds_max[node], max_distance):
                node = int(escapes[node])
                continue
            count = int(counts[node])
            if count > 0:
                start = int(starts[node])
                candidates.extend(int(value) for value in primitives[start:start + count])
            node += 1
        return candidates

    def _build_batch_arrays(self) -> None:
        if self._batch_ready:
            return
        self._corners = np.asarray([c[:2] for c in self.room.corners], dtype=float)
        self._surf_abs = np.asarray([s.absorption for s in self.surfaces], dtype=float)
        self._surf_sca_mean = np.asarray([float(np.mean(s.scattering)) for s in self.surfaces], dtype=float)
        self._surf_names = np.asarray([s.name for s in self.surfaces], dtype=object)
        self._batch_ready = True

    def _point_in_polygon_batch(self, pts: np.ndarray) -> np.ndarray:
        return _points_in_polygon_batch(pts, self._corners)

    def batch_closest_hit(self, origins: np.ndarray, dirs: np.ndarray) -> dict[str, Any]:
        self._build_batch_arrays()
        if self._intersection_backend == "bvh":
            items = [self.closest_hit(origins[index], dirs[index]) for index in range(origins.shape[0])]
            valid = np.asarray([bool(item["valid"]) for item in items], dtype=bool)
            surf = np.asarray([max(0, int(item.get("surface_index", -1))) for item in items], dtype=np.int64)
            t = np.asarray([float(item["distance"]) for item in items], dtype=np.float64)
            points = origins + np.where(valid, t, 0.0)[:, None] * dirs
            normals = np.asarray([
                np.asarray(item.get("normal", (0.0, 0.0, 1.0)), dtype=np.float64)
                for item in items
            ])
            return {"t": t, "valid": valid, "point": points, "normal": normals, "absorption": self._surf_abs[surf], "scattering": self._surf_sca_mean[surf], "surface": self._surf_names[surf]}
        hits = [surface.batch_intersect(origins, dirs, self) for surface in self.surfaces]
        t_all = np.stack([hit[0] for hit in hits], axis=0)
        normal_all = np.stack([hit[1] for hit in hits], axis=0)
        surf = np.argmin(t_all, axis=0)
        t = t_all[surf, np.arange(t_all.shape[1])]
        valid = np.isfinite(t)
        points = origins + np.where(valid, t, 0.0)[:, None] * dirs
        normals = normal_all[surf, np.arange(t_all.shape[1])]
        flip = np.sum(normals * dirs, axis=1) > 0.0
        normals = np.where(flip[:, None], -normals, normals)
        return {"t": t, "valid": valid, "point": points, "normal": normals, "absorption": self._surf_abs[surf], "scattering": self._surf_sca_mean[surf], "surface": self._surf_names[surf]}

    def batch_any_hit(self, origins: np.ndarray, dirs: np.ndarray, max_distance: np.ndarray) -> np.ndarray:
        self._build_batch_arrays()
        if self._intersection_backend == "bvh":
            return np.asarray([
                self.any_hit(origins[index], dirs[index], float(max_distance[index]))
                for index in range(origins.shape[0])
            ], dtype=bool)
        t_all = np.stack([surface.batch_intersect(origins, dirs, self)[0] for surface in self.surfaces], axis=0)
        return np.any((t_all > _EPS) & (t_all < (max_distance - _EPS)[None, :]), axis=0)


def _freeze_cache_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_cache_value(item)) for key, item in value.items()))
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ("ndarray", array.dtype.str, tuple(array.shape), array.tobytes())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    wkb = getattr(value, "wkb", None)
    if isinstance(wkb, bytes):
        return (type(value).__name__, wkb)
    return (type(value).__name__, repr(value))


def _scene_geometry_cache_key(room: Room) -> tuple[Any, ...]:
    material_signature = tuple(
        sorted(
            (
                str(kind),
                str(material.id),
                tuple(float(material.absorption.get(band, 0.2)) for band in FREQUENCY_BANDS),
                tuple(float(material.scattering.get(band, 0.1)) for band in FREQUENCY_BANDS),
                tuple(float(material.transmission.get(band, 10.0 ** (-30.0 / 20.0))) for band in FREQUENCY_BANDS),
            )
            for kind, material in room.materials.items()
        )
    )
    metadata = room.metadata if isinstance(room.metadata, Mapping) else {}
    geometry_metadata = {
        key: metadata.get(key)
        for key in ("surface_segments", "objects", "material_seed")
        if key in metadata
    }
    return (
        tuple((float(corner[0]), float(corner[1])) for corner in room.corners),
        float(room.height_m),
        material_signature,
        _freeze_cache_value(geometry_metadata),
    )


def _cached_room_ray_scene(room: Room) -> RoomRayScene:
    key = _scene_geometry_cache_key(room)
    with _STATIC_CACHE_LOCK:
        cached_surfaces = _SCENE_SURFACE_CACHE.get(key)
        if cached_surfaces is not None:
            _SCENE_SURFACE_CACHE.move_to_end(key)
            _STATIC_CACHE_STATS["scene_hits"] += 1
    if cached_surfaces is None:
        scene = RoomRayScene(room)
        cached_surfaces = tuple(scene.surfaces)
        with _STATIC_CACHE_LOCK:
            _SCENE_SURFACE_CACHE[key] = cached_surfaces
            _SCENE_SURFACE_CACHE.move_to_end(key)
            _STATIC_CACHE_STATS["scene_misses"] += 1
            while len(_SCENE_SURFACE_CACHE) > _STATIC_SCENE_CACHE_LIMIT:
                expired_key, _ = _SCENE_SURFACE_CACHE.popitem(last=False)
                _SCENE_ARRAY_CACHE.pop(expired_key, None)
        return scene

    scene = RoomRayScene.__new__(RoomRayScene)
    scene._set_room_state(room)
    scene.surfaces = list(cached_surfaces)
    scene._batch_ready = False
    scene._intersection_backend = "linear"
    scene._bvh_arrays = None
    return scene


def _points_in_polygon_batch(pts: np.ndarray, corners: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)
    j = corners.shape[0] - 1
    for i in range(corners.shape[0]):
        xi, yi = corners[i]
        xj, yj = corners[j]
        cond = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= cond & (x < x_cross)
        j = i
    return inside


def _boundary_wall_surfaces(room: Room, corners: Sequence[np.ndarray], wall: Any) -> list[_WallSurface]:
    raw_segments = room.metadata.get("surface_segments") if isinstance(room.metadata, Mapping) else None
    segments: list[tuple[np.ndarray, np.ndarray, str, float, float, str]] = []
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if not isinstance(item, Mapping):
                continue
            try:
                a = np.asarray(item.get("a"), dtype=float)
                b = np.asarray(item.get("b"), dtype=float)
                kind = str(item.get("type", "wall")).lower()
                z_min = float(item.get("z_min", 0.0))
                z_max = float(item.get("z_max", room.height_m))
                name = str(item.get("name", f"{kind}_{len(segments)}"))
            except (TypeError, ValueError):
                continue
            valid = (
                a.shape == (2,)
                and b.shape == (2,)
                and np.all(np.isfinite(a))
                and np.all(np.isfinite(b))
                and math.isfinite(z_min)
                and math.isfinite(z_max)
                and z_max - z_min > _EPS
            )
            if valid and float(np.linalg.norm(b - a)) > _EPS:
                segments.append((a, b, kind if kind in {"wall", "door", "window"} else "wall", z_min, z_max, name))
    if not segments:
        segments = [
            (a, corners[(index + 1) % len(corners)], "wall", 0.0, room.height_m, f"wall_{index}")
            for index, a in enumerate(corners)
        ]

    surfaces: list[_WallSurface] = []
    for a, b, kind, z_min, z_max, name in segments:
        material = room.materials.get(kind, wall if kind == "wall" else fallback_material(kind))
        surfaces.append(_WallSurface(
            a,
            b,
            room.height_m,
            name,
            _band_array(material, "absorption", 0.1),
            _band_array(material, "scattering", 0.12),
            _band_array(material, "transmission", 10.0 ** (-30.0 / 20.0)),
            z_min=z_min,
            z_max=z_max,
        ))
    return surfaces


def _box_hit_scalar(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    half_width: float, half_depth: float,
    z_min: float, z_max: float,
    ux: float, uy: float, vx: float, vy: float,
) -> tuple[float, tuple[float, float, float]]:
    t_min = -1.0e30
    t_max = 1.0e30
    nx = ny = nz = 0.0
    exit_nx = exit_ny = exit_nz = 0.0
    for o, d, lo, hi, ax, ay, az in (
        (ox, dx, -half_width, half_width, ux, uy, 0.0),
        (oy, dy, -half_depth, half_depth, vx, vy, 0.0),
        (oz, dz, z_min, z_max, 0.0, 0.0, 1.0),
    ):
        if abs(d) <= 1.0e-12:
            if o < lo or o > hi:
                return np.inf, (0.0, 0.0, 1.0)
            continue
        if d > 0.0:
            enter = (lo - o) / d
            exit_ = (hi - o) / d
            enter_normal = (-ax, -ay, -az)
            exit_normal = (ax, ay, az)
        else:
            enter = (hi - o) / d
            exit_ = (lo - o) / d
            enter_normal = (ax, ay, az)
            exit_normal = (-ax, -ay, -az)
        if enter > t_min:
            t_min = enter
            nx, ny, nz = enter_normal
        if exit_ < t_max:
            t_max = exit_
            exit_nx, exit_ny, exit_nz = exit_normal
        if t_min > t_max:
            return np.inf, (0.0, 0.0, 1.0)
    if t_max <= _EPS:
        return np.inf, (0.0, 0.0, 1.0)
    if t_min > _EPS:
        return float(t_min), (float(nx), float(ny), float(nz))
    return float(t_max), (float(exit_nx), float(exit_ny), float(exit_nz))


def _object_box_surfaces(room: Room, fallback_material: Any) -> list[_BoxSurface]:
    raw_objects = room.metadata.get("objects", []) if isinstance(room.metadata, dict) else []
    if not isinstance(raw_objects, list):
        return []
    surfaces: list[_BoxSurface] = []
    library: MaterialLibrary | None = None
    for index, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            continue
        if str(item.get("semantic", "")) == "small_objects_ignore":
            continue
        try:
            if library is None:
                library = MaterialLibrary.load()
            boxes, material, object_absorption = _resolved_object_acoustics(room, item, index, library)
            if not boxes:
                continue
        except Exception:
            material = fallback_material
            boxes = _object_proxy_boxes(item, room.height_m)
            object_absorption = _band_array(material, "absorption", 0.2)
        for box in boxes:
            center = np.asarray(box["center"], dtype=float)
            if not point_in_polygon(center, room.corners):
                continue
            surfaces.append(_BoxSurface(
                center=center,
                size=np.asarray(box["size"], dtype=float),
                rotation_deg=float(box["rotation"]),
                z_center=float(box["z"]),
                name=f"object_{index}_{str(item.get('type', 'furniture'))}_{str(box['part'])}",
                absorption=object_absorption,
                scattering=_band_array(material, "scattering", 0.18),
                transmission=_band_array(material, "transmission", 10.0 ** (-24.0 / 20.0)),
            ))
    return surfaces


def _resolved_object_acoustics(
    room: Room,
    item: dict[str, Any],
    index: int,
    library: MaterialLibrary,
) -> tuple[list[dict[str, Any]], Any, np.ndarray]:
    boxes = _object_proxy_boxes(item, room.height_m)
    if not boxes:
        return [], fallback_material("structural_element"), np.zeros(_NUM_BANDS, dtype=np.float64)
    semantic = str(item.get("semantic", item.get("type", "structural_element")))
    selected = item.get("material_selection") if isinstance(item.get("material_selection"), Mapping) else {}
    if selected.get("material_id"):
        material = library.resolve(
            {"material_id": str(selected["material_id"])},
            default_semantic=semantic,
        )
    else:
        material = library.sample_geometry(item, seed=int(room.metadata.get("material_seed", 0)) + index + 1)
        item["material_selection"] = material_summary(material)
    object_absorption = _effective_object_absorption(material, boxes)
    if str(material.metadata.get("coefficient_kind", "")) == "equivalent_absorption_area_m2":
        selection = item.get("material_selection")
        if isinstance(selection, dict):
            selection["equivalent_absorption_area_m2"] = dict(material.absorption)
            selection["effective_absorption"] = {
                band: float(value) for band, value in zip(FREQUENCY_BANDS, object_absorption)
            }
    return boxes, material, object_absorption


def object_absorption_areas(room: Room) -> list[dict[str, Any]]:
    """Return furniture absorption areas using the ray tracer's proxy geometry."""
    raw_objects = room.metadata.get("objects") if isinstance(room.metadata, Mapping) else None
    if not isinstance(raw_objects, list):
        return []
    library = MaterialLibrary.load()
    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw_objects):
        if not isinstance(item, dict) or str(item.get("semantic", "")) == "small_objects_ignore":
            continue
        try:
            boxes, _material, absorption = _resolved_object_acoustics(room, item, index, library)
        except Exception:
            continue
        total_area = 0.0
        weighted_center = np.zeros(2, dtype=np.float64)
        for box in boxes:
            size = np.asarray(box.get("size", ()), dtype=np.float64)
            center = np.asarray(box.get("center", ()), dtype=np.float64)
            if size.shape != (3,) or center.shape != (2,):
                continue
            width, depth, height = np.maximum(size, 0.0)
            box_area = float(2.0 * (width * depth + width * height + depth * height))
            total_area += box_area
            weighted_center += box_area * center
        if total_area <= 1e-12:
            continue
        records.append({
            "id": str(item.get("id", f"object_{index}")),
            "center": weighted_center / total_area,
            "surface_area_m2": total_area,
            "absorption_area_m2": np.asarray(absorption, dtype=np.float64) * total_area,
        })
    return records


def _effective_object_absorption(material: Any, boxes: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = _band_array(material, "absorption", 0.2)
    metadata = getattr(material, "metadata", {}) or {}
    if str(metadata.get("coefficient_kind", "")) != "equivalent_absorption_area_m2":
        return values
    total_area = 0.0
    for box in boxes:
        size = np.asarray(box.get("size", (0.0, 0.0, 0.0)), dtype=float)
        if size.shape != (3,) or not np.all(np.isfinite(size)):
            continue
        width, depth, height = np.maximum(size, 0.0)
        total_area += 2.0 * (width * depth + width * height + depth * height)
    return np.clip(values / max(total_area, 1.0e-6), 0.0, 0.99)


def _object_proxy_boxes(item: dict[str, Any], room_height: float) -> list[dict[str, Any]]:
    try:
        size = np.asarray(item.get("size", (1.0, 1.0, 1.0)), dtype=float)
        if size.shape != (3,) or not np.all(np.isfinite(size)) or float(np.min(size)) <= 0.0:
            return []
        position = item.get("position", (0.0, 0.0))
        center = np.asarray((float(position[0]), float(position[1])), dtype=float)
        rotation = float(item.get("rotation", 0.0))
        z_center = float(item.get("z", size[2] * 0.5))
    except Exception:
        return []
    object_type = str(item.get("type", "furniture"))
    height = float(size[2])
    angle = math.radians(rotation)
    axis_u = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
    axis_v = np.asarray([-math.sin(angle), math.cos(angle)], dtype=float)
    if object_type == "table":
        leg_h = max(0.05, height)
        boxes = [{
            "part": "top",
            "center": center,
            "size": np.asarray([size[0], size[1], min(0.08, max(0.04, height * 0.16))], dtype=float),
            "z": min(room_height, height),
            "rotation": rotation,
        }]
        for leg_index, (sx, sy) in enumerate(((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))):
            offset = axis_u * (sx * size[0] * 0.42) + axis_v * (sy * size[1] * 0.38)
            boxes.append({
                "part": f"leg{leg_index}",
                "center": center + offset,
                "size": np.asarray([0.08, 0.08, leg_h], dtype=float),
                "z": min(room_height, leg_h * 0.5),
                "rotation": rotation,
            })
        return boxes
    if object_type == "sanitary_fixture":
        base_z = z_center - height * 0.5
        wall_height = height * 0.72
        boxes = [{
            "part": "base",
            "center": center,
            "size": np.asarray([size[0] * 0.82, size[1] * 0.72, height * 0.16], dtype=float),
            "z": min(room_height, base_z + height * 0.08),
            "rotation": rotation,
        }]
        for part, offset in (("back", -0.44), ("front", 0.44)):
            boxes.append({
                "part": part,
                "center": center + axis_v * (size[1] * offset),
                "size": np.asarray([size[0], size[1] * 0.12, wall_height], dtype=float),
                "z": min(room_height, base_z + wall_height * 0.5),
                "rotation": rotation,
            })
        for part, offset in (("left", -0.445), ("right", 0.445)):
            boxes.append({
                "part": part,
                "center": center + axis_u * (size[0] * offset),
                "size": np.asarray([size[0] * 0.11, size[1] * 0.76, wall_height], dtype=float),
                "z": min(room_height, base_z + wall_height * 0.5),
                "rotation": rotation,
            })
        return boxes
    if object_type == "structural_element":
        base_z = z_center - height * 0.5
        return [
            {
                "part": "base",
                "center": center,
                "size": np.asarray([size[0], size[1], height * 0.06], dtype=float),
                "z": min(room_height, base_z + height * 0.03),
                "rotation": rotation,
            },
            {
                "part": "shaft",
                "center": center,
                "size": np.asarray([size[0] * 0.78, size[1] * 0.78, height * 0.88], dtype=float),
                "z": min(room_height, base_z + height * 0.5),
                "rotation": rotation,
            },
            {
                "part": "capital",
                "center": center,
                "size": np.asarray([size[0], size[1], height * 0.06], dtype=float),
                "z": min(room_height, base_z + height * 0.97),
                "rotation": rotation,
            },
        ]
    return [{
        "part": "body",
        "center": center,
        "size": size,
        "z": z_center,
        "rotation": rotation,
    }]


def _scene_kernel_arrays(scene: RoomRayScene) -> dict[str, Any]:
    cache_key = _scene_geometry_cache_key(scene.room)
    with _STATIC_CACHE_LOCK:
        cached = _SCENE_ARRAY_CACHE.get(cache_key)
        if cached is not None:
            _SCENE_ARRAY_CACHE.move_to_end(cache_key)
            _STATIC_CACHE_STATS["array_hits"] += 1
            return cached
    kinds = np.zeros(len(scene.surfaces), dtype=np.int64)
    wall_a = np.zeros((len(scene.surfaces), 2), dtype=np.float64)
    wall_b = np.zeros((len(scene.surfaces), 2), dtype=np.float64)
    wall_z = np.zeros((len(scene.surfaces), 2), dtype=np.float64)
    wall_z[:, 1] = float(scene.room.height_m)
    z_values = np.zeros(len(scene.surfaces), dtype=np.float64)
    box_center = np.zeros((len(scene.surfaces), 2), dtype=np.float64)
    box_axis_u = np.zeros((len(scene.surfaces), 2), dtype=np.float64)
    box_axis_v = np.zeros((len(scene.surfaces), 2), dtype=np.float64)
    box_half = np.zeros((len(scene.surfaces), 2), dtype=np.float64)
    box_z = np.zeros((len(scene.surfaces), 2), dtype=np.float64)
    normals = np.zeros((len(scene.surfaces), 3), dtype=np.float64)
    absorption = np.zeros((len(scene.surfaces), _NUM_BANDS), dtype=np.float64)
    scattering = np.zeros(len(scene.surfaces), dtype=np.float64)
    names: list[str] = []
    for index, surface in enumerate(scene.surfaces):
        names.append(str(surface.name))
        normals[index] = np.asarray(surface.normal, dtype=np.float64)
        absorption[index] = np.asarray(surface.absorption, dtype=np.float64)
        scattering[index] = float(np.mean(surface.scattering))
        if isinstance(surface, _WallSurface):
            kinds[index] = 0
            wall_a[index] = np.asarray(surface.a, dtype=np.float64)
            wall_b[index] = np.asarray(surface.b, dtype=np.float64)
            wall_z[index] = np.asarray([surface.z_min, surface.z_max], dtype=np.float64)
        elif isinstance(surface, _BoxSurface):
            kinds[index] = 2
            box_center[index] = np.asarray(surface.center, dtype=np.float64)
            box_axis_u[index] = np.asarray(surface.axis_u, dtype=np.float64)
            box_axis_v[index] = np.asarray(surface.axis_v, dtype=np.float64)
            box_half[index] = np.asarray([surface.half_width, surface.half_depth], dtype=np.float64)
            box_z[index] = np.asarray([surface.z_min, surface.z_max], dtype=np.float64)
        else:
            kinds[index] = 1
            z_values[index] = float(surface.z)
    surface_bounds = [_surface_bounds(surface) for surface in scene.surfaces]
    surface_bounds_min = np.asarray([bounds[0] for bounds in surface_bounds], dtype=np.float64)
    surface_bounds_max = np.asarray([bounds[1] for bounds in surface_bounds], dtype=np.float64)
    bvh_started = time.perf_counter()
    bvh = build_bvh(surface_bounds_min, surface_bounds_max)
    bvh_build_time_ms = (time.perf_counter() - bvh_started) * 1000.0
    arrays = {
        "kinds": kinds,
        "wall_a": wall_a,
        "wall_delta": wall_b - wall_a,
        "wall_z": wall_z,
        "z_values": z_values,
        "box_center": box_center,
        "box_axis_u": box_axis_u,
        "box_axis_v": box_axis_v,
        "box_half": box_half,
        "box_z": box_z,
        "normals": normals,
        "absorption": absorption,
        "reflection": 1.0 - absorption,
        "scattering": scattering,
        "names": tuple(names),
        "corners": np.asarray(scene.room.corners, dtype=np.float64)[:, :2],
        "height": float(scene.room.height_m),
        "surface_bounds_min": surface_bounds_min,
        "surface_bounds_max": surface_bounds_max,
        "bvh_bounds_min": bvh["bounds_min"],
        "bvh_bounds_max": bvh["bounds_max"],
        "bvh_start": bvh["start"],
        "bvh_count": bvh["count"],
        "bvh_escape": bvh["escape"],
        "bvh_primitives": bvh["primitives"],
        "bvh_node_count": int(bvh["node_count"]),
        "bvh_leaf_count": int(bvh["leaf_count"]),
        "bvh_max_depth": int(bvh["max_depth"]),
        "bvh_leaf_size": int(bvh["leaf_size"]),
        "bvh_build_time_ms": float(bvh_build_time_ms),
    }
    for value in arrays.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    with _STATIC_CACHE_LOCK:
        _STATIC_CACHE_STATS["array_misses"] += 1
        _SCENE_ARRAY_CACHE[cache_key] = arrays
        _SCENE_ARRAY_CACHE.move_to_end(cache_key)
        while len(_SCENE_ARRAY_CACHE) > _STATIC_SCENE_CACHE_LIMIT:
            _SCENE_ARRAY_CACHE.popitem(last=False)
    return arrays


def simulate_steam_room(
    room: Room,
    source: Sequence[float],
    listener: Sequence[float],
    config: SimConfig,
    source_model: str | Mapping[str, Any] | None = None,
    late_reverb_prior: Mapping[str, float] | None = None,
    render_ambisonics: bool | None = None,
) -> SteamRender:
    src = np.asarray(source, dtype=float)
    rcv = np.asarray(listener, dtype=float)
    emitter = source_directivity(source_model)
    spatial_output = bool(config.render_ambisonics if render_ambisonics is None else render_ambisonics)
    scene = _cached_room_ray_scene(room)
    intersection_backend = scene.configure_intersection(config.intersection_backend, config.bvh_min_surfaces)
    direct = simulate_direct(scene, src, rcv, config, emitter)
    fs = int(config.fs)
    total = max(1, int(round(config.duration_s * fs)))
    discrete_band = np.zeros((_NUM_BANDS, total), dtype=np.float32)
    reflection_band = np.zeros_like(discrete_band)
    direct_sample = int(round(direct["delay_s"] * fs))
    _add_band_impulse(discrete_band, float(direct["delay_s"]), direct["band_gains"], config)

    paths = [_direct_path(src, rcv, direct, config)]
    portal_paths = _multi_room_portal_paths(room, scene, src, rcv, direct, config, emitter)
    diffraction_room = _same_room_diffraction_room(room, scene)
    diffraction_paths = [] if scene.is_cross_room else [
        _apply_source_directivity_to_path(path, emitter)
        for path in _boundary_diffraction_paths(diffraction_room, scene, src, rcv, direct, config)
    ]
    for path in portal_paths:
        sample = int(round(path.delay_s * fs))
        if 0 <= sample < total:
            _add_band_impulse(discrete_band, float(path.delay_s), dict(path.band_gains), config)
    for path in diffraction_paths:
        sample = int(round(path.delay_s * fs))
        if config.diffraction_audio_enabled and 0 <= sample < total:
            _add_band_impulse(discrete_band, float(path.delay_s), path.band_gains, config)
    paths.extend(portal_paths)
    paths.extend(diffraction_paths)
    rt_visual: dict[str, Any] = {
        "paths": [],
        "metadata": {
            "enabled": False,
            "model": "energy_trace_representatives_pending",
            "ray_count": int(config.rt_num_rays),
            "max_bounces": int(config.rt_num_bounces),
            "follows_simulation": True,
            "retain_limit": int(_RT_VISUAL_RETAIN_LIMIT),
            "retained_path_count": 0,
        },
    }
    rt60_bands = {band: 0.0 for band in FREQUENCY_BANDS}
    late_tail_target_rt60_bands = {band: 0.0 for band in FREQUENCY_BANDS}
    late_decay_profiles: dict[str, Any] = {}
    rendered_late_decay_profiles: dict[str, Any] = {}
    hybrid_rt60_bands = {band: 0.0 for band in FREQUENCY_BANDS}
    reconstructed_rt60_bands = {band: 0.0 for band in FREQUENCY_BANDS}
    reflection_metadata: dict[str, Any] = {"enabled": False}

    if config.reflections_enabled:
        reflection_config, adaptive_bounce_meta = _adaptive_reflection_config(scene, config)
        field = trace_energy_field(
            scene,
            src,
            rcv,
            reflection_config,
            emitter,
            render_ambisonics=spatial_output,
        )
        if config.collect_visual_paths:
            rt_visual = _visual_rt_paths_from_energy_field(field, src, rcv, reflection_config)
            paths.extend(rt_visual["paths"])
        else:
            rt_visual = {
                "paths": [],
                "metadata": {
                    "enabled": False,
                    "model": "disabled_for_headless_api",
                    "ray_count": int(config.rt_num_rays),
                    "max_bounces": int(reflection_config.rt_num_bounces),
                    "follows_simulation": True,
                    "retain_limit": int(_RT_VISUAL_RETAIN_LIMIT),
                    "retained_path_count": 0,
                    "shares_energy_trace": True,
                },
            }
        rt60_bands = estimate_reverb_times(field, reflection_config)
        if scene.is_multi_room:
            late_tail_target_rt60_bands, late_decay_profiles = estimate_late_reverb_times(
                field,
                reflection_config,
                fallback=rt60_bands,
            )
            late_tail_target_rt60_bands, late_decay_profiles = _apply_coupled_late_reverb_prior(
                late_tail_target_rt60_bands,
                late_decay_profiles,
                late_reverb_prior,
            )
        else:
            late_tail_target_rt60_bands = dict(rt60_bands)
        traced_band_irs = reconstruct_band_irs(field, config)
        band_irs, fdn_component, late_tail_meta = _render_parametric_fdn_late_reverb(
            traced_band_irs,
            field,
            late_tail_target_rt60_bands,
            config,
            decay_profiles=late_decay_profiles,
        )
        hybrid_rt60_bands = estimate_reconstructed_reverb_times(band_irs, config)
        rendered_late_decay_profiles = {
            band: estimate_signal_decay_profile(band_irs[band_index], config)
            for band_index, band in enumerate(FREQUENCY_BANDS)
        }
        seg_len = min(band_irs.shape[1], total - direct_sample) if direct_sample < total else 0
        ambisonic_rir = np.zeros((4, total), dtype=np.float32) if spatial_output else None
        if seg_len > 0:
            reflection_band[:, direct_sample:direct_sample + seg_len] += band_irs[:, :seg_len]
            if spatial_output:
                traced_ambisonic_band_irs = reconstruct_ambisonic_band_irs(field, config)
                ambisonic_band_irs = _hybridize_ambisonic_tail(
                    traced_ambisonic_band_irs,
                    fdn_component,
                    late_tail_meta,
                    config,
                )
                ambisonic_rir[:, direct_sample:direct_sample + seg_len] += np.sum(ambisonic_band_irs[:, :, :seg_len], axis=0)
        quality_warnings: list[str] = []
        if int(config.rt_num_rays) < 4096:
            quality_warnings.append("ray count is below the Steam Audio realtime reference")
        if int(reflection_config.rt_num_bounces) < 64:
            quality_warnings.append("RT60 is biased by bounce truncation; use at least 64 bounces")
        max_rt60 = max(rt60_bands.values(), default=0.0)
        if max_rt60 > 0.0 and float(config.rt_duration_s) < 1.2 * max_rt60:
            quality_warnings.append("reflection duration is shorter than 1.2 times the estimated RT60")
        reflection_metadata = {
            "enabled": True,
            "num_rays": int(config.rt_num_rays),
            "requested_num_bounces": int(config.rt_num_bounces),
            "num_bounces": int(reflection_config.rt_num_bounces),
            "adaptive_bounces": adaptive_bounce_meta,
            "num_bins": int(field["num_bins"]),
            "bin_duration_s": float(field["bin_duration_s"]),
            "actual_bounces": int(field.get("actual_bounces", 0)),
            "active_ray_count": int(field.get("active_ray_count", 0)),
            "last_energy_time_s": float(field.get("last_energy_time_s", 0.0)),
            "traced_energy": float(np.sum(field["echogram"])),
            "late_tail_energy": float(late_tail_meta.get("rendered_tail_energy", 0.0)),
            "traced_late_tail_energy": float(late_tail_meta.get("traced_tail_energy", 0.0)),
            "model": "monte_carlo_path_tracing_energy_field",
            "late_tail_enabled": bool(config.late_tail),
            "late_tail_cutoff_s": float(late_tail_meta.get("transition_start_s", 0.0)),
            "late_tail": late_tail_meta,
            "late_tail_target_rt60_bands": late_tail_target_rt60_bands,
            "late_decay_profiles": late_decay_profiles,
            "rendered_late_decay_profiles": rendered_late_decay_profiles,
            "hybrid_rt60_bands": hybrid_rt60_bands,
            "quality": _reflection_quality_label(reflection_config),
            "quality_warnings": quality_warnings,
            "surface_hit_count": field.get("surface_hit_count", {}),
            "surface_contribution_count": field.get("surface_contribution_count", {}),
            "surface_energy": field.get("surface_energy", {}),
            "accelerator": field.get("accelerator", "numpy"),
            "precision": field.get("precision", "float64"),
            "cuda": field.get("cuda"),
            "kernel_time_s": field.get("kernel_time_s"),
            "transfer_time_s": field.get("transfer_time_s"),
            "device_input_cache": field.get("device_input_cache"),
            "ambisonics": {
                "enabled": bool(spatial_output),
                "order": 1,
                "channels": ["W", "X", "Y", "Z"],
                "normalization": "acoustic_agent_foa_unit_vector",
                "energy": float(np.sum(ambisonic_rir * ambisonic_rir)) if ambisonic_rir is not None else 0.0,
            },
        }
    else:
        ambisonic_rir = None

    band_matrix = bandlimit_band_signals(discrete_band, fs) + reflection_band
    rir = np.sum(band_matrix, axis=0, dtype=np.float32)
    values = [v for v in rt60_bands.values() if v > 0.0]
    rt60_s = float(np.mean(values)) if values else 0.0
    rir_rt60_bands = estimate_reconstructed_reverb_times(band_matrix, config)
    rir_rt60_s = estimate_signal_reverb_time(rir, config)
    steam_audio_rt60_bands = estimate_steam_audio_default_reverb_times(rir, config)
    hybrid_values = [v for v in hybrid_rt60_bands.values() if v > 0.0]
    hybrid_rt60_s = float(np.mean(hybrid_values)) if hybrid_values else 0.0
    return SteamRender(
        rir=rir.astype(np.float32),
        band_rirs={band: band_matrix[i] for i, band in enumerate(FREQUENCY_BANDS)},
        ambisonic_rir=ambisonic_rir,
        rt60_bands=rt60_bands,
        rt60_s=round(rt60_s, 4),
        rir_rt60_bands=rir_rt60_bands,
        rir_rt60_s=round(rir_rt60_s, 4),
        steam_audio_rt60_bands=steam_audio_rt60_bands,
        hybrid_rt60_bands=hybrid_rt60_bands,
        hybrid_rt60_s=round(hybrid_rt60_s, 4),
        direct=direct,
        paths=tuple(paths),
        metadata={
            "model": "steam_audio_style_direct_plus_pathtraced_reflections",
            "tiers": ["direct", "edge_diffraction", "reflections_energy_field", "reverb_estimate"],
            "reverb_estimator": {
                "traced_model": "schroeder_fit_from_path_traced_energy_field",
                "rir_model": "schroeder_fit_from_final_reconstructed_band_rirs",
                "tail_target_model": (
                    "coupled_room_energy_matrix_prior"
                    if scene.is_multi_room and late_reverb_prior
                    else "coupled_space_late_slope_from_traced_energy_field"
                    if scene.is_multi_room
                    else "steam_style_single_slope_from_traced_energy_field"
                ),
                "fit_range_db": [-5.0, -25.0],
                "extrapolation_db": -60.0,
                "bin_duration_s": float(config.rt_bin_duration_s),
                "rir_rt60_s": float(rir_rt60_s),
                "rir_rt60_bands": rir_rt60_bands,
                "steam_audio_default_bands": {
                    "model": "default_three_band_low_mid_high",
                    "cutoffs_hz": {"low": [0.0, 800.0], "mid": [800.0, 8000.0], "high": [8000.0, 22000.0]},
                    "rt60_bands": steam_audio_rt60_bands,
                },
                "tail_target_rt60_bands": late_tail_target_rt60_bands,
                "rendered_hybrid_rt60_bands": hybrid_rt60_bands,
            },
            "direct": {
                "distance_m": round(float(direct["distance_m"]), 5),
                "delay_s": round(float(direct["delay_s"]), 6),
                "distance_attenuation": float(direct["distance_attenuation"]),
                "occlusion": float(direct["occlusion"]),
                "occlusion_surface": direct.get("occlusion_surface"),
                "transmission": {band: float(direct["transmission"][i]) for i, band in enumerate(FREQUENCY_BANDS)},
                "band_gains": {band: float(direct["band_gains"][band]) for band in FREQUENCY_BANDS},
                "source_directivity_gain": float(direct["source_directivity_gain"]),
            },
            "diffraction": {
                "enabled": bool(config.diffraction_enabled),
                "path_count": len(diffraction_paths),
                "model": _diffraction_metadata_model(diffraction_paths),
                "order": _diffraction_max_order(diffraction_paths, int(config.diffraction_order)),
                "order_counts": _diffraction_order_counts(diffraction_paths),
                "skipped_reason": _diffraction_skip_reason(direct, config, diffraction_paths),
                "contributes_to_rir": bool(config.diffraction_audio_enabled and diffraction_paths),
            },
            "portal_propagation": {
                "enabled": bool(scene.is_multi_room),
                "path_count": len(portal_paths),
                "model": "verified_portal_visibility_graph_pathing_v2" if scene.is_multi_room else "not_applicable",
                "accelerator": "python_visibility_graph" if scene.is_multi_room else None,
                "contributes_to_rir": bool(portal_paths),
            },
            "reflections": reflection_metadata,
            "rt_visual": rt_visual["metadata"],
            "intersection": {
                "requested_backend": _normalize_intersection_backend(config.intersection_backend),
                "backend": intersection_backend,
                "surface_count": len(scene.surfaces),
                "auto_threshold": max(1, int(config.bvh_min_surfaces)),
                "bvh_node_count": int(scene._bvh_arrays["bvh_node_count"]) if scene._bvh_arrays is not None else 0,
                "bvh_leaf_count": int(scene._bvh_arrays["bvh_leaf_count"]) if scene._bvh_arrays is not None else 0,
                "bvh_max_depth": int(scene._bvh_arrays["bvh_max_depth"]) if scene._bvh_arrays is not None else 0,
                "bvh_build_time_ms": float(scene._bvh_arrays["bvh_build_time_ms"]) if scene._bvh_arrays is not None else 0.0,
            },
            "rt60_bands": rt60_bands,
            "source_directivity": dict(emitter),
        },
    )


def _diffraction_order_counts(paths: Sequence[AcousticPath]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        order = int(path.metadata.get("diffraction_order", 1))
        key = str(order)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _reflection_quality_label(config: SimConfig) -> str:
    rays = int(config.rt_num_rays)
    bounces = int(config.rt_num_bounces)
    if rays >= 131072 and bounces >= 96:
        return "reference"
    if rays >= 65536 and bounces >= 96:
        return "fine"
    if rays >= 32768 and bounces >= 64:
        return "simulation"
    if rays >= 8192 and bounces >= 32:
        return "preview"
    return "custom"


def _adaptive_reflection_config(scene: RoomRayScene, config: SimConfig) -> tuple[SimConfig, dict[str, Any]]:
    requested = max(1, int(config.rt_num_bounces))
    metadata: dict[str, Any] = {
        "enabled": bool(config.adaptive_cross_room_bounces or config.adaptive_geometry_bounces),
        "applied": False,
        "requested": requested,
        "effective": requested,
        "reason": "not_applicable",
    }
    if int(config.rt_num_rays) < 32768 or requested < 64:
        metadata["reason"] = "explicit_or_preview_budget"
        return config, metadata

    absorption = np.asarray([surface.absorption for surface in scene.surfaces], dtype=np.float64)
    mean_survival = float(np.max(np.mean(1.0 - absorption, axis=0))) if absorption.size else 0.0
    if not scene.is_multi_room:
        metadata["enabled"] = bool(config.adaptive_geometry_bounces)
        if not config.adaptive_geometry_bounces:
            metadata["reason"] = "disabled"
            return config, metadata
        if not 0.0 < mean_survival < 1.0:
            metadata["reason"] = "no_decay_estimate"
            return config, metadata
        target_energy_ratio = 10.0 ** (-25.0 / 10.0)
        required = int(math.ceil(math.log(target_energy_ratio) / math.log(mean_survival)))
        required = 16 * int(math.ceil(max(1, required) / 16.0))
        maximum = max(requested, int(config.geometry_max_bounces))
        effective = min(maximum, max(requested, required))
        metadata.update({
            "effective": effective,
            "maximum": maximum,
            "estimated_bounces_to_fit_floor": required,
            "fit_floor_db": -25.0,
            "mean_max_band_survival": round(mean_survival, 4),
        })
        if effective <= requested:
            metadata["reason"] = "requested_budget_sufficient"
            return config, metadata
        metadata.update({
            "applied": True,
            "reason": "geometry_tail_convergence",
        })
        return replace(config, rt_num_bounces=effective), metadata

    if not scene.is_cross_room:
        metadata["reason"] = "same_room_floorplan"
        return config, metadata
    metadata["enabled"] = bool(config.adaptive_cross_room_bounces)
    if not config.adaptive_cross_room_bounces:
        metadata["reason"] = "disabled"
        return config, metadata

    minimum = max(96, int(config.cross_room_min_bounces))
    maximum = max(minimum, int(config.cross_room_max_bounces))
    multi_room = scene.room.metadata.get("multi_room") if isinstance(scene.room.metadata, Mapping) else None
    route_portals = multi_room.get("route_portal_ids", []) if isinstance(multi_room, Mapping) else []
    if len(route_portals) > 1 or mean_survival >= 0.90:
        extra = 32
    elif mean_survival >= 0.82:
        extra = 16
    else:
        extra = 0
    effective = min(maximum, max(minimum, minimum + extra))
    if requested >= effective:
        metadata.update({"reason": "requested_budget_sufficient", "effective": requested})
        return config, metadata
    metadata.update({
        "applied": True,
        "effective": effective,
        "minimum": minimum,
        "maximum": maximum,
        "route_portal_count": len(route_portals),
        "mean_max_band_survival": round(mean_survival, 4),
        "reason": "cross_room_tail_convergence",
    })
    return replace(config, rt_num_bounces=effective), metadata


def _diffraction_max_order(paths: Sequence[AcousticPath], fallback: int) -> int:
    if not paths:
        return int(fallback)
    return max(int(path.metadata.get("diffraction_order", 1)) for path in paths)


def _diffraction_metadata_model(paths: Sequence[AcousticPath]) -> str:
    models = {str(path.metadata.get("model", "")) for path in paths}
    models.discard("")
    if not models:
        return "none"
    if len(models) == 1:
        return next(iter(models))
    return "mixed_edge_diffraction"


def simulate_direct(
    scene: RoomRayScene,
    source: np.ndarray,
    listener: np.ndarray,
    config: SimConfig,
    source_model: str | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    emitter = source_directivity(source_model)
    delta = source - listener
    distance = float(np.linalg.norm(delta))
    directivity = source_directivity_gain(listener - source, emitter)
    distance_attenuation = 1.0 / max(distance, config.min_distance_m)
    air = np.asarray([math.exp(-AIR_ABSORPTION_NP_PER_M[b] * distance) for b in FREQUENCY_BANDS], dtype=float)
    delay = distance / config.c
    occlusion = 1.0
    transmission = np.ones(_NUM_BANDS, dtype=float)
    occlusion_surface = None
    if config.direct_occlusion and distance > 1e-6:
        direction = delta / distance
        hit = scene.closest_hit(listener, direction)
        occlusion_surface = hit["surface"] if hit["valid"] and hit["distance"] < distance - 1e-6 else None
        occlusion = _volumetric_occlusion(scene, listener, source, config) if config.direct_occlusion_mode == "volumetric" else (0.0 if occlusion_surface is not None else 1.0)
        if occlusion < 1.0:
            transmission = (
                _transmission(scene, listener, source, distance, config)
                if config.direct_transmission
                else np.zeros(_NUM_BANDS, dtype=float)
            )
    band_gains = {
        band: float(directivity * distance_attenuation * air[i] * (occlusion + (1.0 - occlusion) * transmission[i]))
        for i, band in enumerate(FREQUENCY_BANDS)
    }
    return {
        "distance_m": distance,
        "delay_s": delay,
        "distance_attenuation": distance_attenuation,
        "air_absorption": air,
        "occlusion": occlusion,
        "transmission": transmission,
        "occlusion_surface": occlusion_surface,
        "source_directivity_gain": directivity,
        "band_gains": band_gains,
    }


def scan_visual_rt_paths(room: Room, scene: RoomRayScene, src: np.ndarray, rcv: np.ndarray, config: SimConfig) -> dict[str, Any]:
    ray_count = int(config.rt_visual_num_rays if config.rt_visual_num_rays is not None else config.rt_num_rays)
    max_bounces = int(config.rt_visual_num_bounces if config.rt_visual_num_bounces is not None else config.rt_num_bounces)
    directions = _sphere_samples(ray_count, int(config.seed))
    if njit is not None:
        return _scan_visual_rt_paths_numba(scene, src, rcv, config, directions, ray_count, max_bounces)
    return _scan_visual_rt_paths_python(room, scene, src, rcv, config, directions, ray_count, max_bounces)


def _scan_visual_rt_paths_python(
    room: Room,
    scene: RoomRayScene,
    src: np.ndarray,
    rcv: np.ndarray,
    config: SimConfig,
    directions: np.ndarray,
    ray_count: int,
    max_bounces: int,
) -> dict[str, Any]:
    receiver_radius = float(config.rt_receiver_radius_m)
    retained: list[AcousticPath] = []
    accepted = 0
    hit_rays: set[int] = set()
    bounce_histogram: dict[int, int] = {}
    for ray_index, initial in enumerate(directions):
        origin = src.copy()
        direction = np.asarray(initial, dtype=float)
        distance_so_far = 0.0
        survival = 1.0
        hit_points: list[np.ndarray] = []
        hit_surfaces: list[str] = []
        for bounce in range(max_bounces + 1):
            hit = scene.closest_hit(origin, direction)
            max_t = float(hit["distance"]) if hit["valid"] else np.inf
            receiver_t = _ray_sphere_intersection(origin, direction, rcv, receiver_radius)
            if receiver_t is not None and receiver_t < max_t and bounce > 0:
                accepted += 1
                hit_rays.add(ray_index)
                bounce_histogram[bounce] = bounce_histogram.get(bounce, 0) + 1
                if len(retained) < _RT_VISUAL_RETAIN_LIMIT:
                    path_distance = float(distance_so_far + receiver_t)
                    gain = float(survival / max(path_distance, config.min_distance_m))
                    bands = {band: gain for band in FREQUENCY_BANDS}
                    retained.append(
                        AcousticPath(
                            "rt_reflection",
                            path_distance,
                            path_distance / float(config.c),
                            gain,
                            bands,
                            tuple(tuple(float(v) for v in point) for point in (src, *hit_points, rcv)),
                            {
                                "model": "source_space_specular_ray_scan_diagnostic",
                                "ray_index": int(ray_index),
                                "order": int(bounce),
                                "surfaces": list(hit_surfaces),
                                "contributes_to_rir": False,
                            },
                        )
                    )
            if not hit["valid"]:
                break
            energy_survival = float(np.mean(np.sqrt(np.clip(1.0 - hit["absorption"], 0.0, 1.0))))
            if energy_survival < 1e-5:
                break
            survival *= energy_survival
            hit_points.append(np.asarray(hit["point"], dtype=float).copy())
            hit_surfaces.append(str(hit["surface"]))
            normal = np.asarray(hit["normal"], dtype=float)
            direction = direction - 2.0 * float(np.dot(direction, normal)) * normal
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            distance_so_far += float(hit["distance"])
            if distance_so_far > float(config.rt_duration_s) * float(config.c):
                break
            origin = np.asarray(hit["point"], dtype=float) + 1e-5 * direction
    return {
        "paths": retained,
        "metadata": {
            "enabled": True,
            "model": "source_space_specular_ray_scan_diagnostic",
            "ray_count": ray_count,
            "max_bounces": max_bounces,
            "follows_simulation": config.rt_visual_num_rays is None and config.rt_visual_num_bounces is None,
            "receiver_radius_m": receiver_radius,
            "accepted_event_count": accepted,
            "receiver_hit_ray_count": len(hit_rays),
            "retain_limit": _RT_VISUAL_RETAIN_LIMIT,
            "retained_path_count": len(retained),
            "retention_policy": "first_accepted_python_fallback",
            "bounce_count_histogram": {str(key): int(value) for key, value in sorted(bounce_histogram.items())},
        },
    }


def _scan_visual_rt_paths_numba(
    scene: RoomRayScene,
    src: np.ndarray,
    rcv: np.ndarray,
    config: SimConfig,
    directions: np.ndarray,
    ray_count: int,
    max_bounces: int,
) -> dict[str, Any]:
    arrays = _scene_kernel_arrays(scene)
    use_bvh = _resolve_intersection_backend(
        config.intersection_backend,
        len(scene.surfaces),
        config.bvh_min_surfaces,
    ) == "bvh"
    candidate_limit = max(int(_RT_VISUAL_RETAIN_LIMIT), int(_RT_VISUAL_RETAIN_LIMIT * _RT_VISUAL_CANDIDATE_FACTOR))
    src_array = np.asarray(src, dtype=np.float64)
    rcv_array = np.asarray(rcv, dtype=np.float64)
    direction_array = np.asarray(directions, dtype=np.float64)
    surface_survival = _surface_survival_kernel(arrays["absorption"])
    max_path_len = float(config.rt_duration_s) * float(config.c)
    event_flags = _visual_event_count_kernel(
        src_array,
        rcv_array,
        direction_array,
        arrays["kinds"],
        arrays["wall_a"],
        arrays["wall_delta"],
        arrays["wall_z"],
        arrays["z_values"],
        arrays["box_center"],
        arrays["box_axis_u"],
        arrays["box_axis_v"],
        arrays["box_half"],
        arrays["box_z"],
        arrays["normals"],
        arrays["bvh_bounds_min"],
        arrays["bvh_bounds_max"],
        arrays["bvh_start"],
        arrays["bvh_count"],
        arrays["bvh_escape"],
        arrays["bvh_primitives"],
        bool(use_bvh),
        surface_survival,
        arrays["corners"],
        float(arrays["height"]),
        float(config.rt_receiver_radius_m),
        max_path_len,
        int(max_bounces),
    )
    events_per_ray = np.sum(event_flags, axis=1, dtype=np.int64)
    event_offsets = np.zeros(events_per_ray.shape[0], dtype=np.int64)
    if events_per_ray.shape[0] > 1:
        np.cumsum(events_per_ray[:-1], out=event_offsets[1:])
    accepted = int(np.sum(events_per_ray))
    hit_count = int(np.count_nonzero(events_per_ray))
    bounce_hist = np.sum(event_flags, axis=0, dtype=np.int64)
    retained_points, point_counts, ray_indices, orders, distances, gains, surface_indices = _visual_record_kernel(
        src_array,
        rcv_array,
        direction_array,
        arrays["kinds"],
        arrays["wall_a"],
        arrays["wall_delta"],
        arrays["wall_z"],
        arrays["z_values"],
        arrays["box_center"],
        arrays["box_axis_u"],
        arrays["box_axis_v"],
        arrays["box_half"],
        arrays["box_z"],
        arrays["normals"],
        arrays["bvh_bounds_min"],
        arrays["bvh_bounds_max"],
        arrays["bvh_start"],
        arrays["bvh_count"],
        arrays["bvh_escape"],
        arrays["bvh_primitives"],
        bool(use_bvh),
        surface_survival,
        arrays["corners"],
        float(arrays["height"]),
        float(config.rt_receiver_radius_m),
        float(config.min_distance_m),
        max_path_len,
        int(max_bounces),
        events_per_ray,
        event_offsets,
        int(candidate_limit),
    )
    names = arrays["names"]
    paths: list[AcousticPath] = []
    candidate_indices = [int(index) for index in np.flatnonzero(point_counts)]
    retained_indices = _select_visual_candidate_indices(candidate_indices, orders, gains, distances, int(_RT_VISUAL_RETAIN_LIMIT))
    for index in retained_indices:
        count = int(point_counts[index])
        points = tuple(tuple(float(v) for v in retained_points[index, pi]) for pi in range(count))
        surf_count = max(0, count - 2)
        surfaces = [names[int(surface_indices[index, si])] for si in range(surf_count) if int(surface_indices[index, si]) >= 0]
        gain = float(gains[index])
        bands = {band: gain for band in FREQUENCY_BANDS}
        paths.append(
            AcousticPath(
                "rt_reflection",
                float(distances[index]),
                float(distances[index]) / float(config.c),
                gain,
                bands,
                points,
                {
                    "model": "source_space_specular_ray_scan_jit",
                    "ray_index": int(ray_indices[index]),
                    "order": int(orders[index]),
                    "surfaces": surfaces,
                    "contributes_to_rir": False,
                },
            )
        )
    return {
        "paths": paths,
        "metadata": {
            "enabled": True,
            "model": "source_space_specular_ray_scan_jit",
            "accelerator": "numba",
            "ray_count": int(ray_count),
            "max_bounces": int(max_bounces),
            "follows_simulation": config.rt_visual_num_rays is None and config.rt_visual_num_bounces is None,
            "receiver_radius_m": float(config.rt_receiver_radius_m),
            "accepted_event_count": int(accepted),
            "receiver_hit_ray_count": int(hit_count),
            "retain_limit": _RT_VISUAL_RETAIN_LIMIT,
            "retained_path_count": len(paths),
            "candidate_limit": int(candidate_limit),
            "candidate_path_count": len(candidate_indices),
            "retention_policy": "stratified_order_then_strongest_gain",
            "bounce_count_histogram": {str(index): int(value) for index, value in enumerate(bounce_hist) if int(value) > 0},
        },
    }


def _select_visual_candidate_indices(candidate_indices: list[int], orders: np.ndarray, gains: np.ndarray, distances: np.ndarray, retain_limit: int) -> list[int]:
    if len(candidate_indices) <= retain_limit:
        return sorted(candidate_indices, key=lambda index: (int(orders[index]), float(distances[index])))
    groups: dict[int, list[int]] = {}
    for index in candidate_indices:
        order = int(orders[index])
        groups.setdefault(order, []).append(index)
    for group in groups.values():
        group.sort(key=lambda index: (-float(gains[index]), float(distances[index])))

    active_orders = sorted(groups)
    raw_quotas: dict[int, int] = {}
    for order in active_orders:
        if order <= 2:
            quota = 48
        elif order <= 8:
            quota = 32
        else:
            quota = 12
        raw_quotas[order] = min(quota, len(groups[order]))
    raw_total = sum(raw_quotas.values())
    if raw_total > retain_limit:
        scale = retain_limit / max(raw_total, 1)
        quotas = {order: max(1, min(len(groups[order]), int(math.floor(raw_quotas[order] * scale)))) for order in active_orders}
    else:
        quotas = dict(raw_quotas)

    selected: list[int] = []
    cursors: dict[int, int] = {}
    for order in active_orders:
        take = min(quotas[order], len(groups[order]), retain_limit - len(selected))
        selected.extend(groups[order][:take])
        cursors[order] = take
        if len(selected) >= retain_limit:
            break

    while len(selected) < retain_limit:
        added = False
        for order in active_orders:
            cursor = cursors.get(order, 0)
            if cursor < len(groups[order]):
                selected.append(groups[order][cursor])
                cursors[order] = cursor + 1
                added = True
                if len(selected) >= retain_limit:
                    break
        if not added:
            break
    return sorted(selected[:retain_limit], key=lambda index: (int(orders[index]), float(distances[index])))


def _visual_rt_paths_from_energy_field(
    field: Mapping[str, Any],
    source: np.ndarray,
    listener: np.ndarray,
    config: SimConfig,
) -> dict[str, Any]:
    candidates = field.get("visual_candidates")
    if not isinstance(candidates, Mapping):
        return {
            "paths": [],
            "metadata": {
                "enabled": False,
                "model": "energy_trace_representatives_unavailable",
                "ray_count": int(config.rt_num_rays),
                "max_bounces": int(field.get("actual_bounces", config.rt_num_bounces)),
                "follows_simulation": True,
                "retain_limit": int(_RT_VISUAL_RETAIN_LIMIT),
                "retained_path_count": 0,
            },
        }

    hit_points = np.asarray(candidates["hit_points"], dtype=np.float64)
    surface_indices = np.asarray(candidates["surface_indices"], dtype=np.int64)
    ray_indices = np.asarray(candidates["ray_indices"], dtype=np.int64)
    orders = np.asarray(candidates["orders"], dtype=np.int64)
    distances = np.asarray(candidates["distances"], dtype=np.float64)
    gains = np.asarray(candidates["gains"], dtype=np.float64)
    surface_names = tuple(str(value) for value in candidates.get("surface_names", ()))
    energy_trace_bounces = int(field.get("actual_bounces", config.rt_num_bounces))
    visual_max_bounces = (
        energy_trace_bounces
        if config.rt_visual_num_bounces is None
        else min(energy_trace_bounces, max(1, int(config.rt_visual_num_bounces)))
    )
    candidate_indices = [
        int(index)
        for index in np.flatnonzero(
            (orders > 0)
            & (orders <= visual_max_bounces)
            & np.isfinite(gains)
            & (gains > 0.0)
        )
    ]
    retained_indices = _select_visual_candidate_indices(
        candidate_indices,
        orders,
        gains,
        distances,
        int(_RT_VISUAL_RETAIN_LIMIT),
    )
    paths: list[AcousticPath] = []
    for index in retained_indices:
        order = int(orders[index])
        points: list[tuple[float, float, float]] = [tuple(float(value) for value in source)]
        surfaces: list[str] = []
        for bounce in range(order - 1, -1, -1):
            points.append(tuple(float(value) for value in hit_points[index, bounce]))
            surface_index = int(surface_indices[index, bounce])
            if 0 <= surface_index < len(surface_names):
                surfaces.append(surface_names[surface_index])
        points.append(tuple(float(value) for value in listener))
        gain = float(gains[index])
        bands = {band: gain for band in FREQUENCY_BANDS}
        paths.append(
            AcousticPath(
                "rt_reflection",
                float(distances[index]),
                float(distances[index]) / float(config.c),
                gain,
                bands,
                tuple(points),
                {
                    "model": "listener_space_energy_trace_representative",
                    "ray_index": int(ray_indices[index]),
                    "order": order,
                    "surfaces": surfaces,
                    "contributes_to_rir": True,
                },
            )
        )

    bounce_histogram: dict[str, int] = {}
    for index in candidate_indices:
        key = str(int(orders[index]))
        bounce_histogram[key] = bounce_histogram.get(key, 0) + 1
    contribution_count = sum(int(value) for value in field.get("surface_contribution_count", {}).values())
    return {
        "paths": paths,
        "metadata": {
            "enabled": True,
            "model": "listener_space_energy_trace_representatives",
            "accelerator": field.get("accelerator", "numpy"),
            "ray_count": int(config.rt_num_rays),
            "max_bounces": int(visual_max_bounces),
            "energy_trace_max_bounces": int(energy_trace_bounces),
            "follows_simulation": True,
            "receiver_radius_m": float(config.rt_receiver_radius_m),
            "accepted_event_count": int(contribution_count),
            "receiver_hit_ray_count": len(candidate_indices),
            "representative_ray_count": len(candidate_indices),
            "retain_limit": int(_RT_VISUAL_RETAIN_LIMIT),
            "retained_path_count": len(paths),
            "candidate_limit": int(candidates.get("candidate_limit", len(orders))),
            "candidate_path_count": len(candidate_indices),
            "candidate_stride": int(candidates.get("stride", 1)),
            "retention_policy": "stratified_order_then_strongest_gain",
            "bounce_count_histogram": dict(sorted(bounce_histogram.items(), key=lambda item: int(item[0]))),
            "shares_energy_trace": True,
        },
    }


def trace_energy_field(
    scene: RoomRayScene,
    source: np.ndarray,
    listener: np.ndarray,
    config: SimConfig,
    source_model: str | Mapping[str, Any] | None = None,
    *,
    render_ambisonics: bool = True,
) -> dict[str, Any]:
    emitter = source_directivity(source_model)
    accelerator = _normalize_rt_accelerator(config.rt_accelerator)
    precision = _normalize_rt_precision(config.rt_precision)
    if accelerator in {"cuda", "auto"}:
        from .steam_rt_cuda import cuda_available

        available = cuda_available(int(config.rt_cuda_device))
        if accelerator == "cuda" and not available:
            raise RuntimeError(
                f"CUDA accelerator requested for device {config.rt_cuda_device}, but no compatible CUDA device is available"
            )
        if accelerator == "cuda" and precision != "float32":
            raise ValueError("CUDA tracing currently supports rt_precision='float32' only")
        if available and precision == "float32":
            return _trace_energy_field_cuda(
                scene,
                source,
                listener,
                config,
                emitter,
                render_ambisonics=render_ambisonics,
            )
    if njit is not None:
        return _trace_energy_field_numba(
            scene,
            source,
            listener,
            config,
            emitter,
            render_ambisonics=render_ambisonics,
        )
    return _trace_energy_field_numpy(
        scene,
        source,
        listener,
        config,
        emitter,
        render_ambisonics=render_ambisonics,
    )


def _normalize_rt_accelerator(value: str) -> str:
    accelerator = str(value).strip().lower()
    if accelerator not in {"auto", "numba", "cuda"}:
        raise ValueError("rt_accelerator must be auto, numba, or cuda")
    return accelerator


def _normalize_rt_precision(value: str) -> str:
    aliases = {"fp32": "float32", "single": "float32", "fp64": "float64", "double": "float64"}
    precision = aliases.get(str(value).strip().lower(), str(value).strip().lower())
    if precision not in {"float32", "float64"}:
        raise ValueError("rt_precision must be float32 or float64")
    return precision


def _trace_energy_field_cuda(
    scene: RoomRayScene,
    source: np.ndarray,
    listener: np.ndarray,
    config: SimConfig,
    source_model: Mapping[str, Any] | None = None,
    *,
    render_ambisonics: bool = True,
) -> dict[str, Any]:
    from .steam_rt_cuda import trace_energy_field_cuda

    emitter = source_directivity(source_model)
    arrays = _scene_kernel_arrays(scene)
    intersection_backend = _resolve_intersection_backend(
        config.intersection_backend,
        len(scene.surfaces),
        config.bvh_min_surfaces,
    )
    num_rays = int(config.rt_num_rays)
    num_bounces = int(config.rt_num_bounces)
    bin_dur = float(config.rt_bin_duration_s)
    duration = max(float(config.rt_duration_s), float(config.duration_s))
    num_bins = max(1, int(math.ceil(duration / bin_dur)))
    directions = _sphere_samples(num_rays, int(config.seed))
    diffuse_bank = _diffuse_sample_bank(config.rt_num_diffuse_samples)
    diffuse_random, diffuse_indices = _diffuse_random_sequence(
        num_rays,
        num_bounces,
        diffuse_bank.shape[0],
        int(config.seed),
    )
    default_visual_candidates = int(_RT_VISUAL_RETAIN_LIMIT * 4)
    requested_visual_candidates = (
        default_visual_candidates
        if config.rt_visual_num_rays is None
        else max(1, min(int(config.rt_visual_num_rays), default_visual_candidates))
    )
    visual_candidate_limit = min(num_rays, requested_visual_candidates) if config.collect_visual_paths else 0
    visual_stride = max(1, num_rays // max(visual_candidate_limit, 1))
    direct_delay = float(np.linalg.norm(np.asarray(source, dtype=float) - np.asarray(listener, dtype=float))) / float(config.c)
    raw = trace_energy_field_cuda(
        source=np.asarray(source, dtype=np.float32),
        listener=np.asarray(listener, dtype=np.float32),
        directions=directions,
        diffuse_bank=diffuse_bank,
        diffuse_random=diffuse_random,
        diffuse_indices=diffuse_indices,
        arrays=arrays,
        use_bvh=intersection_backend == "bvh",
        num_bounces=num_bounces,
        num_bins=num_bins,
        bin_dur=bin_dur,
        speed_of_sound=float(config.c),
        max_path_len=duration * float(config.c),
        direct_delay=direct_delay,
        listener_radius=float(config.rt_listener_radius),
        source_radius=float(config.rt_source_radius),
        irradiance_min_distance=float(config.rt_irradiance_min_distance),
        specular_exponent=float(config.rt_specular_exponent),
        source_forward_vector=np.asarray(source_forward(emitter), dtype=np.float32),
        dipole_weight=float(emitter["dipole_weight"]),
        dipole_power=float(emitter["dipole_power"]),
        visual_candidate_limit=visual_candidate_limit,
        visual_stride=visual_stride,
        render_ambisonics=render_ambisonics,
        device_id=int(config.rt_cuda_device),
    )
    names = arrays["names"]
    total_energy = np.sum(raw["echogram"], axis=0)
    nonzero_bins = np.flatnonzero(total_energy > 0.0)
    return {
        "echogram": raw["echogram"],
        "ambisonic_echogram": raw["ambisonic"] if render_ambisonics else None,
        "num_bins": num_bins,
        "bin_duration_s": bin_dur,
        "direct_delay_s": direct_delay,
        "actual_bounces": int(raw["actual_bounces"]),
        "active_ray_count": int(raw["active_count"]),
        "last_energy_time_s": float(nonzero_bins[-1] * bin_dur) if nonzero_bins.size else 0.0,
        "surface_hit_count": {names[i]: int(raw["hit_counts"][i]) for i in range(len(names)) if int(raw["hit_counts"][i]) > 0},
        "surface_contribution_count": {names[i]: int(raw["contrib_counts"][i]) for i in range(len(names)) if int(raw["contrib_counts"][i]) > 0},
        "surface_energy": {names[i]: float(raw["surface_energy"][i]) for i in range(len(names)) if float(raw["surface_energy"][i]) > 0.0},
        "visual_candidates": {
            "hit_points": raw["visual_hit_points"],
            "surface_indices": raw["visual_surface_indices"],
            "ray_indices": raw["visual_ray_indices"],
            "orders": raw["visual_orders"],
            "distances": raw["visual_distances"],
            "gains": raw["visual_gains"],
            "surface_names": tuple(names),
            "candidate_limit": int(visual_candidate_limit),
            "stride": int(visual_stride),
        } if config.collect_visual_paths else None,
        "accelerator": "cuda",
        "precision": "float32",
        "cuda": raw["device"],
        "device_input_cache": raw["device_input_cache"],
        "kernel_time_s": float(raw["kernel_time_s"]),
        "transfer_time_s": float(raw["transfer_time_s"]),
        "intersection_backend": intersection_backend,
        "source_directivity": dict(emitter),
    }


def _trace_energy_field_numba(
    scene: RoomRayScene,
    source: np.ndarray,
    listener: np.ndarray,
    config: SimConfig,
    source_model: Mapping[str, Any] | None = None,
    *,
    render_ambisonics: bool = True,
) -> dict[str, Any]:
    emitter = source_directivity(source_model)
    precision = _normalize_rt_precision(config.rt_precision)
    compute_dtype = np.float32 if precision == "float32" else np.float64
    arrays = _scene_kernel_arrays(scene)
    intersection_backend = _resolve_intersection_backend(
        config.intersection_backend,
        len(scene.surfaces),
        config.bvh_min_surfaces,
    )
    use_bvh = intersection_backend == "bvh"
    num_rays = int(config.rt_num_rays)
    num_bounces = int(config.rt_num_bounces)
    bin_dur = float(config.rt_bin_duration_s)
    duration = max(float(config.rt_duration_s), float(config.duration_s))
    num_bins = max(1, int(math.ceil(duration / bin_dur)))
    directions = _sphere_samples(num_rays, int(config.seed))
    diffuse_bank = _diffuse_sample_bank(config.rt_num_diffuse_samples)
    diffuse_random, diffuse_indices = _diffuse_random_sequence(
        num_rays,
        num_bounces,
        diffuse_bank.shape[0],
        int(config.seed),
    )
    direct_delay = float(np.linalg.norm(np.asarray(source, dtype=float) - np.asarray(listener, dtype=float))) / float(config.c)
    default_visual_candidates = int(_RT_VISUAL_RETAIN_LIMIT * 4)
    requested_visual_candidates = (
        default_visual_candidates
        if config.rt_visual_num_rays is None
        else max(1, min(int(config.rt_visual_num_rays), default_visual_candidates))
    )
    visual_candidate_limit = min(num_rays, requested_visual_candidates) if config.collect_visual_paths else 0
    visual_stride = max(1, num_rays // max(visual_candidate_limit, 1))
    (
        echogram,
        ambisonic,
        hit_counts,
        contrib_counts,
        surface_energy,
        actual_bounces,
        active_count,
        visual_hit_points,
        visual_surface_indices,
        visual_ray_indices,
        visual_orders,
        visual_distances,
        visual_gains,
    ) = _trace_energy_kernel(
        np.asarray(source, dtype=compute_dtype),
        np.asarray(listener, dtype=compute_dtype),
        _as_precision_array(directions, compute_dtype),
        _as_precision_array(diffuse_bank, compute_dtype),
        _as_precision_array(diffuse_random, compute_dtype),
        diffuse_indices,
        arrays["kinds"],
        _as_precision_array(arrays["wall_a"], compute_dtype),
        _as_precision_array(arrays["wall_delta"], compute_dtype),
        _as_precision_array(arrays["wall_z"], compute_dtype),
        _as_precision_array(arrays["z_values"], compute_dtype),
        _as_precision_array(arrays["box_center"], compute_dtype),
        _as_precision_array(arrays["box_axis_u"], compute_dtype),
        _as_precision_array(arrays["box_axis_v"], compute_dtype),
        _as_precision_array(arrays["box_half"], compute_dtype),
        _as_precision_array(arrays["box_z"], compute_dtype),
        _as_precision_array(arrays["normals"], compute_dtype),
        _as_precision_array(arrays["bvh_bounds_min"], compute_dtype),
        _as_precision_array(arrays["bvh_bounds_max"], compute_dtype),
        arrays["bvh_start"],
        arrays["bvh_count"],
        arrays["bvh_escape"],
        arrays["bvh_primitives"],
        bool(use_bvh),
        _as_precision_array(arrays["reflection"], compute_dtype),
        _as_precision_array(arrays["scattering"], compute_dtype),
        _as_precision_array(arrays["corners"], compute_dtype),
        compute_dtype(arrays["height"]),
        int(num_bounces),
        int(num_bins),
        compute_dtype(bin_dur),
        compute_dtype(config.c),
        compute_dtype(duration * float(config.c)),
        compute_dtype(direct_delay),
        compute_dtype(config.rt_listener_radius),
        compute_dtype(config.rt_source_radius),
        compute_dtype(config.rt_irradiance_min_distance),
        compute_dtype(config.rt_specular_exponent),
        np.asarray(source_forward(emitter), dtype=compute_dtype),
        compute_dtype(emitter["dipole_weight"]),
        compute_dtype(emitter["dipole_power"]),
        int(visual_candidate_limit),
        int(visual_stride),
        bool(render_ambisonics),
    )
    names = arrays["names"]
    total_energy = np.sum(echogram, axis=0)
    nonzero_bins = np.flatnonzero(total_energy > 0.0)
    return {
        "echogram": echogram,
        "ambisonic_echogram": ambisonic if render_ambisonics else None,
        "num_bins": num_bins,
        "bin_duration_s": bin_dur,
        "direct_delay_s": direct_delay,
        "actual_bounces": int(actual_bounces),
        "active_ray_count": int(active_count),
        "last_energy_time_s": float(nonzero_bins[-1] * bin_dur) if nonzero_bins.size else 0.0,
        "surface_hit_count": {names[i]: int(hit_counts[i]) for i in range(len(names)) if int(hit_counts[i]) > 0},
        "surface_contribution_count": {names[i]: int(contrib_counts[i]) for i in range(len(names)) if int(contrib_counts[i]) > 0},
        "surface_energy": {names[i]: float(surface_energy[i]) for i in range(len(names)) if float(surface_energy[i]) > 0.0},
        "visual_candidates": {
            "hit_points": visual_hit_points,
            "surface_indices": visual_surface_indices,
            "ray_indices": visual_ray_indices,
            "orders": visual_orders,
            "distances": visual_distances,
            "gains": visual_gains,
            "surface_names": tuple(names),
            "candidate_limit": int(visual_candidate_limit),
            "stride": int(visual_stride),
        } if config.collect_visual_paths else None,
        "accelerator": "numba",
        "precision": precision,
        "intersection_backend": intersection_backend,
        "source_directivity": dict(emitter),
    }


def _trace_energy_field_numpy(
    scene: RoomRayScene,
    source: np.ndarray,
    listener: np.ndarray,
    config: SimConfig,
    source_model: Mapping[str, Any] | None = None,
    *,
    render_ambisonics: bool = True,
) -> dict[str, Any]:
    emitter = source_directivity(source_model)
    c = config.c
    num_rays = int(config.rt_num_rays)
    num_bounces = int(config.rt_num_bounces)
    bin_dur = float(config.rt_bin_duration_s)
    duration = max(float(config.rt_duration_s), float(config.duration_s))
    num_bins = max(1, int(math.ceil(duration / bin_dur)))
    max_path_len = duration * c
    direct_delay = float(np.linalg.norm(source - listener)) / c
    echogram = np.zeros((_NUM_BANDS, num_bins), dtype=np.float64)
    ambisonic_echogram = np.zeros((_NUM_BANDS, 4, num_bins), dtype=np.float64) if render_ambisonics else None
    origins = np.tile(listener, (num_rays, 1)).astype(float)
    dirs = _sphere_samples(num_rays, config.seed)
    listener_dirs = dirs.copy()
    accum_energy = np.ones((num_rays, _NUM_BANDS), dtype=np.float64)
    accum_distance = np.zeros(num_rays, dtype=np.float64)
    alive = np.ones(num_rays, dtype=bool)
    rng = np.random.default_rng(config.seed + 1)
    diffuse_bank = _diffuse_sample_bank(config.rt_num_diffuse_samples)
    emitter_forward = source_forward(emitter)
    dipole_weight = float(emitter["dipole_weight"])
    dipole_power = float(emitter["dipole_power"])
    surface_hit_count: dict[str, int] = {}
    surface_contribution_count: dict[str, int] = {}
    surface_energy: dict[str, float] = {}
    actual_bounces = 0

    for bounce in range(num_bounces):
        actual_bounces = bounce + 1
        if not np.any(alive):
            break
        hit = scene.batch_closest_hit(origins, dirs)
        t = hit["t"]
        alive = alive & hit["valid"] & (t > float(config.rt_listener_radius)) & (accum_distance <= max_path_len)
        if bounce > 0 and np.any(alive):
            intercepted = _ray_hits_sphere_before(origins, dirs, listener, float(config.rt_listener_radius), t)
            if float(np.linalg.norm(listener - source)) > float(config.rt_source_radius):
                intercepted |= _ray_hits_sphere_before(origins, dirs, source, float(config.rt_source_radius), t)
            alive &= ~intercepted
        if not np.any(alive):
            break
        normal = hit["normal"]
        absorption = hit["absorption"]
        scatter = hit["scattering"]
        surfaces = hit.get("surface")
        hit_point = hit["point"] + _HIT_OFFSET * normal
        if surfaces is not None:
            for surface, count in zip(*np.unique(surfaces[alive], return_counts=True)):
                key = str(surface)
                surface_hit_count[key] = surface_hit_count.get(key, 0) + int(count)
        to_source = source[None, :] - hit_point
        dist_to_source = np.linalg.norm(to_source, axis=1)
        facing = (np.sum(normal * to_source, axis=1) > 0.0) & (dist_to_source > float(config.rt_irradiance_min_distance))
        shadow_dir = to_source / np.maximum(dist_to_source, 1e-9)[:, None]
        shade_mask = alive & facing
        if np.any(shade_mask):
            occluded = np.zeros(num_rays, dtype=bool)
            occluded[shade_mask] = scene.batch_any_hit(hit_point[shade_mask], shadow_dir[shade_mask], dist_to_source[shade_mask])
            contribute = shade_mask & ~occluded
            if np.any(contribute):
                cos_in = np.clip(np.sum(normal * shadow_dir, axis=1), 0.0, None)
                diffuse = (1.0 / math.pi) * scatter * cos_in
                half = _normalize_rows(shadow_dir - dirs)
                cos_half = np.clip(np.sum(half * normal, axis=1), 0.0, None)
                specular = ((float(config.rt_specular_exponent) + 2.0) / (8.0 * math.pi)) * (1.0 - scatter) * (cos_half ** float(config.rt_specular_exponent))
                distance_term = (1.0 / (4.0 * math.pi)) * (1.0 / np.maximum(dist_to_source, float(config.rt_irradiance_min_distance))) ** 2
                source_cosine = np.clip((-shadow_dir) @ emitter_forward, -1.0, 1.0)
                source_gain = np.abs((1.0 - dipole_weight) + dipole_weight * source_cosine) ** dipole_power
                energy = ((4.0 * math.pi) / max(num_rays, 1)) * source_gain[:, None] * distance_term[:, None] * (diffuse + specular)[:, None] * (1.0 - absorption) * accum_energy
                rel_delay = np.where(
                    contribute,
                    (accum_distance + t + dist_to_source) / c - direct_delay,
                    -1.0,
                )
                rel_delay = np.where(np.isfinite(rel_delay), rel_delay, -1.0)
                bin_index = np.floor(rel_delay / bin_dur).astype(int)
                valid = contribute & (bin_index >= 0) & (bin_index < num_bins)
                if np.any(valid):
                    bins_v = bin_index[valid]
                    energy_v = energy[valid]
                    coeffs_v = _foa_coefficients(listener_dirs[valid]) if render_ambisonics else None
                    if surfaces is not None:
                        surface_v = surfaces[valid]
                        energy_sum_v = np.sum(energy_v, axis=1)
                        for surface in np.unique(surface_v):
                            key = str(surface)
                            mask = surface_v == surface
                            surface_contribution_count[key] = surface_contribution_count.get(key, 0) + int(np.count_nonzero(mask))
                            surface_energy[key] = surface_energy.get(key, 0.0) + float(np.sum(energy_sum_v[mask]))
                    for bi in range(_NUM_BANDS):
                        np.add.at(echogram[bi], bins_v, energy_v[:, bi])
                        if render_ambisonics:
                            for ci in range(4):
                                np.add.at(ambisonic_echogram[bi, ci], bins_v, energy_v[:, bi] * coeffs_v[:, ci])
        accum_energy = np.where(alive[:, None], accum_energy * (1.0 - absorption), accum_energy)
        accum_distance = np.where(alive, accum_distance + t, accum_distance)
        origins = np.where(alive[:, None], hit_point, origins)
        diffuse_pick = rng.random(num_rays) < scatter
        sample_indices = rng.integers(0, diffuse_bank.shape[0], size=num_rays)
        diffuse_dir = _transform_hemisphere(diffuse_bank[sample_indices], normal)
        spec_dir = _normalize_rows(dirs - 2.0 * np.sum(dirs * normal, axis=1)[:, None] * normal)
        dirs = np.where(alive[:, None], np.where(diffuse_pick[:, None], diffuse_dir, spec_dir), dirs)

    nonzero_bins = np.flatnonzero(np.sum(echogram, axis=0) > 0.0)
    return {
        "echogram": echogram,
        "ambisonic_echogram": ambisonic_echogram,
        "num_bins": num_bins,
        "bin_duration_s": bin_dur,
        "direct_delay_s": direct_delay,
        "actual_bounces": actual_bounces,
        "active_ray_count": int(np.count_nonzero(alive)),
        "last_energy_time_s": float(nonzero_bins[-1] * bin_dur) if nonzero_bins.size else 0.0,
        "surface_hit_count": dict(sorted(surface_hit_count.items())),
        "surface_contribution_count": dict(sorted(surface_contribution_count.items())),
        "surface_energy": {key: float(value) for key, value in sorted(surface_energy.items())},
        "accelerator": "numpy",
        "intersection_backend": scene._intersection_backend,
        "source_directivity": dict(emitter),
    }


def _air_absorption_amplitude(coefficient_per_m: float, travel_time_s: float, speed_m_s: float) -> float:
    distance_m = max(0.0, float(travel_time_s)) * max(0.0, float(speed_m_s))
    return math.exp(-max(0.0, float(coefficient_per_m)) * distance_m)


def _air_absorption_energy_weights(
    coefficient_per_m: float,
    travel_times_s: np.ndarray,
    speed_m_s: float,
) -> np.ndarray:
    distances_m = np.maximum(np.asarray(travel_times_s, dtype=np.float64), 0.0) * max(0.0, float(speed_m_s))
    return np.exp(-2.0 * max(0.0, float(coefficient_per_m)) * distances_m)


def reconstruct_band_irs(field: dict[str, Any], config: SimConfig) -> np.ndarray:
    echogram = field["echogram"]
    samples_per_bin = max(1, int(math.ceil(float(field["bin_duration_s"]) * int(config.fs))))
    num_samples = int(field["num_bins"]) * samples_per_bin
    rng = np.random.default_rng(config.seed + 7)
    white = rng.uniform(-1.0, 1.0, size=num_samples).astype(np.float64)
    raw_band_irs = np.zeros((_NUM_BANDS, num_samples), dtype=np.float64)
    sample_weights = np.arange(samples_per_bin, dtype=np.float64) / samples_per_bin
    direct_delay_s = max(0.0, float(field.get("direct_delay_s", 0.0)))
    for band_index, band in enumerate(FREQUENCY_BANDS):
        coeff = AIR_ABSORPTION_NP_PER_M[band]
        # The local echogram stores the unnormalized ray energy. Steam Audio's
        # order-0 EnergyField coefficient is Y00 * E, so its reconstruction
        # formula reduces to sqrt(E / (4*pi)), not sqrt(E / sqrt(4*pi)).
        amps = np.sqrt(np.clip(echogram[band_index], 0.0, None) / (4.0 * math.pi))
        amps[echogram[band_index] < _ENERGY_THRESHOLD] = 0.0
        sample_amp = np.zeros(num_samples, dtype=np.float64)
        for b in range(int(field["num_bins"])):
            lo = b * samples_per_bin
            hi = min(num_samples, lo + samples_per_bin)
            prev = amps[b] if b == 0 else amps[b - 1]
            w = sample_weights[:hi - lo]
            seg = (1.0 - w) * prev + w * amps[b]
            path_time_s = direct_delay_s + (b + 0.5) * samples_per_bin / int(config.fs)
            seg *= _air_absorption_amplitude(coeff, path_time_s, float(config.c))
            sample_amp[lo:hi] = seg
        raw_band_irs[band_index] = sample_amp * white
    return bandlimit_band_signals(raw_band_irs, int(config.fs))


def _fdn_impulse_kernel(
    delays: np.ndarray,
    absorptive_coeffs: np.ndarray,
    num_samples: int,
    allpass_delays: np.ndarray,
    tone_coeffs: np.ndarray,
) -> np.ndarray:
    """Render Steam Audio's multiband 16-line Hadamard FDN topology."""
    num_delays = int(delays.size)
    num_bands = int(absorptive_coeffs.shape[1])
    max_delay = int(np.max(delays)) + 1
    delay_buffers = np.zeros((num_delays, max_delay), dtype=np.float64)
    delay_positions = np.zeros(num_delays, dtype=np.int64)
    delayed = np.zeros(num_delays, dtype=np.float64)
    mixed = np.zeros(num_delays, dtype=np.float64)
    output = np.zeros(num_samples, dtype=np.float64)
    absorptive_state = np.zeros((num_delays, num_bands, 4), dtype=np.float64)
    tone_state = np.zeros((num_bands, 4), dtype=np.float64)

    max_allpass = int(np.max(allpass_delays)) + 1
    allpass_buffers = np.zeros((allpass_delays.size, max_allpass), dtype=np.float64)
    allpass_positions = np.zeros(allpass_delays.size, dtype=np.int64)
    for sample in range(num_samples):
        for line in range(num_delays):
            value = delay_buffers[line, delay_positions[line]]
            for band in range(num_bands):
                b0 = absorptive_coeffs[line, band, 0]
                b1 = absorptive_coeffs[line, band, 1]
                b2 = absorptive_coeffs[line, band, 2]
                a1 = absorptive_coeffs[line, band, 3]
                a2 = absorptive_coeffs[line, band, 4]
                x1 = absorptive_state[line, band, 0]
                x2 = absorptive_state[line, band, 1]
                y1 = absorptive_state[line, band, 2]
                y2 = absorptive_state[line, band, 3]
                filtered = b0 * value + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                absorptive_state[line, band, 1] = x1
                absorptive_state[line, band, 0] = value
                absorptive_state[line, band, 3] = y1
                absorptive_state[line, band, 2] = filtered
                value = filtered
            delayed[line] = value
            mixed[line] = value

        stride = 1
        while stride < num_delays:
            block = stride * 2
            for base in range(0, num_delays, block):
                for offset in range(stride):
                    first = mixed[base + offset]
                    second = mixed[base + offset + stride]
                    mixed[base + offset] = first + second
                    mixed[base + offset + stride] = first - second
            stride = block

        injection = 1.0 if sample == 0 else 0.0
        for line in range(num_delays):
            delay_buffers[line, delay_positions[line]] = injection + 0.25 * mixed[line]
            delay_positions[line] += 1
            if delay_positions[line] >= delays[line]:
                delay_positions[line] = 0

        value = 0.0
        for line in range(num_delays):
            value += delayed[line]
        value /= num_delays
        for stage in range(allpass_delays.size):
            position = allpass_positions[stage]
            previous = allpass_buffers[stage, position]
            internal = value + 0.5 * previous
            allpass_buffers[stage, position] = internal
            value = previous - 0.5 * internal
            position += 1
            if position >= allpass_delays[stage]:
                position = 0
            allpass_positions[stage] = position
        for band in range(num_bands):
            b0 = tone_coeffs[band, 0]
            b1 = tone_coeffs[band, 1]
            b2 = tone_coeffs[band, 2]
            a1 = tone_coeffs[band, 3]
            a2 = tone_coeffs[band, 4]
            x1 = tone_state[band, 0]
            x2 = tone_state[band, 1]
            y1 = tone_state[band, 2]
            y2 = tone_state[band, 3]
            filtered = b0 * value + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            tone_state[band, 1] = x1
            tone_state[band, 0] = value
            tone_state[band, 3] = y1
            tone_state[band, 2] = filtered
            value = filtered
        output[sample] = value
    return output


_fdn_impulse_kernel_jit = njit(cache=True)(_fdn_impulse_kernel) if njit is not None else None


def _steam_fdn_delays(fs: int, seed: int) -> np.ndarray:
    primes = np.asarray((2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53), dtype=np.int64)
    delay_min = max(1, int(0.15 * 10.0 * max(int(fs), 1) / len(primes)))
    offsets = np.random.default_rng(int(seed) + 1701).integers(0, 101, size=primes.size)
    delays = np.empty(primes.size, dtype=np.int64)
    for index, prime in enumerate(primes):
        target = max(1, delay_min + int(offsets[index]))
        exponent = max(1, int(round(math.log(target) / math.log(int(prime)))))
        delays[index] = max(1, int(prime) ** exponent)
    return delays


def _low_shelf_coefficients(cutoff: float, gain: float, fs: int) -> np.ndarray:
    q = 0.707
    w0 = 2.0 * math.pi * float(cutoff) / max(float(fs), 1.0)
    cw0, sw0 = math.cos(w0), math.sin(w0)
    alpha = sw0 / (2.0 * q)
    a = math.sqrt(max(float(gain), 1e-6))
    root_a = math.sqrt(a)
    a0 = (a + 1.0) + (a - 1.0) * cw0 + 2.0 * root_a * alpha
    return np.asarray((
        a * ((a + 1.0) - (a - 1.0) * cw0 + 2.0 * root_a * alpha) / a0,
        2.0 * a * ((a - 1.0) - (a + 1.0) * cw0) / a0,
        a * ((a + 1.0) - (a - 1.0) * cw0 - 2.0 * root_a * alpha) / a0,
        -2.0 * ((a - 1.0) + (a + 1.0) * cw0) / a0,
        ((a + 1.0) + (a - 1.0) * cw0 - 2.0 * root_a * alpha) / a0,
    ), dtype=np.float64)


def _high_shelf_coefficients(cutoff: float, gain: float, fs: int) -> np.ndarray:
    q = 0.707
    w0 = 2.0 * math.pi * float(cutoff) / max(float(fs), 1.0)
    cw0, sw0 = math.cos(w0), math.sin(w0)
    alpha = sw0 / (2.0 * q)
    a = math.sqrt(max(float(gain), 1e-6))
    root_a = math.sqrt(a)
    a0 = (a + 1.0) - (a - 1.0) * cw0 + 2.0 * root_a * alpha
    return np.asarray((
        a * ((a + 1.0) + (a - 1.0) * cw0 + 2.0 * root_a * alpha) / a0,
        -2.0 * a * ((a - 1.0) + (a + 1.0) * cw0) / a0,
        a * ((a + 1.0) + (a - 1.0) * cw0 - 2.0 * root_a * alpha) / a0,
        2.0 * ((a - 1.0) - (a + 1.0) * cw0) / a0,
        ((a + 1.0) - (a - 1.0) * cw0 - 2.0 * root_a * alpha) / a0,
    ), dtype=np.float64)


def _peaking_coefficients(low: float, high: float, gain: float, fs: int) -> np.ndarray:
    center = math.sqrt(max(float(low) * float(high), 1e-12))
    q_inverse = (float(high) - float(low)) / center
    w0 = 2.0 * math.pi * center / max(float(fs), 1.0)
    cw0, sw0 = math.cos(w0), math.sin(w0)
    alpha = sw0 * q_inverse / 2.0
    a = math.sqrt(max(float(gain), 1e-6))
    a0 = 1.0 + alpha / a
    return np.asarray((
        (1.0 + alpha * a) / a0,
        -2.0 * cw0 / a0,
        (1.0 - alpha * a) / a0,
        -2.0 * cw0 / a0,
        (1.0 - alpha / a) / a0,
    ), dtype=np.float64)


def _steam_multiband_filter_coefficients(gains: np.ndarray, fs: int) -> np.ndarray:
    values = np.asarray(gains, dtype=np.float64).reshape(_NUM_BANDS)
    edges = _band_edges(fs)
    nyquist = max(1.0, 0.5 * float(fs))
    coeffs = np.empty((_NUM_BANDS, 5), dtype=np.float64)
    for band_index, (low, high) in enumerate(edges):
        if band_index == 0:
            cutoff = float(np.clip(high, 1e-3, nyquist * 0.999))
            coeffs[band_index] = _low_shelf_coefficients(cutoff, values[band_index], fs)
        elif band_index == _NUM_BANDS - 1:
            cutoff = float(np.clip(low, 1e-3, nyquist * 0.999))
            coeffs[band_index] = _high_shelf_coefficients(cutoff, values[band_index], fs)
        else:
            lower = float(np.clip(low, 1e-3, nyquist * 0.998))
            upper = float(np.clip(high, lower + 1e-3, nyquist * 0.999))
            coeffs[band_index] = _peaking_coefficients(lower, upper, values[band_index], fs)
    return coeffs


def _steam_multiband_response_matrix(fs: int) -> np.ndarray:
    centers = np.minimum(
        np.asarray([float(band) for band in FREQUENCY_BANDS], dtype=np.float64),
        0.9 * 0.5 * max(float(fs), 1.0),
    )
    z = np.exp(-2.0j * math.pi * centers / max(float(fs), 1.0))
    probe_gain = 0.5
    response = np.empty((_NUM_BANDS, _NUM_BANDS), dtype=np.float64)
    for filter_index in range(_NUM_BANDS):
        gains = np.ones(_NUM_BANDS, dtype=np.float64)
        gains[filter_index] = probe_gain
        coeffs = _steam_multiband_filter_coefficients(gains, fs)
        transfer = np.ones(_NUM_BANDS, dtype=np.complex128)
        for band_coeffs in coeffs:
            numerator = band_coeffs[0] + band_coeffs[1] * z + band_coeffs[2] * z * z
            denominator = 1.0 + band_coeffs[3] * z + band_coeffs[4] * z * z
            transfer *= numerator / denominator
        response[:, filter_index] = np.log(np.maximum(np.abs(transfer), 1e-12)) / math.log(probe_gain)
    return response


def _steam_compensated_multiband_coefficients(
    gains: np.ndarray,
    fs: int,
    response: np.ndarray | None = None,
) -> np.ndarray:
    desired = np.clip(np.asarray(gains, dtype=np.float64).reshape(_NUM_BANDS), 1e-3, 0.999999)
    response_matrix = response if response is not None else _steam_multiband_response_matrix(fs)
    try:
        filter_log_gains = np.linalg.solve(response_matrix, np.log(desired))
    except np.linalg.LinAlgError:
        filter_log_gains = np.linalg.pinv(response_matrix) @ np.log(desired)
    # Individual EQ sections may need a small boost to offset neighboring
    # cuts; the solved cascade still follows sub-unity feedback targets.
    filter_gains = np.clip(np.exp(filter_log_gains), 1e-3, 4.0)
    return _steam_multiband_filter_coefficients(filter_gains, fs)


def _steam_fdn_filter_coefficients(
    delays: np.ndarray,
    rt60_bands: Mapping[str, float],
    fs: int,
) -> tuple[np.ndarray, np.ndarray]:
    rt60 = np.asarray(
        [max(0.1, float(rt60_bands.get(band, 0.0) or 0.0)) for band in FREQUENCY_BANDS],
        dtype=np.float64,
    )
    absorptive = np.empty((delays.size, _NUM_BANDS, 5), dtype=np.float64)
    response = _steam_multiband_response_matrix(fs)
    for line, delay in enumerate(np.asarray(delays, dtype=np.float64)):
        gains = np.maximum(np.exp(-(6.91 * delay) / (rt60 * max(float(fs), 1.0))), 1e-3)
        absorptive[line] = _steam_compensated_multiband_coefficients(gains, fs, response)
    tone_gains = np.sqrt(1.0 / rt60)
    tone_gains /= max(float(np.max(tone_gains)), 1e-12)
    return absorptive, _steam_multiband_filter_coefficients(tone_gains, fs)


def _late_tail_start_power(signal: np.ndarray, start_sample: int, rt60: float, fs: int, bin_samples: int) -> float:
    squared = np.square(np.asarray(signal, dtype=np.float64))
    start_sample = min(max(0, int(start_sample)), squared.size)
    measured = squared[start_sample:min(squared.size, start_sample + bin_samples)]
    measured_power = float(np.mean(measured)) if measured.size else 0.0
    if start_sample <= 0:
        return measured_power
    prior = squared[:start_sample]
    num_bins = int(math.ceil(prior.size / bin_samples))
    energies = np.zeros(num_bins, dtype=np.float64)
    for index in range(num_bins):
        lo = index * bin_samples
        hi = min(prior.size, lo + bin_samples)
        energies[index] = float(np.sum(prior[lo:hi]))
    positive = np.flatnonzero(energies > 1e-18)
    if positive.size == 0:
        return measured_power
    recent = positive[-min(5, positive.size):]
    decay = 6.0 * math.log(10.0) / max(float(rt60), 0.1)
    estimates = []
    for index in recent:
        center_sample = (float(index) + 0.5) * bin_samples
        elapsed = max(0.0, (start_sample - center_sample) / max(float(fs), 1.0))
        estimates.append((energies[index] / bin_samples) * math.exp(-decay * elapsed))
    return max(measured_power, float(np.median(estimates)))


def _render_parametric_fdn_late_reverb(
    traced_band_irs: np.ndarray,
    field: Mapping[str, Any],
    rt60_bands: Mapping[str, float],
    config: SimConfig,
    *,
    decay_profiles: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Crossfade traced convolution into a calibrated parametric FDN tail."""
    traced = np.asarray(traced_band_irs, dtype=np.float64)
    if traced.ndim != 2 or traced.shape[0] != _NUM_BANDS:
        raise ValueError(f"expected {_NUM_BANDS} traced band IRs, got {traced.shape}")
    fs = max(1, int(config.fs))
    num_samples = int(traced.shape[1])
    duration_s = num_samples / fs
    transition_s = min(max(1.0 / fs, float(config.hybrid_transition_s)), max(1.0 / fs, (num_samples - 1) / fs))
    overlap = float(np.clip(config.hybrid_overlap_fraction, 0.0, 0.95))
    start_s = (1.0 - overlap) * transition_s
    start_samples = np.full(_NUM_BANDS, min(num_samples, max(0, int(math.floor(start_s * fs)))), dtype=np.int64)
    end_samples = np.full(
        _NUM_BANDS,
        min(num_samples, max(int(start_samples[0]) + 1, int(math.ceil(transition_s * fs)))),
        dtype=np.int64,
    )
    transition_by_band: dict[str, dict[str, float | str]] = {}
    profiles = decay_profiles if isinstance(decay_profiles, Mapping) else {}
    for band_index, band in enumerate(FREQUENCY_BANDS):
        profile = profiles.get(band)
        source = str(profile.get("selected_target_source", "")) if isinstance(profile, Mapping) else ""
        breakpoint = float(profile.get("transition_time_s", 0.0) or 0.0) if isinstance(profile, Mapping) else 0.0
        if source == "fitted_late_slope" and 0.0 < breakpoint < duration_s:
            band_start_s = (1.0 - overlap) * breakpoint
            band_start = min(num_samples - 1, max(0, int(math.floor(band_start_s * fs))))
            band_end = min(num_samples, max(band_start + 1, int(math.ceil(breakpoint * fs))))
            start_samples[band_index] = band_start
            end_samples[band_index] = band_end
            model = "coupled_space_breakpoint"
        else:
            model = "configured_hybrid_transition"
        transition_by_band[band] = {
            "start_s": round(float(start_samples[band_index]) / fs, 6),
            "end_s": round(float(end_samples[band_index]) / fs, 6),
            "model": model,
        }
    start_sample = int(np.min(start_samples))
    end_sample = int(np.max(end_samples))
    empty_tail = np.zeros_like(traced, dtype=np.float32)
    if not config.late_tail or start_sample >= num_samples:
        return traced.astype(np.float32), empty_tail, {
            "applied": False,
            "added": False,
            "reason": "disabled" if not config.late_tail else "tail_start_outside_rir",
            "model": "steam_style_16_line_hadamard_fdn",
            "transition_start_s": round(start_s, 6),
            "transition_end_s": round(transition_s, 6),
            "overlap_fraction": overlap,
        }

    delays = _steam_fdn_delays(fs, config.seed)
    allpass_delays = np.asarray((225, 341, 441, 556), dtype=np.int64)
    rt60_used = {
        band: round(float(rt60_bands.get(band, 0.0) or 0.0), 4)
        for band in FREQUENCY_BANDS
        if float(rt60_bands.get(band, 0.0) or 0.0) > 0.0
    }
    absorptive_coeffs, tone_coeffs = _steam_fdn_filter_coefficients(delays, rt60_bands, fs)
    kernel = _fdn_impulse_kernel_jit or _fdn_impulse_kernel
    # Start the statistical tail with populated delay and allpass states. This
    # is equivalent to running the FDN before the hybrid window, and is needed
    # when a coupled-space breakpoint occurs before the longest delay line.
    preroll_samples = int(np.max(delays) + np.sum(allpass_delays))
    raw_fdn = kernel(
        delays,
        absorptive_coeffs,
        num_samples + preroll_samples,
        allpass_delays,
        tone_coeffs,
    )[preroll_samples:]
    fdn = np.asarray(
        bandlimit_band_signals(np.repeat(raw_fdn[None, :], _NUM_BANDS, axis=0), fs),
        dtype=np.float64,
    )

    bin_samples = max(1, int(round(float(field.get("bin_duration_s", config.rt_bin_duration_s)) * fs)))
    scales: dict[str, float] = {}
    anchor_power: dict[str, float] = {}
    fdn_energy_by_band: dict[str, float] = {}
    for band_index, band in enumerate(FREQUENCY_BANDS):
        rt60 = float(rt60_bands.get(band, 0.0) or 0.0)
        if rt60 <= 0.0:
            continue
        band_start = int(start_samples[band_index])
        transition_model = str(transition_by_band[band]["model"])
        anchor_sample = int(end_samples[band_index]) if transition_model == "coupled_space_breakpoint" else band_start
        remaining = num_samples - anchor_sample
        power = _late_tail_start_power(traced[band_index], anchor_sample, rt60, fs, bin_samples)
        decay = 6.0 * math.log(10.0) / max(rt60, 0.1)
        expected = power * np.exp(-decay * np.arange(remaining, dtype=np.float64) / fs)
        target_energy = float(np.sum(expected))
        unit_energy = float(np.sum(np.square(fdn[band_index, anchor_sample:])))
        scale = math.sqrt(target_energy / max(unit_energy, 1e-24)) if target_energy > 0.0 else 0.0
        fdn[band_index, band_start:] *= scale
        scales[band] = round(scale, 8)
        anchor_power[band] = round(power, 14)
        transition_by_band[band]["anchor_s"] = round(float(anchor_sample) / fs, 6)
        fdn_energy_by_band[band] = round(float(np.sum(np.square(fdn[band_index, band_start:]))), 12)

    out = np.array(traced, copy=True)
    fdn_component = np.zeros_like(traced)
    traced_tail_energy = 0.0
    rendered_tail_energy = 0.0
    for band_index in range(_NUM_BANDS):
        band_start = int(start_samples[band_index])
        band_end = int(end_samples[band_index])
        ramp_count = max(1, band_end - band_start)
        alpha = np.linspace(1.0, 0.0, ramp_count, endpoint=False, dtype=np.float64)
        out[band_index, band_start:band_end] *= np.sqrt(alpha)
        if band_end < num_samples:
            out[band_index, band_end:] = 0.0
        parametric_weight = np.ones(num_samples - band_start, dtype=np.float64)
        parametric_weight[:ramp_count] = np.sqrt(1.0 - alpha)
        fdn_component[band_index, band_start:] = fdn[band_index, band_start:] * parametric_weight
        traced_tail_energy += float(np.sum(np.square(traced[band_index, band_start:])))
    out += fdn_component
    for band_index in range(_NUM_BANDS):
        rendered_tail_energy += float(np.sum(np.square(out[band_index, int(start_samples[band_index]):])))

    fdn_energy = float(np.sum(np.square(fdn_component)))
    return out.astype(np.float32), fdn_component.astype(np.float32), {
        "applied": bool(rt60_used and fdn_energy > 1e-18),
        "changed": bool(not np.allclose(out[:, start_sample:], traced[:, start_sample:], rtol=1e-7, atol=1e-12)),
        "added": bool(fdn_energy > 1e-18),
        "model": "steam_style_16_line_hadamard_fdn",
        "rt60_source": "steam_audio_reverb_estimator_from_traced_energy_field",
        "calibration": "steady-state pre-rolled feedback-loop multiband absorption, traced cutoff power, duration-integrated energy normalization, and equal-power hybrid crossfade",
        "transition_start_s": round(float(start_sample) / fs, 6),
        "transition_end_s": round(float(end_sample) / fs, 6),
        "configured_transition_end_s": round(transition_s, 6),
        "transition_by_band": transition_by_band,
        "overlap_fraction": overlap,
        "delay_line_count": int(delays.size),
        "delay_samples": [int(value) for value in delays],
        "allpass_delay_samples": [int(value) for value in allpass_delays],
        "preroll_samples": preroll_samples,
        "traced_tail_energy": round(traced_tail_energy, 12),
        "fdn_tail_energy": round(fdn_energy, 12),
        "rendered_tail_energy": round(rendered_tail_energy, 12),
        "anchor_power_by_band": anchor_power,
        "scale_by_band": scales,
        "fdn_energy_by_band": fdn_energy_by_band,
        "rt60_bands": rt60_used,
    }


def _hybridize_ambisonic_tail(
    traced: np.ndarray,
    fdn_component: np.ndarray,
    metadata: Mapping[str, Any],
    config: SimConfig,
) -> np.ndarray:
    out = np.asarray(traced, dtype=np.float64).copy()
    if not bool(metadata.get("applied", False)):
        return out.astype(np.float32)
    fs = max(1, int(config.fs))
    num_samples = int(out.shape[-1])
    transitions = metadata.get("transition_by_band") if isinstance(metadata.get("transition_by_band"), Mapping) else {}
    for band_index, band in enumerate(FREQUENCY_BANDS):
        transition = transitions.get(band) if isinstance(transitions, Mapping) else None
        start_s = float(transition.get("start_s", metadata["transition_start_s"])) if isinstance(transition, Mapping) else float(metadata["transition_start_s"])
        end_s = float(transition.get("end_s", metadata["transition_end_s"])) if isinstance(transition, Mapping) else float(metadata["transition_end_s"])
        start = min(num_samples, max(0, int(math.floor(start_s * fs))))
        end = min(num_samples, max(start + 1, int(math.ceil(end_s * fs))))
        ramp_count = max(1, end - start)
        alpha = np.linspace(1.0, 0.0, ramp_count, endpoint=False, dtype=np.float64)
        out[band_index, :, start:end] *= np.sqrt(alpha)[None, :]
        if end < num_samples:
            out[band_index, :, end:] = 0.0
    out[:, 0, :] += np.asarray(fdn_component, dtype=np.float64)
    return out.astype(np.float32)


def reconstruct_ambisonic_band_irs(field: dict[str, Any], config: SimConfig) -> np.ndarray:
    echogram = field["echogram"]
    ambisonic = field.get("ambisonic_echogram")
    if ambisonic is None:
        mono = reconstruct_band_irs(field, config)
        out = np.zeros((_NUM_BANDS, 4, mono.shape[1]), dtype=np.float32)
        out[:, 0, :] = mono
        return out
    samples_per_bin = max(1, int(math.ceil(float(field["bin_duration_s"]) * int(config.fs))))
    num_samples = int(field["num_bins"]) * samples_per_bin
    rng = np.random.default_rng(config.seed + 11)
    white = rng.uniform(-1.0, 1.0, size=num_samples).astype(np.float64)
    raw = np.zeros((_NUM_BANDS, 4, num_samples), dtype=np.float64)
    sample_weights = np.arange(samples_per_bin, dtype=np.float64) / samples_per_bin
    direct_delay_s = max(0.0, float(field.get("direct_delay_s", 0.0)))
    for band_index, band in enumerate(FREQUENCY_BANDS):
        coeff = AIR_ABSORPTION_NP_PER_M[band]
        energy = np.clip(echogram[band_index], 0.0, None)
        amps = np.sqrt(energy / (4.0 * math.pi))
        amps[energy < _ENERGY_THRESHOLD] = 0.0
        ratios = np.zeros((4, int(field["num_bins"])), dtype=np.float64)
        valid = energy > _ENERGY_THRESHOLD
        ratios[0, valid] = 1.0
        for ci in range(1, 4):
            ratios[ci, valid] = np.clip(ambisonic[band_index, ci, valid] / np.maximum(energy[valid], 1e-12), -1.0, 1.0)
        channel_amp = np.zeros((4, num_samples), dtype=np.float64)
        for b in range(int(field["num_bins"])):
            lo = b * samples_per_bin
            hi = min(num_samples, lo + samples_per_bin)
            prev = amps[b] if b == 0 else amps[b - 1]
            w = sample_weights[:hi - lo]
            seg = (1.0 - w) * prev + w * amps[b]
            path_time_s = direct_delay_s + (b + 0.5) * samples_per_bin / int(config.fs)
            seg *= _air_absorption_amplitude(coeff, path_time_s, float(config.c))
            for ci in range(4):
                prev_ratio = ratios[ci, b] if b == 0 else ratios[ci, b - 1]
                ratio_seg = (1.0 - w) * prev_ratio + w * ratios[ci, b]
                channel_amp[ci, lo:hi] = seg * ratio_seg
        for ci in range(4):
            raw[band_index, ci] = channel_amp[ci] * white
    out = np.zeros((_NUM_BANDS, 4, num_samples), dtype=np.float32)
    for ci in range(4):
        out[:, ci] = bandlimit_band_signals(raw[:, ci], int(config.fs))
    return out


def estimate_reverb_times(field: dict[str, Any], config: SimConfig) -> dict[str, float]:
    echogram = field["echogram"]
    bin_dur = float(field["bin_duration_s"])
    num_bins = int(field["num_bins"])
    out: dict[str, float] = {}
    times = bin_dur * np.arange(num_bins, dtype=np.float64)
    for band_index, band in enumerate(FREQUENCY_BANDS):
        weights = _air_absorption_energy_weights(
            AIR_ABSORPTION_NP_PER_M[band],
            times,
            float(config.c),
        )
        # Steam Audio stores Y00 * E in channel 0 of the EnergyField.
        weighted = (_SH_Y00 * echogram[band_index]) * weights
        out[band] = _schroeder_fit_rt60(weighted, bin_dur, min_total_energy=1e-4)
    return out


def estimate_late_reverb_times(
    field: dict[str, Any],
    config: SimConfig,
    *,
    fallback: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Select the actual late EDC slope for a coupled-space parametric tail."""
    echogram = np.asarray(field["echogram"], dtype=np.float64)
    bin_dur = float(field["bin_duration_s"])
    num_bins = int(field["num_bins"])
    standard = dict(fallback or estimate_reverb_times(field, config))
    targets: dict[str, float] = {}
    profiles: dict[str, Any] = {}
    times = bin_dur * np.arange(num_bins, dtype=np.float64)
    for band_index, band in enumerate(FREQUENCY_BANDS):
        weighted = (
            _SH_Y00
            * echogram[band_index]
            * _air_absorption_energy_weights(
                AIR_ABSORPTION_NP_PER_M[band],
                times,
                float(config.c),
            )
        )
        profile = _energy_decay_profile(weighted, bin_dur)
        target = float(standard.get(band, 0.0) or 0.0)
        source = "steam_single_slope"
        if profile.get("model") == "double_slope":
            late = next(
                (item for item in profile.get("segments", []) if item.get("label") == "late"),
                None,
            )
            late_rt60 = float(late.get("equivalent_rt60_s", 0.0)) if isinstance(late, Mapping) else 0.0
            if late_rt60 > 0.0:
                target = late_rt60
                source = "fitted_late_slope"
        elif target <= 0.0 and profile.get("model") == "single_slope":
            fitted = float(profile.get("single_rt60_s", 0.0) or 0.0)
            if fitted > 0.0:
                target = fitted
                source = "fitted_wide_range_single_slope"
        targets[band] = target
        profiles[band] = {
            **profile,
            "selected_target_rt60_s": round(target, 4),
            "selected_target_source": source,
        }
    return targets, profiles


def _apply_coupled_late_reverb_prior(
    traced_targets: Mapping[str, float],
    profiles: Mapping[str, Any],
    prior: Mapping[str, float] | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Use room-system decay constants for the statistical late tail.

    In a connected apartment, moving through a portal changes modal
    amplitudes and early decay, but it does not replace the enclosure's decay
    constants. Sparse receiver hits can make a per-frame traced late slope
    visibly jump, so the FDN target uses the coupled-room energy model while
    retaining the traced fit as a diagnostic.
    """
    targets = {band: float(traced_targets.get(band, 0.0) or 0.0) for band in FREQUENCY_BANDS}
    updated_profiles = {band: dict(profiles.get(band, {})) for band in FREQUENCY_BANDS}
    if not isinstance(prior, Mapping):
        return targets, updated_profiles
    for band in FREQUENCY_BANDS:
        coupled = float(prior.get(band, 0.0) or 0.0)
        if not math.isfinite(coupled) or coupled <= 0.0:
            continue
        profile = updated_profiles[band]
        profile["traced_target_rt60_s"] = round(float(targets[band]), 4)
        profile["selected_target_rt60_s"] = round(coupled, 4)
        profile["selected_target_source"] = "coupled_room_energy_matrix"
        targets[band] = coupled
    return targets, updated_profiles


def estimate_reconstructed_reverb_times(band_rirs: np.ndarray, config: SimConfig) -> dict[str, float]:
    """Estimate RT60 from the reconstructed final band RIRs.

    This mirrors the Steam Audio ReverbEstimator convention used above for the
    traced EnergyField: Schroeder reverse integration, fit roughly -5 dB to
    -25 dB, then extrapolate the fitted slope to -60 dB.  The important
    difference is that this runs after reconstruction and the calibrated late
    tail, so it is the RT60 of the rendered/hybrid impulse response rather than
    the diagnostic early traced energy field.
    """
    fs = max(1, int(config.fs))
    signals = np.asarray(band_rirs, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[0] != _NUM_BANDS:
        return {band: 0.0 for band in FREQUENCY_BANDS}
    bin_samples = max(1, int(round(float(config.rt_bin_duration_s) * fs)))
    num_samples = int(signals.shape[1])
    num_bins = max(1, int(math.ceil(num_samples / bin_samples)))
    out: dict[str, float] = {}
    for band_index, band in enumerate(FREQUENCY_BANDS):
        energy = np.zeros(num_bins, dtype=np.float64)
        squared = np.square(signals[band_index])
        for bin_index in range(num_bins):
            lo = bin_index * bin_samples
            hi = min(num_samples, lo + bin_samples)
            if hi > lo:
                energy[bin_index] = float(np.sum(squared[lo:hi]))
        out[band] = _schroeder_fit_rt60(energy, bin_samples / fs)
    return out


def estimate_signal_reverb_time(signal: np.ndarray, config: SimConfig) -> float:
    """Estimate RT60 from the final rendered RIR signal."""
    fs = max(1, int(config.fs))
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    bin_samples = max(1, int(round(float(config.rt_bin_duration_s) * fs)))
    num_bins = max(1, int(math.ceil(values.size / bin_samples)))
    energy = np.zeros(num_bins, dtype=np.float64)
    squared = values * values
    for bin_index in range(num_bins):
        lo = bin_index * bin_samples
        hi = min(values.size, lo + bin_samples)
        if hi > lo:
            energy[bin_index] = float(np.sum(squared[lo:hi]))
    return _schroeder_fit_rt60(energy, bin_samples / fs)


def estimate_signal_decay_profile(signal: np.ndarray, config: SimConfig) -> dict[str, Any]:
    """Describe statistically significant slope changes in a rendered RIR EDC.

    The regular RT60 remains the -5 to -25 dB single-line estimate.  This
    diagnostic uses the wider -5 to -35 dB range and reports a second segment
    only when it materially improves the fit and the slopes differ enough to
    represent coupled-space decay rather than harmless numerical curvature.
    """
    fs = max(1, int(config.fs))
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return _energy_decay_profile(np.zeros(0, dtype=np.float64), 0.0)
    bin_samples = max(1, int(round(float(config.rt_bin_duration_s) * fs)))
    num_bins = max(1, int(math.ceil(values.size / bin_samples)))
    energy = np.zeros(num_bins, dtype=np.float64)
    squared = values * values
    for bin_index in range(num_bins):
        lo = bin_index * bin_samples
        hi = min(values.size, lo + bin_samples)
        if hi > lo:
            energy[bin_index] = float(np.sum(squared[lo:hi]))
    return _energy_decay_profile(energy, bin_samples / fs)


def _energy_decay_profile(
    energy: np.ndarray,
    bin_dur: float,
    *,
    fit_start_db: float = -5.0,
    fit_end_db: float = -35.0,
) -> dict[str, Any]:
    values = np.asarray(energy, dtype=np.float64).reshape(-1)
    total = float(np.sum(values))
    unavailable = {
        "model": "unavailable",
        "segments": [],
        "fit_range_db": [float(fit_start_db), float(fit_end_db)],
    }
    if values.size < 2 or total < 1e-12 or bin_dur <= 0.0:
        return unavailable

    edc = np.cumsum(values[::-1])[::-1]
    db = 10.0 * np.log10(np.clip(edc / total, 1e-12, None))
    times = float(bin_dur) * np.arange(values.size, dtype=np.float64)
    fit_mask = (db <= float(fit_start_db)) & (db >= float(fit_end_db))
    xs = times[fit_mask]
    ys = db[fit_mask]
    if xs.size < 10 or float(np.ptp(ys)) < 14.0:
        return unavailable

    single = _linear_decay_fit(xs, ys)
    if single is None:
        return unavailable
    point_count = int(xs.size)
    single_bic = point_count * math.log(max(single[2] / point_count, 1e-15)) + 2.0 * math.log(point_count)
    best: tuple[float, int, tuple[float, float, float, float], tuple[float, float, float, float]] | None = None
    min_points = 5
    min_span_db = 7.0
    for split in range(min_points, point_count - min_points + 1):
        early_y = ys[:split]
        late_y = ys[split:]
        if float(np.ptp(early_y)) < min_span_db or float(np.ptp(late_y)) < min_span_db:
            continue
        early = _linear_decay_fit(xs[:split], early_y)
        late = _linear_decay_fit(xs[split:], late_y)
        if early is None or late is None:
            continue
        combined_sse = early[2] + late[2]
        dual_bic = point_count * math.log(max(combined_sse / point_count, 1e-15)) + 5.0 * math.log(point_count)
        candidate = (dual_bic, split, early, late)
        if best is None or dual_bic < best[0]:
            best = candidate

    single_segment = _decay_segment("full", xs, ys, single)
    result: dict[str, Any] = {
        "model": "single_slope",
        "segments": [single_segment],
        "fit_range_db": [float(fit_start_db), float(fit_end_db)],
        "single_rt60_s": single_segment["equivalent_rt60_s"],
        "single_r2": single_segment["r2"],
        "dynamic_range_db": round(float(np.ptp(ys)), 4),
    }
    if best is None:
        return result

    dual_bic, split, early, late = best
    early_rt60 = -60.0 / early[0]
    late_rt60 = -60.0 / late[0]
    slope_ratio = max(early_rt60, late_rt60) / max(min(early_rt60, late_rt60), 1e-12)
    bic_improvement = single_bic - dual_bic
    significant = (
        bic_improvement >= 10.0
        and slope_ratio >= 1.35
        and early[3] >= 0.95
        and late[3] >= 0.95
    )
    result["candidate_bic_improvement"] = round(float(bic_improvement), 4)
    result["candidate_slope_ratio"] = round(float(slope_ratio), 4)
    if not significant:
        return result

    early_segment = _decay_segment("early", xs[:split], ys[:split], early)
    late_segment = _decay_segment("late", xs[split:], ys[split:], late)
    return {
        **result,
        "model": "double_slope",
        "segments": [early_segment, late_segment],
        "transition_time_s": round(float(xs[split]), 4),
        "transition_level_db": round(float(ys[split]), 4),
        "slope_ratio": round(float(slope_ratio), 4),
        "bic_improvement": round(float(bic_improvement), 4),
        "slope_order": "early_slower" if early_rt60 > late_rt60 else "late_slower",
    }


def _linear_decay_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float] | None:
    count = int(x.size)
    if count < 2:
        return None
    sum_x = float(np.sum(x))
    sum_y = float(np.sum(y))
    denom = count * float(np.sum(x * x)) - sum_x * sum_x
    if abs(denom) <= 1e-12:
        return None
    slope = (count * float(np.sum(x * y)) - sum_x * sum_y) / denom
    if slope >= -1e-9:
        return None
    intercept = (sum_y - slope * sum_x) / count
    residual = y - (slope * x + intercept)
    sse = float(np.sum(residual * residual))
    centered = y - float(np.mean(y))
    total_variation = float(np.sum(centered * centered))
    r2 = 1.0 - sse / max(total_variation, 1e-15)
    return float(slope), float(intercept), sse, float(r2)


def _decay_segment(
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    fit: tuple[float, float, float, float],
) -> dict[str, Any]:
    slope, _, _, r2 = fit
    return {
        "label": str(label),
        "equivalent_rt60_s": round(float(-60.0 / slope), 4),
        "slope_db_per_s": round(float(slope), 4),
        "r2": round(float(r2), 4),
        "start_time_s": round(float(x[0]), 4),
        "end_time_s": round(float(x[-1]), 4),
        "start_level_db": round(float(y[0]), 4),
        "end_level_db": round(float(y[-1]), 4),
    }


def estimate_steam_audio_default_reverb_times(signal: np.ndarray, config: SimConfig) -> dict[str, float]:
    """Steam Audio default frequency subdivision: low/mid/high.

    Steam Audio's default build uses three broad bands rather than octave
    bands: low 0-800 Hz, mid 800-8 kHz, high 8-22 kHz.  At 16 kHz sampling
    rate the high band is above Nyquist, so it is omitted.
    """
    fs = max(1, int(config.fs))
    nyquist = fs * 0.5
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    bands = {
        "low": (0.0, min(800.0, nyquist)),
        "mid": (800.0, min(8000.0, nyquist)),
        "high": (8000.0, min(22000.0, nyquist)),
    }
    out: dict[str, float] = {}
    modes = {"low": "lowpass", "mid": "bandpass", "high": "highpass"}
    for name, (lo, hi) in bands.items():
        if hi <= lo or values.size == 0:
            continue
        filtered = _band_limit(values, lo, hi, fs, mode=modes[name])
        out[name] = estimate_signal_reverb_time(filtered, config)
    return out


def _schroeder_fit_rt60(energy: np.ndarray, bin_dur: float, *, min_total_energy: float = 1e-12) -> float:
    total = float(np.sum(energy))
    if total < float(min_total_energy):
        return 0.0
    edc = np.cumsum(np.asarray(energy, dtype=np.float64)[::-1])[::-1]
    db = 10.0 * np.log10(np.clip(edc / total, 1e-12, None))
    x = float(bin_dur) * np.arange(edc.size, dtype=np.float64)
    fit = (db <= -5.0) & (db >= -25.0)
    if int(np.count_nonzero(fit)) < 2:
        return 0.0
    xs, ys = x[fit], db[fit]
    n = xs.size
    denom = n * np.sum(xs * xs) - np.sum(xs) ** 2
    numer = n * np.sum(xs * ys) - np.sum(xs) * np.sum(ys)
    slope_db_per_s = numer / denom if abs(denom) > 1e-12 else 0.0
    return float(max(0.0, -60.0 / slope_db_per_s)) if slope_db_per_s < -1e-9 else 0.0


def bandlimit_band_signals(signals: np.ndarray, fs: int) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != _NUM_BANDS:
        raise ValueError(f"expected {_NUM_BANDS} band signals, got shape {values.shape}")

    crossovers = [edge[1] for edge in _band_edges(fs)[:-1]]
    out = np.zeros_like(values, dtype=np.float32)
    out[0] = _band_limit(values[0], 0.0, crossovers[0], fs, mode="lowpass")
    for band_index in range(1, _NUM_BANDS - 1):
        upper = _band_limit(values[band_index], 0.0, crossovers[band_index], fs, mode="lowpass")
        lower = _band_limit(values[band_index], 0.0, crossovers[band_index - 1], fs, mode="lowpass")
        out[band_index] = upper - lower
    out[-1] = values[-1] - _band_limit(values[-1], 0.0, crossovers[-1], fs, mode="lowpass")
    return out


def _foa_coefficients(directions: np.ndarray) -> np.ndarray:
    dirs = _normalize_rows(np.asarray(directions, dtype=float))
    return np.column_stack([
        np.ones(dirs.shape[0], dtype=np.float64),
        dirs[:, 0],
        dirs[:, 1],
        dirs[:, 2],
    ])


def _add_band_impulse(target: np.ndarray, delay_s: float, band_gains: dict[str, float], config: SimConfig) -> None:
    duration_s = target.shape[1] / max(float(config.fs), 1.0)
    for i, band in enumerate(FREQUENCY_BANDS):
        target[i] += render_impulses(
            np.asarray([float(delay_s)], dtype=np.float64),
            np.asarray([float(band_gains[band])], dtype=np.float64),
            fs=int(config.fs),
            duration_s=duration_s,
            fractional=bool(config.fractional_delay),
            sinc_half_width=int(config.sinc_half_width),
        )


def _direct_path(src: np.ndarray, rcv: np.ndarray, direct: dict[str, Any], config: SimConfig) -> AcousticPath:
    kind = "direct" if float(direct["occlusion"]) >= 1.0 else "direct_transmitted"
    return AcousticPath(
        kind,
        float(direct["distance_m"]),
        float(direct["delay_s"]),
        band_mean(direct["band_gains"]),
        direct["band_gains"],
        (tuple(src), tuple(rcv)),
        {
            "model": "direct_with_geometry_occlusion_transmission",
            "occlusion": float(direct["occlusion"]),
            "occlusion_surface": direct.get("occlusion_surface"),
            "source_directivity_gain": float(direct.get("source_directivity_gain", 1.0)),
            "contributes_to_rir": True,
        },
    )


def _multi_room_portal_paths(
    room: Room,
    scene: RoomRayScene,
    src: np.ndarray,
    rcv: np.ndarray,
    direct: Mapping[str, Any],
    config: SimConfig,
    emitter: Mapping[str, Any],
) -> list[AcousticPath]:
    if not scene.is_multi_room or float(direct.get("occlusion", 1.0)) >= 1.0:
        return []
    metadata = room.metadata.get("multi_room") if isinstance(room.metadata, Mapping) else None
    if not isinstance(metadata, Mapping):
        return []
    room_records = [item for item in metadata.get("rooms", []) if isinstance(item, Mapping)]
    portals = [item for item in metadata.get("portals", []) if isinstance(item, Mapping) and bool(item.get("open", False))]
    source_room_id = _multi_room_id_for_point(src[:2], room_records)
    receiver_room_id = _multi_room_id_for_point(rcv[:2], room_records)
    if source_room_id is None or receiver_room_id is None or source_room_id == receiver_room_id:
        return []
    route = _shortest_portal_route(portals, source_room_id, receiver_room_id)
    if not route:
        return []
    room_by_id = {str(item.get("id")): item for item in room_records}
    portal_by_id = {str(item.get("id")): item for item in portals}
    route_rooms, route_portals = route
    xy_points: list[np.ndarray] = [np.asarray(src[:2], dtype=float)]
    current_xy = xy_points[0]
    for route_index, portal_id in enumerate(route_portals):
        current_room_id = route_rooms[route_index]
        next_room_id = route_rooms[route_index + 1]
        portal = portal_by_id[portal_id]
        room_points = portal.get("room_points", {})
        current_side = np.asarray(room_points.get(current_room_id, portal.get("center")), dtype=float)
        next_side = np.asarray(room_points.get(next_room_id, portal.get("center")), dtype=float)
        corners = room_by_id[current_room_id].get("corners", [])
        segment_path = _room_visibility_path(current_xy, current_side, corners)
        xy_points.extend(segment_path[1:])
        if float(np.linalg.norm(next_side - xy_points[-1])) > 1e-6:
            xy_points.append(next_side)
        current_xy = next_side
    final_segment = _room_visibility_path(
        current_xy,
        np.asarray(rcv[:2], dtype=float),
        room_by_id[receiver_room_id].get("corners", []),
    )
    xy_points.extend(final_segment[1:])
    xy_points = _deduplicate_xy_points(xy_points)
    if len(xy_points) < 2:
        return []

    lower = max(float(portal_by_id[portal_id].get("sill_height_m", 0.0)) + 0.05 for portal_id in route_portals)
    upper = min(
        float(portal_by_id[portal_id].get("sill_height_m", 0.0))
        + float(portal_by_id[portal_id].get("height_m", room.height_m))
        - 0.05
        for portal_id in route_portals
    )
    aperture_z = float(np.clip(0.5 * (float(src[2]) + float(rcv[2])), lower, max(lower, upper)))
    points: list[np.ndarray] = [np.asarray(src, dtype=float)]
    for xy in xy_points[1:-1]:
        points.append(np.asarray([xy[0], xy[1], aperture_z], dtype=float))
    points.append(np.asarray(rcv, dtype=float))
    points = _deduplicate_3d_points(points)
    distance = float(sum(np.linalg.norm(points[index + 1] - points[index]) for index in range(len(points) - 1)))
    if distance <= _EPS or not _portal_path_segments_clear(scene, points):
        return []
    deviation = _path_deviation_angle(points)
    pathing = steam_audio_pathing_deviation(deviation) if deviation > 1e-6 else {band: 1.0 for band in FREQUENCY_BANDS}
    propagation = propagation_band_gains(distance, min_distance_m=float(config.min_distance_m))
    directivity = source_directivity_gain(points[1] - points[0], emitter)
    aperture_estimate, aperture_details = _portal_aperture_coupling(
        route_portals,
        portal_by_id,
        src,
        rcv,
        aperture_z,
    )
    aperture_gain = aperture_estimate if config.portal_aperture_attenuation else 1.0
    band_gains = {
        band: float(propagation[band] * pathing[band] * directivity * aperture_gain)
        for band in FREQUENCY_BANDS
    }
    return [AcousticPath(
        "portal_path",
        distance,
        distance / float(config.c),
        band_mean(band_gains),
        band_gains,
        tuple(tuple(float(value) for value in point) for point in points),
        {
            "model": "verified_portal_visibility_graph_pathing_v2",
            "source_room_id": source_room_id,
            "receiver_room_id": receiver_room_id,
            "route_room_ids": route_rooms,
            "route_portal_ids": route_portals,
            "portal_count": len(route_portals),
            "aperture_pressure_gain": round(aperture_gain, 8),
            "aperture_pressure_gain_estimate": round(aperture_estimate, 8),
            "aperture_attenuation_applied": bool(config.portal_aperture_attenuation),
            "aperture_coupling": aperture_details,
            "segment_visibility_verified": True,
            "total_deviation_deg": math.degrees(deviation),
            "source_directivity_gain": directivity,
            "contributes_to_rir": True,
        },
    )]


def _portal_path_segments_clear(scene: RoomRayScene, points: Sequence[np.ndarray]) -> bool:
    for start, end in zip(points[:-1], points[1:]):
        segment = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        distance = float(np.linalg.norm(segment))
        if distance <= 2.0e-3:
            continue
        direction = segment / distance
        origin = np.asarray(start, dtype=float) + 1.0e-3 * direction
        if scene.any_hit(origin, direction, distance - 2.0e-3):
            return False
    return True


def _portal_aperture_coupling(
    route_portals: Sequence[str],
    portal_by_id: Mapping[str, Mapping[str, Any]],
    source: np.ndarray,
    receiver: np.ndarray,
    aperture_z: float,
) -> tuple[float, list[dict[str, float | str]]]:
    centers = []
    for portal_id in route_portals:
        center = np.asarray(portal_by_id[portal_id].get("center", (0.0, 0.0)), dtype=float)
        centers.append(np.asarray((center[0], center[1], aperture_z), dtype=float))
    anchors = [np.asarray(source, dtype=float), *centers, np.asarray(receiver, dtype=float)]
    total_pressure_gain = 1.0
    details: list[dict[str, float | str]] = []
    for index, portal_id in enumerate(route_portals):
        portal = portal_by_id[portal_id]
        area = max(
            0.01,
            float(portal.get("width_m", 0.8)) * float(portal.get("height_m", 2.0)),
        )
        incoming = max(0.25, float(np.linalg.norm(anchors[index + 1] - anchors[index])))
        outgoing = max(0.25, float(np.linalg.norm(anchors[index + 2] - anchors[index + 1])))
        energy_fraction = float(np.clip(area / (area + 2.0 * math.pi * incoming * outgoing), 0.0, 1.0))
        pressure_gain = math.sqrt(energy_fraction)
        total_pressure_gain *= pressure_gain
        details.append({
            "portal_id": str(portal_id),
            "area_m2": round(area, 5),
            "incoming_distance_m": round(incoming, 5),
            "outgoing_distance_m": round(outgoing, 5),
            "energy_fraction": round(energy_fraction, 8),
            "pressure_gain": round(pressure_gain, 8),
        })
    return float(total_pressure_gain), details


def _multi_room_id_for_point(point: Sequence[float], rooms: Sequence[Mapping[str, Any]]) -> str | None:
    for item in rooms:
        corners = item.get("corners")
        if isinstance(corners, Sequence) and len(corners) >= 3 and point_in_polygon(point, corners):
            return str(item.get("id"))
    return None


def _shortest_portal_route(
    portals: Sequence[Mapping[str, Any]],
    source_room_id: str,
    receiver_room_id: str,
) -> tuple[list[str], list[str]] | None:
    adjacency: dict[str, list[tuple[str, str, float]]] = {}
    for portal in portals:
        room_ids = portal.get("room_ids")
        if not isinstance(room_ids, Sequence) or len(room_ids) != 2:
            continue
        first, second = str(room_ids[0]), str(room_ids[1])
        width = max(float(portal.get("width_m", 0.8)), 0.1)
        cost = 1.0 + 0.05 / width
        portal_id = str(portal.get("id"))
        adjacency.setdefault(first, []).append((second, portal_id, cost))
        adjacency.setdefault(second, []).append((first, portal_id, cost))
    queue: list[tuple[float, str]] = [(0.0, source_room_id)]
    distance = {source_room_id: 0.0}
    previous: dict[str, tuple[str, str]] = {}
    while queue:
        cost, room_id = heapq.heappop(queue)
        if cost > distance.get(room_id, math.inf):
            continue
        if room_id == receiver_room_id:
            break
        for neighbor, portal_id, edge_cost in adjacency.get(room_id, []):
            candidate = cost + edge_cost
            if candidate < distance.get(neighbor, math.inf):
                distance[neighbor] = candidate
                previous[neighbor] = (room_id, portal_id)
                heapq.heappush(queue, (candidate, neighbor))
    if receiver_room_id not in distance:
        return None
    rooms = [receiver_room_id]
    portal_ids: list[str] = []
    while rooms[-1] != source_room_id:
        prior_room, portal_id = previous[rooms[-1]]
        portal_ids.append(portal_id)
        rooms.append(prior_room)
    rooms.reverse()
    portal_ids.reverse()
    return rooms, portal_ids


def _room_visibility_path(
    start: np.ndarray,
    end: np.ndarray,
    corners: Sequence[Sequence[float]],
) -> list[np.ndarray]:
    polygon = Polygon(corners)
    if polygon.is_empty or not polygon.is_valid:
        return [np.asarray(start, dtype=float), np.asarray(end, dtype=float)]
    domain = polygon.buffer(1e-4, join_style=2)
    direct_line = LineString((start, end))
    if domain.covers(direct_line):
        return [np.asarray(start, dtype=float), np.asarray(end, dtype=float)]
    nodes = [np.asarray(start, dtype=float), np.asarray(end, dtype=float)]
    nodes.extend(np.asarray(point, dtype=float) for point in list(polygon.exterior.coords)[:-1])
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for first_index, first in enumerate(nodes):
        for second_index in range(first_index + 1, len(nodes)):
            second = nodes[second_index]
            line = LineString((first, second))
            if domain.covers(line):
                length = float(np.linalg.norm(second - first))
                adjacency[first_index].append((second_index, length))
                adjacency[second_index].append((first_index, length))
    queue: list[tuple[float, int]] = [(0.0, 0)]
    distance = {0: 0.0}
    previous: dict[int, int] = {}
    while queue:
        cost, node_index = heapq.heappop(queue)
        if node_index == 1:
            break
        if cost > distance.get(node_index, math.inf):
            continue
        for neighbor, edge_length in adjacency[node_index]:
            candidate = cost + edge_length
            if candidate < distance.get(neighbor, math.inf):
                distance[neighbor] = candidate
                previous[neighbor] = node_index
                heapq.heappush(queue, (candidate, neighbor))
    if 1 not in distance:
        return [np.asarray(start, dtype=float), np.asarray(end, dtype=float)]
    indices = [1]
    while indices[-1] != 0:
        indices.append(previous[indices[-1]])
    indices.reverse()
    return [nodes[index] for index in indices]


def _deduplicate_xy_points(points: Sequence[np.ndarray]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for point in points:
        value = np.asarray(point, dtype=float)
        if not result or float(np.linalg.norm(value - result[-1])) > 1e-6:
            result.append(value)
    return result


def _deduplicate_3d_points(points: Sequence[np.ndarray]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for point in points:
        value = np.asarray(point, dtype=float)
        if not result or float(np.linalg.norm(value - result[-1])) > 1e-6:
            result.append(value)
    return result


def _path_deviation_angle(points: Sequence[np.ndarray]) -> float:
    deviation = 0.0
    for index in range(1, len(points) - 1):
        incoming = points[index] - points[index - 1]
        outgoing = points[index + 1] - points[index]
        incoming /= max(float(np.linalg.norm(incoming)), 1e-12)
        outgoing /= max(float(np.linalg.norm(outgoing)), 1e-12)
        deviation += math.acos(float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0)))
    return deviation


def _apply_source_directivity_to_path(path: AcousticPath, model: Mapping[str, Any]) -> AcousticPath:
    if len(path.points) < 2:
        return path
    gain = source_directivity_gain(
        np.asarray(path.points[1], dtype=np.float64) - np.asarray(path.points[0], dtype=np.float64),
        model,
    )
    return AcousticPath(
        path.kind,
        path.distance_m,
        path.delay_s,
        float(path.gain) * gain,
        {band: float(value) * gain for band, value in path.band_gains.items()},
        path.points,
        {**dict(path.metadata), "source_directivity_gain": gain},
    )


def _band_array(material: Any, attr: str, default: float) -> np.ndarray:
    table = getattr(material, attr, {}) or {}
    return np.asarray([float(table.get(b, default)) for b in FREQUENCY_BANDS], dtype=float)


def _transmission(scene: RoomRayScene, listener: np.ndarray, source: np.ndarray, distance: float, config: SimConfig) -> np.ndarray:
    accumulated = np.ones(_NUM_BANDS, dtype=float)
    directions = ((source - listener) / distance, (listener - source) / distance)
    endpoints = (listener, source)
    min_distances = [0.0, 0.0]
    current = 0
    surface_names: set[str] = set()
    max_layers = max(1, int(config.num_transmission_rays))
    max_iterations = max_layers * 2 + 2
    for _ in range(max_iterations):
        origin = endpoints[current] + min_distances[current] * directions[current]
        hit = scene.closest_hit(origin, directions[current])
        absolute_distance = min_distances[current] + float(hit["distance"])
        if not hit["valid"] or absolute_distance >= distance:
            break
        surface_name = str(hit.get("surface") or "")
        if surface_name not in surface_names:
            accumulated *= np.clip(hit["transmission"], 0.0, 1.0)
            surface_names.add(surface_name)
            if len(surface_names) >= max_layers:
                break
        min_distances[current] = absolute_distance + 1e-2
        if min_distances[current] >= distance or sum(min_distances) >= distance:
            break
        current = 1 - current
    return accumulated


def _boundary_diffraction_paths(
    room: Room,
    scene: RoomRayScene,
    src: np.ndarray,
    rcv: np.ndarray,
    direct: dict[str, Any],
    config: SimConfig,
) -> list[AcousticPath]:
    if not config.diffraction_enabled or float(direct.get("occlusion", 1.0)) >= 1.0:
        return []
    occlusion_surface = str(direct.get("occlusion_surface") or "")
    if occlusion_surface.startswith("object_"):
        return _object_diffraction_paths(room, scene, src, rcv, direct, config)
    points = np.asarray(room.corners, dtype=float)
    reflex_vertices: list[tuple[int, np.ndarray]] = []
    for index, point in enumerate(points):
        incoming = point - points[index - 1]
        outgoing = points[(index + 1) % len(points)] - point
        if float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]) < -1e-9:
            reflex_vertices.append((index, point))

    direct_distance = max(float(direct["distance_m"]), 1e-9)
    paths: list[AcousticPath] = []
    max_order = min(max(1, int(config.diffraction_order)), max(1, len(reflex_vertices)))
    seen: set[tuple[int, ...]] = set()
    for order in range(1, max_order + 1):
        for sequence in permutations(reflex_vertices, order):
            edge_indices = tuple(int(item[0]) for item in sequence)
            if edge_indices in seen:
                continue
            seen.add(edge_indices)
            candidate = _diffraction_path_for_sequence(
                room,
                src,
                rcv,
                sequence,
                direct_distance,
                config,
            )
            if candidate is not None:
                paths.append(candidate)
    paths.sort(key=lambda path: (path.delay_s, -abs(path.gain)))
    return paths[: max(0, int(config.max_diffraction_paths))]


def _same_room_diffraction_room(room: Room, scene: RoomRayScene) -> Room:
    if not scene.is_multi_room or scene.is_cross_room:
        return room
    multi_room = room.metadata.get("multi_room") if isinstance(room.metadata, Mapping) else None
    if not isinstance(multi_room, Mapping):
        return room
    source_room_id = str(multi_room.get("source_room_id", ""))
    room_record = next(
        (
            item
            for item in multi_room.get("rooms", [])
            if isinstance(item, Mapping) and str(item.get("id", "")) == source_room_id
        ),
        None,
    )
    corners = room_record.get("corners") if isinstance(room_record, Mapping) else None
    if not isinstance(corners, Sequence) or len(corners) < 3:
        return room
    try:
        local_corners = tuple((float(point[0]), float(point[1])) for point in corners)
    except (IndexError, TypeError, ValueError):
        return room
    return Room(
        id=room.id,
        name=room.name,
        corners=local_corners,
        height_m=room.height_m,
        materials=room.materials,
        metadata=room.metadata,
    )


def _diffraction_skip_reason(direct: dict[str, Any], config: SimConfig, paths: Sequence[AcousticPath] = ()) -> str | None:
    if not config.diffraction_enabled:
        return "disabled"
    if float(direct.get("occlusion", 1.0)) >= 1.0:
        return "direct_path_visible"
    surface = str(direct.get("occlusion_surface") or "")
    if surface.startswith("object_") and not paths:
        return "object_edge_no_valid_path"
    return None


def _object_diffraction_paths(
    room: Room,
    scene: RoomRayScene,
    src: np.ndarray,
    rcv: np.ndarray,
    direct: dict[str, Any],
    config: SimConfig,
) -> list[AcousticPath]:
    occlusion_surface = str(direct.get("occlusion_surface") or "")
    object_index = _object_index_from_surface(occlusion_surface)
    object_part = _object_part_from_surface(occlusion_surface)
    raw_objects = room.metadata.get("objects", []) if isinstance(room.metadata, dict) else []
    if object_index is None or not isinstance(raw_objects, list) or object_index >= len(raw_objects):
        return []
    item = raw_objects[object_index]
    if not isinstance(item, dict):
        return []
    solid_body = _uses_solid_body_diffraction(item, object_part)
    geometry = _object_edge_geometry(item, float(room.height_m), object_part)
    if geometry is None:
        return []
    direct_distance = max(float(direct["distance_m"]), 1e-9)
    candidates = _object_diffraction_candidates(src, rcv, geometry, solid_body=solid_body)
    paths: list[AcousticPath] = []
    for candidate in candidates:
        path = _object_diffraction_path_for_candidate(
            room,
            scene,
            src,
            rcv,
            candidate,
            item,
            object_index,
            direct_distance,
            config,
        )
        if path is not None:
            paths.append(path)
    paths.sort(key=lambda path: (path.delay_s, -abs(path.gain)))
    if solid_body:
        paths = [path for path in paths if path.metadata.get("object_edge_type") == "side_pair"][:2]
    return paths[: max(0, int(config.max_diffraction_paths))]


def _object_index_from_surface(surface: str) -> int | None:
    parts = surface.split("_", 2)
    if len(parts) < 2 or parts[0] != "object":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _object_part_from_surface(surface: str) -> str | None:
    parts = surface.split("_", 3)
    return parts[3] if len(parts) >= 4 else None


def _uses_solid_body_diffraction(item: dict[str, Any], part: str | None) -> bool:
    object_type = str(item.get("type", "furniture"))
    return part in {None, "body"} and object_type in {"cuboid", "low_block", "cabinet", "bookshelf", "sofa", "door"}


def _object_edge_geometry(item: dict[str, Any], room_height: float, part: str | None = None) -> dict[str, Any] | None:
    if part:
        for box in _object_proxy_boxes(item, room_height):
            if str(box.get("part")) == part:
                return _box_edge_geometry(box, room_height)
    boxes = _object_proxy_boxes(item, room_height)
    if len(boxes) == 1:
        return _box_edge_geometry(boxes[0], room_height)
    try:
        return _box_edge_geometry(_object_proxy_boxes(item, room_height)[0], room_height)
    except Exception:
        return None


def _box_edge_geometry(box: dict[str, Any], room_height: float) -> dict[str, Any] | None:
    try:
        size = np.asarray(box["size"], dtype=float)
        center = np.asarray(box["center"], dtype=float)
        angle = math.radians(float(box["rotation"]))
        axis_u = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
        axis_v = np.asarray([-math.sin(angle), math.cos(angle)], dtype=float)
        half_w = float(size[0] * 0.5)
        half_d = float(size[1] * 0.5)
        half_h = float(size[2] * 0.5)
        z_center = float(box["z"])
    except Exception:
        return None
    z_min = float(np.clip(z_center - half_h, 0.0, room_height))
    z_max = float(np.clip(z_center + half_h, z_min + 1e-3, room_height))
    corners = [
        center + sx * axis_u * half_w + sy * axis_v * half_d
        for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    ]
    return {
        "center": center,
        "corners": corners,
        "z_min": z_min,
        "z_max": z_max,
    }


def _object_diffraction_candidates(src: np.ndarray, rcv: np.ndarray, geometry: dict[str, Any], *, solid_body: bool = False) -> list[dict[str, Any]]:
    corners = geometry["corners"]
    z_min = float(geometry["z_min"])
    z_max = float(geometry["z_max"])
    candidates: list[dict[str, Any]] = []
    if not solid_body:
        for index, corner in enumerate(corners):
            z = _shortest_z_between_bounds(src, rcv, corner, z_min, z_max)
            candidates.append({
                "edge_id": f"vertical_{index}",
                "edge_type": "vertical",
                "point": np.asarray([corner[0], corner[1], z], dtype=float),
            })
        for index in range(len(corners)):
            a = corners[index]
            b = corners[(index + 1) % len(corners)]
            midpoint = 0.5 * (a + b)
            candidates.append({
                "edge_id": f"top_{index}",
                "edge_type": "top",
                "point": np.asarray([midpoint[0], midpoint[1], z_max], dtype=float),
            })
    side_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for index, (a_index, b_index) in enumerate(side_pairs):
        a = corners[a_index]
        b = corners[b_index]
        midpoint = 0.5 * (a + b)
        z = _shortest_z_between_bounds(src, rcv, midpoint, z_min, z_max)
        p_a = np.asarray([a[0], a[1], z], dtype=float)
        p_b = np.asarray([b[0], b[1], z], dtype=float)
        a_first = np.linalg.norm(src - p_a) <= np.linalg.norm(src - p_b)
        first, second = (p_a, p_b) if a_first else (p_b, p_a)
        first_id, second_id = (a_index, b_index) if a_first else (b_index, a_index)
        candidates.append({
            "edge_id": f"side_{index}",
            "edge_ids": [f"vertical_{first_id}", f"vertical_{second_id}"],
            "edge_type": "side_pair",
            "points": [first, second],
        })
        if not solid_body:
            top_a = np.asarray([a[0], a[1], z_max], dtype=float)
            top_b = np.asarray([b[0], b[1], z_max], dtype=float)
            top_a_first = np.linalg.norm(src - top_a) <= np.linalg.norm(src - top_b)
            top_first, top_second = (top_a, top_b) if top_a_first else (top_b, top_a)
            top_first_id, top_second_id = (a_index, b_index) if top_a_first else (b_index, a_index)
            candidates.append({
                "edge_id": f"top_side_{index}",
                "edge_ids": [f"top_corner_{top_first_id}", f"top_corner_{top_second_id}"],
                "edge_type": "top_side_pair",
                "points": [top_first, top_second],
            })
    return candidates


def _object_diffraction_path_for_candidate(
    room: Room,
    scene: RoomRayScene,
    src: np.ndarray,
    rcv: np.ndarray,
    candidate: dict[str, Any],
    item: dict[str, Any],
    object_index: int,
    direct_distance: float,
    config: SimConfig,
) -> AcousticPath | None:
    raw_points = candidate["points"] if "points" in candidate else [candidate["point"]]
    diffraction_points = [np.asarray(point, dtype=float) for point in raw_points]
    route = [src, *diffraction_points, rcv]
    object_surface_prefix = f"object_{object_index}_"
    for start, end in zip(route, route[1:]):
        if float(np.linalg.norm(end - start)) <= 1e-9:
            return None
        if not _segment_inside_room(start, end, room):
            return None
    if not _segment_clear_to_diffraction_edge(scene, src, diffraction_points[0], object_surface_prefix):
        return None
    if not _segment_clear_from_diffraction_edge(scene, diffraction_points[-1], rcv):
        return None
    legs = [route[i + 1] - route[i] for i in range(len(route) - 1)]
    lengths = [float(np.linalg.norm(leg)) for leg in legs]
    if min(lengths) <= 1e-9:
        return None
    directions = [leg / length for leg, length in zip(legs, lengths)]
    deviations = [
        max(1e-8, float(np.arccos(np.clip(np.dot(directions[i], directions[i + 1]), -1.0, 1.0))))
        for i in range(len(directions) - 1)
    ]
    distance_m = float(sum(lengths))
    if distance_m <= direct_distance + 1e-6:
        return None
    bands = multiply_bands(
        propagation_band_gains(distance_m, min_distance_m=config.min_distance_m),
        steam_audio_pathing_deviation(sum(deviations)),
    )
    order = len(diffraction_points)
    return AcousticPath(
        "diffraction",
        distance_m,
        distance_m / float(config.c),
        band_mean(bands),
        bands,
        tuple(tuple(float(v) for v in p) for p in route),
        {
            "model": "steam_audio_utd_object_edge_approx",
            "diffraction_order": int(order),
            "object_index": int(object_index),
            "object_id": str(item.get("id", f"object_{object_index}")),
            "object_type": str(item.get("type", "furniture")),
            "object_edge_id": str(candidate["edge_id"]),
            "object_edge_ids": list(candidate.get("edge_ids", [str(candidate["edge_id"])])),
            "object_edge_type": str(candidate["edge_type"]),
            "deviation_angle_rad": round(float(sum(deviations)), 7),
            "per_edge_deviation_rad": [round(float(deviation), 7) for deviation in deviations],
            "deviation_application": "total_angle_reference_normalized",
            "path_excess_m": round(distance_m - direct_distance, 6),
            "contributes_to_rir": bool(config.diffraction_audio_enabled),
        },
    )


def _segment_clear_to_diffraction_edge(scene: RoomRayScene, start: np.ndarray, edge_point: np.ndarray, object_surface_prefix: str) -> bool:
    delta = edge_point - start
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        return False
    direction = delta / distance
    hit = scene.closest_hit(start, direction)
    if not hit["valid"] or float(hit["distance"]) >= distance - 0.035:
        return True
    surface = str(hit.get("surface") or "")
    return surface.startswith(object_surface_prefix) and float(hit["distance"]) >= distance - 0.08


def _segment_clear_from_diffraction_edge(scene: RoomRayScene, edge_point: np.ndarray, end: np.ndarray) -> bool:
    delta = end - edge_point
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        return False
    direction = delta / distance
    offset = min(_HIT_OFFSET * 2.0, max(distance * 0.25, 1e-4))
    if distance <= offset + 1e-6:
        return True
    origin = edge_point + direction * offset
    hit = scene.closest_hit(origin, direction)
    return not (hit["valid"] and float(hit["distance"]) < distance - offset - 0.035)


def _diffraction_path_for_sequence(
    room: Room,
    src: np.ndarray,
    rcv: np.ndarray,
    sequence: Sequence[tuple[int, np.ndarray]],
    direct_distance: float,
    config: SimConfig,
) -> AcousticPath | None:
    diffraction_points = _diffraction_points_for_sequence(src, rcv, sequence, float(room.height_m))
    route = [src, *diffraction_points, rcv]
    for start, end in zip(route, route[1:]):
        if float(np.linalg.norm(end - start)) <= 1e-9:
            return None
        if not _segment_inside_room(start, end, room):
            return None

    legs = [route[i + 1] - route[i] for i in range(len(route) - 1)]
    lengths = [float(np.linalg.norm(leg)) for leg in legs]
    if min(lengths) <= 1e-9:
        return None
    directions = [leg / length for leg, length in zip(legs, lengths)]
    deviations = [
        max(1e-8, float(np.arccos(np.clip(np.dot(directions[i], directions[i + 1]), -1.0, 1.0))))
        for i in range(len(directions) - 1)
    ]
    distance_m = float(sum(lengths))
    bands = multiply_bands(
        propagation_band_gains(distance_m, min_distance_m=config.min_distance_m),
        steam_audio_pathing_deviation(sum(deviations)),
    )
    edge_indices = [int(item[0]) for item in sequence]
    order = len(edge_indices)
    return AcousticPath(
        "diffraction",
        distance_m,
        distance_m / float(config.c),
        band_mean(bands),
        bands,
        tuple(tuple(float(v) for v in point) for point in route),
        {
            "model": "steam_audio_utd_multi_edge" if order > 1 else "steam_audio_utd_deviation",
            "diffraction_order": int(order),
            "boundary_edge_index": edge_indices[0] if order == 1 else None,
            "boundary_edge_indices": edge_indices,
            "deviation_angle_rad": round(float(sum(deviations)), 7),
            "per_edge_deviation_rad": [round(float(angle), 7) for angle in deviations],
            "deviation_application": "total_angle_reference_normalized",
            "path_excess_m": round(distance_m - direct_distance, 6),
            "contributes_to_rir": bool(config.diffraction_audio_enabled),
        },
    )


def _diffraction_points_for_sequence(
    src: np.ndarray,
    rcv: np.ndarray,
    sequence: Sequence[tuple[int, np.ndarray]],
    height: float,
) -> list[np.ndarray]:
    vertices = [np.asarray(item[1], dtype=float) for item in sequence]
    xy_points = [src[:2], *vertices, rcv[:2]]
    horizontal_lengths = [
        float(np.linalg.norm(np.asarray(xy_points[index + 1]) - np.asarray(xy_points[index])))
        for index in range(len(xy_points) - 1)
    ]
    total = max(float(sum(horizontal_lengths)), 1e-9)
    cumulative = 0.0
    out: list[np.ndarray] = []
    for index, vertex in enumerate(vertices):
        cumulative += horizontal_lengths[index]
        ratio = cumulative / total
        z_value = float(src[2] + (rcv[2] - src[2]) * ratio)
        out.append(np.asarray([vertex[0], vertex[1], np.clip(z_value, 0.0, height)], dtype=float))
    return out


def _shortest_vertical_edge_z(src: np.ndarray, rcv: np.ndarray, height: float, vertex: np.ndarray) -> float:
    lo, hi = 0.0, float(height)
    for _ in range(36):
        third = (hi - lo) / 3.0
        z1, z2 = lo + third, hi - third
        p1 = np.asarray([vertex[0], vertex[1], z1])
        p2 = np.asarray([vertex[0], vertex[1], z2])
        d1 = float(np.linalg.norm(src - p1) + np.linalg.norm(rcv - p1))
        d2 = float(np.linalg.norm(src - p2) + np.linalg.norm(rcv - p2))
        if d1 <= d2:
            hi = z2
        else:
            lo = z1
    return 0.5 * (lo + hi)


def _shortest_z_between_bounds(src: np.ndarray, rcv: np.ndarray, xy: np.ndarray, z_min: float, z_max: float) -> float:
    lo, hi = float(z_min), float(z_max)
    if hi <= lo + 1e-6:
        return lo
    for _ in range(36):
        third = (hi - lo) / 3.0
        z1, z2 = lo + third, hi - third
        p1 = np.asarray([xy[0], xy[1], z1])
        p2 = np.asarray([xy[0], xy[1], z2])
        d1 = float(np.linalg.norm(src - p1) + np.linalg.norm(rcv - p1))
        d2 = float(np.linalg.norm(src - p2) + np.linalg.norm(rcv - p2))
        if d1 <= d2:
            hi = z2
        else:
            lo = z1
    return float(np.clip(0.5 * (lo + hi), z_min, z_max))


def _segment_inside_room(start: np.ndarray, end: np.ndarray, room: Room) -> bool:
    if min(float(start[2]), float(end[2])) < -1e-6 or max(float(start[2]), float(end[2])) > float(room.height_m) + 1e-6:
        return False
    start_xy = np.asarray(start[:2], dtype=float)
    end_xy = np.asarray(end[:2], dtype=float)
    delta = end_xy - start_xy
    length_sq = float(np.dot(delta, delta))
    if length_sq <= 1e-18:
        return point_in_polygon(start_xy, room.corners)

    breakpoints = [0.0, 1.0]
    corners = np.asarray(room.corners, dtype=float)
    for edge_start, edge_end in zip(corners, np.roll(corners, -1, axis=0)):
        edge = edge_end - edge_start
        rel = edge_start - start_xy
        denominator = float(delta[0] * edge[1] - delta[1] * edge[0])
        if abs(denominator) > 1e-12:
            t = float((rel[0] * edge[1] - rel[1] * edge[0]) / denominator)
            u = float((rel[0] * delta[1] - rel[1] * delta[0]) / denominator)
            if -1e-10 <= t <= 1.0 + 1e-10 and -1e-10 <= u <= 1.0 + 1e-10:
                breakpoints.append(float(np.clip(t, 0.0, 1.0)))
        else:
            collinear = abs(float(rel[0] * delta[1] - rel[1] * delta[0])) <= 1e-10
            if collinear:
                breakpoints.extend([
                    float(np.clip(np.dot(edge_start - start_xy, delta) / length_sq, 0.0, 1.0)),
                    float(np.clip(np.dot(edge_end - start_xy, delta) / length_sq, 0.0, 1.0)),
                ])

    ordered = sorted(breakpoints)
    unique = [ordered[0]]
    for value in ordered[1:]:
        if value - unique[-1] > 1e-10:
            unique.append(value)
    for left, right in zip(unique, unique[1:]):
        if right - left <= 1e-10:
            continue
        midpoint = start_xy + (0.5 * (left + right)) * delta
        if not point_in_polygon(midpoint, room.corners):
            return False
    return True


@lru_cache(maxsize=8)
def _sphere_samples(n: int, seed: int) -> np.ndarray:
    n = max(1, int(n))
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = i * (math.pi * (3.0 - math.sqrt(5.0))) + rng.uniform(-0.02, 0.02, n)
    samples = np.stack([np.cos(theta) * r, np.sin(theta) * r, z], axis=1)
    samples.setflags(write=False)
    return samples


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def _ray_hits_sphere_before(
    origins: np.ndarray,
    directions: np.ndarray,
    center: np.ndarray,
    radius: float,
    max_distance: np.ndarray,
) -> np.ndarray:
    offset = origins - center[None, :]
    b = np.sum(offset * directions, axis=1)
    c = np.sum(offset * offset, axis=1) - radius * radius
    discriminant = b * b - c
    root = -b - np.sqrt(np.clip(discriminant, 0.0, None))
    return (discriminant >= 0.0) & (root >= 0.0) & (root < max_distance)


@lru_cache(maxsize=16)
def _diffuse_sample_bank(count: int) -> np.ndarray:
    count = max(1, int(count))
    i = np.arange(count, dtype=np.uint64)
    bits = i.copy()
    inverse = np.zeros(count, dtype=np.float64)
    factor = 0.5
    while np.any(bits):
        inverse += factor * (bits & 1)
        bits >>= 1
        factor *= 0.5
    u = (np.arange(count, dtype=np.float64) + 0.5) / count
    r = np.sqrt(u)
    theta = 2.0 * math.pi * inverse
    samples = np.stack([r * np.cos(theta), r * np.sin(theta), np.sqrt(np.clip(1.0 - u, 0.0, 1.0))], axis=1)
    samples.setflags(write=False)
    return samples


def _diffuse_random_sequence(
    num_rays: int,
    num_bounces: int,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    global _RANDOM_WORKSPACE_BYTES
    key = (int(num_rays), int(num_bounces), int(sample_count), int(seed))
    with _STATIC_CACHE_LOCK:
        cached = _RANDOM_WORKSPACE_CACHE.get(key)
        if cached is not None:
            _RANDOM_WORKSPACE_CACHE.move_to_end(key)
            _STATIC_CACHE_STATS["workspace_hits"] += 1
            return cached
        _STATIC_CACHE_STATS["workspace_misses"] += 1
    rng = np.random.default_rng(int(seed) + 1)
    random_values = np.empty((num_bounces, num_rays), dtype=np.float64)
    sample_indices = np.empty((num_bounces, num_rays), dtype=np.int32)
    for bounce in range(num_bounces):
        random_values[bounce] = rng.random(num_rays)
        sample_indices[bounce] = rng.integers(0, sample_count, size=num_rays, dtype=np.int32)
    random_values.setflags(write=False)
    sample_indices.setflags(write=False)
    workspace = (random_values, sample_indices)
    workspace_bytes = int(random_values.nbytes + sample_indices.nbytes)
    if workspace_bytes <= _WORKSPACE_CACHE_BYTES:
        with _STATIC_CACHE_LOCK:
            previous = _RANDOM_WORKSPACE_CACHE.pop(key, None)
            if previous is not None:
                _RANDOM_WORKSPACE_BYTES -= int(previous[0].nbytes + previous[1].nbytes)
            while _RANDOM_WORKSPACE_CACHE and _RANDOM_WORKSPACE_BYTES + workspace_bytes > _WORKSPACE_CACHE_BYTES:
                _, expired = _RANDOM_WORKSPACE_CACHE.popitem(last=False)
                _RANDOM_WORKSPACE_BYTES -= int(expired[0].nbytes + expired[1].nbytes)
            _RANDOM_WORKSPACE_CACHE[key] = workspace
            _RANDOM_WORKSPACE_BYTES += workspace_bytes
    return workspace


def _as_precision_array(value: np.ndarray, dtype: Any) -> np.ndarray:
    global _PRECISION_ARRAY_BYTES
    source = np.asarray(value)
    target_dtype = np.dtype(dtype)
    if source.dtype == target_dtype and source.flags.c_contiguous:
        return source
    key = (
        int(source.__array_interface__["data"][0]),
        source.shape,
        source.strides,
        source.dtype.str,
        target_dtype.str,
    )
    with _STATIC_CACHE_LOCK:
        cached = _PRECISION_ARRAY_CACHE.get(key)
        if cached is not None and cached[0] is value:
            _PRECISION_ARRAY_CACHE.move_to_end(key)
            _STATIC_CACHE_STATS["precision_hits"] += 1
            return cached[1]

    converted = np.ascontiguousarray(value, dtype=target_dtype)
    converted.setflags(write=False)
    size = int(converted.nbytes)
    with _STATIC_CACHE_LOCK:
        _STATIC_CACHE_STATS["precision_misses"] += 1
        previous = _PRECISION_ARRAY_CACHE.pop(key, None)
        if previous is not None:
            _PRECISION_ARRAY_BYTES -= previous[2]
        while _PRECISION_ARRAY_CACHE and _PRECISION_ARRAY_BYTES + size > _PRECISION_CACHE_BYTES:
            _, expired = _PRECISION_ARRAY_CACHE.popitem(last=False)
            _PRECISION_ARRAY_BYTES -= expired[2]
        if size <= _PRECISION_CACHE_BYTES:
            _PRECISION_ARRAY_CACHE[key] = (value, converted, size)
            _PRECISION_ARRAY_BYTES += size
    return converted


def _clear_static_caches() -> None:
    global _PRECISION_ARRAY_BYTES, _RANDOM_WORKSPACE_BYTES
    with _STATIC_CACHE_LOCK:
        _SCENE_SURFACE_CACHE.clear()
        _SCENE_ARRAY_CACHE.clear()
        _RANDOM_WORKSPACE_CACHE.clear()
        _PRECISION_ARRAY_CACHE.clear()
        _RANDOM_WORKSPACE_BYTES = 0
        _PRECISION_ARRAY_BYTES = 0
        for key in _STATIC_CACHE_STATS:
            _STATIC_CACHE_STATS[key] = 0
    _sphere_samples.cache_clear()
    _diffuse_sample_bank.cache_clear()


def _static_cache_info() -> dict[str, int]:
    with _STATIC_CACHE_LOCK:
        return {
            **{key: int(value) for key, value in _STATIC_CACHE_STATS.items()},
            "scene_entries": len(_SCENE_SURFACE_CACHE),
            "array_entries": len(_SCENE_ARRAY_CACHE),
            "workspace_entries": len(_RANDOM_WORKSPACE_CACHE),
            "workspace_bytes": int(_RANDOM_WORKSPACE_BYTES),
            "workspace_byte_limit": int(_WORKSPACE_CACHE_BYTES),
            "precision_entries": len(_PRECISION_ARRAY_CACHE),
            "precision_bytes": int(_PRECISION_ARRAY_BYTES),
            "precision_byte_limit": int(_PRECISION_CACHE_BYTES),
            "sphere_entries": int(_sphere_samples.cache_info().currsize),
            "diffuse_bank_entries": int(_diffuse_sample_bank.cache_info().currsize),
        }


def _transform_hemisphere(samples: np.ndarray, normals: np.ndarray) -> np.ndarray:
    helper = np.where((np.abs(normals[:, 0]) < 0.9)[:, None], np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    tangent = _normalize_rows(np.cross(helper, normals))
    bitangent = np.cross(normals, tangent)
    return _normalize_rows(samples[:, 0, None] * tangent + samples[:, 1, None] * bitangent + samples[:, 2, None] * normals)


def _band_edges(fs: int) -> list[tuple[float, float]]:
    nyquist = fs * 0.5
    inner = [176.78, 353.55, 707.11, 1414.2, 2828.4]
    bounds = [0.0, *inner, nyquist]
    return [(bounds[i], bounds[i + 1]) for i in range(_NUM_BANDS)]


if njit is not None:

    @njit(cache=True)
    def _biquad_filter_jit(values, b0, b1, b2, a1, a2):
        out = np.zeros(values.shape[0], dtype=np.float32)
        x1 = 0.0
        x2 = 0.0
        y1 = 0.0
        y2 = 0.0
        for index in range(values.shape[0]):
            x0 = values[index]
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[index] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0
        return out


def _band_limit(signal: np.ndarray, lo: float, hi: float, fs: int, *, mode: str = "bandpass") -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.astype(np.float32)
    sampling_rate = max(1, int(fs))
    nyquist = 0.5 * sampling_rate
    q = 0.707
    if mode == "lowpass":
        cutoff = float(np.clip(float(hi), 1e-3, nyquist * 0.999))
        out = values
        order = 8
        for section in range(order // 2):
            section_q = 1.0 / (2.0 * math.cos((2.0 * section + 1.0) * math.pi / (2.0 * order)))
            w0 = 2.0 * math.pi * cutoff / sampling_rate
            cw0, sw0 = math.cos(w0), math.sin(w0)
            alpha = sw0 / (2.0 * section_q)
            a0 = 1.0 + alpha
            b0 = ((1.0 - cw0) / 2.0) / a0
            b1 = (1.0 - cw0) / a0
            b2 = b0
            a1 = (-2.0 * cw0) / a0
            a2 = (1.0 - alpha) / a0
            out = _apply_biquad(out, b0, b1, b2, a1, a2)
        return out
    elif mode == "highpass":
        cutoff = float(np.clip(float(lo), 1e-3, nyquist * 0.999))
        w0 = 2.0 * math.pi * cutoff / sampling_rate
        cw0, sw0 = math.cos(w0), math.sin(w0)
        alpha = sw0 / (2.0 * q)
        a0 = 1.0 + alpha
        b0 = ((1.0 + cw0) / 2.0) / a0
        b1 = (-(1.0 + cw0)) / a0
        b2 = b0
        a1 = (-2.0 * cw0) / a0
        a2 = (1.0 - alpha) / a0
    else:
        lower = float(np.clip(float(lo), 1e-3, nyquist * 0.998))
        upper = float(np.clip(float(hi), lower + 1e-3, nyquist * 0.999))
        cutoff = math.sqrt(max(lower * upper, 1e-12))
        q_inverse = (upper - lower) / cutoff
        w0 = 2.0 * math.pi * cutoff / sampling_rate
        cw0, sw0 = math.cos(w0), math.sin(w0)
        alpha = sw0 * q_inverse / 2.0
        a0 = 1.0 + alpha
        b0 = alpha / a0
        b1 = 0.0
        b2 = -alpha / a0
        a1 = (-2.0 * cw0) / a0
        a2 = (1.0 - alpha) / a0

    return _apply_biquad(values, b0, b1, b2, a1, a2)


def _apply_biquad(values: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
    if njit is not None:
        return _biquad_filter_jit(np.asarray(values, dtype=np.float64), b0, b1, b2, a1, a2)

    out = np.zeros_like(values, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for index, x0 in enumerate(values):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[index] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return out.astype(np.float32)


def _volumetric_occlusion(scene: RoomRayScene, listener: np.ndarray, source: np.ndarray, config: SimConfig) -> float:
    count = max(1, int(config.direct_occlusion_samples))
    radius = max(1e-4, float(config.direct_occlusion_radius_m))
    samples = _sphere_volume_samples(count) * radius + source[None, :]
    visible = 0
    valid = 0
    for sample in samples:
        source_leg = sample - source
        source_distance = float(np.linalg.norm(source_leg))
        if source_distance > 1e-9 and scene.any_hit(source, source_leg / source_distance, source_distance):
            continue
        listener_leg = sample - listener
        listener_distance = float(np.linalg.norm(listener_leg))
        valid += 1
        if listener_distance <= 1e-9 or not scene.any_hit(listener, listener_leg / max(listener_distance, 1e-9), listener_distance):
            visible += 1
    return float(visible / valid) if valid else 0.0


def _sphere_volume_samples(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=float) + 0.5
    phi = 2.0 * math.pi * np.mod(indices * 0.6180339887498949, 1.0)
    cos_theta = 1.0 - 2.0 * indices / count
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))
    radius = np.power(indices / count, 1.0 / 3.0)
    return np.stack([radius * sin_theta * np.cos(phi), radius * sin_theta * np.sin(phi), radius * cos_theta], axis=1)


def _ray_sphere_intersection(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float) -> float | None:
    rel = origin - center
    b = 2.0 * float(np.dot(direction, rel))
    c = float(np.dot(rel, rel) - radius * radius)
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    sqrt_disc = float(np.sqrt(disc))
    candidates = [(-b - sqrt_disc) * 0.5, (-b + sqrt_disc) * 0.5]
    positives = [t for t in candidates if t > 1e-6]
    return min(positives) if positives else None


if njit is not None:

    @njit(cache=True)
    def _point_in_polygon_jit(x, y, corners):
        inside = False
        j = corners.shape[0] - 1
        for i in range(corners.shape[0]):
            xi = corners[i, 0]
            yi = corners[i, 1]
            xj = corners[j, 0]
            yj = corners[j, 1]
            if (yi > y) != (yj > y):
                denom = yj - yi
                if abs(denom) > 1e-12:
                    x_cross = (xj - xi) * (y - yi) / denom + xi
                    if x < x_cross:
                        inside = not inside
            j = i
        return inside


    @njit(cache=True)
    def _box_hit_jit(origin, direction, center, axis_u, axis_v, half, z_range):
        relx = origin[0] - center[0]
        rely = origin[1] - center[1]
        ox = relx * axis_u[0] + rely * axis_u[1]
        oy = relx * axis_v[0] + rely * axis_v[1]
        oz = origin[2]
        dx = direction[0] * axis_u[0] + direction[1] * axis_u[1]
        dy = direction[0] * axis_v[0] + direction[1] * axis_v[1]
        dz = direction[2]
        t_min = -1.0e30
        t_max = 1.0e30
        nx = 0.0
        ny = 0.0
        nz = 0.0
        ex_nx = 0.0
        ex_ny = 0.0
        ex_nz = 0.0
        for axis_i in range(3):
            if axis_i == 0:
                o = ox
                d = dx
                lo = -half[0]
                hi = half[0]
                ax = axis_u[0]
                ay = axis_u[1]
                az = 0.0
            elif axis_i == 1:
                o = oy
                d = dy
                lo = -half[1]
                hi = half[1]
                ax = axis_v[0]
                ay = axis_v[1]
                az = 0.0
            else:
                o = oz
                d = dz
                lo = z_range[0]
                hi = z_range[1]
                ax = 0.0
                ay = 0.0
                az = 1.0
            if abs(d) <= 1.0e-12:
                if o < lo or o > hi:
                    return 1.0e30, 0.0, 0.0, 1.0
                continue
            if d > 0.0:
                enter = (lo - o) / d
                exit_ = (hi - o) / d
                en_x = -ax
                en_y = -ay
                en_z = -az
                out_x = ax
                out_y = ay
                out_z = az
            else:
                enter = (hi - o) / d
                exit_ = (lo - o) / d
                en_x = ax
                en_y = ay
                en_z = az
                out_x = -ax
                out_y = -ay
                out_z = -az
            if enter > t_min:
                t_min = enter
                nx = en_x
                ny = en_y
                nz = en_z
            if exit_ < t_max:
                t_max = exit_
                ex_nx = out_x
                ex_ny = out_y
                ex_nz = out_z
            if t_min > t_max:
                return 1.0e30, 0.0, 0.0, 1.0
        if t_max <= _EPS:
            return 1.0e30, 0.0, 0.0, 1.0
        if t_min > _EPS:
            return t_min, nx, ny, nz
        return t_max, ex_nx, ex_ny, ex_nz


    @njit(cache=True, inline="always")
    def _surface_hit_jit(si, origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners):
        t = 1e30
        surf_nx = normals[si, 0]
        surf_ny = normals[si, 1]
        surf_nz = normals[si, 2]
        if kinds[si] == 0:
            sx = wall_delta[si, 0]
            sy = wall_delta[si, 1]
            det = sx * direction[1] - sy * direction[0]
            if abs(det) > 1e-12:
                relx = origin[0] - wall_a[si, 0]
                rely = origin[1] - wall_a[si, 1]
                cand_t = (relx * sy - rely * sx) / det
                u = (relx * direction[1] - rely * direction[0]) / det
                z = origin[2] + cand_t * direction[2]
                if cand_t > _EPS and u >= -1e-6 and u <= 1.0 + 1e-6 and z >= wall_z[si, 0] - 1e-6 and z <= wall_z[si, 1] + 1e-6:
                    t = cand_t
        elif kinds[si] == 2:
            cand_t, cand_nx, cand_ny, cand_nz = _box_hit_jit(origin, direction, box_center[si], box_axis_u[si], box_axis_v[si], box_half[si], box_z[si])
            if cand_t > _EPS and cand_t < 1.0e29:
                t = cand_t
                surf_nx = cand_nx
                surf_ny = cand_ny
                surf_nz = cand_nz
        elif abs(direction[2]) > 1e-12:
            cand_t = (z_values[si] - origin[2]) / direction[2]
            px = origin[0] + cand_t * direction[0]
            py = origin[1] + cand_t * direction[1]
            if cand_t > _EPS and _point_in_polygon_jit(px, py, corners):
                t = cand_t
        return t, surf_nx, surf_ny, surf_nz


    @njit(cache=True, inline="always")
    def _orient_hit_normal_jit(best_surface, best_t, nx, ny, nz, direction):
        if best_surface >= 0:
            dot = nx * direction[0] + ny * direction[1] + nz * direction[2]
            if dot > 0.0:
                nx = -nx
                ny = -ny
                nz = -nz
        return best_surface, best_t, nx, ny, nz


    @njit(cache=True, inline="always")
    def _ray_aabb_intersects_jit(origin, direction, lower, upper, max_distance):
        enter = 0.0
        exit_ = max_distance + _BVH_BOUNDS_EPS
        for axis in range(3):
            value = direction[axis]
            if abs(value) <= 1e-15:
                if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                    return False
                continue
            first = (lower[axis] - origin[axis]) / value
            second = (upper[axis] - origin[axis]) / value
            if first > second:
                first, second = second, first
            if first > enter:
                enter = first
            if second < exit_:
                exit_ = second
            if enter > exit_:
                return False
        return exit_ > _EPS


    @njit(cache=True)
    def _closest_hit_jit(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, height):
        best_t = 1e30
        best_surface = -1
        best_nx = 0.0
        best_ny = 0.0
        best_nz = 0.0
        for si in range(kinds.shape[0]):
            t = 1e30
            surf_nx = normals[si, 0]
            surf_ny = normals[si, 1]
            surf_nz = normals[si, 2]
            if kinds[si] == 0:
                sx = wall_delta[si, 0]
                sy = wall_delta[si, 1]
                det = sx * direction[1] - sy * direction[0]
                if abs(det) > 1e-12:
                    relx = origin[0] - wall_a[si, 0]
                    rely = origin[1] - wall_a[si, 1]
                    cand_t = (relx * sy - rely * sx) / det
                    u = (relx * direction[1] - rely * direction[0]) / det
                    z = origin[2] + cand_t * direction[2]
                    if cand_t > _EPS and u >= -1e-6 and u <= 1.0 + 1e-6 and z >= wall_z[si, 0] - 1e-6 and z <= wall_z[si, 1] + 1e-6:
                        t = cand_t
            elif kinds[si] == 2:
                cand_t, cand_nx, cand_ny, cand_nz = _box_hit_jit(origin, direction, box_center[si], box_axis_u[si], box_axis_v[si], box_half[si], box_z[si])
                if cand_t > _EPS and cand_t < 1.0e29:
                    t = cand_t
                    surf_nx = cand_nx
                    surf_ny = cand_ny
                    surf_nz = cand_nz
            elif abs(direction[2]) > 1e-12:
                cand_t = (z_values[si] - origin[2]) / direction[2]
                px = origin[0] + cand_t * direction[0]
                py = origin[1] + cand_t * direction[1]
                if cand_t > _EPS and _point_in_polygon_jit(px, py, corners):
                    t = cand_t
            if t < best_t:
                best_t = t
                best_surface = si
                best_nx = surf_nx
                best_ny = surf_ny
                best_nz = surf_nz
        if best_surface >= 0:
            dot = best_nx * direction[0] + best_ny * direction[1] + best_nz * direction[2]
            if dot > 0.0:
                best_nx = -best_nx
                best_ny = -best_ny
                best_nz = -best_nz
        return best_surface, best_t, best_nx, best_ny, best_nz


    @njit(cache=True, inline="always")
    def _closest_hit_bvh_jit(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives):
        best_t = 1e30
        best_surface = -1
        best_nx = 0.0
        best_ny = 0.0
        best_nz = 0.0
        node = 0
        while node < bvh_start.shape[0]:
            if not _ray_aabb_intersects_jit(origin, direction, bvh_bounds_min[node], bvh_bounds_max[node], best_t):
                node = bvh_escape[node]
                continue
            count = bvh_count[node]
            if count > 0:
                start = bvh_start[node]
                for offset in range(count):
                    si = bvh_primitives[start + offset]
                    t, surf_nx, surf_ny, surf_nz = _surface_hit_jit(si, origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners)
                    if t < best_t or (t == best_t and (best_surface < 0 or si < best_surface)):
                        best_t = t
                        best_surface = si
                        best_nx = surf_nx
                        best_ny = surf_ny
                        best_nz = surf_nz
            node += 1
        return _orient_hit_normal_jit(best_surface, best_t, best_nx, best_ny, best_nz, direction)


    @njit(cache=True, inline="always")
    def _closest_hit_backend_jit(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives, use_bvh):
        if use_bvh:
            return _closest_hit_bvh_jit(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives)
        return _closest_hit_jit(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, 0.0)


    @njit(cache=True)
    def _any_hit_jit(origin, direction, max_distance, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives, use_bvh):
        surf, t, nx, ny, nz = _closest_hit_backend_jit(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives, use_bvh)
        return surf >= 0 and t > _EPS and t < max_distance - _EPS


    @njit(cache=True)
    def _ray_sphere_before_jit(origin, direction, cx, cy, cz, radius, max_distance):
        ox = origin[0] - cx
        oy = origin[1] - cy
        oz = origin[2] - cz
        b = ox * direction[0] + oy * direction[1] + oz * direction[2]
        c = ox * ox + oy * oy + oz * oz - radius * radius
        disc = b * b - c
        if disc < 0.0:
            return False
        root = -b - math.sqrt(max(disc, 0.0))
        return root >= 0.0 and root < max_distance


    @njit(cache=True)
    def _ray_sphere_intersection_jit(origin, direction, center, radius):
        relx = origin[0] - center[0]
        rely = origin[1] - center[1]
        relz = origin[2] - center[2]
        b = 2.0 * (direction[0] * relx + direction[1] * rely + direction[2] * relz)
        c = relx * relx + rely * rely + relz * relz - radius * radius
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return -1.0
        root = math.sqrt(disc)
        t0 = (-b - root) * 0.5
        t1 = (-b + root) * 0.5
        if t0 > 1e-6 and (t0 <= t1 or t1 <= 1e-6):
            return t0
        if t1 > 1e-6:
            return t1
        return -1.0


    @njit(cache=True)
    def _normalize3_jit(x, y, z):
        n = math.sqrt(x * x + y * y + z * z)
        if n <= 1e-12:
            return 0.0, 0.0, 0.0
        return x / n, y / n, z / n


    @njit(cache=True)
    def _diffuse_direction_jit(sample, nx, ny, nz):
        hx = 1.0
        hy = 0.0
        hz = 0.0
        if abs(nx) >= 0.9:
            hx = 0.0
            hy = 1.0
        tx = hy * nz - hz * ny
        ty = hz * nx - hx * nz
        tz = hx * ny - hy * nx
        tx, ty, tz = _normalize3_jit(tx, ty, tz)
        bx = ny * tz - nz * ty
        by = nz * tx - nx * tz
        bz = nx * ty - ny * tx
        dx = sample[0] * tx + sample[1] * bx + sample[2] * nx
        dy = sample[0] * ty + sample[1] * by + sample[2] * ny
        dz = sample[0] * tz + sample[1] * bz + sample[2] * nz
        return _normalize3_jit(dx, dy, dz)


    @njit(cache=True)
    def _surface_survival_kernel(absorption):
        survival = np.empty(absorption.shape[0], dtype=np.float64)
        for si in range(absorption.shape[0]):
            value = 0.0
            for bi in range(absorption.shape[1]):
                value += math.sqrt(max(0.0, 1.0 - absorption[si, bi]))
            survival[si] = value / absorption.shape[1]
        return survival


    @njit(parallel=True, cache=True)
    def _visual_event_count_kernel(src, rcv, directions, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives, use_bvh, surface_survival, corners, height, receiver_radius, max_path_len, max_bounces):
        event_flags = np.zeros((directions.shape[0], max_bounces + 1), dtype=np.bool_)
        for ri in prange(directions.shape[0]):
            origin = src.copy()
            direction = directions[ri].copy()
            distance_so_far = 0.0
            for bounce in range(max_bounces + 1):
                surf, t, nx, ny, nz = _closest_hit_backend_jit(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives, use_bvh)
                receiver_t = _ray_sphere_intersection_jit(origin, direction, rcv, receiver_radius)
                max_t = t if surf >= 0 else 1e30
                if receiver_t > 0.0 and receiver_t < max_t and bounce > 0:
                    event_flags[ri, bounce] = True
                if surf < 0:
                    break
                e = surface_survival[surf]
                if e < 1e-5:
                    break
                hx = origin[0] + t * direction[0]
                hy = origin[1] + t * direction[1]
                hz = origin[2] + t * direction[2]
                dot = direction[0] * nx + direction[1] * ny + direction[2] * nz
                direction[0] = direction[0] - 2.0 * dot * nx
                direction[1] = direction[1] - 2.0 * dot * ny
                direction[2] = direction[2] - 2.0 * dot * nz
                direction[0], direction[1], direction[2] = _normalize3_jit(direction[0], direction[1], direction[2])
                distance_so_far += t
                if distance_so_far > max_path_len:
                    break
                origin[0] = hx + 1e-5 * direction[0]
                origin[1] = hy + 1e-5 * direction[1]
                origin[2] = hz + 1e-5 * direction[2]
        return event_flags


    @njit(parallel=True, cache=True)
    def _visual_record_kernel(src, rcv, directions, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives, use_bvh, surface_survival, corners, height, receiver_radius, min_distance, max_path_len, max_bounces, events_per_ray, event_offsets, retain_limit):
        retained_points = np.zeros((retain_limit, max_bounces + 2, 3), dtype=np.float64)
        point_counts = np.zeros(retain_limit, dtype=np.int64)
        ray_indices = np.zeros(retain_limit, dtype=np.int64)
        orders = np.zeros(retain_limit, dtype=np.int64)
        distances = np.zeros(retain_limit, dtype=np.float64)
        gains = np.zeros(retain_limit, dtype=np.float64)
        surface_indices = -np.ones((retain_limit, max_bounces + 1), dtype=np.int64)
        for ri in prange(directions.shape[0]):
            if events_per_ray[ri] <= 0 or event_offsets[ri] >= retain_limit:
                continue
            origin = src.copy()
            direction = directions[ri].copy()
            distance_so_far = 0.0
            survival = 1.0
            hit_points = np.zeros((max_bounces + 1, 3), dtype=np.float64)
            hit_surfaces = -np.ones(max_bounces + 1, dtype=np.int64)
            hit_count = 0
            accepted_for_ray = 0
            for bounce in range(max_bounces + 1):
                surf, t, nx, ny, nz = _closest_hit_backend_jit(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives, use_bvh)
                receiver_t = _ray_sphere_intersection_jit(origin, direction, rcv, receiver_radius)
                max_t = t if surf >= 0 else 1e30
                if receiver_t > 0.0 and receiver_t < max_t and bounce > 0:
                    slot = event_offsets[ri] + accepted_for_ray
                    accepted_for_ray += 1
                    if slot >= retain_limit:
                        break
                    path_distance = distance_so_far + receiver_t
                    gain = survival / max(path_distance, min_distance)
                    retained_points[slot, 0, 0] = src[0]
                    retained_points[slot, 0, 1] = src[1]
                    retained_points[slot, 0, 2] = src[2]
                    for pi in range(hit_count):
                        retained_points[slot, pi + 1, 0] = hit_points[pi, 0]
                        retained_points[slot, pi + 1, 1] = hit_points[pi, 1]
                        retained_points[slot, pi + 1, 2] = hit_points[pi, 2]
                        surface_indices[slot, pi] = hit_surfaces[pi]
                    end_index = hit_count + 1
                    retained_points[slot, end_index, 0] = rcv[0]
                    retained_points[slot, end_index, 1] = rcv[1]
                    retained_points[slot, end_index, 2] = rcv[2]
                    point_counts[slot] = hit_count + 2
                    ray_indices[slot] = ri
                    orders[slot] = bounce
                    distances[slot] = path_distance
                    gains[slot] = gain
                if surf < 0:
                    break
                e = surface_survival[surf]
                if e < 1e-5:
                    break
                survival *= e
                hx = origin[0] + t * direction[0]
                hy = origin[1] + t * direction[1]
                hz = origin[2] + t * direction[2]
                if hit_count < hit_points.shape[0]:
                    hit_points[hit_count, 0] = hx
                    hit_points[hit_count, 1] = hy
                    hit_points[hit_count, 2] = hz
                    hit_surfaces[hit_count] = surf
                    hit_count += 1
                dot = direction[0] * nx + direction[1] * ny + direction[2] * nz
                direction[0] = direction[0] - 2.0 * dot * nx
                direction[1] = direction[1] - 2.0 * dot * ny
                direction[2] = direction[2] - 2.0 * dot * nz
                direction[0], direction[1], direction[2] = _normalize3_jit(direction[0], direction[1], direction[2])
                distance_so_far += t
                if distance_so_far > max_path_len:
                    break
                origin[0] = hx + 1e-5 * direction[0]
                origin[1] = hy + 1e-5 * direction[1]
                origin[2] = hz + 1e-5 * direction[2]
        return retained_points, point_counts, ray_indices, orders, distances, gains, surface_indices


    @njit(parallel=True)
    def _trace_energy_kernel(source, listener, directions, diffuse_bank, diffuse_random, diffuse_indices, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives, use_bvh, reflection, scattering, corners, height, num_bounces, num_bins, bin_dur, c, max_path_len, direct_delay, listener_radius, source_radius, irradiance_min_distance, specular_exponent, source_forward_vector, dipole_weight, dipole_power, visual_candidate_limit, visual_stride, render_ambisonics):
        thread_count = get_num_threads()
        local_echogram = np.zeros((thread_count, _NUM_BANDS, num_bins), dtype=directions.dtype)
        local_ambisonic = np.zeros((thread_count, _NUM_BANDS, 4, num_bins), dtype=directions.dtype)
        local_hit_counts = np.zeros((thread_count, kinds.shape[0]), dtype=np.int64)
        local_contrib_counts = np.zeros((thread_count, kinds.shape[0]), dtype=np.int64)
        local_surface_energy = np.zeros((thread_count, kinds.shape[0]), dtype=directions.dtype)
        local_active_count = np.zeros(thread_count, dtype=np.int64)
        local_actual_bounces = np.zeros(thread_count, dtype=np.int64)
        visual_hit_points = np.zeros((visual_candidate_limit, num_bounces, 3), dtype=directions.dtype)
        visual_surface_indices = -np.ones((visual_candidate_limit, num_bounces), dtype=np.int64)
        visual_ray_indices = -np.ones(visual_candidate_limit, dtype=np.int64)
        visual_orders = np.zeros(visual_candidate_limit, dtype=np.int64)
        visual_distances = np.zeros(visual_candidate_limit, dtype=directions.dtype)
        visual_gains = np.zeros(visual_candidate_limit, dtype=directions.dtype)
        for ri in prange(directions.shape[0]):
            tid = get_thread_id()
            visual_slot = -1
            if visual_candidate_limit > 0 and ri % visual_stride == 0:
                candidate_slot = ri // visual_stride
                if candidate_slot < visual_candidate_limit:
                    visual_slot = candidate_slot
                    visual_ray_indices[visual_slot] = ri
            origin = np.empty(3, dtype=directions.dtype)
            origin[0] = listener[0]
            origin[1] = listener[1]
            origin[2] = listener[2]
            direction = np.empty(3, dtype=directions.dtype)
            direction[0] = directions[ri, 0]
            direction[1] = directions[ri, 1]
            direction[2] = directions[ri, 2]
            accum_distance = 0.0
            accum_energy = np.ones(_NUM_BANDS, dtype=directions.dtype)
            alive = True
            for bounce in range(num_bounces):
                if bounce + 1 > local_actual_bounces[tid]:
                    local_actual_bounces[tid] = bounce + 1
                if use_bvh:
                    surf, t, nx, ny, nz = _closest_hit_bvh_jit(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives)
                else:
                    surf, t, nx, ny, nz = _closest_hit_jit(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, height)
                if surf < 0 or t <= listener_radius or accum_distance > max_path_len:
                    alive = False
                    break
                if bounce > 0:
                    if _ray_sphere_before_jit(origin, direction, listener[0], listener[1], listener[2], listener_radius, t):
                        alive = False
                        break
                    source_distance = math.sqrt((listener[0] - source[0]) ** 2 + (listener[1] - source[1]) ** 2 + (listener[2] - source[2]) ** 2)
                    if source_distance > source_radius and _ray_sphere_before_jit(origin, direction, source[0], source[1], source[2], source_radius, t):
                        alive = False
                        break
                local_hit_counts[tid, surf] += 1
                hx = origin[0] + t * direction[0] + _HIT_OFFSET * nx
                hy = origin[1] + t * direction[1] + _HIT_OFFSET * ny
                hz = origin[2] + t * direction[2] + _HIT_OFFSET * nz
                if visual_slot >= 0:
                    visual_hit_points[visual_slot, bounce, 0] = hx
                    visual_hit_points[visual_slot, bounce, 1] = hy
                    visual_hit_points[visual_slot, bounce, 2] = hz
                    visual_surface_indices[visual_slot, bounce] = surf
                tsx = source[0] - hx
                tsy = source[1] - hy
                tsz = source[2] - hz
                dist_to_source = math.sqrt(tsx * tsx + tsy * tsy + tsz * tsz)
                if dist_to_source > irradiance_min_distance:
                    sdx = tsx / dist_to_source
                    sdy = tsy / dist_to_source
                    sdz = tsz / dist_to_source
                    facing = nx * sdx + ny * sdy + nz * sdz
                    if facing > 0.0:
                        shadow_origin = np.empty(3, dtype=directions.dtype)
                        shadow_origin[0] = hx
                        shadow_origin[1] = hy
                        shadow_origin[2] = hz
                        shadow_dir = np.empty(3, dtype=directions.dtype)
                        shadow_dir[0] = sdx
                        shadow_dir[1] = sdy
                        shadow_dir[2] = sdz
                        if use_bvh:
                            shadow_surface, shadow_t, _, _, _ = _closest_hit_bvh_jit(shadow_origin, shadow_dir, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives)
                        else:
                            shadow_surface, shadow_t, _, _, _ = _closest_hit_jit(shadow_origin, shadow_dir, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, height)
                        occluded = shadow_surface >= 0 and shadow_t > _EPS and shadow_t < dist_to_source - _EPS
                        if not occluded:
                            cos_in = facing
                            diffuse = (1.0 / math.pi) * scattering[surf] * cos_in
                            halfx = sdx - direction[0]
                            halfy = sdy - direction[1]
                            halfz = sdz - direction[2]
                            halfx, halfy, halfz = _normalize3_jit(halfx, halfy, halfz)
                            cos_half = max(0.0, halfx * nx + halfy * ny + halfz * nz)
                            specular = ((specular_exponent + 2.0) / (8.0 * math.pi)) * (1.0 - scattering[surf]) * (cos_half ** specular_exponent)
                            distance_term = (1.0 / (4.0 * math.pi)) * (1.0 / max(dist_to_source, irradiance_min_distance)) ** 2
                            source_cosine = -(source_forward_vector[0] * sdx + source_forward_vector[1] * sdy + source_forward_vector[2] * sdz)
                            source_gain = abs((1.0 - dipole_weight) + dipole_weight * source_cosine) ** dipole_power
                            rel_delay = (accum_distance + t + dist_to_source) / c - direct_delay
                            bin_index = int(math.floor(rel_delay / bin_dur))
                            if bin_index >= 0 and bin_index < num_bins:
                                coeff_x = directions[ri, 0]
                                coeff_y = directions[ri, 1]
                                coeff_z = directions[ri, 2]
                                energy_sum = 0.0
                                for bi in range(_NUM_BANDS):
                                    energy = ((4.0 * math.pi) / max(directions.shape[0], 1)) * source_gain * distance_term * (diffuse + specular) * reflection[surf, bi] * accum_energy[bi]
                                    local_echogram[tid, bi, bin_index] += energy
                                    if render_ambisonics:
                                        local_ambisonic[tid, bi, 0, bin_index] += energy
                                        local_ambisonic[tid, bi, 1, bin_index] += energy * coeff_x
                                        local_ambisonic[tid, bi, 2, bin_index] += energy * coeff_y
                                        local_ambisonic[tid, bi, 3, bin_index] += energy * coeff_z
                                    energy_sum += energy
                                if visual_slot >= 0 and energy_sum > visual_gains[visual_slot] * visual_gains[visual_slot]:
                                    visual_orders[visual_slot] = bounce + 1
                                    visual_distances[visual_slot] = accum_distance + t + dist_to_source
                                    visual_gains[visual_slot] = math.sqrt(max(energy_sum, 0.0))
                                local_contrib_counts[tid, surf] += 1
                                local_surface_energy[tid, surf] += energy_sum
                for bi in range(_NUM_BANDS):
                    accum_energy[bi] *= reflection[surf, bi]
                accum_distance += t
                origin[0] = hx
                origin[1] = hy
                origin[2] = hz
                use_diffuse = diffuse_random[bounce, ri] < scattering[surf]
                if use_diffuse:
                    sample_index = diffuse_indices[bounce, ri]
                    dx, dy, dz = _diffuse_direction_jit(diffuse_bank[sample_index], nx, ny, nz)
                    direction[0] = dx
                    direction[1] = dy
                    direction[2] = dz
                else:
                    dot = direction[0] * nx + direction[1] * ny + direction[2] * nz
                    direction[0] = direction[0] - 2.0 * dot * nx
                    direction[1] = direction[1] - 2.0 * dot * ny
                    direction[2] = direction[2] - 2.0 * dot * nz
                    direction[0], direction[1], direction[2] = _normalize3_jit(direction[0], direction[1], direction[2])
                if accum_distance > max_path_len:
                    alive = False
                    break
            if alive:
                local_active_count[tid] += 1
        echogram = np.zeros((_NUM_BANDS, num_bins), dtype=directions.dtype)
        ambisonic = np.zeros((_NUM_BANDS, 4, num_bins), dtype=directions.dtype)
        hit_counts = np.zeros(kinds.shape[0], dtype=np.int64)
        contrib_counts = np.zeros(kinds.shape[0], dtype=np.int64)
        surface_energy = np.zeros(kinds.shape[0], dtype=directions.dtype)
        active_count = 0
        actual_bounces = 0
        for ti in range(thread_count):
            active_count += local_active_count[ti]
            if local_actual_bounces[ti] > actual_bounces:
                actual_bounces = local_actual_bounces[ti]
            for si in range(kinds.shape[0]):
                hit_counts[si] += local_hit_counts[ti, si]
                contrib_counts[si] += local_contrib_counts[ti, si]
                surface_energy[si] += local_surface_energy[ti, si]
            for bi in range(_NUM_BANDS):
                for b in range(num_bins):
                    echogram[bi, b] += local_echogram[ti, bi, b]
                    if render_ambisonics:
                        for ci in range(4):
                            ambisonic[bi, ci, b] += local_ambisonic[ti, bi, ci, b]
        return (
            echogram,
            ambisonic,
            hit_counts,
            contrib_counts,
            surface_energy,
            actual_bounces,
            active_count,
            visual_hit_points,
            visual_surface_indices,
            visual_ray_indices,
            visual_orders,
            visual_distances,
            visual_gains,
        )
