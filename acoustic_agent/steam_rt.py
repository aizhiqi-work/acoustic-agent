from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
import math
from typing import Any, Mapping, Sequence

import numpy as np

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
from .directivity import source_directivity, source_directivity_gain, source_forward
from .geometry import point_in_polygon
from .materials import MaterialLibrary
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
    def __init__(self, a: np.ndarray, b: np.ndarray, height: float, name: str, absorption: np.ndarray, scattering: np.ndarray, transmission: np.ndarray) -> None:
        super().__init__("wall", name, absorption, scattering, transmission)
        self.a = a
        self.b = b
        self.height = height
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
        if z < -1e-6 or z > self.height + 1e-6:
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
        valid = (np.abs(det) > 1e-12) & (t > _EPS) & (u >= -1e-6) & (u <= 1.0 + 1e-6) & (z >= -1e-6) & (z <= self.height + 1e-6)
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
        inside = scene._point_in_polygon_batch(p[:, :2])
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


class RoomRayScene:
    def __init__(self, room: Room) -> None:
        self.room = room
        corners = [np.asarray(c[:2], dtype=float) for c in room.corners]
        wall = room.materials.get("wall") or next(iter(room.materials.values()))
        floor = room.materials.get("floor", wall)
        ceiling = room.materials.get("ceiling", wall)
        self.surfaces: list[Any] = []
        for i, a in enumerate(corners):
            self.surfaces.append(_WallSurface(a, corners[(i + 1) % len(corners)], room.height_m, f"wall_{i}", _band_array(wall, "absorption", 0.1), _band_array(wall, "scattering", 0.12), _band_array(wall, "transmission", 10.0 ** (-30.0 / 20.0))))
        self.surfaces.append(_HorizontalSurface(0.0, True, room.corners, "floor", _band_array(floor, "absorption", 0.12), _band_array(floor, "scattering", 0.1), _band_array(floor, "transmission", 10.0 ** (-35.0 / 20.0))))
        self.surfaces.append(_HorizontalSurface(room.height_m, False, room.corners, "ceiling", _band_array(ceiling, "absorption", 0.1), _band_array(ceiling, "scattering", 0.1), _band_array(ceiling, "transmission", 10.0 ** (-30.0 / 20.0))))
        self.surfaces.extend(_object_box_surfaces(room, wall))
        self._batch_ready = False

    def closest_hit(self, origin: np.ndarray, direction: np.ndarray) -> dict[str, Any]:
        best_t = np.inf
        best = None
        normal = None
        for surface in self.surfaces:
            t = surface.intersect(origin, direction)
            if t < best_t:
                best_t = t
                best = surface
                if hasattr(surface, "normal_at"):
                    normal = surface.normal_at(origin, direction, t)
                else:
                    normal = surface.normal.copy()
        if best is None:
            return {"valid": False, "distance": np.inf, "transmission": np.ones(_NUM_BANDS), "surface": None}
        if float(np.dot(normal, direction)) > 0.0:
            normal = -normal
        return {"valid": True, "distance": best_t, "point": origin + best_t * direction, "normal": normal, "absorption": best.absorption, "scattering": best.scattering, "transmission": best.transmission, "surface": best.name}

    def any_hit(self, origin: np.ndarray, direction: np.ndarray, max_distance: float) -> bool:
        for surface in self.surfaces:
            t = surface.intersect(origin, direction)
            if _EPS < t < max_distance - _EPS:
                return True
        return False

    def _build_batch_arrays(self) -> None:
        if self._batch_ready:
            return
        self._corners = np.asarray([c[:2] for c in self.room.corners], dtype=float)
        self._surf_abs = np.asarray([s.absorption for s in self.surfaces], dtype=float)
        self._surf_sca_mean = np.asarray([float(np.mean(s.scattering)) for s in self.surfaces], dtype=float)
        self._surf_names = np.asarray([s.name for s in self.surfaces], dtype=object)
        self._batch_ready = True

    def _point_in_polygon_batch(self, pts: np.ndarray) -> np.ndarray:
        corners = self._corners
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

    def batch_closest_hit(self, origins: np.ndarray, dirs: np.ndarray) -> dict[str, Any]:
        self._build_batch_arrays()
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
        t_all = np.stack([surface.batch_intersect(origins, dirs, self)[0] for surface in self.surfaces], axis=0)
        return np.any((t_all > _EPS) & (t_all < (max_distance - _EPS)[None, :]), axis=0)


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
        try:
            boxes = _object_proxy_boxes(item, room.height_m)
            if not boxes:
                continue
            material_key = str(item.get("material", "wood"))
            if library is None:
                library = MaterialLibrary.load()
            material = library.sample_object(material_key)
        except Exception:
            material = fallback_material
            boxes = _object_proxy_boxes(item, room.height_m)
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
                absorption=_band_array(material, "absorption", 0.2),
                scattering=_band_array(material, "scattering", 0.18),
                transmission=_band_array(material, "transmission", 10.0 ** (-24.0 / 20.0)),
            ))
    return surfaces


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
    if object_type == "table":
        angle = math.radians(rotation)
        axis_u = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
        axis_v = np.asarray([-math.sin(angle), math.cos(angle)], dtype=float)
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
    return [{
        "part": "body",
        "center": center,
        "size": size,
        "z": z_center,
        "rotation": rotation,
    }]


def _scene_kernel_arrays(scene: RoomRayScene) -> dict[str, Any]:
    kinds = np.zeros(len(scene.surfaces), dtype=np.int64)
    wall_a = np.zeros((len(scene.surfaces), 2), dtype=np.float64)
    wall_b = np.zeros((len(scene.surfaces), 2), dtype=np.float64)
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
    return {
        "kinds": kinds,
        "wall_a": wall_a,
        "wall_delta": wall_b - wall_a,
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
        "names": names,
        "corners": np.asarray(scene.room.corners, dtype=np.float64)[:, :2],
        "height": float(scene.room.height_m),
    }


def simulate_steam_room(
    room: Room,
    source: Sequence[float],
    listener: Sequence[float],
    config: SimConfig,
    source_model: str | Mapping[str, Any] | None = None,
) -> SteamRender:
    src = np.asarray(source, dtype=float)
    rcv = np.asarray(listener, dtype=float)
    emitter = source_directivity(source_model)
    scene = RoomRayScene(room)
    direct = simulate_direct(scene, src, rcv, config, emitter)
    fs = int(config.fs)
    total = max(1, int(round(config.duration_s * fs)))
    discrete_band = np.zeros((_NUM_BANDS, total), dtype=np.float32)
    reflection_band = np.zeros_like(discrete_band)
    direct_sample = int(round(direct["delay_s"] * fs))
    _add_band_impulse(discrete_band, float(direct["delay_s"]), direct["band_gains"], config)

    paths = [_direct_path(src, rcv, direct, config)]
    diffraction_paths = [
        _apply_source_directivity_to_path(path, emitter)
        for path in _boundary_diffraction_paths(room, scene, src, rcv, direct, config)
    ]
    for path in diffraction_paths:
        sample = int(round(path.delay_s * fs))
        if config.diffraction_audio_enabled and 0 <= sample < total:
            _add_band_impulse(discrete_band, float(path.delay_s), path.band_gains, config)
    paths.extend(diffraction_paths)
    rt_visual = scan_visual_rt_paths(room, scene, src, rcv, config)
    rt_visual["paths"] = [
        _apply_source_directivity_to_path(path, emitter)
        for path in rt_visual["paths"]
    ]
    paths.extend(rt_visual["paths"])
    rt60_bands = {band: 0.0 for band in FREQUENCY_BANDS}
    hybrid_rt60_bands = {band: 0.0 for band in FREQUENCY_BANDS}
    reconstructed_rt60_bands = {band: 0.0 for band in FREQUENCY_BANDS}
    reflection_metadata: dict[str, Any] = {"enabled": False}

    if config.reflections_enabled:
        field = trace_energy_field(scene, src, rcv, config, emitter)
        rt60_bands = estimate_reverb_times(field, config)
        reconstruction_field, late_tail_meta = _extend_energy_field_late_tail(field, rt60_bands, config)
        hybrid_rt60_bands = estimate_reverb_times(reconstruction_field, config) if config.late_tail else dict(rt60_bands)
        band_irs = reconstruct_band_irs(reconstruction_field, config)
        ambisonic_band_irs = reconstruct_ambisonic_band_irs(reconstruction_field, config)
        seg_len = min(band_irs.shape[1], total - direct_sample) if direct_sample < total else 0
        ambisonic_rir = np.zeros((4, total), dtype=np.float32)
        if seg_len > 0:
            reflection_band[:, direct_sample:direct_sample + seg_len] += band_irs[:, :seg_len]
            ambisonic_rir[:, direct_sample:direct_sample + seg_len] += np.sum(ambisonic_band_irs[:, :, :seg_len], axis=0)
        quality_warnings: list[str] = []
        if int(config.rt_num_rays) < 4096:
            quality_warnings.append("ray count is below the Steam Audio realtime reference")
        if int(config.rt_num_bounces) < 64:
            quality_warnings.append("RT60 is biased by bounce truncation; use at least 64 bounces")
        max_rt60 = max(rt60_bands.values(), default=0.0)
        if max_rt60 > 0.0 and float(config.rt_duration_s) < 1.2 * max_rt60:
            quality_warnings.append("reflection duration is shorter than 1.2 times the estimated RT60")
        late_start_bin = min(
            int(max(0.0, float(late_tail_meta.get("transition_start_s", 0.0))) / max(float(field["bin_duration_s"]), 1e-9)),
            field["echogram"].shape[1],
        )
        reflection_metadata = {
            "enabled": True,
            "num_rays": int(config.rt_num_rays),
            "num_bounces": int(config.rt_num_bounces),
            "num_bins": int(field["num_bins"]),
            "bin_duration_s": float(field["bin_duration_s"]),
            "actual_bounces": int(field.get("actual_bounces", 0)),
            "active_ray_count": int(field.get("active_ray_count", 0)),
            "last_energy_time_s": float(field.get("last_energy_time_s", 0.0)),
            "traced_energy": float(np.sum(field["echogram"])),
            "late_tail_energy": float(np.sum(reconstruction_field["echogram"][:, late_start_bin:])),
            "traced_late_tail_energy": float(np.sum(field["echogram"][:, late_start_bin:])),
            "model": "monte_carlo_path_tracing_energy_field",
            "late_tail_enabled": bool(config.late_tail),
            "late_tail_cutoff_s": float(late_tail_meta.get("transition_start_s", 0.0)),
            "late_tail": late_tail_meta,
            "hybrid_rt60_bands": hybrid_rt60_bands,
            "quality": _reflection_quality_label(config),
            "quality_warnings": quality_warnings,
            "surface_hit_count": field.get("surface_hit_count", {}),
            "surface_contribution_count": field.get("surface_contribution_count", {}),
            "surface_energy": field.get("surface_energy", {}),
            "ambisonics": {
                "enabled": True,
                "order": 1,
                "channels": ["W", "X", "Y", "Z"],
                "normalization": "acoustic_agent_foa_unit_vector",
                "energy": float(np.sum(ambisonic_rir * ambisonic_rir)),
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
                "tail_target_model": "steam_hybrid_energy_envelope_from_traced_energy_field",
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
                "tail_target_rt60_bands": hybrid_rt60_bands,
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
            "reflections": reflection_metadata,
            "rt_visual": rt_visual["metadata"],
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
        arrays["z_values"],
        arrays["box_center"],
        arrays["box_axis_u"],
        arrays["box_axis_v"],
        arrays["box_half"],
        arrays["box_z"],
        arrays["normals"],
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
        arrays["z_values"],
        arrays["box_center"],
        arrays["box_axis_u"],
        arrays["box_axis_v"],
        arrays["box_half"],
        arrays["box_z"],
        arrays["normals"],
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


def trace_energy_field(
    scene: RoomRayScene,
    source: np.ndarray,
    listener: np.ndarray,
    config: SimConfig,
    source_model: str | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    emitter = source_directivity(source_model)
    if njit is not None:
        return _trace_energy_field_numba(scene, source, listener, config, emitter)
    return _trace_energy_field_numpy(scene, source, listener, config, emitter)


def _trace_energy_field_numba(
    scene: RoomRayScene,
    source: np.ndarray,
    listener: np.ndarray,
    config: SimConfig,
    source_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    emitter = source_directivity(source_model)
    arrays = _scene_kernel_arrays(scene)
    num_rays = int(config.rt_num_rays)
    num_bounces = int(config.rt_num_bounces)
    bin_dur = float(config.rt_bin_duration_s)
    duration = max(float(config.rt_duration_s), float(config.duration_s))
    num_bins = max(1, int(math.ceil(duration / bin_dur)))
    directions = _sphere_samples(num_rays, int(config.seed))
    diffuse_bank = _diffuse_sample_bank(config.rt_num_diffuse_samples)
    direct_delay = float(np.linalg.norm(np.asarray(source, dtype=float) - np.asarray(listener, dtype=float))) / float(config.c)
    echogram, ambisonic, hit_counts, contrib_counts, surface_energy, actual_bounces, active_count = _trace_energy_kernel(
        np.asarray(source, dtype=np.float64),
        np.asarray(listener, dtype=np.float64),
        np.asarray(directions, dtype=np.float64),
        np.asarray(diffuse_bank, dtype=np.float64),
        arrays["kinds"],
        arrays["wall_a"],
        arrays["wall_delta"],
        arrays["z_values"],
        arrays["box_center"],
        arrays["box_axis_u"],
        arrays["box_axis_v"],
        arrays["box_half"],
        arrays["box_z"],
        arrays["normals"],
        arrays["reflection"],
        arrays["scattering"],
        arrays["corners"],
        float(arrays["height"]),
        int(num_bounces),
        int(num_bins),
        float(bin_dur),
        float(config.c),
        float(duration) * float(config.c),
        float(direct_delay),
        float(config.rt_listener_radius),
        float(config.rt_source_radius),
        float(config.rt_irradiance_min_distance),
        float(config.rt_specular_exponent),
        source_forward(emitter),
        float(emitter["dipole_weight"]),
        float(emitter["dipole_power"]),
        float(config.seed),
    )
    names = arrays["names"]
    total_energy = np.sum(echogram, axis=0)
    nonzero_bins = np.flatnonzero(total_energy > 0.0)
    return {
        "echogram": echogram,
        "ambisonic_echogram": ambisonic,
        "num_bins": num_bins,
        "bin_duration_s": bin_dur,
        "direct_delay_s": direct_delay,
        "actual_bounces": int(actual_bounces),
        "active_ray_count": int(active_count),
        "last_energy_time_s": float(nonzero_bins[-1] * bin_dur) if nonzero_bins.size else 0.0,
        "surface_hit_count": {names[i]: int(hit_counts[i]) for i in range(len(names)) if int(hit_counts[i]) > 0},
        "surface_contribution_count": {names[i]: int(contrib_counts[i]) for i in range(len(names)) if int(contrib_counts[i]) > 0},
        "surface_energy": {names[i]: float(surface_energy[i]) for i in range(len(names)) if float(surface_energy[i]) > 0.0},
        "accelerator": "numba",
        "source_directivity": dict(emitter),
    }


def _trace_energy_field_numpy(
    scene: RoomRayScene,
    source: np.ndarray,
    listener: np.ndarray,
    config: SimConfig,
    source_model: Mapping[str, Any] | None = None,
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
    ambisonic_echogram = np.zeros((_NUM_BANDS, 4, num_bins), dtype=np.float64)
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
                    coeffs_v = _foa_coefficients(listener_dirs[valid])
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
        "source_directivity": dict(emitter),
    }


def reconstruct_band_irs(field: dict[str, Any], config: SimConfig) -> np.ndarray:
    echogram = field["echogram"]
    samples_per_bin = max(1, int(math.ceil(float(field["bin_duration_s"]) * int(config.fs))))
    num_samples = int(field["num_bins"]) * samples_per_bin
    rng = np.random.default_rng(config.seed + 7)
    white = rng.uniform(-1.0, 1.0, size=num_samples).astype(np.float64)
    raw_band_irs = np.zeros((_NUM_BANDS, num_samples), dtype=np.float64)
    sample_weights = np.arange(samples_per_bin, dtype=np.float64) / samples_per_bin
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
            bin_time = (b + 0.5) * samples_per_bin / int(config.fs)
            seg *= math.exp(-coeff * (0.5 * config.c * bin_time))
            sample_amp[lo:hi] = seg
        raw_band_irs[band_index] = sample_amp * white
    return bandlimit_band_signals(raw_band_irs, int(config.fs))


def _extend_energy_field_late_tail(field: dict[str, Any], rt60_bands: Mapping[str, float], config: SimConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project Steam-style hybrid late energy from the traced EnergyField.

    Steam Audio estimates per-band reverb time from the traced EnergyField, then
    starts its parametric branch at ``(1 - overlap) * transitionTime``.  The
    production engine renders that branch with a feedback-delay network.  This
    offline RIR renderer uses the same transition, overlap, traced RT60, and
    cutoff-bin energy to project an equivalent diffuse energy envelope before
    stochastic reconstruction.  Material/Sabine RT60 is deliberately not used
    as a target.
    """
    out = dict(field)
    echogram = np.array(field["echogram"], dtype=np.float64, copy=True)
    ambisonic = field.get("ambisonic_echogram")
    if ambisonic is not None:
        out["ambisonic_echogram"] = np.array(ambisonic, dtype=np.float64, copy=True)
    out["echogram"] = echogram

    bin_dur = float(field["bin_duration_s"])
    num_bins = int(field["num_bins"])
    duration_s = num_bins * bin_dur
    transition_s = min(duration_s, max(bin_dur, float(config.hybrid_transition_s)))
    overlap = float(np.clip(float(config.hybrid_overlap_fraction), 0.0, 0.95))
    start_s = (1.0 - overlap) * transition_s
    start_bin = min(num_bins, max(0, int(math.ceil(start_s / max(bin_dur, 1e-9)))))
    transition_bin = min(num_bins, max(start_bin + 1, int(math.ceil(transition_s / max(bin_dur, 1e-9)))))
    if not config.late_tail or start_bin >= num_bins:
        return out, {
            "applied": False,
            "added": False,
            "reason": "disabled" if not config.late_tail else "tail_start_outside_field",
            "model": "steam_hybrid_energy_envelope_projection",
            "transition_start_s": round(start_s, 6),
            "transition_end_s": round(transition_s, 6),
            "overlap_fraction": overlap,
        }

    added_by_band: dict[str, float] = {}
    target_by_band: dict[str, float] = {}
    anchor_by_band: dict[str, float] = {}
    anchor_model_by_band: dict[str, str] = {}
    traced_by_band: dict[str, float] = {}
    early_by_band: dict[str, float] = {}
    rt60_used: dict[str, float] = {}

    for band_index, band in enumerate(FREQUENCY_BANDS):
        rt60 = float(rt60_bands.get(band, 0.0) or 0.0)
        if rt60 <= 0.0:
            continue
        decay = 6.0 * math.log(10.0) / max(rt60, 0.1)
        early_energy = float(np.sum(echogram[band_index, :start_bin]))
        traced_late_energy = float(np.sum(echogram[band_index, start_bin:]))
        anchor = float(echogram[band_index, start_bin]) if start_bin < num_bins else 0.0
        anchor_model = "cutoff_bin"
        if anchor <= 0.0:
            lo = max(0, start_bin - 2)
            hi = min(num_bins, start_bin + 3)
            positive = echogram[band_index, lo:hi]
            positive = positive[positive > 0.0]
            if positive.size:
                anchor = float(np.mean(positive))
                anchor_model = "five_bin_nonzero_mean_fallback"
        if anchor <= 0.0 and start_bin > 0:
            prior_indices = np.flatnonzero(echogram[band_index, :start_bin] > 0.0)
            if prior_indices.size:
                recent = prior_indices[-5:]
                estimates = echogram[band_index, recent] * np.exp(-decay * (start_bin - recent) * bin_dur)
                anchor = float(np.median(estimates))
                anchor_model = "rt60_extrapolated_recent_bins"
        relative_t = np.arange(num_bins - start_bin, dtype=np.float64) * bin_dur
        target = anchor * np.exp(-decay * relative_t)
        traced = np.array(echogram[band_index, start_bin:], dtype=np.float64, copy=True)
        blend = np.ones_like(relative_t)
        overlap_bins = max(1, transition_bin - start_bin)
        blend[:overlap_bins] = np.arange(overlap_bins, dtype=np.float64) / overlap_bins
        projected = (1.0 - blend) * traced + blend * target
        target_late_energy = float(np.sum(projected))
        delta_energy = target_late_energy - traced_late_energy
        early_by_band[band] = round(early_energy, 12)
        traced_by_band[band] = round(traced_late_energy, 12)
        target_by_band[band] = round(target_late_energy, 12)
        anchor_by_band[band] = round(anchor, 12)
        anchor_model_by_band[band] = anchor_model
        rt60_used[band] = round(rt60, 4)
        echogram[band_index, start_bin:] = projected
        added_by_band[band] = round(delta_energy, 12)

    if ambisonic is not None and start_bin < num_bins:
        early_weight = np.ones(num_bins - start_bin, dtype=np.float64)
        overlap_bins = max(1, transition_bin - start_bin)
        early_weight[:overlap_bins] = 1.0 - np.arange(overlap_bins, dtype=np.float64) / overlap_bins
        early_weight[overlap_bins:] = 0.0
        out["ambisonic_echogram"][:, 1:, start_bin:] *= early_weight[None, None, :]

    added_energy = float(sum(max(0.0, value) for value in added_by_band.values()))
    removed_energy = float(sum(max(0.0, -value) for value in added_by_band.values()))
    net_energy_delta = added_energy - removed_energy
    changed_energy = added_energy + removed_energy
    nonzero_bins = np.flatnonzero(np.sum(echogram, axis=0) > 0.0)
    out["last_energy_time_s"] = float(nonzero_bins[-1] * bin_dur) if nonzero_bins.size else float(field.get("last_energy_time_s", 0.0))
    return out, {
        "applied": bool(rt60_used),
        "changed": bool(changed_energy > 1e-14),
        "added": bool(added_energy > 1e-14),
        "transition_start_s": round(start_s, 6),
        "transition_end_s": round(transition_s, 6),
        "overlap_fraction": overlap,
        "model": "steam_hybrid_energy_envelope_projection",
        "rt60_source": "steam_audio_reverb_estimator_from_traced_energy_field",
        "calibration": "cutoff-bin energy and traced RT60; power crossfade over Steam hybrid overlap",
        "added_energy": round(added_energy, 12),
        "removed_energy": round(removed_energy, 12),
        "net_energy_delta": round(net_energy_delta, 12),
        "early_energy_by_band": early_by_band,
        "traced_late_energy_by_band": traced_by_band,
        "target_late_energy_by_band": target_by_band,
        "anchor_energy_by_band": anchor_by_band,
        "anchor_model_by_band": anchor_model_by_band,
        "added_energy_by_band": added_by_band,
        "rt60_bands": rt60_used,
    }


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
            bin_time = (b + 0.5) * samples_per_bin / int(config.fs)
            seg *= math.exp(-coeff * (0.5 * config.c * bin_time))
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
    for band_index, band in enumerate(FREQUENCY_BANDS):
        weights = np.exp(-AIR_ABSORPTION_NP_PER_M[band] * (bin_dur * np.arange(num_bins)))
        # Steam Audio stores Y00 * E in channel 0 of the EnergyField.
        weighted = (_SH_Y00 * echogram[band_index]) * weights
        out[band] = _schroeder_fit_rt60(weighted, bin_dur, min_total_energy=1e-4)
    return out


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
            "model": "same_room_direct_with_occlusion_transmission",
            "occlusion": float(direct["occlusion"]),
            "occlusion_surface": direct.get("occlusion_surface"),
            "source_directivity_gain": float(direct.get("source_directivity_gain", 1.0)),
            "contributes_to_rir": True,
        },
    )


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


def _sphere_samples(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = i * (math.pi * (3.0 - math.sqrt(5.0))) + rng.uniform(-0.02, 0.02, n)
    return np.stack([np.cos(theta) * r, np.sin(theta) * r, z], axis=1)


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
    return np.stack([r * np.cos(theta), r * np.sin(theta), np.sqrt(np.clip(1.0 - u, 0.0, 1.0))], axis=1)


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


    @njit(cache=True)
    def _closest_hit_jit(origin, direction, kinds, wall_a, wall_delta, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, height):
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
                    if cand_t > _EPS and u >= -1e-6 and u <= 1.0 + 1e-6 and z >= -1e-6 and z <= height + 1e-6:
                        t = cand_t
            elif kinds[si] == 2:
                cand_t, cand_nx, cand_ny, cand_nz = _box_hit_jit(origin, direction, box_center[si], box_axis_u[si], box_axis_v[si], box_half[si], box_z[si])
                if cand_t > _EPS and cand_t < 1.0e29:
                    t = cand_t
                    surf_nx = cand_nx
                    surf_ny = cand_ny
                    surf_nz = cand_nz
            else:
                if abs(direction[2]) > 1e-12:
                    cand_t = (z_values[si] - origin[2]) / direction[2]
                    if cand_t > _EPS:
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


    @njit(cache=True)
    def _any_hit_jit(origin, direction, max_distance, kinds, wall_a, wall_delta, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, height):
        surf, t, nx, ny, nz = _closest_hit_jit(origin, direction, kinds, wall_a, wall_delta, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, height)
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
    def _hash_unit_jit(a, b, seed):
        value = math.sin((a + 1.0) * 12.9898 + (b + 1.0) * 78.233 + seed * 0.037719) * 43758.5453123
        return value - math.floor(value)


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
    def _visual_event_count_kernel(src, rcv, directions, kinds, wall_a, wall_delta, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, surface_survival, corners, height, receiver_radius, max_path_len, max_bounces):
        event_flags = np.zeros((directions.shape[0], max_bounces + 1), dtype=np.bool_)
        for ri in prange(directions.shape[0]):
            origin = src.copy()
            direction = directions[ri].copy()
            distance_so_far = 0.0
            for bounce in range(max_bounces + 1):
                surf, t, nx, ny, nz = _closest_hit_jit(origin, direction, kinds, wall_a, wall_delta, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, height)
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
    def _visual_record_kernel(src, rcv, directions, kinds, wall_a, wall_delta, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, surface_survival, corners, height, receiver_radius, min_distance, max_path_len, max_bounces, events_per_ray, event_offsets, retain_limit):
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
                surf, t, nx, ny, nz = _closest_hit_jit(origin, direction, kinds, wall_a, wall_delta, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, height)
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
    def _trace_energy_kernel(source, listener, directions, diffuse_bank, kinds, wall_a, wall_delta, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, reflection, scattering, corners, height, num_bounces, num_bins, bin_dur, c, max_path_len, direct_delay, listener_radius, source_radius, irradiance_min_distance, specular_exponent, source_forward_vector, dipole_weight, dipole_power, seed):
        thread_count = get_num_threads()
        local_echogram = np.zeros((thread_count, _NUM_BANDS, num_bins), dtype=np.float64)
        local_ambisonic = np.zeros((thread_count, _NUM_BANDS, 4, num_bins), dtype=np.float64)
        local_hit_counts = np.zeros((thread_count, kinds.shape[0]), dtype=np.int64)
        local_contrib_counts = np.zeros((thread_count, kinds.shape[0]), dtype=np.int64)
        local_surface_energy = np.zeros((thread_count, kinds.shape[0]), dtype=np.float64)
        local_active_count = np.zeros(thread_count, dtype=np.int64)
        local_actual_bounces = np.zeros(thread_count, dtype=np.int64)
        for ri in prange(directions.shape[0]):
            tid = get_thread_id()
            origin = np.empty(3, dtype=np.float64)
            origin[0] = listener[0]
            origin[1] = listener[1]
            origin[2] = listener[2]
            direction = np.empty(3, dtype=np.float64)
            direction[0] = directions[ri, 0]
            direction[1] = directions[ri, 1]
            direction[2] = directions[ri, 2]
            accum_distance = 0.0
            accum_energy = np.ones(_NUM_BANDS, dtype=np.float64)
            alive = True
            for bounce in range(num_bounces):
                if bounce + 1 > local_actual_bounces[tid]:
                    local_actual_bounces[tid] = bounce + 1
                surf, t, nx, ny, nz = _closest_hit_jit(origin, direction, kinds, wall_a, wall_delta, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, height)
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
                        shadow_origin = np.empty(3, dtype=np.float64)
                        shadow_origin[0] = hx
                        shadow_origin[1] = hy
                        shadow_origin[2] = hz
                        shadow_dir = np.empty(3, dtype=np.float64)
                        shadow_dir[0] = sdx
                        shadow_dir[1] = sdy
                        shadow_dir[2] = sdz
                        occluded = _any_hit_jit(shadow_origin, shadow_dir, dist_to_source, kinds, wall_a, wall_delta, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, height)
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
                                    local_ambisonic[tid, bi, 0, bin_index] += energy
                                    local_ambisonic[tid, bi, 1, bin_index] += energy * coeff_x
                                    local_ambisonic[tid, bi, 2, bin_index] += energy * coeff_y
                                    local_ambisonic[tid, bi, 3, bin_index] += energy * coeff_z
                                    energy_sum += energy
                                local_contrib_counts[tid, surf] += 1
                                local_surface_energy[tid, surf] += energy_sum
                for bi in range(_NUM_BANDS):
                    accum_energy[bi] *= reflection[surf, bi]
                accum_distance += t
                origin[0] = hx
                origin[1] = hy
                origin[2] = hz
                use_diffuse = _hash_unit_jit(ri, bounce, seed) < scattering[surf]
                if use_diffuse:
                    sample_index = int(_hash_unit_jit(ri + 13, bounce + 17, seed) * diffuse_bank.shape[0])
                    if sample_index >= diffuse_bank.shape[0]:
                        sample_index = diffuse_bank.shape[0] - 1
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
        echogram = np.zeros((_NUM_BANDS, num_bins), dtype=np.float64)
        ambisonic = np.zeros((_NUM_BANDS, 4, num_bins), dtype=np.float64)
        hit_counts = np.zeros(kinds.shape[0], dtype=np.int64)
        contrib_counts = np.zeros(kinds.shape[0], dtype=np.int64)
        surface_energy = np.zeros(kinds.shape[0], dtype=np.float64)
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
                    for ci in range(4):
                        ambisonic[bi, ci, b] += local_ambisonic[ti, bi, ci, b]
        return echogram, ambisonic, hit_counts, contrib_counts, surface_energy, actual_bounces, active_count
