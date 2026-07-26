from __future__ import annotations

from collections import OrderedDict
import math
from threading import RLock
import time
from typing import Any, Mapping

import numpy as np

try:
    from numba import cuda, float32
except Exception:  # pragma: no cover - exercised on installations without CUDA support
    cuda = None
    float32 = None


_NUM_BANDS = 6
_EPS = np.float32(1e-6)
_HIT_OFFSET = np.float32(1e-2)
_BVH_BOUNDS_EPS = np.float32(1e-5)
_INF = np.float32(1e30)
_INV_PI = np.float32(1.0 / math.pi)
_INV_4PI = np.float32(1.0 / (4.0 * math.pi))
_INV_8PI = np.float32(1.0 / (8.0 * math.pi))
_FOUR_PI = np.float32(4.0 * math.pi)
_DEVICE_CACHE_LIMIT_BYTES = 512 * 1024 * 1024
_DEVICE_CACHE_LOCK = RLock()
_DEVICE_INPUT_CACHE: OrderedDict[tuple[Any, ...], tuple[np.ndarray, Any, int]] = OrderedDict()
_DEVICE_INPUT_CACHE_BYTES = 0


def cuda_available(device_id: int = 0) -> bool:
    if cuda is None:
        return False
    try:
        if not cuda.is_available():
            return False
        return 0 <= int(device_id) < len(cuda.gpus)
    except Exception:
        return False


def trace_energy_field_cuda(
    *,
    source: np.ndarray,
    listener: np.ndarray,
    directions: np.ndarray,
    diffuse_bank: np.ndarray,
    diffuse_random: np.ndarray,
    diffuse_indices: np.ndarray,
    arrays: Mapping[str, Any],
    use_bvh: bool,
    num_bounces: int,
    num_bins: int,
    bin_dur: float,
    speed_of_sound: float,
    max_path_len: float,
    direct_delay: float,
    listener_radius: float,
    source_radius: float,
    irradiance_min_distance: float,
    specular_exponent: float,
    source_forward_vector: np.ndarray,
    dipole_weight: float,
    dipole_power: float,
    visual_candidate_limit: int,
    visual_stride: int,
    render_ambisonics: bool,
    device_id: int,
) -> dict[str, Any]:
    if not cuda_available(device_id):
        raise RuntimeError(f"CUDA device {device_id} is not available")

    cuda.select_device(int(device_id))
    device = cuda.get_current_device()
    transfer_started = time.perf_counter()
    cache_stats = {"hits": 0, "misses": 0}

    def to_float32(value: Any, *, cache: bool = False) -> Any:
        if cache:
            return _cached_device_array(value, np.float32, int(device_id), cache_stats)
        return cuda.to_device(np.ascontiguousarray(value, dtype=np.float32))

    def to_int32(value: Any, *, cache: bool = False) -> Any:
        if cache:
            return _cached_device_array(value, np.int32, int(device_id), cache_stats)
        return cuda.to_device(np.ascontiguousarray(value, dtype=np.int32))

    d_source = to_float32(source)
    d_listener = to_float32(listener)
    d_directions = to_float32(directions, cache=True)
    d_diffuse_bank = to_float32(diffuse_bank, cache=True)
    d_diffuse_random = to_float32(diffuse_random, cache=True)
    d_diffuse_indices = to_int32(diffuse_indices, cache=True)
    d_kinds = to_int32(arrays["kinds"], cache=True)
    d_wall_a = to_float32(arrays["wall_a"], cache=True)
    d_wall_delta = to_float32(arrays["wall_delta"], cache=True)
    d_wall_z = to_float32(arrays["wall_z"], cache=True)
    d_z_values = to_float32(arrays["z_values"], cache=True)
    d_box_center = to_float32(arrays["box_center"], cache=True)
    d_box_axis_u = to_float32(arrays["box_axis_u"], cache=True)
    d_box_axis_v = to_float32(arrays["box_axis_v"], cache=True)
    d_box_half = to_float32(arrays["box_half"], cache=True)
    d_box_z = to_float32(arrays["box_z"], cache=True)
    d_normals = to_float32(arrays["normals"], cache=True)
    d_bvh_bounds_min = to_float32(arrays["bvh_bounds_min"], cache=True)
    d_bvh_bounds_max = to_float32(arrays["bvh_bounds_max"], cache=True)
    d_bvh_start = to_int32(arrays["bvh_start"], cache=True)
    d_bvh_count = to_int32(arrays["bvh_count"], cache=True)
    d_bvh_escape = to_int32(arrays["bvh_escape"], cache=True)
    d_bvh_primitives = to_int32(arrays["bvh_primitives"], cache=True)
    d_reflection = to_float32(arrays["reflection"], cache=True)
    d_scattering = to_float32(arrays["scattering"], cache=True)
    d_corners = to_float32(arrays["corners"], cache=True)
    d_source_forward = to_float32(source_forward_vector)

    surface_count = int(np.asarray(arrays["kinds"]).shape[0])
    visual_alloc = max(1, int(visual_candidate_limit))
    d_echogram = cuda.to_device(np.zeros((_NUM_BANDS, num_bins), dtype=np.float32))
    d_ambisonic = cuda.to_device(np.zeros((_NUM_BANDS, 4, num_bins), dtype=np.float32))
    d_hit_counts = cuda.to_device(np.zeros(surface_count, dtype=np.int32))
    d_contrib_counts = cuda.to_device(np.zeros(surface_count, dtype=np.int32))
    d_surface_energy = cuda.to_device(np.zeros(surface_count, dtype=np.float32))
    d_actual_bounces = cuda.to_device(np.zeros(1, dtype=np.int32))
    d_active_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    d_visual_hit_points = cuda.to_device(np.zeros((visual_alloc, num_bounces, 3), dtype=np.float32))
    d_visual_surface_indices = cuda.to_device(-np.ones((visual_alloc, num_bounces), dtype=np.int32))
    d_visual_ray_indices = cuda.to_device(-np.ones(visual_alloc, dtype=np.int32))
    d_visual_orders = cuda.to_device(np.zeros(visual_alloc, dtype=np.int32))
    d_visual_distances = cuda.to_device(np.zeros(visual_alloc, dtype=np.float32))
    d_visual_gains = cuda.to_device(np.zeros(visual_alloc, dtype=np.float32))
    cuda.synchronize()
    upload_time = time.perf_counter() - transfer_started

    threads = 128
    blocks = (directions.shape[0] + threads - 1) // threads
    kernel_started = time.perf_counter()
    _trace_energy_cuda_kernel[blocks, threads](
        d_source,
        d_listener,
        d_directions,
        d_diffuse_bank,
        d_diffuse_random,
        d_diffuse_indices,
        d_kinds,
        d_wall_a,
        d_wall_delta,
        d_wall_z,
        d_z_values,
        d_box_center,
        d_box_axis_u,
        d_box_axis_v,
        d_box_half,
        d_box_z,
        d_normals,
        d_bvh_bounds_min,
        d_bvh_bounds_max,
        d_bvh_start,
        d_bvh_count,
        d_bvh_escape,
        d_bvh_primitives,
        bool(use_bvh),
        d_reflection,
        d_scattering,
        d_corners,
        int(num_bounces),
        int(num_bins),
        np.float32(bin_dur),
        np.float32(speed_of_sound),
        np.float32(max_path_len),
        np.float32(direct_delay),
        np.float32(listener_radius),
        np.float32(source_radius),
        np.float32(irradiance_min_distance),
        np.float32(specular_exponent),
        d_source_forward,
        np.float32(dipole_weight),
        np.float32(dipole_power),
        int(visual_candidate_limit),
        int(visual_stride),
        bool(render_ambisonics),
        d_echogram,
        d_ambisonic,
        d_hit_counts,
        d_contrib_counts,
        d_surface_energy,
        d_actual_bounces,
        d_active_count,
        d_visual_hit_points,
        d_visual_surface_indices,
        d_visual_ray_indices,
        d_visual_orders,
        d_visual_distances,
        d_visual_gains,
    )
    cuda.synchronize()
    kernel_time = time.perf_counter() - kernel_started

    download_started = time.perf_counter()
    result = {
        "echogram": d_echogram.copy_to_host(),
        "ambisonic": d_ambisonic.copy_to_host(),
        "hit_counts": d_hit_counts.copy_to_host(),
        "contrib_counts": d_contrib_counts.copy_to_host(),
        "surface_energy": d_surface_energy.copy_to_host(),
        "actual_bounces": int(d_actual_bounces.copy_to_host()[0]),
        "active_count": int(d_active_count.copy_to_host()[0]),
        "visual_hit_points": d_visual_hit_points.copy_to_host()[:visual_candidate_limit],
        "visual_surface_indices": d_visual_surface_indices.copy_to_host()[:visual_candidate_limit],
        "visual_ray_indices": d_visual_ray_indices.copy_to_host()[:visual_candidate_limit],
        "visual_orders": d_visual_orders.copy_to_host()[:visual_candidate_limit],
        "visual_distances": d_visual_distances.copy_to_host()[:visual_candidate_limit],
        "visual_gains": d_visual_gains.copy_to_host()[:visual_candidate_limit],
    }
    cuda.synchronize()
    result["kernel_time_s"] = kernel_time
    result["transfer_time_s"] = upload_time + (time.perf_counter() - download_started)
    name = device.name.decode("utf-8") if isinstance(device.name, bytes) else str(device.name)
    result["device"] = {
        "id": int(device_id),
        "name": name,
        "compute_capability": ".".join(str(value) for value in device.compute_capability),
    }
    result["device_input_cache"] = cache_stats
    return result


def _cached_device_array(value: Any, dtype: Any, device_id: int, stats: dict[str, int]) -> Any:
    global _DEVICE_INPUT_CACHE_BYTES
    host = np.asarray(value)
    key = (
        int(device_id),
        int(host.__array_interface__["data"][0]),
        host.shape,
        host.strides,
        host.dtype.str,
        np.dtype(dtype).str,
    )
    with _DEVICE_CACHE_LOCK:
        cached = _DEVICE_INPUT_CACHE.get(key)
        if cached is not None and cached[0] is value:
            _DEVICE_INPUT_CACHE.move_to_end(key)
            stats["hits"] += 1
            return cached[1]

    converted = np.ascontiguousarray(value, dtype=dtype)
    device_value = cuda.to_device(converted)
    size = int(converted.nbytes)
    with _DEVICE_CACHE_LOCK:
        stats["misses"] += 1
        previous = _DEVICE_INPUT_CACHE.pop(key, None)
        if previous is not None:
            _DEVICE_INPUT_CACHE_BYTES -= previous[2]
        while _DEVICE_INPUT_CACHE and _DEVICE_INPUT_CACHE_BYTES + size > _DEVICE_CACHE_LIMIT_BYTES:
            _, expired = _DEVICE_INPUT_CACHE.popitem(last=False)
            _DEVICE_INPUT_CACHE_BYTES -= expired[2]
        if size <= _DEVICE_CACHE_LIMIT_BYTES:
            _DEVICE_INPUT_CACHE[key] = (value, device_value, size)
            _DEVICE_INPUT_CACHE_BYTES += size
    return device_value


if cuda is not None:

    @cuda.jit(device=True, inline=True)
    def _point_in_polygon_cuda(x, y, corners):
        inside = False
        j = corners.shape[0] - 1
        for i in range(corners.shape[0]):
            xi = corners[i, 0]
            yi = corners[i, 1]
            xj = corners[j, 0]
            yj = corners[j, 1]
            if (yi > y) != (yj > y):
                denom = yj - yi
                if abs(denom) > np.float32(1e-12):
                    x_cross = (xj - xi) * (y - yi) / denom + xi
                    if x < x_cross:
                        inside = not inside
            j = i
        return inside


    @cuda.jit(device=True, inline=True)
    def _normalize3_cuda(x, y, z):
        norm = math.sqrt(x * x + y * y + z * z)
        if norm <= np.float32(1e-12):
            return np.float32(0.0), np.float32(0.0), np.float32(0.0)
        return x / norm, y / norm, z / norm


    @cuda.jit(device=True, inline=True)
    def _box_hit_cuda(origin, direction, center, axis_u, axis_v, half, z_range):
        relx = origin[0] - center[0]
        rely = origin[1] - center[1]
        ox = relx * axis_u[0] + rely * axis_u[1]
        oy = relx * axis_v[0] + rely * axis_v[1]
        oz = origin[2]
        dx = direction[0] * axis_u[0] + direction[1] * axis_u[1]
        dy = direction[0] * axis_v[0] + direction[1] * axis_v[1]
        dz = direction[2]
        t_min = -_INF
        t_max = _INF
        nx = ny = nz = np.float32(0.0)
        ex_nx = ex_ny = np.float32(0.0)
        ex_nz = np.float32(1.0)
        for axis_i in range(3):
            if axis_i == 0:
                o, d = ox, dx
                lo, hi = -half[0], half[0]
                ax, ay, az = axis_u[0], axis_u[1], np.float32(0.0)
            elif axis_i == 1:
                o, d = oy, dy
                lo, hi = -half[1], half[1]
                ax, ay, az = axis_v[0], axis_v[1], np.float32(0.0)
            else:
                o, d = oz, dz
                lo, hi = z_range[0], z_range[1]
                ax, ay, az = np.float32(0.0), np.float32(0.0), np.float32(1.0)
            if abs(d) <= np.float32(1e-12):
                if o < lo or o > hi:
                    return _INF, np.float32(0.0), np.float32(0.0), np.float32(1.0)
                continue
            if d > np.float32(0.0):
                enter = (lo - o) / d
                exit_ = (hi - o) / d
                en_x, en_y, en_z = -ax, -ay, -az
                out_x, out_y, out_z = ax, ay, az
            else:
                enter = (hi - o) / d
                exit_ = (lo - o) / d
                en_x, en_y, en_z = ax, ay, az
                out_x, out_y, out_z = -ax, -ay, -az
            if enter > t_min:
                t_min = enter
                nx, ny, nz = en_x, en_y, en_z
            if exit_ < t_max:
                t_max = exit_
                ex_nx, ex_ny, ex_nz = out_x, out_y, out_z
            if t_min > t_max:
                return _INF, np.float32(0.0), np.float32(0.0), np.float32(1.0)
        if t_max <= _EPS:
            return _INF, np.float32(0.0), np.float32(0.0), np.float32(1.0)
        if t_min > _EPS:
            return t_min, nx, ny, nz
        return t_max, ex_nx, ex_ny, ex_nz


    @cuda.jit(device=True, inline=True)
    def _surface_hit_cuda(si, origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners):
        t = _INF
        surf_nx = normals[si, 0]
        surf_ny = normals[si, 1]
        surf_nz = normals[si, 2]
        if kinds[si] == 0:
            sx = wall_delta[si, 0]
            sy = wall_delta[si, 1]
            det = sx * direction[1] - sy * direction[0]
            if abs(det) > np.float32(1e-12):
                relx = origin[0] - wall_a[si, 0]
                rely = origin[1] - wall_a[si, 1]
                candidate = (relx * sy - rely * sx) / det
                u = (relx * direction[1] - rely * direction[0]) / det
                z = origin[2] + candidate * direction[2]
                if candidate > _EPS and u >= np.float32(-1e-6) and u <= np.float32(1.0 + 1e-6) and z >= wall_z[si, 0] - np.float32(1e-6) and z <= wall_z[si, 1] + np.float32(1e-6):
                    t = candidate
        elif kinds[si] == 2:
            candidate, cand_nx, cand_ny, cand_nz = _box_hit_cuda(
                origin, direction, box_center[si], box_axis_u[si], box_axis_v[si], box_half[si], box_z[si]
            )
            if candidate > _EPS and candidate < np.float32(1e29):
                t = candidate
                surf_nx, surf_ny, surf_nz = cand_nx, cand_ny, cand_nz
        elif abs(direction[2]) > np.float32(1e-12):
            candidate = (z_values[si] - origin[2]) / direction[2]
            px = origin[0] + candidate * direction[0]
            py = origin[1] + candidate * direction[1]
            if candidate > _EPS and _point_in_polygon_cuda(px, py, corners):
                t = candidate
        return t, surf_nx, surf_ny, surf_nz


    @cuda.jit(device=True, inline=True)
    def _orient_hit_cuda(best_surface, best_t, nx, ny, nz, direction):
        if best_surface >= 0 and nx * direction[0] + ny * direction[1] + nz * direction[2] > np.float32(0.0):
            nx, ny, nz = -nx, -ny, -nz
        return best_surface, best_t, nx, ny, nz


    @cuda.jit(device=True, inline=True)
    def _ray_aabb_intersects_cuda(origin, direction, lower, upper, max_distance):
        enter = np.float32(0.0)
        exit_ = max_distance + _BVH_BOUNDS_EPS
        for axis in range(3):
            value = direction[axis]
            if abs(value) <= np.float32(1e-12):
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


    @cuda.jit(device=True, inline=True)
    def _closest_hit_linear_cuda(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners):
        best_t = _INF
        best_surface = -1
        best_nx = best_ny = best_nz = np.float32(0.0)
        for si in range(kinds.shape[0]):
            t, nx, ny, nz = _surface_hit_cuda(si, origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners)
            if t < best_t:
                best_t = t
                best_surface = si
                best_nx, best_ny, best_nz = nx, ny, nz
        return _orient_hit_cuda(best_surface, best_t, best_nx, best_ny, best_nz, direction)


    @cuda.jit(device=True, inline=True)
    def _closest_hit_bvh_cuda(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives):
        best_t = _INF
        best_surface = -1
        best_nx = best_ny = best_nz = np.float32(0.0)
        node = 0
        while node < bvh_start.shape[0]:
            if not _ray_aabb_intersects_cuda(origin, direction, bvh_bounds_min[node], bvh_bounds_max[node], best_t):
                node = bvh_escape[node]
                continue
            count = bvh_count[node]
            if count > 0:
                start = bvh_start[node]
                for offset in range(count):
                    si = bvh_primitives[start + offset]
                    t, nx, ny, nz = _surface_hit_cuda(si, origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners)
                    if t < best_t or (t == best_t and (best_surface < 0 or si < best_surface)):
                        best_t = t
                        best_surface = si
                        best_nx, best_ny, best_nz = nx, ny, nz
            node += 1
        return _orient_hit_cuda(best_surface, best_t, best_nx, best_ny, best_nz, direction)


    @cuda.jit(device=True, inline=True)
    def _closest_hit_cuda(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives, use_bvh):
        if use_bvh:
            return _closest_hit_bvh_cuda(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners, bvh_bounds_min, bvh_bounds_max, bvh_start, bvh_count, bvh_escape, bvh_primitives)
        return _closest_hit_linear_cuda(origin, direction, kinds, wall_a, wall_delta, wall_z, z_values, box_center, box_axis_u, box_axis_v, box_half, box_z, normals, corners)


    @cuda.jit(device=True, inline=True)
    def _ray_sphere_before_cuda(origin, direction, cx, cy, cz, radius, max_distance):
        ox = origin[0] - cx
        oy = origin[1] - cy
        oz = origin[2] - cz
        b = ox * direction[0] + oy * direction[1] + oz * direction[2]
        c = ox * ox + oy * oy + oz * oz - radius * radius
        disc = b * b - c
        if disc < np.float32(0.0):
            return False
        clipped_disc = disc if disc > np.float32(0.0) else np.float32(0.0)
        root = -b - math.sqrt(clipped_disc)
        return root >= np.float32(0.0) and root < max_distance


    @cuda.jit(device=True, inline=True)
    def _diffuse_direction_cuda(sample, nx, ny, nz):
        hx, hy, hz = np.float32(1.0), np.float32(0.0), np.float32(0.0)
        if abs(nx) >= np.float32(0.9):
            hx, hy = np.float32(0.0), np.float32(1.0)
        tx = hy * nz - hz * ny
        ty = hz * nx - hx * nz
        tz = hx * ny - hy * nx
        tx, ty, tz = _normalize3_cuda(tx, ty, tz)
        bx = ny * tz - nz * ty
        by = nz * tx - nx * tz
        bz = nx * ty - ny * tx
        dx = sample[0] * tx + sample[1] * bx + sample[2] * nx
        dy = sample[0] * ty + sample[1] * by + sample[2] * ny
        dz = sample[0] * tz + sample[1] * bz + sample[2] * nz
        return _normalize3_cuda(dx, dy, dz)


    @cuda.jit
    def _trace_energy_cuda_kernel(
        source,
        listener,
        directions,
        diffuse_bank,
        diffuse_random,
        diffuse_indices,
        kinds,
        wall_a,
        wall_delta,
        wall_z,
        z_values,
        box_center,
        box_axis_u,
        box_axis_v,
        box_half,
        box_z,
        normals,
        bvh_bounds_min,
        bvh_bounds_max,
        bvh_start,
        bvh_count,
        bvh_escape,
        bvh_primitives,
        use_bvh,
        reflection,
        scattering,
        corners,
        num_bounces,
        num_bins,
        bin_dur,
        speed_of_sound,
        max_path_len,
        direct_delay,
        listener_radius,
        source_radius,
        irradiance_min_distance,
        specular_exponent,
        source_forward_vector,
        dipole_weight,
        dipole_power,
        visual_candidate_limit,
        visual_stride,
        render_ambisonics,
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
    ):
        ri = cuda.grid(1)
        if ri >= directions.shape[0]:
            return

        origin = cuda.local.array(3, dtype=float32)
        direction = cuda.local.array(3, dtype=float32)
        shadow_origin = cuda.local.array(3, dtype=float32)
        shadow_direction = cuda.local.array(3, dtype=float32)
        accum_energy = cuda.local.array(_NUM_BANDS, dtype=float32)
        for axis in range(3):
            origin[axis] = listener[axis]
            direction[axis] = directions[ri, axis]
        for bi in range(_NUM_BANDS):
            accum_energy[bi] = np.float32(1.0)

        visual_slot = -1
        if visual_candidate_limit > 0 and ri % visual_stride == 0:
            candidate_slot = ri // visual_stride
            if candidate_slot < visual_candidate_limit:
                visual_slot = candidate_slot
                visual_ray_indices[visual_slot] = ri

        source_distance = math.sqrt(
            (listener[0] - source[0]) ** 2
            + (listener[1] - source[1]) ** 2
            + (listener[2] - source[2]) ** 2
        )
        accum_distance = np.float32(0.0)
        alive = True
        ray_count = directions.shape[0] if directions.shape[0] > 1 else 1
        ray_scale = _FOUR_PI / ray_count

        for bounce in range(num_bounces):
            cuda.atomic.max(actual_bounces, 0, bounce + 1)
            surf, distance, nx, ny, nz = _closest_hit_cuda(
                origin,
                direction,
                kinds,
                wall_a,
                wall_delta,
                wall_z,
                z_values,
                box_center,
                box_axis_u,
                box_axis_v,
                box_half,
                box_z,
                normals,
                corners,
                bvh_bounds_min,
                bvh_bounds_max,
                bvh_start,
                bvh_count,
                bvh_escape,
                bvh_primitives,
                use_bvh,
            )
            if surf < 0 or distance <= listener_radius or accum_distance > max_path_len:
                alive = False
                break
            if bounce > 0:
                if _ray_sphere_before_cuda(origin, direction, listener[0], listener[1], listener[2], listener_radius, distance):
                    alive = False
                    break
                if source_distance > source_radius and _ray_sphere_before_cuda(origin, direction, source[0], source[1], source[2], source_radius, distance):
                    alive = False
                    break

            cuda.atomic.add(hit_counts, surf, 1)
            hx = origin[0] + distance * direction[0] + _HIT_OFFSET * nx
            hy = origin[1] + distance * direction[1] + _HIT_OFFSET * ny
            hz = origin[2] + distance * direction[2] + _HIT_OFFSET * nz
            if visual_slot >= 0:
                visual_hit_points[visual_slot, bounce, 0] = hx
                visual_hit_points[visual_slot, bounce, 1] = hy
                visual_hit_points[visual_slot, bounce, 2] = hz
                visual_surface_indices[visual_slot, bounce] = surf

            tsx = source[0] - hx
            tsy = source[1] - hy
            tsz = source[2] - hz
            distance_to_source = math.sqrt(tsx * tsx + tsy * tsy + tsz * tsz)
            if distance_to_source > irradiance_min_distance:
                sdx = tsx / distance_to_source
                sdy = tsy / distance_to_source
                sdz = tsz / distance_to_source
                facing = nx * sdx + ny * sdy + nz * sdz
                if facing > np.float32(0.0):
                    shadow_origin[0], shadow_origin[1], shadow_origin[2] = hx, hy, hz
                    shadow_direction[0], shadow_direction[1], shadow_direction[2] = sdx, sdy, sdz
                    shadow_surface, shadow_distance, _, _, _ = _closest_hit_cuda(
                        shadow_origin,
                        shadow_direction,
                        kinds,
                        wall_a,
                        wall_delta,
                        wall_z,
                        z_values,
                        box_center,
                        box_axis_u,
                        box_axis_v,
                        box_half,
                        box_z,
                        normals,
                        corners,
                        bvh_bounds_min,
                        bvh_bounds_max,
                        bvh_start,
                        bvh_count,
                        bvh_escape,
                        bvh_primitives,
                        use_bvh,
                    )
                    occluded = shadow_surface >= 0 and shadow_distance > _EPS and shadow_distance < distance_to_source - _EPS
                    if not occluded:
                        diffuse = _INV_PI * scattering[surf] * facing
                        halfx, halfy, halfz = _normalize3_cuda(sdx - direction[0], sdy - direction[1], sdz - direction[2])
                        raw_cos_half = halfx * nx + halfy * ny + halfz * nz
                        cos_half = raw_cos_half if raw_cos_half > np.float32(0.0) else np.float32(0.0)
                        specular = (specular_exponent + np.float32(2.0)) * _INV_8PI * (np.float32(1.0) - scattering[surf]) * (cos_half ** specular_exponent)
                        irradiance_distance = distance_to_source if distance_to_source > irradiance_min_distance else irradiance_min_distance
                        distance_term = _INV_4PI / (irradiance_distance ** 2)
                        source_cosine = -(source_forward_vector[0] * sdx + source_forward_vector[1] * sdy + source_forward_vector[2] * sdz)
                        source_gain = abs((np.float32(1.0) - dipole_weight) + dipole_weight * source_cosine) ** dipole_power
                        relative_delay = (accum_distance + distance + distance_to_source) / speed_of_sound - direct_delay
                        bin_index = int(math.floor(relative_delay / bin_dur))
                        if 0 <= bin_index < num_bins:
                            energy_sum = np.float32(0.0)
                            for bi in range(_NUM_BANDS):
                                energy = ray_scale * source_gain * distance_term * (diffuse + specular) * reflection[surf, bi] * accum_energy[bi]
                                cuda.atomic.add(echogram, (bi, bin_index), energy)
                                if render_ambisonics:
                                    cuda.atomic.add(ambisonic, (bi, 0, bin_index), energy)
                                    cuda.atomic.add(ambisonic, (bi, 1, bin_index), energy * directions[ri, 0])
                                    cuda.atomic.add(ambisonic, (bi, 2, bin_index), energy * directions[ri, 1])
                                    cuda.atomic.add(ambisonic, (bi, 3, bin_index), energy * directions[ri, 2])
                                energy_sum += energy
                            if visual_slot >= 0 and energy_sum > visual_gains[visual_slot] * visual_gains[visual_slot]:
                                visual_orders[visual_slot] = bounce + 1
                                visual_distances[visual_slot] = accum_distance + distance + distance_to_source
                                clipped_energy = energy_sum if energy_sum > np.float32(0.0) else np.float32(0.0)
                                visual_gains[visual_slot] = math.sqrt(clipped_energy)
                            cuda.atomic.add(contrib_counts, surf, 1)
                            cuda.atomic.add(surface_energy, surf, energy_sum)

            for bi in range(_NUM_BANDS):
                accum_energy[bi] *= reflection[surf, bi]
            accum_distance += distance
            origin[0], origin[1], origin[2] = hx, hy, hz
            if diffuse_random[bounce, ri] < scattering[surf]:
                sample_index = diffuse_indices[bounce, ri]
                dx, dy, dz = _diffuse_direction_cuda(diffuse_bank[sample_index], nx, ny, nz)
                direction[0], direction[1], direction[2] = dx, dy, dz
            else:
                dot = direction[0] * nx + direction[1] * ny + direction[2] * nz
                dx = direction[0] - np.float32(2.0) * dot * nx
                dy = direction[1] - np.float32(2.0) * dot * ny
                dz = direction[2] - np.float32(2.0) * dot * nz
                direction[0], direction[1], direction[2] = _normalize3_cuda(dx, dy, dz)
            if accum_distance > max_path_len:
                alive = False
                break

        if alive:
            cuda.atomic.add(active_count, 0, 1)
