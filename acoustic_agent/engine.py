from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .acoustics import estimate_rt60, estimate_surface_rt60
from .directivity import source_directivity
from .geometry import point_in_polygon, polygon_area, polygon_perimeter
from .hrtf import render_binaural_sofa
from .mic import channel_positions, microphone_array
from .models import FREQUENCY_BANDS, AcousticPath, Room, SimConfig, vec3
from .motion import room_for_motion_frame
from .steam_rt import estimate_signal_decay_profile, object_absorption_areas, simulate_steam_room


@dataclass(frozen=True)
class SimulationResult:
    rir: np.ndarray
    paths: tuple[AcousticPath, ...]
    rt60: Mapping[str, Any]
    receiver_model: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ambisonic_rir: np.ndarray | None = None
    source_model: Mapping[str, Any] = field(default_factory=dict)


def simulate_rir(
    room: Room,
    source: Sequence[float],
    receiver: Sequence[float],
    *,
    config: SimConfig | None = None,
    receiver_model: Mapping[str, Any] | None = None,
    source_model: str | Mapping[str, Any] | None = None,
) -> SimulationResult:
    cfg = config or SimConfig()
    room = room_for_motion_frame(room, source, receiver)
    src = _validate_point(source, room, "source")
    rcv = _validate_point(receiver, room, "receiver")
    model = dict(receiver_model or microphone_array("mono"))
    emitter = source_directivity(source_model)
    kind = str(model.get("type", "mono"))
    if kind == "hrtf":
        mono = _simulate_mono(room, src, rcv, cfg, emitter)
        rendered, hrtf_meta = render_binaural_sofa(
            mono.rir,
            source=src,
            receiver=rcv,
            fs=cfg.fs,
            paths=mono.paths,
            sofa_path=model.get("sofa_path"),
            interpolation=str(model.get("interpolation", "bilinear")),
            orientation_deg=float(model.get("orientation_deg", 0.0)),
            spatial_blend=float(model.get("spatial_blend", 1.0)),
            loudness_normalization=str(model.get("loudness_normalization", "energy")),
            seed=int(cfg.seed),
            ambisonic_rir=mono.ambisonic_rir,
        )
        merged_model = {**model, "render_metadata": hrtf_meta}
        return SimulationResult(rendered, mono.paths, mono.rt60, merged_model, mono.metadata, mono.ambisonic_rir, emitter)
    if kind in {"linear_array", "circular_array"}:
        channel_results = [_simulate_mono(room, src, position, cfg, emitter) for position in channel_positions(rcv, model)]
        rir = np.stack([item.rir for item in channel_results], axis=0).astype(np.float32)
        paths = tuple(path for item in channel_results for path in item.paths)
        meta = dict(channel_results[0].metadata)
        meta["array_channel_count"] = int(rir.shape[0])
        return SimulationResult(rir, paths, channel_results[0].rt60, model, meta, source_model=emitter)
    mono = _simulate_mono(room, src, rcv, cfg, emitter)
    return SimulationResult(mono.rir.reshape(1, -1), mono.paths, mono.rt60, model, mono.metadata, mono.ambisonic_rir, emitter)


def solve_paths(
    room: Room,
    source: Sequence[float],
    receiver: Sequence[float],
    config: SimConfig | None = None,
    source_model: str | Mapping[str, Any] | None = None,
) -> tuple[AcousticPath, ...]:
    cfg = config or SimConfig()
    room = room_for_motion_frame(room, source, receiver)
    src = _validate_point(source, room, "source")
    rcv = _validate_point(receiver, room, "receiver")
    result = simulate_steam_room(room, src, rcv, cfg, source_model=source_directivity(source_model))
    return tuple(sorted(result.paths, key=lambda path: (path.delay_s, path.kind, -abs(path.gain))))


def _simulate_mono(
    room: Room,
    source: tuple[float, float, float],
    receiver: tuple[float, float, float],
    config: SimConfig,
    source_model: Mapping[str, Any],
) -> SimulationResult:
    material_rt60 = _estimate_material_rt60(room)
    steam = simulate_steam_room(
        room,
        source,
        receiver,
        config,
        source_model=source_model,
        late_reverb_prior=material_rt60.get("coupled_rt60_bands"),
    )
    paths = tuple(sorted(tuple(steam.paths), key=lambda path: (path.delay_s, path.kind, -abs(path.gain))))
    rir = steam.rir
    decay_profile = _decay_profile_with_context(room, estimate_signal_decay_profile(rir, config))
    rir_rt60_bands = {band: round(float(value), 4) for band, value in steam.rir_rt60_bands.items()}
    rir_rt60_s = round(float(steam.rir_rt60_s), 4)
    steam_audio_rt60_bands = {band: round(float(value), 4) for band, value in steam.steam_audio_rt60_bands.items()}
    material_steam_audio_bands = _steam_audio_default_material_bands(material_rt60.get("rt60_bands", {}))
    hybrid_rt60_bands = {band: round(float(value), 4) for band, value in steam.hybrid_rt60_bands.items()}
    hybrid_rt60_s = round(float(steam.hybrid_rt60_s), 4)
    rt60 = {
        "rt60_bands": dict(rir_rt60_bands),
        "rt60_s": rir_rt60_s,
        "model": "schroeder_fit_from_final_rir",
        "rir_rt60_bands": dict(rir_rt60_bands),
        "rir_rt60_s": rir_rt60_s,
        "rir_model": "schroeder_fit_from_final_rir",
        "steam_audio_rt60_bands": dict(steam_audio_rt60_bands),
        "steam_audio_band_model": "default_low_mid_high",
        "hybrid_rt60_bands": dict(hybrid_rt60_bands),
        "hybrid_rt60_s": hybrid_rt60_s,
        "hybrid_model": "steam_style_16_line_hadamard_fdn",
        "traced_rt60_bands": {band: round(float(value), 4) for band, value in steam.rt60_bands.items()},
        "traced_rt60_s": round(float(steam.rt60_s), 4),
        "traced_model": "schroeder_fit_from_path_traced_echogram",
        "material_rt60_bands": dict(material_rt60.get("rt60_bands", {})),
        "material_rt60_s": float(material_rt60.get("rt60_s", 0.0)),
        "material_steam_audio_rt60_bands": dict(material_steam_audio_bands),
        "material_model": str(material_rt60.get("model", "sabine_extruded_polygon")),
        "material_scope": str(material_rt60.get("scope", "room")),
        "material_room_id": material_rt60.get("room_id"),
        "material_opening_area_m2": float(material_rt60.get("opening_area_m2", 0.0)),
        "coupled_material_rt60_bands": dict(material_rt60.get("coupled_rt60_bands", {})),
        "coupled_material_rt60_s": float(material_rt60.get("coupled_rt60_s", 0.0)),
        "coupled_decay": dict(material_rt60.get("coupled_decay", {})),
        "decay_profile": decay_profile,
    }
    if rt60["rt60_s"] <= 0.0:
        rt60 = {
            **material_rt60,
            "rir_rt60_bands": dict(rir_rt60_bands),
            "rir_rt60_s": rir_rt60_s,
            "rir_model": "schroeder_fit_from_final_rir",
            "steam_audio_rt60_bands": dict(steam_audio_rt60_bands),
            "steam_audio_band_model": "default_low_mid_high",
            "hybrid_rt60_bands": dict(hybrid_rt60_bands),
            "hybrid_rt60_s": hybrid_rt60_s,
            "hybrid_model": "steam_style_16_line_hadamard_fdn",
            "traced_rt60_bands": {band: round(float(value), 4) for band, value in steam.rt60_bands.items()},
            "traced_rt60_s": round(float(steam.rt60_s), 4),
            "traced_model": "schroeder_fit_from_path_traced_echogram",
            "material_rt60_bands": dict(material_rt60.get("rt60_bands", {})),
            "material_rt60_s": float(material_rt60.get("rt60_s", 0.0)),
            "material_steam_audio_rt60_bands": dict(material_steam_audio_bands),
            "material_model": str(material_rt60.get("model", "sabine_extruded_polygon")),
            "material_scope": str(material_rt60.get("scope", "room")),
            "material_room_id": material_rt60.get("room_id"),
            "material_opening_area_m2": float(material_rt60.get("opening_area_m2", 0.0)),
            "coupled_material_rt60_bands": dict(material_rt60.get("coupled_rt60_bands", {})),
            "coupled_material_rt60_s": float(material_rt60.get("coupled_rt60_s", 0.0)),
            "coupled_decay": dict(material_rt60.get("coupled_decay", {})),
            "decay_profile": decay_profile,
            "model": "sabine_extruded_polygon_fallback",
        }
    return SimulationResult(
        rir.astype(np.float32),
        tuple(paths),
        rt60,
        microphone_array("mono"),
        {
            "sample_rate": int(config.fs),
            "duration_s": float(config.duration_s),
            "path_count": len(paths),
            "room_id": room.id,
            "room_shape": room.metadata.get("shape"),
            "geometry_model": room.metadata.get("geometry_model"),
            "late_tail": {
                "model": "steam_style_16_line_hadamard_fdn",
                "enabled": bool(config.late_tail),
                "transition_s": float(config.hybrid_transition_s),
                "overlap_fraction": float(config.hybrid_overlap_fraction),
            },
            "steam_audio": steam.metadata,
            "source_model": dict(source_model),
            "solver_pipeline": (
                ["direct", "portal_pathing", "rt_energy_field", "reverb_estimate", "hybrid_late_reverb", "receiver_reconstruction"]
                if bool(room.metadata.get("multi_room", {}).get("enabled"))
                else ["direct", "diffraction", "rt_energy_field", "reverb_estimate", "hybrid_late_reverb", "receiver_reconstruction"]
            ),
            "multi_room": dict(room.metadata.get("multi_room", {})) if isinstance(room.metadata.get("multi_room"), Mapping) else None,
        },
        steam.ambisonic_rir,
        dict(source_model),
    )


def _decay_profile_with_context(room: Room, profile: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(profile)
    multi_room = room.metadata.get("multi_room") if isinstance(room.metadata, Mapping) else None
    if not isinstance(multi_room, Mapping) or not bool(multi_room.get("enabled")):
        result["space_context"] = "single_space_geometry"
        return result
    source_room_id = str(multi_room.get("source_room_id", ""))
    receiver_room_id = str(multi_room.get("receiver_room_id", ""))
    result["space_context"] = "cross_room" if source_room_id != receiver_room_id else "same_room"
    result["source_room_id"] = source_room_id
    result["receiver_room_id"] = receiver_room_id
    if result.get("model") == "double_slope":
        result["physical_model"] = "coupled_space_decay"
    return result


def _estimate_material_rt60(room: Room) -> dict[str, Any]:
    metadata = room.metadata
    multi_room = metadata.get("multi_room")
    segments = metadata.get("surface_segments")
    source_room_id = metadata.get("source_room_id")
    if not isinstance(multi_room, Mapping) or not isinstance(segments, list) or not source_room_id:
        area = abs(polygon_area(room.corners))
        perimeter = polygon_perimeter(room.corners)
        return {
            **estimate_rt60(area, perimeter, room.height_m, room.materials),
            "scope": "room",
            "room_id": room.id,
        }

    source_room = next(
        (
            candidate for candidate in multi_room.get("rooms", [])
            if isinstance(candidate, Mapping) and candidate.get("id") == source_room_id
        ),
        None,
    )
    if not isinstance(source_room, Mapping):
        area = abs(polygon_area(room.corners))
        perimeter = polygon_perimeter(room.corners)
        return {
            **estimate_rt60(area, perimeter, room.height_m, room.materials),
            "scope": "room",
            "room_id": room.id,
        }

    corners = source_room.get("corners", [])
    floor_area = float(source_room.get("area_m2") or abs(polygon_area(corners)))
    full_vertical_area = polygon_perimeter(corners) * float(room.height_m)
    vertical_areas: dict[str, float] = {}
    for segment in segments:
        if not isinstance(segment, Mapping) or segment.get("room_id") != source_room_id:
            continue
        a = np.asarray(segment.get("a", ()), dtype=float)
        b = np.asarray(segment.get("b", ()), dtype=float)
        if a.shape != (2,) or b.shape != (2,):
            continue
        height = max(0.0, float(segment.get("z_max", room.height_m)) - float(segment.get("z_min", 0.0)))
        area = float(np.linalg.norm(b - a)) * height
        semantic = str(segment.get("type", "wall"))
        vertical_areas[semantic] = vertical_areas.get(semantic, 0.0) + area
    opening_area = max(0.0, full_vertical_area - sum(vertical_areas.values()))
    result = {
        **estimate_surface_rt60(
            floor_area,
            room.height_m,
            room.materials,
            vertical_areas,
            opening_area_m2=opening_area,
        ),
        "scope": "source_room",
        "room_id": str(source_room_id),
    }
    coupled_decay = _estimate_coupled_room_decay(room)
    if coupled_decay:
        result["coupled_decay"] = coupled_decay
        result["coupled_rt60_bands"] = dict(coupled_decay["rt60_bands"])
        values = [float(value) for value in coupled_decay["rt60_bands"].values() if float(value) > 0.0]
        result["coupled_rt60_s"] = round(float(np.mean(values)), 4) if values else 0.0
    return result


def _estimate_coupled_room_decay(room: Room) -> dict[str, Any]:
    metadata = room.metadata if isinstance(room.metadata, Mapping) else {}
    multi_room = metadata.get("multi_room")
    segments = metadata.get("surface_segments")
    if not isinstance(multi_room, Mapping) or not isinstance(segments, list):
        return {}
    room_records = [item for item in multi_room.get("rooms", []) if isinstance(item, Mapping)]
    if len(room_records) < 2:
        return {}
    room_ids = [str(item.get("id")) for item in room_records]
    room_index = {room_id: index for index, room_id in enumerate(room_ids)}
    source_room_id = str(multi_room.get("source_room_id", ""))
    receiver_room_id = str(multi_room.get("receiver_room_id", source_room_id))
    if source_room_id not in room_index or receiver_room_id not in room_index:
        return {}

    height = max(float(room.height_m), 1e-6)
    floor_areas = np.asarray([
        max(float(item.get("area_m2") or abs(polygon_area(item.get("corners", [])))), 1e-6)
        for item in room_records
    ], dtype=np.float64)
    volumes = floor_areas * height
    absorption_area = np.zeros((len(room_ids), len(FREQUENCY_BANDS)), dtype=np.float64)
    floor_material = room.materials.get("floor") or next(iter(room.materials.values()))
    ceiling_material = room.materials.get("ceiling") or floor_material
    wall_material = room.materials.get("wall") or floor_material
    for room_i, floor_area in enumerate(floor_areas):
        for band_i, band in enumerate(FREQUENCY_BANDS):
            absorption_area[room_i, band_i] = floor_area * (
                float(floor_material.absorption.get(band, 0.1))
                + float(ceiling_material.absorption.get(band, 0.08))
            )
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        segment_room_id = str(segment.get("room_id", ""))
        if segment_room_id not in room_index:
            continue
        a = np.asarray(segment.get("a", ()), dtype=np.float64)
        b = np.asarray(segment.get("b", ()), dtype=np.float64)
        if a.shape != (2,) or b.shape != (2,):
            continue
        segment_height = max(0.0, float(segment.get("z_max", height)) - float(segment.get("z_min", 0.0)))
        area = float(np.linalg.norm(b - a)) * segment_height
        semantic = str(segment.get("type", "wall"))
        material = room.materials.get(semantic) or wall_material
        for band_i, band in enumerate(FREQUENCY_BANDS):
            absorption_area[room_index[segment_room_id], band_i] += area * float(material.absorption.get(band, 0.08))

    object_area_totals = np.zeros(len(FREQUENCY_BANDS), dtype=np.float64)
    for record in object_absorption_areas(room):
        center = np.asarray(record.get("center", ()), dtype=np.float64)
        values = np.asarray(record.get("absorption_area_m2", ()), dtype=np.float64)
        if center.shape != (2,) or values.shape != (len(FREQUENCY_BANDS),):
            continue
        containing = next(
            (
                room_i for room_i, item in enumerate(room_records)
                if point_in_polygon(center, item.get("corners", ()))
            ),
            None,
        )
        if containing is None:
            continue
        absorption_area[containing] += values
        object_area_totals += values

    portal_records: list[tuple[int, int, float, str]] = []
    exterior_opening_area = np.zeros(len(room_ids), dtype=np.float64)
    for portal in multi_room.get("portals", []):
        if not isinstance(portal, Mapping) or not bool(portal.get("open", False)):
            continue
        connected = [str(value) for value in portal.get("room_ids", []) if str(value) in room_index]
        area = max(0.0, float(portal.get("width_m", 0.0)) * float(portal.get("height_m", 0.0)))
        if len(connected) == 2 and area > 0.0:
            portal_records.append((room_index[connected[0]], room_index[connected[1]], area, str(portal.get("id", ""))))
        elif len(connected) == 1:
            exterior_opening_area[room_index[connected[0]]] += area

    speed_factor = 343.0 / 4.0
    initial = np.zeros(len(room_ids), dtype=np.float64)
    initial[room_index[source_room_id]] = 1.0
    receiver_i = room_index[receiver_room_id]
    modes_by_band: dict[str, list[dict[str, float]]] = {}
    rt60_bands: dict[str, float] = {}
    for band_i, band in enumerate(FREQUENCY_BANDS):
        matrix = np.zeros((len(room_ids), len(room_ids)), dtype=np.float64)
        for room_i in range(len(room_ids)):
            loss_area = absorption_area[room_i, band_i] + exterior_opening_area[room_i]
            matrix[room_i, room_i] -= speed_factor * loss_area / volumes[room_i]
        for first, second, area, _portal_id in portal_records:
            matrix[first, first] -= speed_factor * area / volumes[first]
            matrix[second, second] -= speed_factor * area / volumes[second]
            matrix[first, second] += speed_factor * area / volumes[second]
            matrix[second, first] += speed_factor * area / volumes[first]

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        try:
            coefficients = np.linalg.solve(eigenvectors, initial)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(eigenvectors) @ initial
        candidates: list[tuple[float, float, float]] = []
        for mode_i, eigenvalue in enumerate(eigenvalues):
            rate = -float(np.real(eigenvalue))
            if rate <= 1e-9 or abs(float(np.imag(eigenvalue))) > 1e-7:
                continue
            contribution = abs(float(np.real(eigenvectors[receiver_i, mode_i] * coefficients[mode_i] / volumes[receiver_i])))
            candidates.append((math.log(1e6) / rate, rate, contribution))
        contribution_total = sum(item[2] for item in candidates)
        normalized = [
            (rt60, rate, contribution / max(contribution_total, 1e-18))
            for rt60, rate, contribution in candidates
        ]
        relevant = [item for item in normalized if item[2] >= 0.01] or normalized
        rt60_bands[band] = round(max((item[0] for item in relevant), default=0.0), 4)
        modes_by_band[band] = [
            {
                "rt60_s": round(rt60, 4),
                "decay_rate_per_s": round(rate, 6),
                "relative_receiver_contribution": round(contribution, 6),
            }
            for rt60, rate, contribution in sorted(normalized, key=lambda item: item[2], reverse=True)[:3]
        ]
    return {
        "model": "coupled_room_energy_matrix",
        "room_ids": room_ids,
        "source_room_id": source_room_id,
        "receiver_room_id": receiver_room_id,
        "portal_count": len(portal_records),
        "portal_coupling_area_m2": round(float(sum(item[2] for item in portal_records)), 4),
        "object_absorption_area_m2": {
            band: round(float(object_area_totals[index]), 6)
            for index, band in enumerate(FREQUENCY_BANDS)
        },
        "rt60_bands": rt60_bands,
        "modes_by_band": modes_by_band,
    }


def _steam_audio_default_material_bands(rt60_bands: Mapping[str, Any]) -> dict[str, float]:
    def mean(keys: tuple[str, ...]) -> float:
        values = [float(rt60_bands[key]) for key in keys if key in rt60_bands and float(rt60_bands[key]) > 0.0]
        return round(float(np.mean(values)), 4) if values else 0.0

    return {
        "low": mean(("125", "250", "500")),
        "mid": mean(("1000", "2000", "4000")),
    }


def _validate_point(point: Sequence[float], room: Room, label: str) -> tuple[float, float, float]:
    value = vec3(point)
    multi_room = room.metadata.get("multi_room") if isinstance(room.metadata, Mapping) else None
    valid_xy = point_in_polygon(value[:2], room.corners)
    if isinstance(multi_room, Mapping) and bool(multi_room.get("enabled")):
        polygons = [
            item.get("corners")
            for item in multi_room.get("rooms", [])
            if isinstance(item, Mapping) and isinstance(item.get("corners"), Sequence)
        ]
        valid_xy = any(point_in_polygon(value[:2], corners) for corners in polygons)
    if not valid_xy:
        raise ValueError(f"{label} point is outside the room polygon: {tuple(value)}")
    if not (0.0 <= float(value[2]) <= float(room.height_m)):
        raise ValueError(f"{label} z must be inside [0, {room.height_m}]")
    return tuple(float(v) for v in value)
