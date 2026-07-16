from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .acoustics import estimate_rt60
from .directivity import source_directivity
from .geometry import point_in_polygon, polygon_area, polygon_perimeter
from .hrtf import render_binaural_sofa
from .mic import channel_positions, microphone_array
from .models import AcousticPath, Room, SimConfig, vec3
from .steam_rt import simulate_steam_room


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
    steam = simulate_steam_room(room, source, receiver, config, source_model=source_model)
    paths = tuple(sorted(tuple(steam.paths), key=lambda path: (path.delay_s, path.kind, -abs(path.gain))))
    rir = steam.rir
    area = abs(polygon_area(room.corners))
    perimeter = polygon_perimeter(room.corners)
    material_rt60 = estimate_rt60(area, perimeter, room.height_m, room.materials)
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
        "hybrid_model": "steam_hybrid_energy_envelope_from_traced_field",
        "traced_rt60_bands": {band: round(float(value), 4) for band, value in steam.rt60_bands.items()},
        "traced_rt60_s": round(float(steam.rt60_s), 4),
        "traced_model": "schroeder_fit_from_path_traced_echogram",
        "material_rt60_bands": dict(material_rt60.get("rt60_bands", {})),
        "material_rt60_s": float(material_rt60.get("rt60_s", 0.0)),
        "material_steam_audio_rt60_bands": dict(material_steam_audio_bands),
        "material_model": str(material_rt60.get("model", "sabine_extruded_polygon")),
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
            "hybrid_model": "steam_hybrid_energy_envelope_from_traced_field",
            "traced_rt60_bands": {band: round(float(value), 4) for band, value in steam.rt60_bands.items()},
            "traced_rt60_s": round(float(steam.rt60_s), 4),
            "traced_model": "schroeder_fit_from_path_traced_echogram",
            "material_rt60_bands": dict(material_rt60.get("rt60_bands", {})),
            "material_rt60_s": float(material_rt60.get("rt60_s", 0.0)),
            "material_steam_audio_rt60_bands": dict(material_steam_audio_bands),
            "material_model": str(material_rt60.get("model", "sabine_extruded_polygon")),
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
                "model": "steam_hybrid_energy_envelope_from_traced_field",
                "enabled": bool(config.late_tail),
                "transition_s": float(config.hybrid_transition_s),
                "overlap_fraction": float(config.hybrid_overlap_fraction),
            },
            "steam_audio": steam.metadata,
            "source_model": dict(source_model),
            "solver_pipeline": ["direct", "diffraction", "rt_energy_field", "reverb_estimate", "hybrid_late_reverb", "receiver_reconstruction"],
        },
        steam.ambisonic_rir,
        dict(source_model),
    )


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
    if not point_in_polygon(value[:2], room.corners):
        raise ValueError(f"{label} point is outside the room polygon: {tuple(value)}")
    if not (0.0 <= float(value[2]) <= float(room.height_m)):
        raise ValueError(f"{label} z must be inside [0, {room.height_m}]")
    return tuple(float(v) for v in value)
