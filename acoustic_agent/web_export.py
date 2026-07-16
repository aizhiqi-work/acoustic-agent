from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .engine import SimulationResult
from .models import Room


def scene_payload(
    room: Room,
    *,
    sources: Sequence[Sequence[float]] = (),
    receivers: Sequence[Sequence[float]] = (),
    result: SimulationResult | None = None,
    include_exact_rir: bool = True,
) -> dict:
    return {
        "room": {
            "id": room.id,
            "name": room.name,
            "corners": [list(point) for point in room.corners],
            "height_m": float(room.height_m),
            "materials": {key: {"id": value.id, "name": value.name, "absorption": dict(value.absorption)} for key, value in room.materials.items()},
            "metadata": dict(room.metadata),
        },
        "sources": [list(point) for point in sources],
        "receivers": [list(point) for point in receivers],
        "paths": [_path_payload(path) for path in (result.paths if result else ())],
        "rir": _rir_payload(result, include_exact=include_exact_rir) if result else {},
        "rt60": dict(result.rt60) if result else {},
        "metadata": dict(result.metadata) if result else {},
    }


def export_scene_json(
    room: Room,
    path: str | Path,
    *,
    sources: Sequence[Sequence[float]] = (),
    receivers: Sequence[Sequence[float]] = (),
    result: SimulationResult | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(scene_payload(room, sources=sources, receivers=receivers, result=result), indent=2), encoding="utf-8")
    return destination


def _path_payload(path) -> Mapping:
    return {
        "kind": path.kind,
        "distance_m": round(float(path.distance_m), 6),
        "delay_s": round(float(path.delay_s), 8),
        "gain": round(float(path.gain), 8),
        "gain_db": round(20.0 * math.log10(max(abs(float(path.gain)), 1e-12)), 3),
        "band_gains": {str(key): round(float(value), 10) for key, value in path.band_gains.items()},
        "points": [list(point) for point in path.points],
        "metadata": dict(path.metadata),
    }


def _rir_payload(result: SimulationResult, *, include_exact: bool = True) -> Mapping:
    values = np.asarray(result.rir, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    exact_values = np.ascontiguousarray(values, dtype="<f4")
    max_points = 48000
    stride = max(1, int(math.ceil(values.shape[1] / max_points)))
    channel_limit = min(int(values.shape[0]), 2)
    channel_samples = [_peak_preserving_preview(values[index], stride) for index in range(channel_limit)]
    preview = channel_samples[0] if channel_samples else []
    fs = int(result.metadata.get("sample_rate", 16000))
    channel_labels = _channel_labels(result, int(values.shape[0]))
    payload = {
        "fs": fs,
        "duration_s": float(values.shape[1] / max(fs, 1)),
        "channel_count": int(values.shape[0]),
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "preview_channel_count": channel_limit,
        "sample_stride": stride,
        "samples": preview,
        "channel_samples": channel_samples,
        "channel_labels": channel_labels[:channel_limit],
        "decay_db": _energy_decay_preview(values, stride),
        "metrics": _rir_metrics(values, fs, result.metadata),
        "representation": "channel_peak_preserving_preview",
    }
    if include_exact:
        payload["encoding"] = "float32-le-base64-planar"
        payload["f32_base64"] = base64.b64encode(exact_values.tobytes(order="C")).decode("ascii")
    return payload


def _peak_preserving_preview(signal: np.ndarray, stride: int) -> list[float]:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if stride > 1:
        preview = []
        for start in range(0, len(values), stride):
            block = values[start:start + stride]
            preview.append(float(block[int(np.argmax(np.abs(block)))]) if len(block) else 0.0)
        return preview
    return [float(value) for value in values]


def _energy_decay_preview(values: np.ndarray, stride: int) -> list[float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.size == 0 or samples.shape[1] == 0:
        return []
    energy = np.sum(samples * samples, axis=0)
    edc = np.cumsum(energy[::-1])[::-1]
    total = max(float(edc[0]), 1e-18)
    db = 10.0 * np.log10(np.maximum(edc, 1e-18) / total)
    return [round(float(db[index]), 3) for index in range(0, len(db), max(1, int(stride)))]


def _channel_labels(result: SimulationResult, channel_count: int) -> list[str]:
    receiver_type = str(result.receiver_model.get("type", "mono"))
    if receiver_type == "hrtf" and channel_count >= 2:
        return ["L", "R"] + [f"Ch {index + 1}" for index in range(2, channel_count)]
    return [f"Ch {index + 1}" for index in range(channel_count)]


def _rir_metrics(values: np.ndarray, fs: int, metadata: Mapping[str, Any]) -> Mapping[str, float | None]:
    fs = max(int(fs), 1)
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.size == 0 or samples.shape[1] == 0:
        return {}

    energy_by_sample = np.sum(samples * samples, axis=0)
    total_energy = float(np.sum(energy_by_sample))
    peak_abs = float(np.max(np.abs(samples)))
    peak_index = int(np.argmax(np.max(np.abs(samples), axis=0)))
    rms = float(np.sqrt(np.mean(samples * samples)))
    direct_delay_s = float(((metadata.get("steam_audio") or {}).get("direct") or {}).get("delay_s", 0.0))
    direct_index = int(np.clip(round(direct_delay_s * fs), 0, samples.shape[1] - 1))

    def db_power(value: float) -> float | None:
        if value <= 1e-18:
            return None
        return round(float(10.0 * math.log10(value)), 2)

    def db_amplitude(value: float) -> float | None:
        if value <= 1e-12:
            return None
        return round(float(20.0 * math.log10(value)), 2)

    def clarity_db(window_s: float) -> float | None:
        split = min(samples.shape[1], direct_index + max(1, int(round(window_s * fs))))
        early = float(np.sum(energy_by_sample[direct_index:split]))
        late = float(np.sum(energy_by_sample[split:]))
        if early <= 1e-18 or late <= 1e-18:
            return None
        return round(float(10.0 * math.log10(early / late)), 2)

    direct_half_window = max(1, int(round(0.0025 * fs)))
    lo = max(0, direct_index - direct_half_window)
    hi = min(samples.shape[1], direct_index + direct_half_window + 1)
    direct_energy = float(np.sum(energy_by_sample[lo:hi]))
    reverb_energy = max(total_energy - direct_energy, 0.0)
    drr_db = None
    if direct_energy > 1e-18 and reverb_energy > 1e-18:
        drr_db = round(float(10.0 * math.log10(direct_energy / reverb_energy)), 2)

    return {
        "peak_dbfs": db_amplitude(peak_abs),
        "peak_time_ms": round(float(peak_index / fs * 1000.0), 2),
        "rms_dbfs": db_amplitude(rms),
        "energy_db": db_power(total_energy),
        "direct_delay_ms": round(float(direct_delay_s * 1000.0), 2),
        "drr_db": drr_db,
        "c50_db": clarity_db(0.05),
        "c80_db": clarity_db(0.08),
    }
