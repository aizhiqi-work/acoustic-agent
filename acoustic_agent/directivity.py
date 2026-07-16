from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


_PRESETS: dict[str, tuple[float, float]] = {
    "omni": (0.0, 1.0),
    "cardioid": (0.5, 1.0),
    "dipole": (1.0, 1.0),
    "focused": (0.5, 4.0),
}


def source_directivity(value: str | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize a Steam Audio compatible weighted-dipole source model."""
    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, str):
        raw = {"type": value}
    else:
        raw = dict(value)

    requested_type = str(raw.get("type", raw.get("pattern", "omni"))).lower().replace("-", "_")
    pattern = str(raw.get("pattern", requested_type)).lower().replace("-", "_")
    if requested_type in _PRESETS:
        pattern = requested_type
    elif requested_type not in {"weighted_dipole", "directivity"}:
        raise ValueError(
            f"unknown source directivity {requested_type!r}; expected omni, cardioid, dipole, focused, or weighted_dipole"
        )
    if pattern not in _PRESETS:
        pattern = "custom"

    default_weight, default_power = _PRESETS.get(pattern, _PRESETS["cardioid"])
    weight = float(raw.get("dipole_weight", default_weight))
    power = float(raw.get("dipole_power", default_power))
    orientation = float(raw.get("orientation_deg", 0.0))
    elevation = float(raw.get("elevation_deg", 0.0))
    values = (weight, power, orientation, elevation)
    if not all(math.isfinite(item) for item in values):
        raise ValueError("source directivity parameters must be finite")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("dipole_weight must be between 0 and 1")
    if power < 0.0:
        raise ValueError("dipole_power must be non-negative")

    return {
        "type": "weighted_dipole",
        "pattern": pattern,
        "orientation_deg": orientation,
        "elevation_deg": max(-90.0, min(90.0, elevation)),
        "dipole_weight": weight,
        "dipole_power": power,
    }


def source_forward(model: Mapping[str, Any]) -> np.ndarray:
    yaw = math.radians(float(model.get("orientation_deg", 0.0)))
    pitch = math.radians(float(model.get("elevation_deg", 0.0)))
    cos_pitch = math.cos(pitch)
    return np.asarray(
        [cos_pitch * math.cos(yaw), cos_pitch * math.sin(yaw), math.sin(pitch)],
        dtype=np.float64,
    )


def source_directivity_gain(direction: Sequence[float], model: Mapping[str, Any]) -> float:
    vector = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        return 1.0
    cosine = float(np.clip(np.dot(source_forward(model), vector / length), -1.0, 1.0))
    weight = float(model.get("dipole_weight", 0.0))
    power = float(model.get("dipole_power", 1.0))
    return float(abs((1.0 - weight) + weight * cosine) ** power)
