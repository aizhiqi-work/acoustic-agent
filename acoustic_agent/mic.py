from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


SUPPORTED_ARRAYS = {"mono", "hrtf", "linear", "linear_array", "circular", "circular_array"}


def microphone_array(kind: str = "mono", **kwargs: Any) -> dict[str, Any]:
    model_type = kind.lower()
    if model_type not in SUPPORTED_ARRAYS:
        raise ValueError(f"unsupported microphone model: {kind!r}")
    orientation = float(kwargs.get("orientation_deg", 0.0))
    if model_type == "hrtf":
        return {
            "type": "hrtf",
            "orientation_deg": orientation,
            "interpolation": str(kwargs.get("interpolation", "bilinear")),
            "spatial_blend": float(np.clip(float(kwargs.get("spatial_blend", 1.0)), 0.0, 1.0)),
            "loudness_normalization": str(kwargs.get("loudness_normalization", "energy")),
            "sofa_path": kwargs.get("sofa_path"),
            "channels": [{"id": "L", "offset": [0.0, 0.0, 0.0]}, {"id": "R", "offset": [0.0, 0.0, 0.0]}],
        }
    if model_type in {"linear", "linear_array"}:
        count = int(np.clip(int(kwargs.get("count", 4)), 2, 64))
        spacing = float(np.clip(float(kwargs.get("spacing_m", 0.08)), 0.005, 2.0))
        return {"type": "linear_array", "orientation_deg": orientation, "channels": _linear_channels(count, spacing, orientation)}
    if model_type in {"circular", "circular_array"}:
        count = int(np.clip(int(kwargs.get("count", 8)), 3, 128))
        radius = float(np.clip(float(kwargs.get("radius_m", 0.12)), 0.005, 5.0))
        return {"type": "circular_array", "orientation_deg": orientation, "channels": _circular_channels(count, radius, orientation)}
    return {"type": "mono", "orientation_deg": orientation, "channels": [{"id": "M", "offset": [0.0, 0.0, 0.0]}]}


def channel_positions(receiver: Sequence[float], model: Mapping[str, Any] | None) -> list[tuple[float, float, float]]:
    normalized = model or microphone_array("mono")
    center = np.asarray(receiver, dtype=float)
    if str(normalized.get("type")) not in {"linear_array", "circular_array"}:
        return [tuple(float(v) for v in center[:3])]
    out = []
    for channel in normalized.get("channels", []):
        offset = np.asarray(channel.get("offset", (0.0, 0.0, 0.0)), dtype=float)
        point = center.copy()
        point[:3] += offset[:3]
        out.append(tuple(float(v) for v in point[:3]))
    return out


def _linear_channels(count: int, spacing_m: float, orientation_deg: float) -> list[dict[str, Any]]:
    angle = math.radians(orientation_deg)
    axis = (math.cos(angle), math.sin(angle))
    center = 0.5 * (count - 1)
    return [
        {
            "id": f"M{index + 1}",
            "offset": [round(axis[0] * (index - center) * spacing_m, 6), round(axis[1] * (index - center) * spacing_m, 6), 0.0],
        }
        for index in range(count)
    ]


def _circular_channels(count: int, radius_m: float, orientation_deg: float) -> list[dict[str, Any]]:
    start = math.radians(orientation_deg)
    return [
        {
            "id": f"M{index + 1}",
            "offset": [
                round(math.cos(start + 2.0 * math.pi * index / count) * radius_m, 6),
                round(math.sin(start + 2.0 * math.pi * index / count) * radius_m, 6),
                0.0,
            ],
        }
        for index in range(count)
    ]
