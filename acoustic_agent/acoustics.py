from __future__ import annotations

import cmath
import math
from typing import Mapping

import numpy as np

from .models import FREQUENCY_BANDS, Material


AIR_ABSORPTION_NP_PER_M = {
    # Steam Audio's octave-band defaults for the six bands represented by
    # this project.  The previous values duplicated the default 3-band
    # low/mid/high coefficients and substantially over-attenuated 2/4 kHz.
    "125": 0.0,
    "250": 0.00011513,
    "500": 0.00034539,
    "1000": 0.00057565,
    "2000": 0.0011513,
    "4000": 0.0034539,
}


def propagation_band_gains(distance_m: float, *, min_distance_m: float) -> dict[str, float]:
    dist = max(float(distance_m), float(min_distance_m))
    real_dist = max(float(distance_m), 0.0)
    return {
        band: (1.0 / dist) * math.exp(-AIR_ABSORPTION_NP_PER_M[band] * real_dist)
        for band in FREQUENCY_BANDS
    }


def apply_surface_reflection(bands: Mapping[str, float], material: Material) -> dict[str, float]:
    reflection = material.reflection
    return {band: float(bands.get(band, 0.0)) * float(reflection[band]) for band in FREQUENCY_BANDS}


def band_mean(curve: Mapping[str, float]) -> float:
    return float(np.mean([float(curve.get(band, 0.0)) for band in FREQUENCY_BANDS]))


def multiply_bands(*curves: Mapping[str, float]) -> dict[str, float]:
    out = {band: 1.0 for band in FREQUENCY_BANDS}
    for curve in curves:
        for band in FREQUENCY_BANDS:
            out[band] *= float(curve.get(band, 1.0))
    return out


def steam_audio_utd_deviation(angle_rad: float) -> dict[str, float]:
    """Steam Audio DeviationModel::utdDeviation octave-band approximation."""
    n = 2.0
    alpha_i = 0.0
    alpha_d = alpha_i + math.pi + max(1e-8, float(angle_rad))
    effective_length = 0.05

    def n_plus(beta: float) -> float:
        return 0.0 if beta <= math.pi * (n - 1.0) else 1.0

    def n_minus(beta: float) -> float:
        if beta < math.pi * (1.0 - n):
            return -1.0
        if beta <= math.pi * (1.0 + n):
            return 0.0
        return 1.0

    def transition(x: float) -> complex:
        phase = cmath.exp(0.25j * math.pi * math.sqrt(x / (x + 1.4)))
        if x < 0.8:
            return math.sqrt(math.pi * x) * (1.0 - math.sqrt(x) / (0.7 * math.sqrt(x) + 1.2)) * phase
        return (1.0 - 0.8 / ((x + 1.25) ** 2)) * phase

    out: dict[str, float] = {}
    for band in FREQUENCY_BANDS:
        wave_number = 2.0 * math.pi * float(band) / 343.0
        base = cmath.exp(-0.25j * math.pi) / (2.0 * n * math.sqrt(2.0 * math.pi * wave_number))
        betas = (alpha_d - alpha_i, alpha_d - alpha_i, alpha_d + alpha_i, alpha_d + alpha_i)
        nearest = (n_plus(betas[0]), n_minus(betas[1]), n_plus(betas[2]), n_minus(betas[3]))
        signs = (1.0, -1.0, 1.0, -1.0)
        terms: list[complex] = []
        for index, (beta, selected) in enumerate(zip(betas, nearest)):
            argument = (math.pi + signs[index] * beta) / (2.0 * n)
            tangent = math.tan(argument)
            cotangent = math.inf if abs(tangent) <= 1e-12 else 1.0 / tangent
            a_value = 2.0 * math.cos(math.pi * n * selected - 0.5 * beta) ** 2
            value = cotangent * transition(wave_number * effective_length * a_value)
            if not math.isfinite(cotangent):
                epsilon = (
                    beta - 2.0 * math.pi * n * selected + math.pi
                    if index in (0, 2)
                    else -(beta - 2.0 * math.pi * n * selected - math.pi)
                )
                epsilon_sign = 1.0 if epsilon > 0.0 else -1.0
                value = n * cmath.exp(-0.25j * math.pi) * (
                    math.sqrt(2.0 * math.pi * wave_number * effective_length) * epsilon_sign
                    - 2.0 * wave_number * effective_length * epsilon * cmath.exp(-0.25j * math.pi)
                )
            terms.append(value)
        out[band] = float(np.clip(abs(base * sum(terms)), 0.0, 1.0))
    return out


def steam_audio_pathing_deviation(angle_rad: float) -> dict[str, float]:
    """Apply Steam Audio's pathing normalization to a total deviation angle."""
    raw = steam_audio_utd_deviation(max(1e-8, float(angle_rad)))
    reference = steam_audio_utd_deviation(1e-8)
    relative = {
        band: float(raw[band] / max(reference[band], 1e-12))
        for band in FREQUENCY_BANDS
    }
    max_gain = max(relative.values(), default=0.0)
    if max_gain <= np.finfo(float).tiny:
        return {band: 0.0 for band in FREQUENCY_BANDS}

    # EQEffect::normalizeGains floors normalized EQ bands at 1/16, then
    # rolls the maximum back into the overall path gain.
    floor = max_gain / 16.0
    return {band: max(float(relative[band]), floor) for band in FREQUENCY_BANDS}


def estimate_rt60(room_area_m2: float, perimeter_m: float, height_m: float, materials: Mapping[str, Material]) -> dict:
    floor_area = max(float(room_area_m2), 1e-6)
    wall_area = max(float(perimeter_m), 1e-6) * max(float(height_m), 1e-6)
    volume = floor_area * max(float(height_m), 1e-6)
    floor = materials.get("floor") or next(iter(materials.values()))
    ceiling = materials.get("ceiling") or floor
    wall = materials.get("wall") or floor
    rt60_bands: dict[str, float] = {}
    for band in FREQUENCY_BANDS:
        absorbed = (
            floor_area * float(floor.absorption.get(band, 0.10))
            + floor_area * float(ceiling.absorption.get(band, 0.08))
            + wall_area * float(wall.absorption.get(band, 0.08))
        )
        rt60_bands[band] = float(0.161 * volume / max(absorbed, 1e-6))
    return {
        "rt60_bands": {band: round(value, 4) for band, value in rt60_bands.items()},
        "rt60_s": round(float(np.mean(list(rt60_bands.values()))), 4),
        "volume_m3": round(volume, 4),
        "surface_area_m2": round(2.0 * floor_area + wall_area, 4),
        "model": "sabine_extruded_polygon",
    }
