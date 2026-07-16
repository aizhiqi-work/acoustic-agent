from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


FREQUENCY_BANDS: tuple[str, ...] = ("125", "250", "500", "1000", "2000", "4000")


@dataclass(frozen=True)
class Material:
    id: str
    name: str
    semantic: str
    absorption: Mapping[str, float]
    scattering: Mapping[str, float] = field(default_factory=dict)
    transmission_loss_db: Mapping[str, float] = field(default_factory=dict)
    source: str = "fallback"

    @property
    def reflection(self) -> dict[str, float]:
        return {
            band: float(np.sqrt(max(0.0, 1.0 - float(self.absorption.get(band, 0.2)))))
            for band in FREQUENCY_BANDS
        }

    @property
    def transmission(self) -> dict[str, float]:
        """Pressure transmission coefficient derived from transmission loss."""
        return {
            band: float(10.0 ** (-float(self.transmission_loss_db.get(band, 30.0)) / 20.0))
            for band in FREQUENCY_BANDS
        }


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    corners: tuple[tuple[float, float], ...]
    height_m: float
    materials: Mapping[str, Material]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcousticPath:
    kind: str
    distance_m: float
    delay_s: float
    gain: float
    band_gains: Mapping[str, float]
    points: tuple[tuple[float, float, float], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimConfig:
    fs: int = 16000
    c: float = 343.0
    duration_s: float = 2.0
    min_distance_m: float = 1.0
    late_tail: bool = True
    late_tail_start_s: float = 0.08
    tail_energy: float = 0.015
    seed: int = 1729
    fractional_delay: bool = True
    sinc_half_width: int = 8
    direct_occlusion: bool = True
    direct_transmission: bool = True
    direct_occlusion_mode: str = "volumetric"
    direct_occlusion_radius_m: float = 0.1
    direct_occlusion_samples: int = 32
    num_transmission_rays: int = 8
    reflections_enabled: bool = True
    rt_num_rays: int = 32768
    rt_num_bounces: int = 96
    rt_num_diffuse_samples: int = 128
    rt_duration_s: float = 2.0
    rt_bin_duration_s: float = 0.01
    rt_specular_exponent: float = 100.0
    rt_irradiance_min_distance: float = 1.0
    rt_source_radius: float = 0.1
    rt_listener_radius: float = 0.1
    rt_receiver_radius_m: float = 0.25
    rt_visual_num_rays: int | None = None
    rt_visual_num_bounces: int | None = None
    late_tail_cutoff_s: float = 0.08
    hybrid_transition_s: float = 1.0
    hybrid_overlap_fraction: float = 0.25
    diffraction_enabled: bool = True
    diffraction_audio_enabled: bool = True
    diffraction_order: int = 3
    max_diffraction_paths: int = 8


def band_constant(value: float) -> dict[str, float]:
    return {band: float(value) for band in FREQUENCY_BANDS}


def vec3(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"expected a 3D point, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("point contains non-finite values")
    return arr
