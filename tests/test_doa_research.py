from __future__ import annotations

import numpy as np

from acoustic_agent import microphone_array
from acoustic_agent.hrtf import render_binaural_sofa
from acoustic_agent.mic import channel_positions
from research.doa.estimators import (
    angular_error_deg,
    estimate_hrtf_template,
    estimate_srp_phat,
    linear_equivalent_azimuth_deg,
)


def _fractional_impulse(delay_samples: float, length: int = 1024) -> np.ndarray:
    indices = np.arange(length, dtype=float)
    values = np.sinc(indices - delay_samples) * np.hanning(length)
    return np.asarray(values, dtype=np.float32)


def test_angle_metrics_wrap_and_preserve_linear_mirror_ambiguity() -> None:
    assert angular_error_deg(359.0, 1.0) == 2.0
    assert np.isclose(linear_equivalent_azimuth_deg(230.0), 130.0)
    assert np.isclose(linear_equivalent_azimuth_deg(30.0), 30.0)


def test_srp_phat_recovers_circular_array_bearing() -> None:
    fs = 16_000
    speed = 343.0
    truth_deg = 67.0
    center = np.asarray([0.0, 0.0, 0.0])
    model = microphone_array("circular", count=8, radius_m=0.05)
    positions = np.asarray(channel_positions(center, model), dtype=float)
    direction = np.asarray([np.cos(np.deg2rad(truth_deg)), np.sin(np.deg2rad(truth_deg)), 0.0])
    relative_delays = -(positions @ direction) / speed * fs
    offset = 64.0 - float(np.min(relative_delays))
    signals = np.stack([_fractional_impulse(offset + delay) for delay in relative_delays])
    estimate, _ = estimate_srp_phat(
        signals,
        positions,
        fs=fs,
        search_deg=np.arange(360.0),
        frequency_range_hz=(300.0, 3500.0),
    )
    assert angular_error_deg(estimate, truth_deg) <= 2.0


def test_hrtf_template_recovers_bundled_sofa_bearing() -> None:
    fs = 16_000
    truth_deg = 118.0
    angle = np.deg2rad(truth_deg)
    impulse = np.zeros(512, dtype=np.float32)
    impulse[0] = 1.0
    binaural, _ = render_binaural_sofa(
        impulse,
        source=[np.cos(angle), np.sin(angle), 0.0],
        receiver=[0.0, 0.0, 0.0],
        fs=fs,
        interpolation="bilinear",
        loudness_normalization="none",
    )
    estimate, _ = estimate_hrtf_template(
        binaural,
        fs=fs,
        search_deg=np.arange(0.0, 360.0, 2.0),
    )
    assert angular_error_deg(estimate, truth_deg) <= 2.0
