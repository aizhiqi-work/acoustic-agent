from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from acoustic_agent.hrtf import DEFAULT_SOFA_PATH, render_binaural_sofa


def angular_error_deg(estimate_deg: float, truth_deg: float) -> float:
    """Return the shortest unsigned distance between two bearings."""
    return abs((float(estimate_deg) - float(truth_deg) + 180.0) % 360.0 - 180.0)


def azimuth_deg(origin: Sequence[float], target: Sequence[float]) -> float:
    """World bearing: +x is 0 degrees, +y is 90 degrees."""
    delta = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
    return float(math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0)


def listener_relative_azimuth_deg(world_deg: float, orientation_deg: float) -> float:
    return float((float(world_deg) - float(orientation_deg)) % 360.0)


def linear_equivalent_azimuth_deg(world_deg: float, orientation_deg: float = 0.0) -> float:
    """Map a bearing to the observable half-plane of a linear array.

    A line array measures the projection onto its axis. Bearings mirrored
    across that axis therefore have identical far-field delays.
    """
    relative = math.radians(float(world_deg) - float(orientation_deg))
    folded = math.degrees(math.acos(float(np.clip(math.cos(relative), -1.0, 1.0))))
    return float((float(orientation_deg) + folded) % 360.0)


def stft_snapshots(signals: np.ndarray, n_fft: int = 512, hop: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = _channel_first(signals)
    hop = int(hop or n_fft // 2)
    if values.shape[1] < n_fft:
        values = np.pad(values, ((0, 0), (0, n_fft - values.shape[1])))
    frame_count = 1 + (values.shape[1] - n_fft) // hop
    starts = np.arange(frame_count, dtype=np.int64) * hop
    frames = np.stack([values[:, start : start + n_fft] for start in starts], axis=1)
    spectra = np.fft.rfft(frames * np.hanning(n_fft)[None, None, :], axis=-1)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0)
    return np.transpose(spectra, (0, 2, 1)), frequencies


def estimate_srp_phat(
    signals: np.ndarray,
    microphone_positions_m: np.ndarray,
    *,
    fs: int,
    search_deg: Sequence[float],
    speed_of_sound_m_s: float = 343.0,
    frequency_range_hz: tuple[float, float] = (300.0, 3500.0),
    n_fft: int = 512,
) -> tuple[float, np.ndarray]:
    """Estimate far-field azimuth with a dependency-free SRP-PHAT scan."""
    values = _channel_first(signals)
    positions = np.asarray(microphone_positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape != (values.shape[0], 3):
        raise ValueError("microphone_positions_m must have shape (channels, 3)")
    if values.shape[0] < 2:
        raise ValueError("SRP-PHAT requires at least two microphone channels")

    spectra, normalized_frequencies = stft_snapshots(values, n_fft=n_fft)
    frequencies = normalized_frequencies * int(fs)
    keep = (frequencies >= float(frequency_range_hz[0])) & (frequencies <= float(frequency_range_hz[1]))
    frequencies = frequencies[keep]
    spectra = spectra[:, keep, :]
    if frequencies.size == 0:
        raise ValueError("frequency_range_hz does not contain an FFT bin")

    search = np.asarray(search_deg, dtype=float).reshape(-1)
    radians = np.deg2rad(search)
    directions = np.column_stack([np.cos(radians), np.sin(radians), np.zeros(search.size)])
    center = np.mean(positions, axis=0)
    offsets = positions - center
    scores = np.zeros(search.size, dtype=np.float64)
    pair_count = 0
    for first in range(values.shape[0] - 1):
        for second in range(first + 1, values.shape[0]):
            cross = spectra[first] * np.conj(spectra[second])
            cross /= np.maximum(np.abs(cross), 1e-12)
            cross = np.mean(cross, axis=1)
            predicted_delay = -((offsets[first] - offsets[second]) @ directions.T) / float(speed_of_sound_m_s)
            steering = np.exp(2j * np.pi * predicted_delay[:, None] * frequencies[None, :])
            scores += np.real(steering @ cross) / frequencies.size
            pair_count += 1
    scores /= max(pair_count, 1)
    return float(search[int(np.argmax(scores))] % 360.0), scores.astype(np.float32)


def estimate_hrtf_template(
    signals: np.ndarray,
    *,
    fs: int,
    search_deg: Sequence[float],
    orientation_deg: float = 0.0,
    sofa_path: str | Path | None = None,
    interpolation: str = "bilinear",
    frequency_range_hz: tuple[float, float] = (500.0, 6000.0),
    n_fft: int = 512,
) -> tuple[float, np.ndarray]:
    """Estimate binaural azimuth by matching interaural HRTF features.

    The source spectrum cancels in the left/right transfer ratio, allowing the
    same estimator to operate on a broadband probe, speech, or an RIR.
    """
    values = _channel_first(signals)
    if values.shape[0] != 2:
        raise ValueError("HRTF template matching requires exactly two channels")
    spectra, normalized_frequencies = stft_snapshots(values, n_fft=n_fft)
    frequencies = normalized_frequencies * int(fs)
    keep = (frequencies >= float(frequency_range_hz[0])) & (frequencies <= min(float(frequency_range_hz[1]), fs / 2.0))
    if not np.any(keep):
        raise ValueError("frequency_range_hz does not contain an FFT bin")

    left_power = np.mean(np.abs(spectra[0]) ** 2, axis=1)
    right_power = np.mean(np.abs(spectra[1]) ** 2, axis=1)
    observed_cross = np.mean(spectra[0] * np.conj(spectra[1]), axis=1)
    observed_phase = observed_cross / np.maximum(np.abs(observed_cross), 1e-12)
    observed_ild = 0.5 * np.log(np.maximum(left_power, 1e-12) / np.maximum(right_power, 1e-12))

    search = np.asarray(search_deg, dtype=float).reshape(-1)
    templates = _hrtf_templates(
        tuple(float(value) for value in search),
        int(fs),
        int(n_fft),
        float(orientation_deg),
        str(Path(sofa_path or DEFAULT_SOFA_PATH).expanduser().resolve()),
        str(interpolation),
    )
    template_left = templates[:, 0]
    template_right = templates[:, 1]
    template_cross = template_left * np.conj(template_right)
    template_phase = template_cross / np.maximum(np.abs(template_cross), 1e-12)
    template_ild = np.log(np.maximum(np.abs(template_left), 1e-12) / np.maximum(np.abs(template_right), 1e-12))

    phase_score = np.mean(np.real(observed_phase[None, keep] * np.conj(template_phase[:, keep])), axis=1)
    ild_scale = max(float(np.std(observed_ild[keep])), 0.25)
    ild_error = np.mean(np.abs(template_ild[:, keep] - observed_ild[None, keep]), axis=1) / ild_scale
    scores = phase_score - 0.08 * ild_error
    return float(search[int(np.argmax(scores))] % 360.0), scores.astype(np.float32)


@lru_cache(maxsize=8)
def _hrtf_templates(
    search_deg: tuple[float, ...],
    fs: int,
    n_fft: int,
    orientation_deg: float,
    sofa_path: str,
    interpolation: str,
) -> np.ndarray:
    impulse = np.zeros(n_fft, dtype=np.float32)
    impulse[0] = 1.0
    receiver = np.zeros(3, dtype=float)
    templates = []
    for angle_deg in search_deg:
        angle = math.radians(angle_deg)
        source = np.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=float)
        rendered, _ = render_binaural_sofa(
            impulse,
            source=source,
            receiver=receiver,
            fs=fs,
            sofa_path=sofa_path,
            interpolation=interpolation,
            orientation_deg=orientation_deg,
            spatial_blend=1.0,
            loudness_normalization="none",
        )
        templates.append(np.fft.rfft(rendered, n=n_fft, axis=1))
    return np.asarray(templates, dtype=np.complex64)


def _channel_first(signals: np.ndarray) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("signals must have shape (channels, samples)")
    return values
