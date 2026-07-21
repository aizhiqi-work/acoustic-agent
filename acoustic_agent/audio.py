from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def render_audio(
    samples: np.ndarray | Sequence[float],
    rir: np.ndarray | Sequence[float],
    *,
    gain_db: float = 0.0,
) -> np.ndarray:
    """Convolve one mono source signal with a receiver RIR.

    Multi-channel source audio is downmixed before convolution because one
    acoustic source represents one emitting point. The returned array always
    has shape ``[receiver_channel, sample]``.
    """
    dry = _mono_source(samples)
    impulse = _channel_first(rir, "rir")
    if dry.size < 1:
        raise ValueError("source audio must contain at least one sample")
    if impulse.shape[1] < 1:
        raise ValueError("RIR must contain at least one sample")

    output_length = dry.size + impulse.shape[1] - 1
    fft_length = 1 << max(0, output_length - 1).bit_length()
    dry_spectrum = np.fft.rfft(dry, fft_length)
    gain = 10.0 ** (float(gain_db) / 20.0)
    output = np.empty((impulse.shape[0], output_length), dtype=np.float32)
    for channel, channel_rir in enumerate(impulse):
        rendered = np.fft.irfft(dry_spectrum * np.fft.rfft(channel_rir, fft_length), fft_length)
        output[channel] = np.asarray(rendered[:output_length] * gain, dtype=np.float32)
    return output


def mix_audio(
    tracks: Sequence[np.ndarray | Sequence[float]],
    *,
    normalize: bool = False,
    peak: float = 0.98,
) -> np.ndarray:
    """Sum rendered receiver-channel tracks while preserving relative level."""
    if not tracks:
        raise ValueError("tracks must contain at least one rendered signal")
    arrays = [_channel_first(track, f"tracks[{index}]") for index, track in enumerate(tracks)]
    channel_count = max(array.shape[0] for array in arrays)
    sample_count = max(array.shape[1] for array in arrays)
    output = np.zeros((channel_count, sample_count), dtype=np.float64)
    for index, array in enumerate(arrays):
        if array.shape[0] == 1 and channel_count > 1:
            array = np.repeat(array, channel_count, axis=0)
        if array.shape[0] != channel_count:
            raise ValueError(
                f"tracks[{index}] has {array.shape[0]} channels; expected 1 or {channel_count}"
            )
        output[:, : array.shape[1]] += array

    if normalize:
        requested_peak = float(peak)
        if not math.isfinite(requested_peak) or not 0.0 < requested_peak <= 1.0:
            raise ValueError("peak must be finite and between 0 and 1")
        current_peak = float(np.max(np.abs(output))) if output.size else 0.0
        if current_peak > requested_peak:
            output *= requested_peak / current_peak
    return np.asarray(output, dtype=np.float32)


def mix_audio_at_snr(
    foreground: np.ndarray | Sequence[float],
    background: np.ndarray | Sequence[float],
    *,
    snr_db: float = 10.0,
    normalize: bool = False,
    peak: float = 0.98,
) -> np.ndarray:
    """Mix rendered signals at a receiver-domain broadband SNR.

    Both inputs should already be convolved with their own RIR. The background
    is scaled so ``20 * log10(foreground_rms / background_rms)`` equals
    ``snr_db`` before optional final peak normalization.
    """
    signal = _channel_first(foreground, "foreground")
    noise = _channel_first(background, "background")
    channel_count = max(signal.shape[0], noise.shape[0])
    signal = _expand_channels(signal, channel_count, "foreground")
    noise = _expand_channels(noise, channel_count, "background")

    target_snr = float(snr_db)
    if not math.isfinite(target_snr):
        raise ValueError("snr_db must be finite")
    signal_rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
    noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    if signal_rms <= 1e-12:
        raise ValueError("foreground RMS must be greater than zero")
    if noise_rms <= 1e-12:
        raise ValueError("background RMS must be greater than zero")
    noise_gain = signal_rms / (noise_rms * 10.0 ** (target_snr / 20.0))
    return mix_audio([signal, noise * noise_gain], normalize=normalize, peak=peak)


def resample_audio(
    samples: np.ndarray | Sequence[float],
    source_fs: int,
    target_fs: int,
) -> np.ndarray:
    """Linearly resample channel-first audio without adding a SciPy dependency."""
    source_rate = int(source_fs)
    target_rate = int(target_fs)
    if source_rate < 1 or target_rate < 1:
        raise ValueError("sample rates must be positive")
    values = _channel_first(samples, "samples")
    if source_rate == target_rate:
        return values.copy()
    output_length = max(1, int(round(values.shape[1] * target_rate / source_rate)))
    source_positions = np.arange(output_length, dtype=np.float64) * source_rate / target_rate
    source_axis = np.arange(values.shape[1], dtype=np.float64)
    output = np.vstack([
        np.interp(source_positions, source_axis, channel).astype(np.float32)
        for channel in values
    ])
    return output


def _mono_source(samples: np.ndarray | Sequence[float]) -> np.ndarray:
    values = _channel_first(samples, "samples")
    return np.asarray(np.mean(values, axis=0), dtype=np.float64)


def _channel_first(value: np.ndarray | Sequence[float], label: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError(f"{label} must have shape [sample] or [channel, sample]")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite samples")
    return values


def _expand_channels(values: np.ndarray, channel_count: int, label: str) -> np.ndarray:
    if values.shape[0] == channel_count:
        return values
    if values.shape[0] == 1:
        return np.repeat(values, channel_count, axis=0)
    raise ValueError(f"{label} has {values.shape[0]} channels; expected 1 or {channel_count}")
