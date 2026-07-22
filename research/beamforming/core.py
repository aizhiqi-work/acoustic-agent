from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


SPEED_OF_SOUND_M_S = 343.0


def channel_first(value: np.ndarray | Sequence[float], label: str = "signal") -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{label} must have shape [sample] or [channel, sample]")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite samples")
    return values


def rir_arrival_times(
    rir: np.ndarray | Sequence[float],
    fs: int,
    *,
    threshold_ratio: float = 0.025,
) -> np.ndarray:
    """Estimate each channel's first meaningful RIR arrival in seconds."""
    responses = channel_first(rir, "rir")
    sample_rate = _positive_sample_rate(fs)
    ratio = float(threshold_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("threshold_ratio must be in (0, 1]")
    arrivals = np.empty(responses.shape[0], dtype=np.float64)
    for index, response in enumerate(responses):
        magnitude = np.abs(response)
        peak = float(np.max(magnitude))
        if peak <= 1e-15:
            raise ValueError(f"rir channel {index} is silent")
        candidates = np.flatnonzero(magnitude >= peak * ratio)
        sample = int(candidates[0]) if candidates.size else int(np.argmax(magnitude))
        arrivals[index] = sample / sample_rate
    return arrivals


def arrival_times_from_source(
    source_position_m: Sequence[float],
    microphone_positions_m: np.ndarray | Sequence[Sequence[float]],
    *,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> np.ndarray:
    source = np.asarray(source_position_m, dtype=np.float64).reshape(-1)
    microphones = np.asarray(microphone_positions_m, dtype=np.float64)
    if source.size < 3 or microphones.ndim != 2 or microphones.shape[1] != 3:
        raise ValueError("source and microphone positions must be three-dimensional")
    speed = _positive_speed(speed_of_sound_m_s)
    return np.linalg.norm(microphones - source[:3], axis=1) / speed


def arrival_times_from_azimuth(
    azimuth_deg: float,
    microphone_positions_m: np.ndarray | Sequence[Sequence[float]],
    *,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> np.ndarray:
    """Return relative far-field arrivals for a world-space source bearing."""
    microphones = np.asarray(microphone_positions_m, dtype=np.float64)
    if microphones.ndim != 2 or microphones.shape[1] != 3:
        raise ValueError("microphone_positions_m must have shape [channel, 3]")
    speed = _positive_speed(speed_of_sound_m_s)
    angle = math.radians(float(azimuth_deg))
    direction = np.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=np.float64)
    offsets = microphones - np.mean(microphones, axis=0)
    arrivals = -(offsets @ direction) / speed
    return arrivals - float(np.min(arrivals))


def fractional_delay(signal: np.ndarray | Sequence[float], delay_samples: float) -> np.ndarray:
    """Apply a zero-padded fractional delay without circular wraparound."""
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    delay = float(delay_samples)
    if not math.isfinite(delay):
        raise ValueError("delay_samples must be finite")
    guard = max(32, int(math.ceil(abs(delay))) + 8)
    padded = np.pad(values, (guard, guard))
    fft_length = 1 << max(1, int(padded.size - 1).bit_length())
    normalized_frequency = np.fft.rfftfreq(fft_length)
    phase = np.exp(-2j * np.pi * normalized_frequency * delay)
    shifted = np.fft.irfft(np.fft.rfft(padded, fft_length) * phase, fft_length)
    return np.asarray(shifted[guard : guard + values.size], dtype=np.float64)


def delay_and_sum(
    signals: np.ndarray | Sequence[Sequence[float]],
    arrival_times_s: Sequence[float],
    fs: int,
    *,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Align channels to the latest arrival and return a normalized mono sum."""
    values = channel_first(signals, "signals")
    arrivals = np.asarray(arrival_times_s, dtype=np.float64).reshape(-1)
    if arrivals.size != values.shape[0] or not np.all(np.isfinite(arrivals)):
        raise ValueError("arrival_times_s must contain one finite value per channel")
    sample_rate = _positive_sample_rate(fs)
    if weights is None:
        normalized_weights = np.full(values.shape[0], 1.0 / values.shape[0], dtype=np.float64)
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if normalized_weights.size != values.shape[0] or np.any(normalized_weights < 0.0):
            raise ValueError("weights must contain one non-negative value per channel")
        weight_sum = float(np.sum(normalized_weights))
        if weight_sum <= 1e-12:
            raise ValueError("at least one weight must be positive")
        normalized_weights /= weight_sum
    extra_delay_samples = (float(np.max(arrivals)) - arrivals) * sample_rate
    output = np.zeros(values.shape[1], dtype=np.float64)
    for channel, delay_samples, weight in zip(values, extra_delay_samples, normalized_weights):
        output += float(weight) * fractional_delay(channel, float(delay_samples))
    return np.asarray(output, dtype=np.float32)


def align_channels(
    signals: np.ndarray | Sequence[Sequence[float]],
    arrival_times_s: Sequence[float],
    fs: int,
) -> np.ndarray:
    """Time-align synchronized channels to the latest target arrival."""
    values = channel_first(signals, "signals")
    arrivals = np.asarray(arrival_times_s, dtype=np.float64).reshape(-1)
    if arrivals.size != values.shape[0] or not np.all(np.isfinite(arrivals)):
        raise ValueError("arrival_times_s must contain one finite value per channel")
    sample_rate = _positive_sample_rate(fs)
    extra_delay_samples = (float(np.max(arrivals)) - arrivals) * sample_rate
    return np.vstack(
        [
            fractional_delay(channel, float(delay_samples))
            for channel, delay_samples in zip(values, extra_delay_samples)
        ]
    ).astype(np.float32)


def estimate_gcc_phat_delays(
    signals: np.ndarray | Sequence[Sequence[float]],
    fs: int,
    *,
    max_delay_s: float | None = None,
    reference_channel: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate channel arrival offsets relative to one synchronized channel."""
    values = channel_first(signals, "signals")
    sample_rate = _positive_sample_rate(fs)
    reference = int(reference_channel)
    if not 0 <= reference < values.shape[0]:
        raise ValueError("reference_channel is outside the channel range")
    max_shift = values.shape[1] - 1
    if max_delay_s is not None:
        requested = float(max_delay_s)
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("max_delay_s must be positive and finite")
        max_shift = min(max_shift, max(1, int(round(requested * sample_rate))))

    fft_length = 1 << max(1, int(2 * values.shape[1] - 1).bit_length())
    reference_spectrum = np.fft.rfft(values[reference], fft_length)
    delays = np.zeros(values.shape[0], dtype=np.float64)
    confidence = np.ones(values.shape[0], dtype=np.float64)
    for index, signal in enumerate(values):
        if index == reference:
            continue
        cross = np.fft.rfft(signal, fft_length) * np.conj(reference_spectrum)
        cross /= np.maximum(np.abs(cross), 1e-12)
        correlation = np.fft.irfft(cross, fft_length)
        centered = np.concatenate((correlation[-max_shift:], correlation[: max_shift + 1]))
        magnitude = np.abs(centered)
        peak_index = int(np.argmax(magnitude))
        fractional_index = float(peak_index)
        if 0 < peak_index < magnitude.size - 1:
            left, center, right = magnitude[peak_index - 1 : peak_index + 2]
            denominator = left - 2.0 * center + right
            if abs(float(denominator)) > 1e-15:
                fractional_index += 0.5 * float(left - right) / float(denominator)
        delays[index] = (fractional_index - max_shift) / sample_rate
        excluded = magnitude.copy()
        excluded[max(0, peak_index - 2) : min(magnitude.size, peak_index + 3)] = 0.0
        second_peak = max(float(np.max(excluded)), 1e-12)
        confidence[index] = float(np.clip(float(magnitude[peak_index]) / second_peak / 4.0, 0.05, 1.0))
    return delays, confidence


def scale_background_to_snr(
    target: np.ndarray | Sequence[float],
    background: np.ndarray | Sequence[float],
    target_snr_db: float,
    *,
    sample_slice: slice | None = None,
) -> tuple[np.ndarray, float]:
    signal = channel_first(target, "target")
    noise = channel_first(background, "background")
    if signal.shape != noise.shape:
        raise ValueError("target and background must have the same shape")
    region = sample_slice or slice(None)
    signal_rms = _rms(signal[:, region])
    noise_rms = _rms(noise[:, region])
    if signal_rms <= 1e-15 or noise_rms <= 1e-15:
        raise ValueError("target and background must be non-silent in sample_slice")
    snr = float(target_snr_db)
    if not math.isfinite(snr):
        raise ValueError("target_snr_db must be finite")
    gain = signal_rms / (noise_rms * 10.0 ** (snr / 20.0))
    return np.asarray(noise * gain, dtype=np.float32), float(gain)


def snr_db(
    target: np.ndarray | Sequence[float],
    interference: np.ndarray | Sequence[float],
    *,
    sample_slice: slice | None = None,
) -> float:
    signal = channel_first(target, "target")
    noise = channel_first(interference, "interference")
    if signal.shape != noise.shape:
        raise ValueError("target and interference must have the same shape")
    region = sample_slice or slice(None)
    signal_power = float(np.mean(np.square(signal[:, region], dtype=np.float64)))
    noise_power = float(np.mean(np.square(noise[:, region], dtype=np.float64)))
    return float(10.0 * math.log10(max(signal_power, 1e-20) / max(noise_power, 1e-20)))


def si_sdr_db(estimate: np.ndarray | Sequence[float], reference: np.ndarray | Sequence[float]) -> float:
    estimated = np.asarray(estimate, dtype=np.float64).reshape(-1)
    truth = np.asarray(reference, dtype=np.float64).reshape(-1)
    length = min(estimated.size, truth.size)
    if length < 1:
        raise ValueError("estimate and reference must be non-empty")
    estimated = estimated[:length] - float(np.mean(estimated[:length]))
    truth = truth[:length] - float(np.mean(truth[:length]))
    projection = truth * (float(np.dot(estimated, truth)) / max(float(np.dot(truth, truth)), 1e-20))
    residual = estimated - projection
    return float(
        10.0
        * math.log10(
            max(float(np.dot(projection, projection)), 1e-20)
            / max(float(np.dot(residual, residual)), 1e-20)
        )
    )


def align_signals(
    estimate: np.ndarray | Sequence[float],
    reference: np.ndarray | Sequence[float],
    *,
    max_shift_samples: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align an estimate to a fixed reference using bounded FFT correlation."""
    estimated = np.asarray(estimate, dtype=np.float64).reshape(-1)
    truth = np.asarray(reference, dtype=np.float64).reshape(-1)
    length = min(estimated.size, truth.size)
    if length < 2:
        raise ValueError("estimate and reference must contain at least two samples")
    estimated = estimated[:length]
    truth = truth[:length]
    limit = min(max(0, int(max_shift_samples)), length - 1)
    fft_length = 1 << max(1, int(2 * length - 1).bit_length())
    correlation = np.fft.irfft(
        np.fft.rfft(estimated, fft_length) * np.conj(np.fft.rfft(truth, fft_length)),
        fft_length,
    )
    centered = np.concatenate((correlation[-limit:], correlation[: limit + 1]))
    shift = int(np.argmax(np.abs(centered))) - limit
    if shift > 0:
        return estimated[shift:], truth[: length - shift], shift
    if shift < 0:
        return estimated[: length + shift], truth[-shift:], shift
    return estimated, truth, 0


def aligned_si_sdr_db(
    estimate: np.ndarray | Sequence[float],
    reference: np.ndarray | Sequence[float],
    *,
    max_shift_samples: int,
) -> float:
    estimated, truth, _ = align_signals(
        estimate,
        reference,
        max_shift_samples=max_shift_samples,
    )
    return si_sdr_db(estimated, truth)


def optional_stoi(estimate: np.ndarray, reference: np.ndarray, fs: int) -> float | None:
    try:
        from pystoi import stoi
    except ImportError:
        return None
    estimated = np.asarray(estimate, dtype=np.float64).reshape(-1)
    truth = np.asarray(reference, dtype=np.float64).reshape(-1)
    length = min(estimated.size, truth.size)
    return float(stoi(truth[:length], estimated[:length], _positive_sample_rate(fs), extended=False))


def optional_pesq(estimate: np.ndarray, reference: np.ndarray, fs: int) -> float | None:
    try:
        from pesq import pesq
    except ImportError:
        return None
    sample_rate = _positive_sample_rate(fs)
    if sample_rate not in {8_000, 16_000}:
        return None
    estimated = np.asarray(estimate, dtype=np.float64).reshape(-1)
    truth = np.asarray(reference, dtype=np.float64).reshape(-1)
    length = min(estimated.size, truth.size)
    if length < int(0.25 * sample_rate):
        return None
    mode = "wb" if sample_rate == 16_000 else "nb"
    try:
        return float(pesq(sample_rate, truth[:length], estimated[:length], mode))
    except Exception:
        return None


def estimate_wiener_gain(
    mixture: np.ndarray | Sequence[float],
    noise_reference: np.ndarray | Sequence[float],
    *,
    n_fft: int = 512,
    hop: int | None = None,
    gain_floor: float = 0.05,
) -> np.ndarray:
    noisy = np.asarray(mixture, dtype=np.float64).reshape(-1)
    noise = np.asarray(noise_reference, dtype=np.float64).reshape(-1)
    fft_size, hop_size, window = _stft_parameters(n_fft, hop)
    noisy_spectra = _stft(noisy, fft_size, hop_size, window)
    noise_spectra = _stft(noise, fft_size, hop_size, window)
    noise_psd = np.mean(np.abs(noise_spectra) ** 2, axis=0)
    posterior = np.abs(noisy_spectra) ** 2
    floor = float(gain_floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("gain_floor must be between 0 and 1")
    return np.asarray(np.maximum(1.0 - noise_psd[None, :] / np.maximum(posterior, 1e-12), floor), dtype=np.float64)


def apply_stft_gain(
    signal: np.ndarray | Sequence[float],
    gain: np.ndarray,
    *,
    n_fft: int = 512,
    hop: int | None = None,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    fft_size, hop_size, window = _stft_parameters(n_fft, hop)
    spectra = _stft(values, fft_size, hop_size, window)
    gains = np.asarray(gain, dtype=np.float64)
    if gains.shape != spectra.shape:
        raise ValueError(f"gain has shape {gains.shape}; expected {spectra.shape}")
    return _istft(spectra * gains, values.size, fft_size, hop_size, window)


def steering_vector(
    arrival_times_s: Sequence[float],
    fs: int,
    *,
    n_fft: int = 512,
) -> np.ndarray:
    arrivals = np.asarray(arrival_times_s, dtype=np.float64).reshape(-1)
    if arrivals.size < 2 or not np.all(np.isfinite(arrivals)):
        raise ValueError("arrival_times_s must contain at least two finite values")
    relative = arrivals - float(np.min(arrivals))
    frequencies = np.fft.rfftfreq(int(n_fft), d=1.0 / _positive_sample_rate(fs))
    return np.exp(-2j * np.pi * frequencies[:, None] * relative[None, :])


def estimate_adaptive_beamformer_weights(
    target_calibration: np.ndarray | Sequence[Sequence[float]],
    noise_calibration: np.ndarray | Sequence[Sequence[float]],
    arrival_times_s: Sequence[float],
    fs: int,
    *,
    algorithm: str,
    reference_channel: int = 0,
    reliability: Sequence[float] | None = None,
    n_fft: int = 512,
    hop: int | None = None,
    diagonal_loading: float = 1e-3,
) -> np.ndarray:
    """Estimate frequency-domain DS, MVDR, GEV, or MWF weights."""
    target = channel_first(target_calibration, "target_calibration")
    noise = channel_first(noise_calibration, "noise_calibration")
    if target.shape[0] != noise.shape[0]:
        raise ValueError("target and noise calibration channel counts must match")
    fft_size, hop_size, window = _stft_parameters(n_fft, hop)
    steering = steering_vector(arrival_times_s, fs, n_fft=fft_size)
    if steering.shape[1] != target.shape[0]:
        raise ValueError("arrival_times_s must contain one value per channel")
    method = str(algorithm).strip().lower()
    if method in {"ds", "delay_and_sum"}:
        return np.asarray(steering / target.shape[0], dtype=np.complex128)
    if method in {"weighted_ds", "wds"}:
        if reliability is None:
            raise ValueError("weighted DS requires reliability values")
        gains = np.asarray(reliability, dtype=np.float64).reshape(-1)
        if gains.size != target.shape[0] or np.any(gains < 0.0):
            raise ValueError("reliability must contain one non-negative value per channel")
        gains = np.maximum(gains, 1e-12)
        gains /= float(np.sum(gains))
        return np.asarray(steering * gains[None, :], dtype=np.complex128)
    if method not in {"mvdr", "gev", "mwf"}:
        raise ValueError("algorithm must be ds, weighted_ds, mvdr, gev, or mwf")

    target_covariance = _spatial_covariance(target, fft_size, hop_size, window)
    noise_covariance = _spatial_covariance(noise, fft_size, hop_size, window)
    channel_count = target.shape[0]
    reference = int(reference_channel)
    if not 0 <= reference < channel_count:
        raise ValueError("reference_channel is outside the channel range")
    loading = float(diagonal_loading)
    if not math.isfinite(loading) or loading < 0.0:
        raise ValueError("diagonal_loading must be non-negative and finite")
    identity = np.eye(channel_count, dtype=np.complex128)
    weights = np.empty((steering.shape[0], channel_count), dtype=np.complex128)
    for frequency_index, guide in enumerate(steering):
        target_cov = target_covariance[frequency_index]
        noise_cov = noise_covariance[frequency_index]
        noise_scale = max(float(np.trace(noise_cov).real) / channel_count, 1e-10)
        target_scale = max(float(np.trace(target_cov).real) / channel_count, 1e-10)
        loaded_noise = noise_cov + identity * (loading * noise_scale + 1e-10)
        if method == "mvdr":
            solved = np.linalg.solve(loaded_noise, guide)
            denominator = max(float(np.vdot(guide, solved).real), 1e-12)
            weights[frequency_index] = solved / denominator
            continue
        if method == "gev":
            loaded_target = target_cov + identity * (loading * target_scale + 1e-10)
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(loaded_noise, loaded_target))
            vector = eigenvectors[:, int(np.argmax(eigenvalues.real))]
            denominator = np.vdot(vector, guide)
            if abs(denominator) <= 1e-10:
                solved = np.linalg.solve(loaded_noise, guide)
                vector = solved / max(float(np.vdot(guide, solved).real), 1e-12)
            else:
                vector = vector / np.conj(denominator)
            weights[frequency_index] = vector
            continue
        mixture_cov = target_cov + noise_cov + identity * (loading * (target_scale + noise_scale) + 1e-10)
        cross_covariance = target_cov[:, reference]
        weights[frequency_index] = np.linalg.solve(mixture_cov, cross_covariance)
    return weights


def apply_stft_beamformer(
    signals: np.ndarray | Sequence[Sequence[float]],
    weights: np.ndarray,
    *,
    n_fft: int = 512,
    hop: int | None = None,
) -> np.ndarray:
    values = channel_first(signals, "signals")
    fft_size, hop_size, window = _stft_parameters(n_fft, hop)
    spectra = np.stack([_stft(channel, fft_size, hop_size, window) for channel in values])
    actual_weights = np.asarray(weights, dtype=np.complex128)
    expected_shape = (spectra.shape[2], values.shape[0])
    if actual_weights.shape != expected_shape:
        raise ValueError(f"weights have shape {actual_weights.shape}; expected {expected_shape}")
    output_spectra = np.einsum("fm,mtf->tf", np.conj(actual_weights), spectra, optimize=True)
    return _istft(output_spectra, values.shape[1], fft_size, hop_size, window)


def beamform_stft_components(
    components: Mapping[str, np.ndarray],
    weights: np.ndarray,
    *,
    n_fft: int = 512,
    hop: int | None = None,
) -> dict[str, np.ndarray]:
    return {
        name: apply_stft_beamformer(value, weights, n_fft=n_fft, hop=hop)
        for name, value in components.items()
    }


def estimate_wpe_filters(
    signals: np.ndarray | Sequence[Sequence[float]],
    *,
    n_fft: int = 512,
    hop: int | None = None,
    delay_frames: int = 3,
    taps: int = 10,
    iterations: int = 2,
    diagonal_loading: float = 1e-4,
) -> np.ndarray:
    """Estimate multichannel WPE late-reverberation prediction filters."""
    values = channel_first(signals, "signals")
    fft_size, hop_size, window = _stft_parameters(n_fft, hop)
    delay, order = _prediction_parameters(delay_frames, taps)
    spectra = np.stack([_stft(channel, fft_size, hop_size, window) for channel in values])
    valid_frames = spectra.shape[1] - delay - order + 1
    if valid_frames < max(4, values.shape[0] * order):
        raise ValueError("signals are too short for the requested WPE configuration")
    filters = np.zeros(
        (spectra.shape[2], values.shape[0] * order, values.shape[0]),
        dtype=np.complex128,
    )
    identity = np.eye(values.shape[0] * order, dtype=np.complex128)
    for frequency_index in range(spectra.shape[2]):
        current, delayed = _prediction_matrices(
            spectra[:, :, frequency_index],
            delay,
            order,
        )
        power = np.maximum(np.mean(np.abs(current) ** 2, axis=0), 1e-8)
        prediction = np.zeros_like(current)
        for _ in range(max(1, int(iterations))):
            inverse_power = 1.0 / np.maximum(power, np.percentile(power, 10) * 1e-2 + 1e-10)
            covariance = (delayed * inverse_power[None, :]) @ np.conj(delayed.T)
            cross = (delayed * inverse_power[None, :]) @ np.conj(current.T)
            scale = max(float(np.trace(covariance).real) / covariance.shape[0], 1e-10)
            loaded = covariance + identity * (float(diagonal_loading) * scale + 1e-10)
            prediction_filter = _stable_solve(loaded, cross)
            prediction = np.conj(prediction_filter.T) @ delayed
            residual = current - prediction
            power = np.maximum(np.mean(np.abs(residual) ** 2, axis=0), 1e-8)
        filters[frequency_index] = prediction_filter
    return filters


def apply_wpe(
    signals: np.ndarray | Sequence[Sequence[float]],
    filters: np.ndarray,
    *,
    n_fft: int = 512,
    hop: int | None = None,
    delay_frames: int = 3,
    taps: int = 10,
) -> np.ndarray:
    values = channel_first(signals, "signals")
    fft_size, hop_size, window = _stft_parameters(n_fft, hop)
    delay, order = _prediction_parameters(delay_frames, taps)
    spectra = np.stack([_stft(channel, fft_size, hop_size, window) for channel in values])
    actual_filters = np.asarray(filters, dtype=np.complex128)
    expected = (spectra.shape[2], values.shape[0] * order, values.shape[0])
    if actual_filters.shape != expected:
        raise ValueError(f"filters have shape {actual_filters.shape}; expected {expected}")
    output = spectra.copy()
    start = delay + order - 1
    for frequency_index in range(spectra.shape[2]):
        current, delayed = _prediction_matrices(
            spectra[:, :, frequency_index],
            delay,
            order,
        )
        prediction = np.conj(actual_filters[frequency_index].T) @ delayed
        output[:, start:, frequency_index] = current - prediction
    return np.vstack(
        [
            _istft(output[channel], values.shape[1], fft_size, hop_size, window)
            for channel in range(values.shape[0])
        ]
    ).astype(np.float32)


def estimate_wpd_weights(
    signals: np.ndarray | Sequence[Sequence[float]],
    arrival_times_s: Sequence[float],
    fs: int,
    *,
    n_fft: int = 512,
    hop: int | None = None,
    delay_frames: int = 3,
    taps: int = 5,
    diagonal_loading: float = 1e-3,
) -> np.ndarray:
    """Estimate weighted-power distortionless beamformer coefficients."""
    values = channel_first(signals, "signals")
    fft_size, hop_size, window = _stft_parameters(n_fft, hop)
    delay, order = _prediction_parameters(delay_frames, taps)
    spectra = np.stack([_stft(channel, fft_size, hop_size, window) for channel in values])
    guide = steering_vector(arrival_times_s, fs, n_fft=fft_size)
    if guide.shape[1] != values.shape[0]:
        raise ValueError("arrival_times_s must contain one value per channel")
    extended_channels = values.shape[0] * (order + 1)
    weights = np.zeros((spectra.shape[2], extended_channels), dtype=np.complex128)
    identity = np.eye(extended_channels, dtype=np.complex128)
    for frequency_index in range(spectra.shape[2]):
        current, delayed = _prediction_matrices(
            spectra[:, :, frequency_index],
            delay,
            order,
        )
        extended = np.vstack([current, delayed])
        initial = np.mean(np.conj(guide[frequency_index])[:, None] * current, axis=0)
        power = np.maximum(np.abs(initial) ** 2, np.percentile(np.abs(initial) ** 2, 10) + 1e-8)
        covariance = (extended / power[None, :]) @ np.conj(extended.T)
        scale = max(float(np.trace(covariance).real) / extended_channels, 1e-10)
        loaded = covariance + identity * (float(diagonal_loading) * scale + 1e-10)
        extended_guide = np.concatenate(
            [guide[frequency_index], np.zeros(values.shape[0] * order, dtype=np.complex128)]
        )
        solved = _stable_solve(loaded, extended_guide)
        denominator = max(float(np.vdot(extended_guide, solved).real), 1e-12)
        weights[frequency_index] = solved / denominator
    return weights


def apply_wpd_beamformer(
    signals: np.ndarray | Sequence[Sequence[float]],
    weights: np.ndarray,
    *,
    n_fft: int = 512,
    hop: int | None = None,
    delay_frames: int = 3,
    taps: int = 5,
) -> np.ndarray:
    values = channel_first(signals, "signals")
    fft_size, hop_size, window = _stft_parameters(n_fft, hop)
    delay, order = _prediction_parameters(delay_frames, taps)
    spectra = np.stack([_stft(channel, fft_size, hop_size, window) for channel in values])
    actual_weights = np.asarray(weights, dtype=np.complex128)
    expected = (spectra.shape[2], values.shape[0] * (order + 1))
    if actual_weights.shape != expected:
        raise ValueError(f"weights have shape {actual_weights.shape}; expected {expected}")
    start = delay + order - 1
    output = np.mean(spectra, axis=0)
    for frequency_index in range(spectra.shape[2]):
        current, delayed = _prediction_matrices(
            spectra[:, :, frequency_index],
            delay,
            order,
        )
        extended = np.vstack([current, delayed])
        output[start:, frequency_index] = np.conj(actual_weights[frequency_index]) @ extended
    return _istft(output, values.shape[1], fft_size, hop_size, window)


def beamform_components(
    components: Mapping[str, np.ndarray],
    arrival_times_s: Sequence[float],
    fs: int,
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    return {
        name: delay_and_sum(value, arrival_times_s, fs, weights=weights)
        for name, value in components.items()
    }


def _stft(signal: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    padding = n_fft // 2
    centered = np.pad(signal, (padding, padding))
    frame_count = max(1, int(math.ceil(max(centered.size - n_fft, 0) / hop)) + 1)
    padded_length = (frame_count - 1) * hop + n_fft
    padded = np.pad(centered, (0, max(0, padded_length - centered.size)))
    frames = np.stack([padded[index * hop : index * hop + n_fft] for index in range(frame_count)])
    return np.fft.rfft(frames * window[None, :], axis=1)


def _spatial_covariance(signal: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    spectra = np.stack([_stft(channel, n_fft, hop, window) for channel in signal])
    return np.asarray(
        np.einsum("mtf,ntf->fmn", spectra, np.conj(spectra), optimize=True) / spectra.shape[1],
        dtype=np.complex128,
    )


def _prediction_parameters(delay_frames: int, taps: int) -> tuple[int, int]:
    delay = int(delay_frames)
    order = int(taps)
    if delay < 1 or order < 1:
        raise ValueError("delay_frames and taps must be positive")
    return delay, order


def _prediction_matrices(
    spectra: np.ndarray,
    delay_frames: int,
    taps: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = delay_frames + taps - 1
    current = spectra[:, start:]
    delayed = np.vstack(
        [
            spectra[:, start - delay_frames - tap : spectra.shape[1] - delay_frames - tap]
            for tap in range(taps)
        ]
    )
    return current, delayed


def _stable_solve(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, right_hand_side)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix, rcond=1e-8) @ right_hand_side


def _istft(
    spectra: np.ndarray,
    output_samples: int,
    n_fft: int,
    hop: int,
    window: np.ndarray,
) -> np.ndarray:
    output_length = (spectra.shape[0] - 1) * hop + n_fft
    output = np.zeros(output_length, dtype=np.float64)
    normalization = np.zeros(output_length, dtype=np.float64)
    for frame_index, spectrum in enumerate(spectra):
        start = frame_index * hop
        frame = np.fft.irfft(spectrum, n_fft) * window
        output[start : start + n_fft] += frame
        normalization[start : start + n_fft] += window * window
    output /= np.maximum(normalization, 1e-12)
    padding = n_fft // 2
    return np.asarray(output[padding : padding + output_samples], dtype=np.float32)


def _stft_parameters(n_fft: int, hop: int | None) -> tuple[int, int, np.ndarray]:
    fft_size = int(n_fft)
    hop_size = int(hop or fft_size // 4)
    if fft_size < 16 or hop_size < 1 or hop_size > fft_size:
        raise ValueError("n_fft and hop define an invalid STFT")
    return fft_size, hop_size, np.sqrt(np.hanning(fft_size))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _positive_sample_rate(fs: int) -> int:
    value = int(fs)
    if value < 1:
        raise ValueError("fs must be positive")
    return value


def _positive_speed(speed: float) -> float:
    value = float(speed)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("speed_of_sound_m_s must be positive and finite")
    return value
