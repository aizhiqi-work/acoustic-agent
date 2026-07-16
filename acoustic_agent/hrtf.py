from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from .models import FREQUENCY_BANDS, AcousticPath
from .rir import render_impulses


DEFAULT_SOFA_PATH = Path(__file__).resolve().parent / "resources" / "hrtf" / "cipic_124.sofa"
MAX_LOUDNESS_GAIN_DB = 24.0


@dataclass(frozen=True)
class SOFAHRTFDatabase:
    path: str
    sampling_rate: int
    ir: np.ndarray
    source_positions: np.ndarray
    source_unit_vectors: np.ndarray


def render_binaural_sofa(
    mono_rir: np.ndarray,
    *,
    source: Sequence[float],
    receiver: Sequence[float],
    fs: int,
    paths: Sequence[AcousticPath] = (),
    sofa_path: str | Path | None = None,
    interpolation: str = "bilinear",
    orientation_deg: float = 0.0,
    spatial_blend: float = 1.0,
    loudness_normalization: str = "energy",
    seed: int = 1729,
    ambisonic_rir: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    mono = np.asarray(mono_rir, dtype=np.float32).reshape(-1)
    db = load_sofa_hrtf(sofa_path or DEFAULT_SOFA_PATH, int(fs))
    if paths:
        return _render_path_aware(
            mono,
            paths=paths,
            db=db,
            fs=int(fs),
            interpolation=interpolation,
            orientation_deg=float(orientation_deg),
            spatial_blend=float(spatial_blend),
            loudness_normalization=loudness_normalization,
            seed=int(seed),
            ambisonic_rir=ambisonic_rir,
        )
    direction = np.asarray(source, dtype=float) - np.asarray(receiver, dtype=float)
    hrir = db.ir[_nearest_index(db.source_unit_vectors, _world_to_listener_direction(direction, orientation_deg))]
    if interpolation == "bilinear":
        hrir = _interpolated_hrir(db.ir, db.source_unit_vectors, _world_to_listener_direction(direction, orientation_deg), count=4)
    left = _fft_convolve_truncated(mono, hrir[0], mono.size)
    right = _fft_convolve_truncated(mono, hrir[1], mono.size)
    rendered = _apply_spatial_blend(np.stack([left, right], axis=0), mono, float(spatial_blend))
    rendered, loudness_meta = _normalize_binaural_loudness(rendered, mono, loudness_normalization)
    return rendered.astype(np.float32), {
        "model": "sofa_hrir_fft_convolution",
        "sofa_path": str(db.path),
        "sofa_database": Path(db.path).stem,
        "sofa_sampling_rate": int(db.sampling_rate),
        "interpolation": interpolation,
        "interpolation_model": "log_magnitude_unwrapped_phase" if interpolation == "bilinear" else "nearest_hrir",
        "resampling_model": "bandlimited_sinc",
        "coordinate_model": "world_x_right_y_front_z_up_to_sofa_x_front_y_left_z_up",
        "orientation_deg": round(float(orientation_deg), 4),
        "spatial_blend": round(float(np.clip(spatial_blend, 0.0, 1.0)), 4),
        **loudness_meta,
    }


@lru_cache(maxsize=4)
def load_sofa_hrtf(path: str | Path, fs: int) -> SOFAHRTFDatabase:
    sofa_path = Path(path).expanduser()
    if not sofa_path.exists():
        raise FileNotFoundError(f"SOFA HRTF file not found: {sofa_path}")
    with h5py.File(sofa_path, "r") as handle:
        ir = np.asarray(handle["Data.IR"][()], dtype=np.float32)
        sampling_rate = int(round(float(np.asarray(handle["Data.SamplingRate"][()]).reshape(-1)[0])))
        source_positions = np.asarray(handle["SourcePosition"][()], dtype=np.float32)
    if ir.ndim != 3 or ir.shape[1] != 2:
        raise ValueError(f"SOFA Data.IR must have shape (measurements, 2, samples), got {ir.shape}")
    if sampling_rate != int(fs):
        ir = _resample_bandlimited(ir, sampling_rate, int(fs))
        sampling_rate = int(fs)
    return SOFAHRTFDatabase(
        path=str(sofa_path),
        sampling_rate=sampling_rate,
        ir=np.ascontiguousarray(ir, dtype=np.float32),
        source_positions=np.ascontiguousarray(source_positions, dtype=np.float32),
        source_unit_vectors=np.ascontiguousarray(_source_positions_to_unit_vectors(source_positions), dtype=np.float32),
    )


def _interpolated_hrir(ir: np.ndarray, vectors: np.ndarray, query: np.ndarray, count: int) -> np.ndarray:
    dots = np.clip(vectors @ query, -1.0, 1.0)
    count = max(1, min(int(count), len(dots)))
    indices = np.argpartition(-dots, count - 1)[:count]
    angles = np.arccos(np.clip(dots[indices], -1.0, 1.0))
    if float(np.min(angles)) < 1e-6:
        return ir[int(indices[int(np.argmin(angles))])]
    weights = 1.0 / np.maximum(angles, 1e-5)
    weights = weights / np.sum(weights)
    spectra = np.fft.rfft(np.asarray(ir[indices], dtype=np.float64), axis=-1)
    log_magnitude = np.log(np.maximum(np.abs(spectra), 1e-9))
    phase = np.unwrap(np.angle(spectra), axis=-1)
    magnitude_out = np.exp(np.tensordot(weights, log_magnitude, axes=(0, 0)))
    phase_out = np.tensordot(weights, phase, axes=(0, 0))
    interpolated = np.fft.irfft(magnitude_out * np.exp(1j * phase_out), n=ir.shape[-1], axis=-1)
    return np.asarray(interpolated, dtype=np.float32)


def _render_path_aware(
    mono: np.ndarray,
    *,
    paths: Sequence[AcousticPath],
    db: SOFAHRTFDatabase,
    fs: int,
    interpolation: str,
    orientation_deg: float,
    spatial_blend: float,
    loudness_normalization: str,
    seed: int,
    ambisonic_rir: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    directional = [path for path in paths if _path_has_explicit_binaural_direction(path)]
    length = int(mono.size)
    ambisonic = _as_ambisonic_rir(ambisonic_rir, length)
    if not directional:
        if ambisonic is not None:
            reflection = _decode_foa_binaural(ambisonic, db, orientation_deg)
            rendered = _apply_spatial_blend(reflection, mono, spatial_blend)
            rendered, loudness_meta = _normalize_binaural_loudness(rendered, mono, loudness_normalization)
            return rendered.astype(np.float32), _hrtf_metadata(
                db,
                interpolation,
                orientation_deg,
                spatial_blend,
                model="sofa_foa_reflections",
                directional_path_count=0,
                ambisonic_order=1,
                ambisonic_channels=["W", "X", "Y", "Z"],
                ambisonic_energy=float(np.sum(ambisonic * ambisonic)),
                residual_energy=0.0,
                **loudness_meta,
            )
        diffuse = _decorrelate_residual(mono, seed)
        rendered = _apply_spatial_blend(diffuse, mono, spatial_blend)
        rendered, loudness_meta = _normalize_binaural_loudness(rendered, mono, loudness_normalization)
        return rendered.astype(np.float32), _hrtf_metadata(
            db,
            interpolation,
            orientation_deg,
            spatial_blend,
            model="sofa_path_aware_decorrelated_residual",
            directional_path_count=0,
            residual_energy=float(np.sum(mono * mono)),
            **loudness_meta,
        )

    directional_mono = np.zeros(length, dtype=np.float32)
    binaural = np.zeros((2, length), dtype=np.float32)
    kind_counts: dict[str, int] = {}

    for path in directional:
        path_signal = _render_directional_path_signal(path, fs, length)
        directional_mono += path_signal
        direction = _path_listener_direction(path)
        hrir = _query_hrir(db, direction, interpolation, orientation_deg)
        binaural[0] += _fft_convolve_truncated(path_signal, hrir[0], length)
        binaural[1] += _fft_convolve_truncated(path_signal, hrir[1], length)
        kind_counts[path.kind] = kind_counts.get(path.kind, 0) + 1

    if ambisonic is not None:
        reflection = _decode_foa_binaural(ambisonic, db, orientation_deg)
        binaural += reflection
        residual_energy = 0.0
        rendered = _apply_spatial_blend(binaural, mono, spatial_blend)
        rendered, loudness_meta = _normalize_binaural_loudness(rendered, mono, loudness_normalization)
        return rendered.astype(np.float32), _hrtf_metadata(
            db,
            interpolation,
            orientation_deg,
            spatial_blend,
            model="sofa_path_aware_direct_diffraction_plus_foa_reflections",
            directional_path_count=len(directional),
            directional_path_kinds=kind_counts,
            directional_filtering="six_band_path_gains",
            ambisonic_order=1,
            ambisonic_channels=["W", "X", "Y", "Z"],
            ambisonic_energy=float(np.sum(ambisonic * ambisonic)),
            residual_energy=residual_energy,
            **loudness_meta,
        )

    residual = mono - directional_mono
    residual_energy = float(np.sum(residual * residual))
    if residual_energy > 1e-12:
        binaural += _decorrelate_residual(residual, seed)
    rendered = _apply_spatial_blend(binaural, mono, spatial_blend)
    rendered, loudness_meta = _normalize_binaural_loudness(rendered, mono, loudness_normalization)
    return rendered.astype(np.float32), _hrtf_metadata(
        db,
        interpolation,
        orientation_deg,
        spatial_blend,
        model="sofa_path_aware_direct_diffraction_plus_decorrelated_residual",
        directional_path_count=len(directional),
        directional_path_kinds=kind_counts,
        directional_filtering="six_band_path_gains",
        residual_energy=residual_energy,
        **loudness_meta,
    )


def _path_has_explicit_binaural_direction(path: AcousticPath) -> bool:
    if not bool(path.metadata.get("contributes_to_rir", False)):
        return False
    if path.kind in {"direct", "direct_transmitted", "diffraction"}:
        return len(path.points) >= 2
    return False


def _path_listener_direction(path: AcousticPath) -> np.ndarray:
    points = np.asarray(path.points, dtype=float)
    return points[-2] - points[-1]


def _render_directional_path_signal(path: AcousticPath, fs: int, length: int) -> np.ndarray:
    from .steam_rt import bandlimit_band_signals

    band_impulses = np.zeros((len(FREQUENCY_BANDS), length), dtype=np.float32)
    duration_s = length / max(int(fs), 1)
    for band_index, band in enumerate(FREQUENCY_BANDS):
        gain = float(path.band_gains.get(band, path.gain))
        band_impulses[band_index] = render_impulses(
            np.asarray([float(path.delay_s)], dtype=np.float64),
            np.asarray([gain], dtype=np.float64),
            fs=int(fs),
            duration_s=duration_s,
            fractional=True,
        )
    return np.sum(bandlimit_band_signals(band_impulses, int(fs)), axis=0, dtype=np.float32)


def _query_hrir(db: SOFAHRTFDatabase, direction: Sequence[float], interpolation: str, orientation_deg: float) -> np.ndarray:
    query = _world_to_listener_direction(np.asarray(direction, dtype=float), orientation_deg)
    if interpolation == "bilinear":
        return _interpolated_hrir(db.ir, db.source_unit_vectors, query, count=4)
    return db.ir[_nearest_index(db.source_unit_vectors, query)]


def _as_ambisonic_rir(value: np.ndarray | None, length: int) -> np.ndarray | None:
    if value is None:
        return None
    ambi = np.asarray(value, dtype=np.float32)
    if ambi.ndim != 2 or ambi.shape[0] != 4:
        return None
    out = np.zeros((4, length), dtype=np.float32)
    n = min(length, ambi.shape[1])
    out[:, :n] = ambi[:, :n]
    return out


def _decode_foa_binaural(ambisonic: np.ndarray, db: SOFAHRTFDatabase, orientation_deg: float) -> np.ndarray:
    rotated = _rotate_foa_yaw(ambisonic, orientation_deg)
    decoder = _foa_decoder_filters(db)
    length = rotated.shape[1]
    out = np.zeros((2, length), dtype=np.float32)
    for ear in range(2):
        for channel in range(4):
            out[ear] += _fft_convolve_truncated(rotated[channel], decoder[ear, channel], length)
    return out.astype(np.float32)


def _rotate_foa_yaw(ambisonic: np.ndarray, orientation_deg: float) -> np.ndarray:
    yaw = math.radians(float(orientation_deg))
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    out = np.asarray(ambisonic, dtype=np.float32).copy()
    x = out[1].copy()
    y = out[2].copy()
    out[1] = cos_yaw * x - sin_yaw * y
    out[2] = sin_yaw * x + cos_yaw * y
    return out


@lru_cache(maxsize=4)
def _foa_decoder_filters_cached(path: str, sampling_rate: int, ir_shape: tuple[int, int, int]) -> np.ndarray:
    db = load_sofa_hrtf(path, sampling_rate)
    return _fit_foa_decoder(db.ir, db.source_unit_vectors)


def _foa_decoder_filters(db: SOFAHRTFDatabase) -> np.ndarray:
    return _foa_decoder_filters_cached(db.path, int(db.sampling_rate), tuple(db.ir.shape))


def _fit_foa_decoder(ir: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    design = np.column_stack([
        np.ones(vectors.shape[0], dtype=np.float64),
        vectors[:, 0],
        vectors[:, 1],
        vectors[:, 2],
    ])
    pinv = np.linalg.pinv(design)
    filters = np.zeros((2, 4, ir.shape[-1]), dtype=np.float32)
    for ear in range(2):
        filters[ear] = (pinv @ ir[:, ear, :]).astype(np.float32)
    return filters


def _decorrelate_residual(signal: np.ndarray, seed: int) -> np.ndarray:
    mono = np.asarray(signal, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return np.zeros((2, 0), dtype=np.float32)
    left_kernel = _diffuse_kernel(seed + 17)
    right_kernel = _diffuse_kernel(seed + 31)
    left = _fft_convolve_truncated(mono, left_kernel, mono.size)
    right = _fft_convolve_truncated(mono, right_kernel, mono.size)
    return _match_energy(np.stack([left, right], axis=0), mono).astype(np.float32)


def _diffuse_kernel(seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    taps = rng.normal(0.0, 1.0, 23).astype(np.float32)
    taps *= np.hanning(taps.size).astype(np.float32)
    taps[0] += 1.0
    norm = float(np.linalg.norm(taps))
    if norm > 1e-12:
        taps /= norm
    return taps


def _match_energy(stereo: np.ndarray, mono: np.ndarray) -> np.ndarray:
    target = float(np.sum(mono * mono))
    current = 0.5 * float(np.sum(stereo[0] * stereo[0]) + np.sum(stereo[1] * stereo[1]))
    if target > 1e-12 and current > 1e-12:
        stereo = stereo * math.sqrt(target / current)
    return stereo


def _normalize_binaural_loudness(stereo: np.ndarray, mono: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    normalization = str(mode).strip().lower()
    if normalization == "rms":
        normalization = "energy"
    if normalization not in {"energy", "none"}:
        raise ValueError("HRTF loudness_normalization must be 'energy' or 'none'")

    target = float(np.sum(np.asarray(mono, dtype=np.float64) ** 2))
    values = np.asarray(stereo, dtype=np.float32)
    current = 0.5 * float(np.sum(values[0].astype(np.float64) ** 2) + np.sum(values[1].astype(np.float64) ** 2))
    raw_db = _energy_ratio_db(current, target)
    gain = 1.0
    if normalization == "energy" and target > 1e-12 and current > 1e-12:
        gain = math.sqrt(target / current)
        gain = min(gain, 10.0 ** (MAX_LOUDNESS_GAIN_DB / 20.0))
        values = values * gain
    final_energy = current * gain * gain
    return values.astype(np.float32), {
        "loudness_normalization": normalization,
        "raw_binaural_energy_db": None if raw_db is None else round(raw_db, 4),
        "loudness_gain_db": round(20.0 * math.log10(max(gain, 1e-12)), 4),
        "binaural_energy_db": None if target <= 1e-12 else round(10.0 * math.log10(max(final_energy, 1e-30) / target), 4),
    }


def _energy_ratio_db(current: float, target: float) -> float | None:
    if target <= 1e-12:
        return None
    return 10.0 * math.log10(max(current, 1e-30) / target)


def _apply_spatial_blend(stereo: np.ndarray, mono: np.ndarray, spatial_blend: float) -> np.ndarray:
    blend = float(np.clip(spatial_blend, 0.0, 1.0))
    if blend >= 1.0:
        return np.asarray(stereo, dtype=np.float32)
    dry = np.stack([mono, mono], axis=0).astype(np.float32)
    return (blend * np.asarray(stereo, dtype=np.float32) + (1.0 - blend) * dry).astype(np.float32)


def _hrtf_metadata(
    db: SOFAHRTFDatabase,
    interpolation: str,
    orientation_deg: float,
    spatial_blend: float,
    **extra,
) -> dict:
    return {
        "sofa_path": str(db.path),
        "sofa_database": Path(db.path).stem,
        "sofa_sampling_rate": int(db.sampling_rate),
        "interpolation": interpolation,
        "interpolation_model": "log_magnitude_unwrapped_phase" if interpolation == "bilinear" else "nearest_hrir",
        "resampling_model": "bandlimited_sinc",
        "coordinate_model": "world_x_right_y_front_z_up_to_sofa_x_front_y_left_z_up",
        "orientation_deg": round(float(orientation_deg), 4),
        "spatial_blend": round(float(np.clip(spatial_blend, 0.0, 1.0)), 4),
        **extra,
    }


def _nearest_index(vectors: np.ndarray, query: np.ndarray) -> int:
    return int(np.argmax(vectors @ query))


def _source_positions_to_unit_vectors(positions: np.ndarray) -> np.ndarray:
    azimuth = np.deg2rad(positions[:, 0].astype(float))
    elevation = np.deg2rad(positions[:, 1].astype(float))
    cos_el = np.cos(elevation)
    vectors = np.stack([-np.sin(azimuth) * cos_el, np.cos(azimuth) * cos_el, np.sin(elevation)], axis=1)
    return (vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _world_to_listener_direction(direction: np.ndarray, orientation_deg: float) -> np.ndarray:
    yaw = math.radians(float(orientation_deg))
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    vec = _unit_vector(direction).astype(np.float64)
    return np.asarray([cos_yaw * vec[0] - sin_yaw * vec[1], sin_yaw * vec[0] + cos_yaw * vec[1], vec[2]], dtype=np.float32)


def _unit_vector(values: Sequence[float]) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float64).reshape(-1)
    if vec.size < 3:
        vec = np.pad(vec, (0, 3 - vec.size))
    norm = float(np.linalg.norm(vec[:3]))
    if norm <= 1e-12:
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    return (vec[:3] / norm).astype(np.float32)


def _fft_convolve_truncated(signal: np.ndarray, ir: np.ndarray, length: int) -> np.ndarray:
    size = 1 << max(1, int(signal.size + ir.size - 2).bit_length())
    result = np.fft.irfft(np.fft.rfft(signal, size) * np.fft.rfft(ir, size), size)
    return np.asarray(result[:length], dtype=np.float32)


def _resample_bandlimited(ir: np.ndarray, in_fs: int, out_fs: int) -> np.ndarray:
    old_n = ir.shape[-1]
    new_n = max(1, int(math.ceil(old_n * float(out_fs) / float(in_fs))))
    ratio = float(out_fs) / float(in_fs)
    cutoff = min(1.0, ratio)
    half_width = 16.0 / cutoff
    flat = np.asarray(ir, dtype=np.float64).reshape(-1, old_n)
    out = np.zeros((flat.shape[0], new_n), dtype=np.float64)
    for output_index in range(new_n):
        position = output_index / ratio
        start = max(0, int(math.ceil(position - half_width)))
        stop = min(old_n, int(math.floor(position + half_width)) + 1)
        sample_indices = np.arange(start, stop, dtype=np.float64)
        offsets = position - sample_indices
        weights = cutoff * np.sinc(cutoff * offsets) * np.sinc(offsets / half_width)
        out[:, output_index] = flat[:, start:stop] @ weights
    return np.asarray(out.reshape(*ir.shape[:-1], new_n), dtype=np.float32)
