from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

from acoustic_agent.audio import resample_audio


def load_wav_mono(path: str | Path, target_fs: int) -> np.ndarray:
    """Load PCM or IEEE-float RIFF/WAVE audio as normalized mono float32."""
    audio_path = Path(path)
    raw = audio_path.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{audio_path} is not a RIFF/WAVE file")
    format_code = channels = sample_rate = bits_per_sample = None
    data = None
    offset = 12
    while offset + 8 <= len(raw):
        chunk_id = raw[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", raw, offset + 4)[0]
        payload = raw[offset + 8 : offset + 8 + chunk_size]
        if chunk_id == b"fmt ":
            format_code, channels, sample_rate, _, _, bits_per_sample = struct.unpack_from("<HHIIHH", payload, 0)
        elif chunk_id == b"data":
            data = payload
        offset += 8 + chunk_size + (chunk_size & 1)
    if None in {format_code, channels, sample_rate, bits_per_sample} or data is None:
        raise ValueError(f"{audio_path} is missing fmt or data chunks")
    values = _decode_samples(data, int(format_code), int(bits_per_sample))
    frame_count = values.size // int(channels)
    values = values[: frame_count * int(channels)].reshape(frame_count, int(channels)).T
    mono = np.mean(values, axis=0, keepdims=True)
    resampled = resample_audio(mono, int(sample_rate), int(target_fs))[0]
    resampled -= float(np.mean(resampled))
    rms = float(np.sqrt(np.mean(np.square(resampled, dtype=np.float64))))
    if rms <= 1e-12:
        raise ValueError(f"{audio_path} is silent")
    return np.asarray(resampled / rms * 0.1, dtype=np.float32)


def write_wav_mono(path: str | Path, samples: np.ndarray, fs: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    peak = max(float(np.max(np.abs(values))), 1e-12)
    if peak > 0.98:
        values = values * (0.98 / peak)
    pcm = np.asarray(np.clip(values, -1.0, 1.0) * 32767.0, dtype="<i2")
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(fs))
        handle.writeframes(pcm.tobytes())


def _decode_samples(data: bytes, format_code: int, bits_per_sample: int) -> np.ndarray:
    if format_code == 1 and bits_per_sample == 16:
        return np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    if format_code == 1 and bits_per_sample == 24:
        packed = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float64) / 8388608.0
    if format_code == 1 and bits_per_sample == 32:
        return np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
    if format_code == 3 and bits_per_sample == 32:
        return np.frombuffer(data, dtype="<f4").astype(np.float64)
    if format_code == 3 and bits_per_sample == 64:
        return np.frombuffer(data, dtype="<f8").astype(np.float64)
    raise ValueError(f"unsupported WAV format code={format_code}, bits={bits_per_sample}")
