from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - exercised only when numba is absent
    njit = None


def render_impulses(
    delays_s: np.ndarray,
    gains: np.ndarray,
    *,
    fs: int,
    duration_s: float,
    fractional: bool = True,
    sinc_half_width: int = 8,
) -> np.ndarray:
    n = max(1, int(round(float(fs) * float(duration_s))))
    out = np.zeros(n, dtype=np.float32)
    delays = np.asarray(delays_s, dtype=np.float64)
    values = np.asarray(gains, dtype=np.float64)
    if fractional:
        _add_fractional_impulses(out, delays, values, int(fs), int(sinc_half_width))
    else:
        _add_integer_impulses(out, delays, values, int(fs))
    return out


if njit is not None:

    @njit(cache=True)
    def _add_integer_impulses(out, delays_s, gains, fs):
        for i in range(delays_s.shape[0]):
            index = int(round(delays_s[i] * fs))
            if 0 <= index < out.shape[0]:
                out[index] += gains[i]

    @njit(cache=True)
    def _add_fractional_impulses(out, delays_s, gains, fs, half_width):
        for i in range(delays_s.shape[0]):
            exact = delays_s[i] * fs
            center = int(np.floor(exact))
            frac = exact - center
            for k in range(-half_width, half_width + 1):
                index = center + k
                if 0 <= index < out.shape[0]:
                    x = k - frac
                    if abs(x) < 1e-12:
                        sinc = 1.0
                    else:
                        sinc = np.sin(np.pi * x) / (np.pi * x)
                    window_pos = (k + half_width) / max(2.0 * half_width, 1.0)
                    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * window_pos)
                    out[index] += gains[i] * sinc * window

else:

    def _add_integer_impulses(out, delays_s, gains, fs):
        for delay, gain in zip(delays_s, gains):
            index = int(round(float(delay) * fs))
            if 0 <= index < out.shape[0]:
                out[index] += float(gain)

    def _add_fractional_impulses(out, delays_s, gains, fs, half_width):
        for delay, gain in zip(delays_s, gains):
            exact = float(delay) * fs
            center = int(np.floor(exact))
            frac = exact - center
            for k in range(-half_width, half_width + 1):
                index = center + k
                if 0 <= index < out.shape[0]:
                    x = k - frac
                    sinc = 1.0 if abs(x) < 1e-12 else np.sin(np.pi * x) / (np.pi * x)
                    window_pos = (k + half_width) / max(2.0 * half_width, 1.0)
                    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * window_pos)
                    out[index] += float(gain) * sinc * window


def add_late_tail(
    rir: np.ndarray,
    *,
    fs: int,
    start_s: float,
    rt60_s: float,
    energy: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    out = np.asarray(rir, dtype=np.float32).copy()
    start = min(len(out), max(0, int(round(start_s * fs))))
    if start >= len(out) or energy <= 0.0:
        return out, {"added": False, "reason": "no_tail_energy"}
    rng = np.random.default_rng(seed)
    t = np.arange(len(out) - start, dtype=np.float32) / float(fs)
    decay = np.exp(-6.9078 * t / max(float(rt60_s), 0.08))
    noise = rng.normal(0.0, 1.0, len(t)).astype(np.float32) * decay
    current = float(np.sum(noise * noise))
    if current > 1e-12:
        noise *= np.sqrt(float(energy) / current)
    out[start:] += noise
    return out, {
        "added": True,
        "start_s": round(float(start / fs), 6),
        "rt60_s": round(float(rt60_s), 4),
        "rendered_energy": round(float(np.sum(noise * noise)), 12),
        "model": "energy_decay_noise_tail",
    }

