"""Audio loading, mono conversion, resampling, and controlled noise augmentation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def prepare_audio(audio: np.ndarray, sample_rate: int, target_rate: int = 16_000) -> np.ndarray:
    """Return finite mono float32 audio at target_rate."""
    if int(sample_rate) <= 0 or int(target_rate) <= 0:
        raise ValueError("sample_rate and target_rate must be positive integers")
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        # soundfile convention: samples x channels; always average channels.
        x = x.mean(axis=1)
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError(f"Expected mono or stereo audio; got shape {x.shape}")
    if x.size == 0:
        return np.empty(0, dtype=np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, copy=False)
    if sample_rate != target_rate:
        from math import gcd
        divisor = gcd(int(sample_rate), int(target_rate))
        x = resample_poly(x, target_rate // divisor, sample_rate // divisor).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def load_audio(path: str | Path, target_rate: int = 16_000) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = prepare_audio(audio, sr, target_rate)
    return audio, target_rate


def add_awgn(audio: np.ndarray, snr_db: float = 10.0, seed: int = 42) -> np.ndarray:
    """Add deterministic white Gaussian noise at the requested signal-to-noise ratio.

    The original signal amplitude is preserved unless the mixed waveform would
    exceed 0.99; any required final scaling is applied to the whole mixture, so
    the SNR remains unchanged. This avoids silently changing the clean signal's
    loudness as part of the noisy condition.
    """
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x.copy()
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, copy=False)
    signal_power = float(np.mean(np.square(x, dtype=np.float64)))
    if signal_power < 1e-16:
        return x.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    noise_power = float(np.mean(np.square(noise, dtype=np.float64)))
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise *= np.sqrt(target_noise_power / max(noise_power, 1e-20))
    mixed = x + noise
    mixed_peak = float(np.max(np.abs(mixed)))
    if mixed_peak > 0.99:
        mixed *= 0.99 / mixed_peak
    return np.ascontiguousarray(mixed, dtype=np.float32)
