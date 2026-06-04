"""Tone-shaping and output safety processors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .analysis import AnalysisResult, AnalysisSettings
from .codec import peak_dbfs


@dataclass(slots=True)
class ToneSettings:
    enable_deesser: bool = True
    enable_dynamic_eq: bool = True
    deesser_freq_low_hz: float = 5500.0
    deesser_freq_high_hz: float = 9000.0
    deesser_max_reduction_db: float = 6.0
    presence_freq_low_hz: float = 2800.0
    presence_freq_high_hz: float = 4500.0
    presence_max_reduction_db: float = 3.0
    sibilance_freq_low_hz: float = 5500.0
    sibilance_freq_high_hz: float = 9000.0
    sibilance_max_reduction_db: float = 4.5


@dataclass(slots=True)
class LimiterSettings:
    enable_limiter: bool = True
    lookahead_ms: float = 8.0
    ceiling_db: float = -1.0
    release_ms: float = 100.0
    peak_normalize_max_gain_db: float = 1.5


def dynamic_deesser(
    samples: np.ndarray,
    sample_rate: int,
    analysis_settings: AnalysisSettings,
    tone_settings: ToneSettings,
) -> tuple[np.ndarray, float]:
    """Apply STFT-domain de-essing."""

    if not tone_settings.enable_deesser:
        return np.asarray(samples, dtype=np.float32), 0.0
    return _dynamic_band_reduction(
        samples=samples,
        sample_rate=sample_rate,
        analysis_settings=analysis_settings,
        band_low=tone_settings.deesser_freq_low_hz,
        band_high=tone_settings.deesser_freq_high_hz,
        max_reduction_db=tone_settings.deesser_max_reduction_db,
    )


def dynamic_eq(
    samples: np.ndarray,
    sample_rate: int,
    analysis_settings: AnalysisSettings,
    tone_settings: ToneSettings,
) -> tuple[np.ndarray, float]:
    """Apply lightweight dynamic EQ for harsh presence and sibilance bands."""

    if not tone_settings.enable_dynamic_eq:
        return np.asarray(samples, dtype=np.float32), 0.0

    presence_processed, presence_reduction = _dynamic_band_reduction(
        samples=samples,
        sample_rate=sample_rate,
        analysis_settings=analysis_settings,
        band_low=tone_settings.presence_freq_low_hz,
        band_high=tone_settings.presence_freq_high_hz,
        max_reduction_db=tone_settings.presence_max_reduction_db,
        threshold_bias=1.15,
    )
    sibilance_processed, sibilance_reduction = _dynamic_band_reduction(
        samples=presence_processed,
        sample_rate=sample_rate,
        analysis_settings=analysis_settings,
        band_low=tone_settings.sibilance_freq_low_hz,
        band_high=tone_settings.sibilance_freq_high_hz,
        max_reduction_db=tone_settings.sibilance_max_reduction_db,
        threshold_bias=1.1,
    )
    return sibilance_processed.astype(np.float32), max(presence_reduction, sibilance_reduction)


def lookahead_limiter(
    samples: np.ndarray,
    sample_rate: int,
    limiter_settings: LimiterSettings,
) -> tuple[np.ndarray, float]:
    """Apply a sample-peak limiter with a short lookahead."""

    if not limiter_settings.enable_limiter:
        return np.asarray(samples, dtype=np.float32), 0.0

    mono = np.asarray(samples, dtype=np.float32)
    ceiling = 10.0 ** (float(limiter_settings.ceiling_db) / 20.0)
    lookahead = max(1, int(sample_rate * limiter_settings.lookahead_ms / 1000.0))
    release_samples = max(1, int(sample_rate * limiter_settings.release_ms / 1000.0))

    abs_samples = np.abs(mono)
    padded = np.pad(abs_samples, (0, lookahead), mode="edge")
    if len(mono) == 1:
        future_peak = padded[:1]
    else:
        windows = np.lib.stride_tricks.sliding_window_view(padded, lookahead + 1)
        future_peak = np.max(windows, axis=1)

    instant_gain = np.minimum(1.0, ceiling / np.maximum(future_peak, 1e-6)).astype(np.float32)
    gain = np.ones_like(instant_gain)
    release_alpha = np.exp(-1.0 / float(release_samples))
    current_gain = 1.0
    for index, target in enumerate(instant_gain):
        if target < current_gain:
            current_gain = float(target)
        else:
            current_gain = float(target + (current_gain - target) * release_alpha)
        gain[index] = current_gain

    delayed = np.pad(mono, (lookahead, 0), mode="constant")[: len(mono)]
    processed = delayed * gain
    gain_reduction_db = -20.0 * float(np.log10(np.clip(np.min(gain), 1e-6, 1.0)))
    return processed.astype(np.float32), gain_reduction_db


def peak_normalize(
    samples: np.ndarray,
    limiter_settings: LimiterSettings,
) -> np.ndarray:
    """Trim or lightly boost to the configured ceiling."""

    mono = np.asarray(samples, dtype=np.float32)
    ceiling = 10.0 ** (float(limiter_settings.ceiling_db) / 20.0)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak <= 1e-8:
        return mono

    max_gain = 10.0 ** (float(limiter_settings.peak_normalize_max_gain_db) / 20.0)
    target_gain = min(ceiling / peak, max_gain)
    if abs(target_gain - 1.0) < 1e-4:
        return mono
    return np.clip(mono * target_gain, -1.0, 1.0).astype(np.float32)


def _dynamic_band_reduction(
    *,
    samples: np.ndarray,
    sample_rate: int,
    analysis_settings: AnalysisSettings,
    band_low: float,
    band_high: float,
    max_reduction_db: float,
    threshold_bias: float = 1.05,
) -> tuple[np.ndarray, float]:
    mono = np.asarray(samples, dtype=np.float32)
    stft_window = max(16, int(sample_rate * analysis_settings.stft_window_ms / 1000.0))
    stft_hop = max(4, int(sample_rate * analysis_settings.stft_hop_ms / 1000.0))
    freqs, _, spectrum = signal.stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=stft_window,
        noverlap=max(0, stft_window - stft_hop),
        boundary="zeros",
        padded=True,
    )

    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    band_mask = (freqs >= band_low) & (freqs <= min(band_high, sample_rate / 2.0))
    base_mask = (freqs >= 300.0) & (freqs <= 4000.0)
    if not np.any(band_mask) or not np.any(base_mask):
        return mono, 0.0

    band_energy = np.mean(magnitude[band_mask], axis=0)
    base_energy = np.mean(magnitude[base_mask], axis=0)
    ratio = band_energy / np.maximum(base_energy, 1e-6)
    baseline = max(float(np.median(ratio)) * threshold_bias, 1e-4)
    over = np.maximum(0.0, ratio / baseline - 1.0)
    if np.max(over) <= 0:
        return mono, 0.0

    reduction_db = np.clip(over * max_reduction_db, 0.0, max_reduction_db).astype(np.float32)
    gains = 10.0 ** (-reduction_db / 20.0)
    updated_mag = magnitude.copy()
    updated_mag[band_mask, :] *= gains[np.newaxis, :]
    repaired = updated_mag * np.exp(1j * phase)
    _, restored = signal.istft(
        repaired,
        fs=sample_rate,
        window="hann",
        nperseg=stft_window,
        noverlap=max(0, stft_window - stft_hop),
        input_onesided=True,
    )
    return restored[: len(mono)].astype(np.float32), float(np.max(reduction_db))


__all__ = [
    "LimiterSettings",
    "ToneSettings",
    "dynamic_deesser",
    "dynamic_eq",
    "lookahead_limiter",
    "peak_normalize",
]
