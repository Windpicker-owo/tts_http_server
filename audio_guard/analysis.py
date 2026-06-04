"""Analysis primitives for the TTS audio guard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(slots=True)
class AnalysisSettings:
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    stft_window_ms: float = 32.0
    stft_hop_ms: float = 8.0
    clip_threshold: float = 0.985
    high_band_min_hz: float = 6000.0
    high_band_max_hz: float = 12000.0
    mid_band_min_hz: float = 300.0
    mid_band_max_hz: float = 4000.0


@dataclass(slots=True)
class ThresholdSettings:
    light_badness: float = 8.0
    severe_badness: float = 14.0
    moderate_frame_artifact: float = 6.0
    severe_frame_artifact: float = 10.0
    clipping_ratio_severe: float = 0.01


@dataclass(slots=True)
class AnalysisResult:
    frame_artifact_scores: np.ndarray
    sentence_badness: float
    max_artifact_score: float
    p95_artifact_score: float
    clipping_ratio: float
    harsh_band_energy_p95: float
    severity: str
    repair_frame_mask: np.ndarray


def analyze_audio(
    samples: np.ndarray,
    sample_rate: int,
    settings: AnalysisSettings,
    thresholds: ThresholdSettings,
) -> AnalysisResult:
    """Run a sentence-level rule-based quality scan."""

    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    frame_size = max(8, int(sample_rate * settings.frame_ms / 1000.0))
    hop_size = max(4, int(sample_rate * settings.hop_ms / 1000.0))
    frames = _frame_signal(mono, frame_size, hop_size)

    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-8)
    peak = np.max(np.abs(frames), axis=1)
    peak_to_rms = peak / np.maximum(rms, 1e-4)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1).astype(np.float32)
    clipping = np.mean(np.abs(frames) >= settings.clip_threshold, axis=1).astype(np.float32)

    stft_window = max(16, int(sample_rate * settings.stft_window_ms / 1000.0))
    stft_hop = max(4, int(sample_rate * settings.stft_hop_ms / 1000.0))
    freqs, _, spectrum = signal.stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=stft_window,
        noverlap=max(0, stft_window - stft_hop),
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(spectrum).astype(np.float32) + 1e-8
    spectral_flatness = np.exp(np.mean(np.log(magnitude), axis=0)) / np.mean(magnitude, axis=0)

    normalized_mag = magnitude / np.maximum(np.sum(magnitude, axis=0, keepdims=True), 1e-8)
    if normalized_mag.shape[1] > 1:
        diff = np.diff(normalized_mag, axis=1)
        spectral_flux = np.concatenate(
            [np.zeros(1, dtype=np.float32), np.sqrt(np.sum(diff * diff, axis=0)).astype(np.float32)]
        )
    else:
        spectral_flux = np.zeros(1, dtype=np.float32)

    high_mask = (freqs >= settings.high_band_min_hz) & (
        freqs <= min(settings.high_band_max_hz, sample_rate / 2.0)
    )
    mid_mask = (freqs >= settings.mid_band_min_hz) & (freqs <= settings.mid_band_max_hz)
    high_energy = np.sum(magnitude[high_mask], axis=0) if np.any(high_mask) else np.zeros(magnitude.shape[1])
    mid_energy = np.sum(magnitude[mid_mask], axis=0) if np.any(mid_mask) else np.ones(magnitude.shape[1])
    high_band_ratio = (high_energy / np.maximum(mid_energy, 1e-6)).astype(np.float32)

    aligned_flatness = _align_feature(spectral_flatness, len(rms))
    aligned_flux = _align_feature(spectral_flux, len(rms))
    aligned_high_band = _align_feature(high_band_ratio, len(rms))

    high_band_outlier = _positive_outlier(aligned_high_band)
    flatness_outlier = _positive_outlier(aligned_flatness)
    zcr_outlier = _positive_outlier(zcr)
    flux_outlier = _positive_outlier(aligned_flux)
    peak_to_rms_outlier = _positive_outlier(peak_to_rms)
    clipping_score = np.maximum(_positive_outlier(clipping), clipping * 100.0)

    frame_artifact_scores = (
        1.5 * high_band_outlier
        + 1.2 * flatness_outlier
        + 1.0 * zcr_outlier
        + 1.2 * flux_outlier
        + 1.8 * clipping_score
        + 1.2 * peak_to_rms_outlier
    ).astype(np.float32)

    max_artifact = float(np.max(frame_artifact_scores)) if frame_artifact_scores.size else 0.0
    p95_artifact = float(np.percentile(frame_artifact_scores, 95)) if frame_artifact_scores.size else 0.0
    clipping_ratio = float(np.mean(np.abs(mono) >= settings.clip_threshold))
    harsh_band_energy_p95 = float(np.percentile(aligned_high_band, 95)) if aligned_high_band.size else 0.0

    sentence_badness = (
        max_artifact
        + 0.5 * p95_artifact
        + 10.0 * clipping_ratio
        + 2.0 * harsh_band_energy_p95
    )
    severity = _classify_severity(
        sentence_badness=sentence_badness,
        max_artifact=max_artifact,
        clipping_ratio=clipping_ratio,
        thresholds=thresholds,
    )
    repair_frame_mask = frame_artifact_scores >= thresholds.moderate_frame_artifact

    return AnalysisResult(
        frame_artifact_scores=frame_artifact_scores,
        sentence_badness=float(sentence_badness),
        max_artifact_score=max_artifact,
        p95_artifact_score=p95_artifact,
        clipping_ratio=clipping_ratio,
        harsh_band_energy_p95=harsh_band_energy_p95,
        severity=severity,
        repair_frame_mask=repair_frame_mask,
    )


def _frame_signal(samples: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((1, frame_size), dtype=np.float32)
    if samples.size < frame_size:
        padded = np.pad(samples, (0, frame_size - samples.size))
        return padded.reshape(1, frame_size)

    starts = np.arange(0, max(1, samples.size - frame_size + 1), hop_size)
    last_end = int(starts[-1] + frame_size)
    if last_end < samples.size:
        starts = np.append(starts, samples.size - frame_size)
    return np.stack([samples[start:start + frame_size] for start in starts]).astype(np.float32)


def _align_feature(values: np.ndarray, target_length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if array.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if array.size == target_length:
        return array
    if array.size == 1:
        return np.full(target_length, float(array[0]), dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, num=array.size)
    target_x = np.linspace(0.0, 1.0, num=target_length)
    return np.interp(target_x, source_x, array).astype(np.float32)


def _positive_outlier(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = max(mad * 1.4826, 1e-4)
    return np.maximum(0.0, (array - median) / scale).astype(np.float32)


def _classify_severity(
    *,
    sentence_badness: float,
    max_artifact: float,
    clipping_ratio: float,
    thresholds: ThresholdSettings,
) -> str:
    if (
        sentence_badness >= thresholds.severe_badness
        or max_artifact >= thresholds.severe_frame_artifact
        or clipping_ratio >= thresholds.clipping_ratio_severe
    ):
        return "severe"
    if sentence_badness >= thresholds.light_badness or max_artifact >= thresholds.moderate_frame_artifact:
        if max_artifact >= thresholds.moderate_frame_artifact:
            return "moderate"
        return "light"
    return "normal"


__all__ = [
    "AnalysisResult",
    "AnalysisSettings",
    "ThresholdSettings",
    "analyze_audio",
]
