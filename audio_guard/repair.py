"""Repair processors for the TTS audio guard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .analysis import AnalysisResult, AnalysisSettings, ThresholdSettings


@dataclass(slots=True)
class RepairSettings:
    enable_declick: bool = True
    enable_spectral_repair: bool = True
    enable_transient_suppressor: bool = True
    declick_max_duration_ms: float = 30.0
    declick_context_ms: float = 4.0
    spectral_freq_min_hz: float = 4500.0
    spectral_threshold_ratio: float = 3.0
    spectral_replace_strength: float = 0.6
    transient_window_ms: float = 4.0
    transient_threshold: float = 3.2
    transient_max_reduction_db: float = 9.0


def repair_clicks(
    samples: np.ndarray,
    sample_rate: int,
    settings: RepairSettings,
) -> tuple[np.ndarray, int]:
    """Repair short impulse-like clicks with local interpolation."""

    if not settings.enable_declick or len(samples) < 5:
        return np.asarray(samples, dtype=np.float32), 0

    mono = np.asarray(samples, dtype=np.float32).copy()
    residual = np.abs(mono[1:-1] - 0.5 * (mono[:-2] + mono[2:]))
    if residual.size == 0:
        return mono, 0
    threshold = max(float(np.percentile(residual, 99.5)) * 1.5, 0.12)
    spike_indices = np.where(residual > threshold)[0] + 1
    if spike_indices.size == 0:
        return mono, 0

    groups = _group_indices(spike_indices.tolist())
    max_len = max(1, int(sample_rate * settings.declick_max_duration_ms / 1000.0))
    context = max(1, int(sample_rate * settings.declick_context_ms / 1000.0))
    repaired = 0
    for start, end in groups:
        if end - start + 1 > max_len:
            continue
        left = max(0, start - context)
        right = min(len(mono) - 1, end + context)
        if right - left < 2:
            continue
        segment_len = right - left + 1
        replacement = np.interp(
            np.arange(segment_len),
            [0, segment_len - 1],
            [mono[left], mono[right]],
        ).astype(np.float32)
        fade = np.linspace(0.0, 1.0, num=segment_len, dtype=np.float32)
        mono[left:right + 1] = mono[left:right + 1] * (1.0 - fade) + replacement * fade
        repaired += segment_len
    return mono, repaired


def spectral_repair(
    samples: np.ndarray,
    sample_rate: int,
    analysis: AnalysisResult,
    analysis_settings: AnalysisSettings,
    threshold_settings: ThresholdSettings,
    repair_settings: RepairSettings,
) -> tuple[np.ndarray, int]:
    """Suppress harsh high-frequency bursts on flagged frames."""

    if not repair_settings.enable_spectral_repair or analysis.repair_frame_mask.size == 0:
        return np.asarray(samples, dtype=np.float32), 0

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
    flagged_frames = np.where(_align_mask(analysis.repair_frame_mask, magnitude.shape[1]))[0]
    freq_mask = freqs >= repair_settings.spectral_freq_min_hz
    if flagged_frames.size == 0 or not np.any(freq_mask):
        return mono, 0

    replacements = 0
    updated_mag = magnitude.copy()
    threshold = max(repair_settings.spectral_threshold_ratio, 1.1)
    blend = float(np.clip(repair_settings.spectral_replace_strength, 0.0, 1.0))
    for frame_index in flagged_frames:
        left = max(0, frame_index - 2)
        right = min(magnitude.shape[1], frame_index + 3)
        neighborhood = magnitude[freq_mask, left:right]
        median_mag = np.median(neighborhood, axis=1)
        current = updated_mag[freq_mask, frame_index]
        burst_mask = current > np.maximum(median_mag * threshold, 1e-6)
        if not np.any(burst_mask):
            continue
        current[burst_mask] = (
            (1.0 - blend) * current[burst_mask] + blend * median_mag[burst_mask]
        )
        updated_mag[freq_mask, frame_index] = current
        replacements += int(np.sum(burst_mask))

    repaired = updated_mag * np.exp(1j * phase)
    _, restored = signal.istft(
        repaired,
        fs=sample_rate,
        window="hann",
        nperseg=stft_window,
        noverlap=max(0, stft_window - stft_hop),
        input_onesided=True,
    )
    return restored[: len(mono)].astype(np.float32), replacements


def transient_suppress(
    samples: np.ndarray,
    sample_rate: int,
    settings: RepairSettings,
) -> tuple[np.ndarray, float]:
    """Reduce short transients that poke above the local RMS floor."""

    if not settings.enable_transient_suppressor or len(samples) < 4:
        return np.asarray(samples, dtype=np.float32), 0.0

    mono = np.asarray(samples, dtype=np.float32)
    window = max(2, int(sample_rate * settings.transient_window_ms / 1000.0))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    local_rms = np.sqrt(np.convolve(np.square(mono), kernel, mode="same") + 1e-8)
    ratio = np.abs(mono) / np.maximum(local_rms, 1e-4)
    if np.max(ratio) <= settings.transient_threshold:
        return mono, 0.0

    max_reduction_gain = 10.0 ** (-float(settings.transient_max_reduction_db) / 20.0)
    target_gain = np.minimum(1.0, settings.transient_threshold / np.maximum(ratio, 1e-4))
    target_gain = np.maximum(target_gain, max_reduction_gain).astype(np.float32)
    smoothed_gain = np.convolve(target_gain, kernel, mode="same")
    processed = mono * smoothed_gain
    reduction_db = -20.0 * float(np.log10(np.clip(np.min(smoothed_gain), 1e-6, 1.0)))
    return processed.astype(np.float32), reduction_db


def _group_indices(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    groups: list[tuple[int, int]] = []
    start = end = indices[0]
    for index in indices[1:]:
        if index == end + 1:
            end = index
            continue
        groups.append((start, end))
        start = end = index
    groups.append((start, end))
    return groups


def _align_mask(mask: np.ndarray, target_length: int) -> np.ndarray:
    source = np.asarray(mask, dtype=np.float32).reshape(-1)
    if target_length <= 0:
        return np.zeros(0, dtype=bool)
    if source.size == 0:
        return np.zeros(target_length, dtype=bool)
    if source.size == target_length:
        return source.astype(bool)
    if source.size == 1:
        return np.full(target_length, bool(source[0]), dtype=bool)
    source_x = np.linspace(0.0, 1.0, num=source.size)
    target_x = np.linspace(0.0, 1.0, num=target_length)
    return (np.interp(target_x, source_x, source) >= 0.5).astype(bool)


__all__ = ["RepairSettings", "repair_clicks", "spectral_repair", "transient_suppress"]
