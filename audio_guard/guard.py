"""High-level sentence guard for provider TTS output."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any

import numpy as np

from plugins.tts_http_server.config import TTSHttpServerConfig
from plugins.tts_http_server.protocol import TTSSynthesisResponse

from .analysis import AnalysisResult, AnalysisSettings, ThresholdSettings, analyze_audio
from .codec import AudioDecodeError, decode_wav_base64, duration_ms, encode_wav_base64, peak_dbfs
from .repair import RepairSettings, repair_clicks, spectral_repair, transient_suppress
from .tone import (
    LimiterSettings,
    ToneSettings,
    dynamic_deesser,
    dynamic_eq,
    lookahead_limiter,
    peak_normalize,
)


@dataclass(slots=True)
class RetrySettings:
    enabled: bool = True
    max_retry: int = 1


@dataclass(slots=True)
class AudioGuardConfig:
    enabled: bool = True
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    thresholds: ThresholdSettings = field(default_factory=ThresholdSettings)
    repair: RepairSettings = field(default_factory=RepairSettings)
    tone: ToneSettings = field(default_factory=ToneSettings)
    limiter: LimiterSettings = field(default_factory=LimiterSettings)
    retry: RetrySettings = field(default_factory=RetrySettings)

    @classmethod
    def from_plugin_config(cls, plugin_config: TTSHttpServerConfig | None) -> "AudioGuardConfig":
        config = cls()
        if not isinstance(plugin_config, TTSHttpServerConfig):
            return config

        config.enabled = bool(plugin_config.audio_guard.enabled)
        config.analysis = AnalysisSettings(
            frame_ms=float(plugin_config.audio_guard_analysis.frame_ms),
            hop_ms=float(plugin_config.audio_guard_analysis.hop_ms),
            stft_window_ms=float(plugin_config.audio_guard_analysis.stft_window_ms),
            stft_hop_ms=float(plugin_config.audio_guard_analysis.stft_hop_ms),
            clip_threshold=float(plugin_config.audio_guard_analysis.clip_threshold),
            high_band_min_hz=float(plugin_config.audio_guard_analysis.high_band_min_hz),
            high_band_max_hz=float(plugin_config.audio_guard_analysis.high_band_max_hz),
            mid_band_min_hz=float(plugin_config.audio_guard_analysis.mid_band_min_hz),
            mid_band_max_hz=float(plugin_config.audio_guard_analysis.mid_band_max_hz),
        )
        config.thresholds = ThresholdSettings(
            light_badness=float(plugin_config.audio_guard_thresholds.light_badness),
            severe_badness=float(plugin_config.audio_guard_thresholds.severe_badness),
            moderate_frame_artifact=float(
                plugin_config.audio_guard_thresholds.moderate_frame_artifact
            ),
            severe_frame_artifact=float(plugin_config.audio_guard_thresholds.severe_frame_artifact),
            clipping_ratio_severe=float(plugin_config.audio_guard_thresholds.clipping_ratio_severe),
        )
        config.repair = RepairSettings(
            enable_declick=bool(plugin_config.audio_guard_repair.enable_declick),
            enable_spectral_repair=bool(plugin_config.audio_guard_repair.enable_spectral_repair),
            enable_transient_suppressor=bool(
                plugin_config.audio_guard_repair.enable_transient_suppressor
            ),
            declick_max_duration_ms=float(
                plugin_config.audio_guard_repair.declick_max_duration_ms
            ),
            declick_context_ms=float(plugin_config.audio_guard_repair.declick_context_ms),
            spectral_freq_min_hz=float(plugin_config.audio_guard_repair.spectral_freq_min_hz),
            spectral_threshold_ratio=float(
                plugin_config.audio_guard_repair.spectral_threshold_ratio
            ),
            spectral_replace_strength=float(
                plugin_config.audio_guard_repair.spectral_replace_strength
            ),
            transient_window_ms=float(plugin_config.audio_guard_repair.transient_window_ms),
            transient_threshold=float(plugin_config.audio_guard_repair.transient_threshold),
            transient_max_reduction_db=float(
                plugin_config.audio_guard_repair.transient_max_reduction_db
            ),
        )
        config.tone = ToneSettings(
            enable_deesser=bool(plugin_config.audio_guard_tone.enable_deesser),
            enable_dynamic_eq=bool(plugin_config.audio_guard_tone.enable_dynamic_eq),
            deesser_freq_low_hz=float(plugin_config.audio_guard_tone.deesser_freq_low_hz),
            deesser_freq_high_hz=float(plugin_config.audio_guard_tone.deesser_freq_high_hz),
            deesser_max_reduction_db=float(
                plugin_config.audio_guard_tone.deesser_max_reduction_db
            ),
            presence_freq_low_hz=float(plugin_config.audio_guard_tone.presence_freq_low_hz),
            presence_freq_high_hz=float(plugin_config.audio_guard_tone.presence_freq_high_hz),
            presence_max_reduction_db=float(
                plugin_config.audio_guard_tone.presence_max_reduction_db
            ),
            sibilance_freq_low_hz=float(plugin_config.audio_guard_tone.sibilance_freq_low_hz),
            sibilance_freq_high_hz=float(plugin_config.audio_guard_tone.sibilance_freq_high_hz),
            sibilance_max_reduction_db=float(
                plugin_config.audio_guard_tone.sibilance_max_reduction_db
            ),
        )
        config.limiter = LimiterSettings(
            enable_limiter=bool(plugin_config.audio_guard_limiter.enable_limiter),
            lookahead_ms=float(plugin_config.audio_guard_limiter.lookahead_ms),
            ceiling_db=float(plugin_config.audio_guard_limiter.ceiling_db),
            release_ms=float(plugin_config.audio_guard_limiter.release_ms),
            peak_normalize_max_gain_db=float(
                plugin_config.audio_guard_limiter.peak_normalize_max_gain_db
            ),
        )
        config.retry = RetrySettings(
            enabled=bool(plugin_config.audio_guard_retry.enabled),
            max_retry=int(plugin_config.audio_guard_retry.max_retry),
        )
        return config


@dataclass(slots=True)
class AudioGuardCandidate:
    response: TTSSynthesisResponse
    analysis: AnalysisResult | None
    changed: bool
    should_retry: bool
    skipped_reason: str | None
    processing_time_ms: float

    @property
    def score(self) -> float:
        if self.analysis is None:
            return float("inf")
        return float(self.analysis.sentence_badness)


class TTSAudioGuard:
    """Apply rule-based DSP repair and safety limiting to TTS responses."""

    def __init__(self, config: AudioGuardConfig) -> None:
        self.config = config

    def process_response(
        self,
        response: TTSSynthesisResponse,
        *,
        retry_count: int = 0,
    ) -> AudioGuardCandidate:
        metadata = dict(response.metadata or {})
        if not self.config.enabled:
            metadata["audio_guard"] = {
                "enabled": False,
                "changed": False,
                "retry_count": retry_count,
                "processors_applied": [],
            }
            return AudioGuardCandidate(
                response=replace(response, metadata=metadata),
                analysis=None,
                changed=False,
                should_retry=False,
                skipped_reason=None,
                processing_time_ms=0.0,
            )

        started = time.perf_counter()
        try:
            decoded = decode_wav_base64(response.audio_base64)
        except AudioDecodeError as exc:
            metadata["audio_guard"] = {
                "enabled": True,
                "changed": False,
                "skipped_reason": str(exc),
                "retry_count": retry_count,
                "processors_applied": [],
            }
            return AudioGuardCandidate(
                response=replace(response, metadata=metadata),
                analysis=None,
                changed=False,
                should_retry=False,
                skipped_reason=str(exc),
                processing_time_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            metadata["audio_guard"] = {
                "enabled": True,
                "changed": False,
                "skipped_reason": "guard_error",
                "error": str(exc),
                "retry_count": retry_count,
                "processors_applied": [],
            }
            return AudioGuardCandidate(
                response=replace(response, metadata=metadata),
                analysis=None,
                changed=False,
                should_retry=False,
                skipped_reason="guard_error",
                processing_time_ms=(time.perf_counter() - started) * 1000.0,
            )

        original = decoded.samples
        working = _remove_dc_offset(original)
        processors_applied: list[str] = []
        metrics = {
            "deesser_gain_reduction_db": 0.0,
            "eq_gain_reduction_db": 0.0,
            "limiter_gain_reduction_db": 0.0,
        }

        analysis = analyze_audio(working, decoded.sample_rate, self.config.analysis, self.config.thresholds)
        severity = analysis.severity

        if severity in {"moderate", "severe"}:
            repaired, repaired_samples = repair_clicks(working, decoded.sample_rate, self.config.repair)
            if repaired_samples > 0:
                working = repaired
                processors_applied.append("declick_depop")

            repaired, repaired_bins = spectral_repair(
                working,
                decoded.sample_rate,
                analysis,
                self.config.analysis,
                self.config.thresholds,
                self.config.repair,
            )
            if repaired_bins > 0:
                working = repaired
                processors_applied.append("spectral_repair")

        if severity in {"light", "moderate", "severe"}:
            processed, deesser_reduction = dynamic_deesser(
                working,
                decoded.sample_rate,
                self.config.analysis,
                self.config.tone,
            )
            if deesser_reduction > 0.05:
                working = processed
                metrics["deesser_gain_reduction_db"] = deesser_reduction
                processors_applied.append("dynamic_deesser")

            processed, eq_reduction = dynamic_eq(
                working,
                decoded.sample_rate,
                self.config.analysis,
                self.config.tone,
            )
            if eq_reduction > 0.05:
                working = processed
                metrics["eq_gain_reduction_db"] = eq_reduction
                processors_applied.append("dynamic_eq")

        if severity in {"moderate", "severe"}:
            processed, transient_reduction = transient_suppress(
                working,
                decoded.sample_rate,
                self.config.repair,
            )
            if transient_reduction > 0.05:
                working = processed
                processors_applied.append("transient_suppressor")

        limited, limiter_reduction = lookahead_limiter(working, decoded.sample_rate, self.config.limiter)
        if limiter_reduction > 0.01:
            processors_applied.append("lookahead_limiter")
        metrics["limiter_gain_reduction_db"] = limiter_reduction
        working = limited

        normalized = peak_normalize(working, self.config.limiter)
        if not np.allclose(normalized, working, atol=1e-5):
            processors_applied.append("peak_normalize")
            working = normalized

        changed = bool(processors_applied) or not np.allclose(working, original, atol=1e-4)
        processing_time_ms = (time.perf_counter() - started) * 1000.0
        audio_guard_metadata = {
            "enabled": True,
            "changed": changed,
            "skipped_reason": None,
            "input_sample_rate": decoded.sample_rate,
            "output_sample_rate": decoded.sample_rate,
            "sentence_badness": analysis.sentence_badness,
            "max_artifact_score": analysis.max_artifact_score,
            "p95_artifact_score": analysis.p95_artifact_score,
            "severity": severity,
            "processors_applied": processors_applied,
            "retry_count": retry_count,
            "processing_time_ms": processing_time_ms,
            "limiter_gain_reduction_db": metrics["limiter_gain_reduction_db"],
            "deesser_gain_reduction_db": metrics["deesser_gain_reduction_db"],
            "eq_gain_reduction_db": metrics["eq_gain_reduction_db"],
            "clipping_ratio": analysis.clipping_ratio,
            "output_peak_db": peak_dbfs(working),
        }
        metadata["audio_guard"] = audio_guard_metadata

        if not changed:
            return AudioGuardCandidate(
                response=replace(response, metadata=metadata),
                analysis=analysis,
                changed=False,
                should_retry=self._should_retry(analysis),
                skipped_reason=None,
                processing_time_ms=processing_time_ms,
            )

        updated_response = replace(
            response,
            audio_base64=encode_wav_base64(working, decoded.sample_rate),
            mime_type="audio/wav",
            format="wav",
            sample_rate=decoded.sample_rate,
            duration_ms=response.duration_ms or duration_ms(working, decoded.sample_rate),
            metadata=metadata,
        )
        return AudioGuardCandidate(
            response=updated_response,
            analysis=analysis,
            changed=True,
            should_retry=self._should_retry(analysis),
            skipped_reason=None,
            processing_time_ms=processing_time_ms,
        )

    def _should_retry(self, analysis: AnalysisResult) -> bool:
        return self.config.retry.enabled and analysis.severity == "severe"


def _remove_dc_offset(samples: np.ndarray) -> np.ndarray:
    mono = np.asarray(samples, dtype=np.float32)
    return np.clip(mono - float(np.mean(mono)), -1.0, 1.0).astype(np.float32)


__all__ = ["AudioGuardCandidate", "AudioGuardConfig", "RetrySettings", "TTSAudioGuard"]
