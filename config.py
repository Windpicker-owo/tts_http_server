"""TTS HTTP 协议服务器插件配置。"""

from __future__ import annotations

from typing import ClassVar

from src.core.components.base.config import BaseConfig, Field, SectionBase, config_section


class TTSHttpServerConfig(BaseConfig):
    """TTS HTTP 协议服务器插件配置。"""

    config_name: ClassVar[str] = "config"
    config_description: ClassVar[str] = "TTS HTTP 协议服务器配置"

    @config_section("action", title="动作设置", tag="plugin")
    class ActionSection(SectionBase):
        """动作暴露配置。"""

        expose_generate_voice_action: bool = Field(
            default=False,
            description="是否注册 generate_voice action；开启后普通 chatter 可直接合成并发送语音",
        )

    action: ActionSection = Field(default_factory=ActionSection)

    @config_section("audio_guard", title="Audio Guard", tag="audio")
    class AudioGuardSection(SectionBase):
        """Top-level switch for post-processing."""

        enabled: bool = Field(
            default=True,
            description="Whether to run the built-in sentence-level TTS audio guard.",
        )

    @config_section("audio_guard_analysis", title="Audio Guard Analysis", tag="audio")
    class AudioGuardAnalysisSection(SectionBase):
        """Analysis window and feature settings."""

        frame_ms: float = Field(default=20.0, description="Waveform analysis frame size in ms.")
        hop_ms: float = Field(default=10.0, description="Waveform analysis hop size in ms.")
        stft_window_ms: float = Field(default=32.0, description="STFT window size in ms.")
        stft_hop_ms: float = Field(default=8.0, description="STFT hop size in ms.")
        clip_threshold: float = Field(
            default=0.985,
            description="Absolute sample threshold used to estimate clipping ratio.",
        )
        high_band_min_hz: float = Field(default=6000.0, description="High-band energy lower bound.")
        high_band_max_hz: float = Field(default=12000.0, description="High-band energy upper bound.")
        mid_band_min_hz: float = Field(default=300.0, description="Reference mid-band lower bound.")
        mid_band_max_hz: float = Field(default=4000.0, description="Reference mid-band upper bound.")

    @config_section("audio_guard_thresholds", title="Audio Guard Thresholds", tag="audio")
    class AudioGuardThresholdsSection(SectionBase):
        """Quality score thresholds."""

        light_badness: float = Field(default=8.0, description="Light sentence badness threshold.")
        severe_badness: float = Field(default=14.0, description="Severe sentence badness threshold.")
        moderate_frame_artifact: float = Field(
            default=6.0,
            description="Per-frame artifact score that unlocks repair processors.",
        )
        severe_frame_artifact: float = Field(
            default=10.0,
            description="Per-frame artifact score that marks a sentence as severe.",
        )
        clipping_ratio_severe: float = Field(
            default=0.01,
            description="Sentence clipping ratio threshold that forces severe classification.",
        )

    @config_section("audio_guard_repair", title="Audio Guard Repair", tag="audio")
    class AudioGuardRepairSection(SectionBase):
        """Repair processor settings."""

        enable_declick: bool = Field(default=True, description="Enable click/pop interpolation repair.")
        enable_spectral_repair: bool = Field(
            default=True,
            description="Enable high-frequency spectral repair on bad frames.",
        )
        enable_transient_suppressor: bool = Field(
            default=True,
            description="Enable transient suppression for short harsh peaks.",
        )
        declick_max_duration_ms: float = Field(
            default=30.0,
            description="Maximum click duration eligible for interpolation repair.",
        )
        declick_context_ms: float = Field(
            default=4.0,
            description="Context range expanded around click regions before interpolation.",
        )
        spectral_freq_min_hz: float = Field(
            default=4500.0,
            description="Only bins above this frequency are eligible for spectral repair.",
        )
        spectral_threshold_ratio: float = Field(
            default=3.0,
            description="Burst threshold relative to neighborhood median magnitude.",
        )
        spectral_replace_strength: float = Field(
            default=0.6,
            description="Blend strength used when replacing harsh spectral bins.",
        )
        transient_window_ms: float = Field(
            default=4.0,
            description="Local RMS window for transient detection.",
        )
        transient_threshold: float = Field(
            default=3.2,
            description="Peak-to-local-RMS ratio above which transient suppression starts.",
        )
        transient_max_reduction_db: float = Field(
            default=9.0,
            description="Maximum attenuation applied by the transient suppressor.",
        )

    @config_section("audio_guard_tone", title="Audio Guard Tone", tag="audio")
    class AudioGuardToneSection(SectionBase):
        """Tone-safety processor settings."""

        enable_deesser: bool = Field(default=True, description="Enable dynamic de-essing.")
        enable_dynamic_eq: bool = Field(default=True, description="Enable dynamic presence/sibilance EQ.")
        deesser_freq_low_hz: float = Field(default=5500.0, description="De-esser band lower bound.")
        deesser_freq_high_hz: float = Field(default=9000.0, description="De-esser band upper bound.")
        deesser_max_reduction_db: float = Field(
            default=6.0,
            description="Maximum de-esser attenuation.",
        )
        presence_freq_low_hz: float = Field(
            default=2800.0,
            description="Dynamic EQ presence band lower bound.",
        )
        presence_freq_high_hz: float = Field(
            default=4500.0,
            description="Dynamic EQ presence band upper bound.",
        )
        presence_max_reduction_db: float = Field(
            default=3.0,
            description="Maximum presence-band attenuation.",
        )
        sibilance_freq_low_hz: float = Field(
            default=5500.0,
            description="Dynamic EQ sibilance band lower bound.",
        )
        sibilance_freq_high_hz: float = Field(
            default=9000.0,
            description="Dynamic EQ sibilance band upper bound.",
        )
        sibilance_max_reduction_db: float = Field(
            default=4.5,
            description="Maximum sibilance-band attenuation.",
        )

    @config_section("audio_guard_limiter", title="Audio Guard Limiter", tag="audio")
    class AudioGuardLimiterSection(SectionBase):
        """Output safety limiter settings."""

        enable_limiter: bool = Field(default=True, description="Enable the look-ahead limiter.")
        lookahead_ms: float = Field(default=8.0, description="Limiter look-ahead in ms.")
        ceiling_db: float = Field(default=-1.0, description="Limiter output ceiling in dBFS.")
        release_ms: float = Field(default=100.0, description="Limiter release in ms.")
        peak_normalize_max_gain_db: float = Field(
            default=1.5,
            description="Maximum makeup gain allowed during final peak normalization.",
        )

    @config_section("audio_guard_retry", title="Audio Guard Retry", tag="audio")
    class AudioGuardRetrySection(SectionBase):
        """Retry settings for severe sentences."""

        enabled: bool = Field(
            default=True,
            description="Whether severe audio may trigger one extra provider synthesis attempt.",
        )
        max_retry: int = Field(default=1, description="Maximum retry attempts for severe audio.")

    audio_guard: AudioGuardSection = Field(default_factory=AudioGuardSection)
    audio_guard_analysis: AudioGuardAnalysisSection = Field(
        default_factory=AudioGuardAnalysisSection
    )
    audio_guard_thresholds: AudioGuardThresholdsSection = Field(
        default_factory=AudioGuardThresholdsSection
    )
    audio_guard_repair: AudioGuardRepairSection = Field(default_factory=AudioGuardRepairSection)
    audio_guard_tone: AudioGuardToneSection = Field(default_factory=AudioGuardToneSection)
    audio_guard_limiter: AudioGuardLimiterSection = Field(default_factory=AudioGuardLimiterSection)
    audio_guard_retry: AudioGuardRetrySection = Field(default_factory=AudioGuardRetrySection)


__all__ = ["TTSHttpServerConfig"]
