"""TTS HTTP v1 协议定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


TTS_PROTOCOL_VERSION = "mfx-tts-http-v1"


@dataclass(slots=True)
class TTSSynthesisRequest:
    """TTS 合成请求。"""

    stream_id: str
    text: str
    emotion: str | None = None
    markers: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TTSSynthesisResponse:
    """TTS 合成响应。"""

    audio_base64: str
    mime_type: str = "audio/wav"
    format: str = "wav"
    sample_rate: int | None = None
    duration_ms: int | None = None
    provider: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class TTSProvider(Protocol):
    """具体 TTS 插件需要实现的 provider 协议。"""

    provider_name: str

    async def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResponse:
        """合成语音并返回 base64 音频。"""
        ...

__all__ = [
    "TTS_PROTOCOL_VERSION",
    "TTSProvider",
    "TTSSynthesisRequest",
    "TTSSynthesisResponse",
]
