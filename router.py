"""TTS HTTP v1 Router。"""

from __future__ import annotations

from typing import Any, cast

from fastapi import HTTPException

from src.core.components.base.router import BaseRouter

from .protocol import TTS_PROTOCOL_VERSION, TTSSynthesisRequest
from .service import TTSProviderRegistryService


class TTSHttpServerRouter(BaseRouter):
    """TTS HTTP 协议路由。"""

    router_name = "tts_http_server"
    router_description = "MoFox TTS HTTP v1 protocol router"
    custom_route_path = "/router/tts_http_server"

    def __init__(self, plugin: Any) -> None:
        """初始化路由。"""

        self._registry: TTSProviderRegistryService | None = None
        super().__init__(plugin)

    def _get_registry(self) -> TTSProviderRegistryService:
        """获取 provider registry 服务。"""

        if self._registry is None:
            self._registry = TTSProviderRegistryService(plugin=self.plugin)
        return self._registry

    def register_endpoints(self) -> None:
        """注册 TTS HTTP 端点。"""

        @self.app.get("/api/tts/v1/status")
        async def status() -> dict[str, Any]:
            registry_status = self._get_registry().list_providers()
            return {
                "protocol": TTS_PROTOCOL_VERSION,
                **registry_status,
            }

        @self.app.post("/api/tts/v1/synthesize")
        async def synthesize(payload: dict[str, Any]) -> dict[str, Any]:
            if payload.get("protocol") != TTS_PROTOCOL_VERSION:
                raise HTTPException(status_code=400, detail="unsupported TTS protocol")

            text = str(payload.get("text") or "").strip()
            if not text:
                raise HTTPException(status_code=400, detail="text is required")

            raw_options = payload.get("options")
            options = cast(dict[str, Any], raw_options) if isinstance(raw_options, dict) else {}
            provider_name = str(options.get("provider") or "").strip() or None
            registry = self._get_registry()
            provider = registry.get_provider(provider_name)
            if provider is None:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "message": "no TTS provider registered",
                        "requested_provider": provider_name,
                        **registry.list_providers(),
                    },
                )

            raw_markers = payload.get("markers")
            markers = cast(dict[str, Any], raw_markers) if isinstance(raw_markers, dict) else {}
            request = TTSSynthesisRequest(
                stream_id=str(payload.get("stream_id") or ""),
                text=text,
                emotion=payload.get("emotion") if isinstance(payload.get("emotion"), str) else None,
                markers=markers,
                options=options,
            )
            try:
                result = await registry.synthesize(request)
            except Exception as error:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "message": "TTS provider synthesis failed",
                        "provider": getattr(provider, "provider_name", provider_name or ""),
                        "error": str(error),
                    },
                ) from error
            return {
                "protocol": TTS_PROTOCOL_VERSION,
                "audio_base64": result.audio_base64,
                "mime_type": result.mime_type,
                "format": result.format,
                "sample_rate": result.sample_rate,
                "duration_ms": result.duration_ms,
                "provider": result.provider or getattr(provider, "provider_name", ""),
                "text": result.text or text,
                "metadata": result.metadata,
            }


__all__ = ["TTSHttpServerRouter"]
