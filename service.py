"""TTS provider registry service."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.core.components.base.service import BaseService

from .audio_guard import AudioGuardCandidate, AudioGuardConfig, TTSAudioGuard
from .config import TTSHttpServerConfig
from .protocol import TTSProvider, TTSSynthesisRequest, TTSSynthesisResponse


_PROVIDERS: dict[str, TTSProvider] = {}
_DEFAULT_PROVIDER: str | None = None


class TTSProviderRegistryService(BaseService):
    """Registry and orchestration service for TTS providers."""

    service_name = "tts_provider_registry"
    service_description = "TTS HTTP server provider registry"
    version = "1.0.0"

    def __init__(self, plugin: Any) -> None:
        super().__init__(plugin)
        plugin_config = self.plugin.config if isinstance(getattr(self.plugin, "config", None), TTSHttpServerConfig) else None
        self._audio_guard = TTSAudioGuard(AudioGuardConfig.from_plugin_config(plugin_config))

    def register_provider(self, provider: TTSProvider, *, default: bool = False) -> None:
        """Register a provider implementation."""

        global _DEFAULT_PROVIDER
        provider_name = str(getattr(provider, "provider_name", "") or "").strip()
        if not provider_name:
            raise ValueError("TTS provider must declare provider_name")
        _PROVIDERS[provider_name] = provider
        if default or _DEFAULT_PROVIDER is None:
            _DEFAULT_PROVIDER = provider_name

    def unregister_provider(self, provider_name: str) -> bool:
        """Unregister a provider implementation."""

        global _DEFAULT_PROVIDER
        removed = _PROVIDERS.pop(provider_name, None) is not None
        if _DEFAULT_PROVIDER == provider_name:
            _DEFAULT_PROVIDER = next(iter(_PROVIDERS), None)
        return removed

    def get_provider(self, provider_name: str | None = None) -> TTSProvider | None:
        """Return the requested or default provider."""

        name = provider_name or _DEFAULT_PROVIDER
        if not name:
            return None
        return _PROVIDERS.get(name)

    async def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResponse:
        """Run provider synthesis and apply the built-in audio guard."""

        provider_name = str(request.options.get("provider") or "").strip() or None
        provider = self.get_provider(provider_name)
        if provider is None:
            raise LookupError("no TTS provider registered")

        initial_response = await provider.synthesize(request)
        return await self._synthesize_with_audio_guard(provider, request, initial_response)

    def list_providers(self) -> dict[str, Any]:
        """List registered providers."""

        return {
            "default_provider": _DEFAULT_PROVIDER,
            "providers": sorted(_PROVIDERS),
        }

    async def _synthesize_with_audio_guard(
        self,
        provider: TTSProvider,
        request: TTSSynthesisRequest,
        initial_response: TTSSynthesisResponse,
    ) -> TTSSynthesisResponse:
        primary = self._audio_guard.process_response(initial_response, retry_count=0)
        retry_attempts = 0
        best = primary

        if primary.should_retry:
            for attempt in range(1, max(0, self._audio_guard.config.retry.max_retry) + 1):
                retry_attempts = attempt
                retry_request = self._build_retry_request(request, attempt=attempt)
                try:
                    retry_response = await provider.synthesize(retry_request)
                except Exception:  # noqa: BLE001
                    break
                retry_candidate = self._audio_guard.process_response(
                    retry_response,
                    retry_count=attempt,
                )
                if retry_candidate.score < best.score:
                    best = retry_candidate

        return self._with_retry_count(best, retry_attempts)

    @staticmethod
    def _build_retry_request(
        request: TTSSynthesisRequest,
        *,
        attempt: int,
    ) -> TTSSynthesisRequest:
        retry_options = dict(request.options or {})
        retry_options["audio_guard_retry_attempt"] = attempt
        retry_options["audio_guard_retry_reason"] = "severe_artifact"
        return replace(request, options=retry_options)

    @staticmethod
    def _with_retry_count(candidate: AudioGuardCandidate, retry_count: int) -> TTSSynthesisResponse:
        metadata = dict(candidate.response.metadata or {})
        audio_guard_meta = dict(metadata.get("audio_guard") or {})
        if audio_guard_meta:
            audio_guard_meta["retry_count"] = retry_count
        else:
            audio_guard_meta = {
                "enabled": False,
                "changed": False,
                "retry_count": retry_count,
                "processors_applied": [],
            }
        metadata["audio_guard"] = audio_guard_meta
        return replace(candidate.response, metadata=metadata)


__all__ = ["TTSProviderRegistryService"]
