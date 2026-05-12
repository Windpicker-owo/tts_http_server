"""TTS provider 注册服务。"""

from __future__ import annotations

from typing import Any

from src.core.components.base.service import BaseService

from .protocol import TTSProvider


_PROVIDERS: dict[str, TTSProvider] = {}
_DEFAULT_PROVIDER: str | None = None


class TTSProviderRegistryService(BaseService):
    """供具体 TTS 插件注册 provider 的服务。"""

    service_name = "tts_provider_registry"
    service_description = "TTS HTTP server provider registry"
    version = "1.0.0"

    def register_provider(self, provider: TTSProvider, *, default: bool = False) -> None:
        """注册一个 TTS provider。"""

        global _DEFAULT_PROVIDER
        provider_name = str(getattr(provider, "provider_name", "") or "").strip()
        if not provider_name:
            raise ValueError("TTS provider 必须声明 provider_name")
        _PROVIDERS[provider_name] = provider
        if default or _DEFAULT_PROVIDER is None:
            _DEFAULT_PROVIDER = provider_name

    def unregister_provider(self, provider_name: str) -> bool:
        """注销 provider。"""

        global _DEFAULT_PROVIDER
        removed = _PROVIDERS.pop(provider_name, None) is not None
        if _DEFAULT_PROVIDER == provider_name:
            _DEFAULT_PROVIDER = next(iter(_PROVIDERS), None)
        return removed

    def get_provider(self, provider_name: str | None = None) -> TTSProvider | None:
        """获取指定或默认 provider。"""

        name = provider_name or _DEFAULT_PROVIDER
        if not name:
            return None
        return _PROVIDERS.get(name)

    def list_providers(self) -> dict[str, Any]:
        """列出 provider 状态。"""

        return {
            "default_provider": _DEFAULT_PROVIDER,
            "providers": sorted(_PROVIDERS),
        }


__all__ = ["TTSProviderRegistryService"]
