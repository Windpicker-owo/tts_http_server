"""TTS HTTP 协议服务器插件。"""

from __future__ import annotations

from src.core.components.base.plugin import BasePlugin
from src.core.components.loader import register_plugin

from .action import GenerateVoiceAction
from .config import TTSHttpServerConfig
from .router import TTSHttpServerRouter
from .service import TTSProviderRegistryService


@register_plugin
class TTSHttpServerPlugin(BasePlugin):
    """只提供 HTTP Router 的 TTS 协议服务器插件。"""

    plugin_name = "tts_http_server"
    plugin_version = "1.0.0"
    plugin_description = "TTS HTTP 协议服务器"
    configs = [TTSHttpServerConfig]

    def get_components(self) -> list[type]:
        """返回插件组件。"""

        return [GenerateVoiceAction, TTSHttpServerRouter, TTSProviderRegistryService]


__all__ = ["TTSHttpServerPlugin"]
