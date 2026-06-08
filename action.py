"""TTS HTTP Server Action 组件。"""

from __future__ import annotations

from typing import Annotated, cast

from src.app.plugin_system.api.send_api import send_voice
from src.app.plugin_system.api.service_api import get_service
from src.core.components.base.action import BaseAction
from src.core.managers import get_chatter_manager

from .protocol import TTSSynthesisRequest
from .service import TTSProviderRegistryService


class GenerateVoiceAction(BaseAction):
    """调用 TTS provider 合成语音并发送到当前会话。"""

    action_name = "generate_voice"
    action_description = (
        "把文本交给 TTS 后端合成为语音并直接发送到当前会话。"
        "适用于普通 chatter 在需要语音表达时调用；"
        "实时语音通话场景由 voice_chatter 自带动作处理，不要在那里调用此动作。"
    )
    associated_types = ["voice"]
    dependencies = ["tts_http_server:service:tts_provider_registry"]

    async def go_activate(self) -> bool:
        """根据配置和 chatter 类型决定是否激活该动作。"""

        plugin_config = getattr(self.plugin, "config", None)
        if plugin_config is not None:
            expose_action = getattr(
                getattr(plugin_config, "action", None),
                "expose_generate_voice_action",
                False,
            )
            if not expose_action:
                return False

        active_chatter = get_chatter_manager().get_chatter_by_stream(
            self.chat_stream.stream_id
        )
        if active_chatter is None:
            return True

        chatter_signature = active_chatter.get_signature()
        if active_chatter.chatter_name == "voice_chatter":
            return False
        return chatter_signature != "voice_chatter:chatter:voice_chatter"

    async def execute(
        self,
        content: Annotated[str, "要合成为语音并发送给当前会话的文本内容"],
        emotion: Annotated[str | None, "可选情绪标签，交由 TTS provider 解释"] = None,
        provider: Annotated[str | None, "可选 provider 名称；为空时使用默认 provider"] = None,
    ) -> tuple[bool, str]:
        """执行语音合成并发送。"""

        text = str(content or "").strip()
        if not text:
            return False, "语音内容不能为空"

        registry = get_service("tts_http_server:service:tts_provider_registry")
        if registry is None:
            return False, "tts_provider_registry service 未加载"

        registry = cast(TTSProviderRegistryService, registry)
        options: dict[str, str] = {}
        if provider:
            options["provider"] = str(provider).strip()

        try:
            result = await registry.synthesize(
                TTSSynthesisRequest(
                    stream_id=self.chat_stream.stream_id,
                    text=text,
                    emotion=emotion,
                    markers={},
                    options=options,
                )
            )
        except LookupError:
            providers = registry.list_providers()
            return False, f"没有可用的 TTS provider: {providers}"
        except Exception as error:
            return False, f"TTS 合成失败: {error}"

        ok = await send_voice(
            voice_data=result.audio_base64,
            stream_id=self.chat_stream.stream_id,
            platform=self.chat_stream.platform,
            processed_plain_text=result.text or text,
        )
        if not ok:
            return False, "语音已合成，但发送失败"

        provider_name = result.provider or options.get("provider") or "default"
        return True, f"已生成并发送语音（provider={provider_name}）"


__all__ = ["GenerateVoiceAction"]
