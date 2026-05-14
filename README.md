# tts_http_server

## 概述

`tts_http_server` 是 Neo-MoFox 的 TTS HTTP 协议入口。

它负责暴露统一的 HTTP Router、TTS provider registry，以及供其他 chatter 调用的 `generate_voice` action，本身不实现具体的 TTS 合成逻辑。真正的语音合成由外部 provider 插件注册到 registry 后提供，例如 `qwen_tts_provider`。

## 提供的组件

- `tts_http_server:router:tts_http_server`
- `tts_http_server:service:tts_provider_registry`
- `tts_http_server:action:generate_voice`

其中：

- router 暴露统一 HTTP 协议
- registry service 管理可用 TTS provider 及默认 provider
- generate_voice action 供普通 chatter 直接合成并发送语音，且不会在 `voice_chatter` 中激活

## HTTP 接口

Router 基础路径：

- `/router/tts_http_server`

当前公开接口：

- `GET /router/tts_http_server/api/tts/v1/status`
- `POST /router/tts_http_server/api/tts/v1/synthesize`

`status` 用于查看协议版本、默认 provider 和已注册 provider 列表。

`synthesize` 接收统一的 TTS 请求负载，并将请求分发给 registry 中选中的 provider。返回值中包含：

- `audio_base64`
- `mime_type`
- `format`
- `sample_rate`
- `duration_ms`
- `provider`
- `text`
- `metadata`

## 依赖

该插件没有单独的配置类，也没有默认配置文件；它依赖外部 provider 注册后才能真正提供合成能力。

如果当前没有任何 provider 注册到 `tts_provider_registry`，`synthesize` 接口会返回 503。

## 配置

当前插件目录下没有独立 `config.py`，也没有默认的 `config/plugins/tts_http_server/config.toml`。

它的职责刻意保持很窄：

- 暴露协议
- 校验请求
- 选择 provider
- 返回统一响应格式

真正与模型、音色、存储、设备相关的配置应放在具体 provider 插件中。

## 典型联动

1. `qwen_tts_provider` 等插件在加载时向 `tts_provider_registry` 注册 provider。
2. `voice_chatter` 之类的调用方通过 HTTP 协议向 `/api/tts/v1/synthesize` 发请求。
3. Router 按 `options.provider` 或默认 provider 分发请求。
4. provider 返回统一音频结果，由调用方自行发送或播放。

## 相关插件

- `plugins/qwen_tts_provider`
- `plugins/voice_chatter`
