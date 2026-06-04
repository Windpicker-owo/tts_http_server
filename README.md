# tts_http_server

## 概述

`tts_http_server` 是 Neo-MoFox 的统一 TTS HTTP 协议入口。

它负责三件事：

- 暴露统一的 HTTP Router
- 维护 TTS provider registry
- 在 provider 输出之后执行内建的 `TTS Audio Guard`

插件自身仍然不实现具体模型合成逻辑，真正的 TTS 能力由外部 provider 插件注册到 registry 后提供，例如 `voxcpm_tts_provider`。

## 提供的组件

- `tts_http_server:router:tts_http_server`
- `tts_http_server:service:tts_provider_registry`
- `tts_http_server:action:generate_voice`

其中：

- router 暴露统一 HTTP 协议
- registry service 管理可用 provider 及默认 provider
- `generate_voice` action 供普通 chatter 直接合成并发送语音
- `TTS Audio Guard` 由 registry service 在 provider 返回整句音频后自动执行

## HTTP 接口

Router 基础路径：

- `/router/tts_http_server`

当前公开接口：

- `GET /router/tts_http_server/api/tts/v1/status`
- `POST /router/tts_http_server/api/tts/v1/synthesize`

`status` 用于查看协议版本、默认 provider 和已注册 provider 列表。

`synthesize` 接收统一的 TTS 请求负载，并返回：

- `audio_base64`
- `mime_type`
- `format`
- `sample_rate`
- `duration_ms`
- `provider`
- `text`
- `metadata`

## 内建 Audio Guard

`tts_http_server` 会在 provider 返回整句可播放音频后，默认执行一句级轻量 DSP 后处理。

当前链路：

```text
provider synthesize
-> decode WAV
-> quality scan
-> declick / depop
-> spectral repair
-> dynamic de-esser
-> dynamic EQ
-> transient suppressor
-> look-ahead limiter
-> peak normalize
-> return guarded result
```

当句子被判定为 severe 时，服务会：

```text
第一次 provider 输出
-> audio guard 评估
-> severe
-> 同 provider retry 1 次
-> 对两次结果分别评分
-> 返回 sentence_badness 更低的一版
```

Audio Guard 当前只处理可解码的 WAV base64。对于空音频、非 WAV、非法 base64 或解码失败的结果，会直接回退原始 provider 输出，并在 metadata 中标注跳过原因。

## metadata.audio_guard

诊断信息写入 `metadata.audio_guard`，主要字段包括：

- `enabled`
- `changed`
- `skipped_reason`
- `input_sample_rate`
- `output_sample_rate`
- `sentence_badness`
- `max_artifact_score`
- `p95_artifact_score`
- `severity`
- `processors_applied`
- `retry_count`
- `processing_time_ms`
- `limiter_gain_reduction_db`
- `deesser_gain_reduction_db`
- `eq_gain_reduction_db`
- `clipping_ratio`
- `output_peak_db`

`voice_chatter` 和普通 `generate_voice` 都会自动拿到处理后的结果，因为它们本来就是通过 `/api/tts/v1/synthesize` 或 registry service 消费整句音频。

## 配置

默认配置文件位于：

- `config/plugins/tts_http_server/config.toml`

除了 `action.expose_generate_voice_action` 外，现在还提供：

- `audio_guard.*`
- `audio_guard_analysis.*`
- `audio_guard_thresholds.*`
- `audio_guard_repair.*`
- `audio_guard_tone.*`
- `audio_guard_limiter.*`
- `audio_guard_retry.*`

其中比较关键的配置是：

- `audio_guard.enabled`：总开关
- `audio_guard_thresholds.severe_badness`：严重异常阈值
- `audio_guard_retry.enabled` / `max_retry`：是否允许 severe retry
- `audio_guard_limiter.ceiling_db`：最终输出峰值上限

## 典型联动

1. provider 插件在加载时向 `tts_provider_registry` 注册 provider
2. `voice_chatter` 或其他调用方发起 `/api/tts/v1/synthesize`
3. registry 选择 provider 并执行合成
4. `TTS Audio Guard` 对整句音频做评估、修复、限幅和必要重试
5. 调用方消费统一格式的安全输出音频

## 相关插件

- `plugins/voxcpm_tts_provider`
- `plugins/voice_chatter`
