"""Codec helpers for TTS audio guard."""

from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass

import numpy as np


class AudioDecodeError(ValueError):
    """Raised when provider audio cannot be decoded as supported WAV."""


@dataclass(slots=True)
class DecodedAudio:
    samples: np.ndarray
    sample_rate: int


def decode_wav_base64(audio_base64: str) -> DecodedAudio:
    """Decode a base64 WAV payload into mono float32 samples."""

    payload = str(audio_base64 or "").strip()
    if not payload:
        raise AudioDecodeError("empty_audio")

    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise AudioDecodeError("invalid_base64") from exc

    try:
        with wave.open(io.BytesIO(raw), "rb") as wav_file:
            sample_rate = int(wav_file.getframerate())
            channels = int(wav_file.getnchannels())
            sample_width = int(wav_file.getsampwidth())
            frames = wav_file.readframes(wav_file.getnframes())
    except wave.Error as exc:
        raise AudioDecodeError("unsupported_format") from exc

    if sample_rate <= 0:
        raise AudioDecodeError("invalid_sample_rate")
    if channels <= 0:
        raise AudioDecodeError("invalid_channels")
    if sample_width not in {1, 2, 4}:
        raise AudioDecodeError("unsupported_sample_width")

    if sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        samples = samples.reshape(-1)
    if samples.size == 0:
        raise AudioDecodeError("empty_audio")

    return DecodedAudio(samples=np.clip(samples, -1.0, 1.0), sample_rate=sample_rate)


def encode_wav_base64(samples: np.ndarray, sample_rate: int) -> str:
    """Encode mono float32 samples as 16-bit PCM WAV base64."""

    mono = normalize_mono_float32(samples)
    clipped = np.clip(mono, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def normalize_mono_float32(samples: np.ndarray) -> np.ndarray:
    """Convert arbitrary numeric samples into mono float32 in [-1, 1]."""

    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1:
        array = array.reshape(array.shape[0], -1).mean(axis=1)
    return np.clip(array.reshape(-1), -1.0, 1.0).astype(np.float32, copy=False)


def peak_dbfs(samples: np.ndarray) -> float:
    """Return peak level in dBFS."""

    peak = float(np.max(np.abs(normalize_mono_float32(samples)))) if np.size(samples) else 0.0
    if peak <= 1e-9:
        return -120.0
    return 20.0 * float(np.log10(peak))


def duration_ms(samples: np.ndarray, sample_rate: int) -> int:
    """Return audio duration in milliseconds."""

    if sample_rate <= 0:
        return 0
    return int(round((len(normalize_mono_float32(samples)) / float(sample_rate)) * 1000.0))


__all__ = [
    "AudioDecodeError",
    "DecodedAudio",
    "decode_wav_base64",
    "duration_ms",
    "encode_wav_base64",
    "normalize_mono_float32",
    "peak_dbfs",
]
