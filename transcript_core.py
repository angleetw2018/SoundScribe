from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from pathlib import Path
import queue
import re
import time

import numpy as np


RECORDING_MODES = {
    "只錄電腦": ("電腦",),
    "只錄麥克風": ("麥克風",),
    "兩者都錄": ("電腦", "麥克風"),
}


def active_sources(mode: str) -> tuple[str, ...]:
    try:
        return RECORDING_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"未知的錄音模式：{mode}") from exc


def device_states(mode: str) -> tuple[str, str]:
    sources = active_sources(mode)
    return (
        "readonly" if "電腦" in sources else "disabled",
        "readonly" if "麥克風" in sources else "disabled",
    )


def split_audio_devices(devices: list[dict]) -> tuple[list[dict], list[dict]]:
    active = [item for item in devices if int(item.get("maxInputChannels", 0)) > 0]
    system = [item for item in active if bool(item.get("isLoopbackDevice"))]
    microphones = [item for item in active if not bool(item.get("isLoopbackDevice"))]
    return system, microphones


def discover_wasapi_devices(devices: list[dict]) -> tuple[list[dict], list[dict]]:
    wasapi_hosts = {
        int(item.get("hostApi", -1))
        for item in devices
        if bool(item.get("isLoopbackDevice"))
    }
    wasapi_devices = [
        item for item in devices if int(item.get("hostApi", -1)) in wasapi_hosts
    ]
    return split_audio_devices(wasapi_devices)


def prefer_default_device(devices: list[dict], default_index: int) -> list[dict]:
    return sorted(devices, key=lambda item: int(item.get("index", -1)) != int(default_index))


def order_audio_devices(
    system: list[dict], microphones: list[dict], default_input_index: int
) -> tuple[list[dict], list[dict]]:
    return system, prefer_default_device(microphones, default_input_index)


def prefer_physical_microphones(microphones: list[dict]) -> list[dict]:
    virtual_markers = ("virtual", "虛擬")
    return sorted(
        microphones,
        key=lambda item: any(
            marker in str(item.get("name", "")).casefold()
            for marker in virtual_markers
        ),
    )


@dataclass(order=True, frozen=True)
class AudioChunk:
    elapsed: float
    sequence: int
    source: str = field(compare=False)
    audio: np.ndarray = field(compare=False, repr=False)


class AudioChunkBuffer:
    def __init__(self, maxsize: int = 12) -> None:
        self._queue: queue.PriorityQueue[AudioChunk] = queue.PriorityQueue(maxsize=maxsize)
        self._sequence = itertools.count()

    def submit(self, source: str, elapsed: float, audio: np.ndarray) -> bool:
        item = AudioChunk(elapsed, next(self._sequence), source, audio)
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def get(self, timeout: float | None = None) -> AudioChunk:
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> AudioChunk:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()


def to_mono_16k(samples: np.ndarray, source_rate: int, channels: int) -> np.ndarray:
    """Convert interleaved float32 audio to mono 16 kHz float32."""
    if channels < 1 or source_rate < 1:
        raise ValueError("音訊取樣率與聲道數必須大於 0")
    audio = np.asarray(samples, dtype=np.float32)
    usable = audio.size - (audio.size % channels)
    if usable == 0:
        return np.empty(0, dtype=np.float32)
    audio = audio[:usable].reshape(-1, channels).mean(axis=1)
    if source_rate == 16_000 or audio.size < 2:
        return audio.astype(np.float32, copy=False)
    output_size = max(1, round(audio.size * 16_000 / source_rate))
    old_x = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    new_x = np.linspace(0.0, 1.0, output_size, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def is_speech(audio: np.ndarray, threshold: float = 0.006) -> bool:
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    return rms >= threshold


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class TranscriptLine:
    elapsed: float
    text: str
    source: str

    def render(self) -> str:
        return f"[{format_timestamp(self.elapsed)}] [{self.source}] {self.text}"


class TranscriptStore:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.path = folder / f"逐字稿_{stamp}.txt"
        self.path.write_text("聲音逐字稿\n\n", encoding="utf-8")

    def append(self, line: TranscriptLine) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line.render() + "\n")

