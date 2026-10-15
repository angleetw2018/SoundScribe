from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time

import numpy as np


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

    def render(self) -> str:
        return f"[{format_timestamp(self.elapsed)}] {self.text}"


class TranscriptStore:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.path = folder / f"逐字稿_{stamp}.txt"
        self.path.write_text("電腦聲音逐字稿\n\n", encoding="utf-8")

    def append(self, line: TranscriptLine) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line.render() + "\n")

