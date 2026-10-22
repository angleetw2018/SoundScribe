from __future__ import annotations

import queue
import shutil
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from transcript_core import (
    AudioChunkBuffer,
    RECORDING_MODES,
    TranscriptLine,
    TranscriptStore,
    active_sources,
    clean_text,
    device_states,
    is_speech,
    split_audio_devices,
    to_mono_16k,
)


APP_DIR = Path(__file__).resolve().parent
TRANSCRIPT_DIR = APP_DIR / "transcripts"
MODELS = ("tiny", "base", "small", "medium", "large-v3", "turbo")
LANGUAGES = {"自動偵測": None, "繁體中文／中文": "zh", "英文": "en", "日文": "ja", "韓文": "ko"}


def is_cuda_runtime_error(error: BaseException) -> bool:
    message = str(error).lower()
    markers = ("cuda", "cublas", "cudnn", "nvrtc", "nvcuda")
    return any(marker in message for marker in markers)


def transcribe_chunk(model, chunk: np.ndarray, language: str | None) -> str:
    # faster-whisper 的 segments 是惰性 generator，CUDA DLL 錯誤通常在迭代時才發生。
    segments, _ = model.transcribe(
        chunk,
        language=language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return clean_text("".join(segment.text for segment in segments))


def load_audio_library():
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise RuntimeError("尚未安裝 PyAudioWPatch，請先執行 install.bat。") from exc
    return pyaudio


def list_audio_devices() -> tuple[list[dict], list[dict]]:
    pyaudio = load_audio_library()
    audio = pyaudio.PyAudio()
    try:
        wasapi = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
        devices = [
            dict(audio.get_device_info_by_index(index))
            for index in range(audio.get_device_count())
        ]
        wasapi_devices = [item for item in devices if int(item.get("hostApi", -1)) == int(wasapi["index"])]
        return split_audio_devices(wasapi_devices)
    finally:
        audio.terminate()


class TranscriptionWorker:
    def __init__(self, events: queue.Queue, stop_event: threading.Event) -> None:
        self.events = events
        self.stop_event = stop_event

    def _capture_source(
        self,
        source: str,
        device: dict,
        started: float,
        chunks: AudioChunkBuffer,
        errors: queue.Queue,
        audio_library=None,
    ) -> None:
        pyaudio = audio_library or load_audio_library()
        audio = stream = None
        try:
            audio = pyaudio.PyAudio()
            rate = int(device["defaultSampleRate"])
            channels = int(device["maxInputChannels"])
            frames = 2048
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=int(device["index"]),
                frames_per_buffer=frames,
            )
            target = rate * channels * 5
            pending: list[np.ndarray] = []
            pending_count = 0
            while not self.stop_event.is_set():
                raw = stream.read(frames, exception_on_overflow=False)
                block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                pending.append(block)
                pending_count += block.size
                if pending_count < target:
                    continue
                chunk = to_mono_16k(np.concatenate(pending), rate, channels)
                pending.clear()
                pending_count = 0
                if not is_speech(chunk):
                    self.events.put(("level", f"{source}：略過靜音片段"))
                    continue
                elapsed = time.monotonic() - started
                if not chunks.submit(source, elapsed, chunk):
                    self.events.put(("level", f"{source}：辨識速度落後，略過一段音訊"))
        except Exception as exc:
            errors.put((source, exc))
            self.stop_event.set()
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            if audio is not None:
                audio.terminate()

    def _next_chunk(self, chunks: AudioChunkBuffer, capture_errors: queue.Queue):
        if not capture_errors.empty():
            source, error = capture_errors.get_nowait()
            raise RuntimeError(f"{source}音訊擷取失敗：{error}") from error
        if self.stop_event.is_set():
            return None
        try:
            return chunks.get(timeout=0.1)
        except queue.Empty:
            return None

    def run(self, devices: dict[str, dict], model_name: str, language: str | None) -> None:
        started = time.monotonic()
        store = TranscriptStore(TRANSCRIPT_DIR)
        capture_threads: list[threading.Thread] = []
        try:
            self.events.put(("status", f"載入 {model_name} 模型（首次使用會下載）…"))
            from faster_whisper import WhisperModel

            try:
                model = WhisperModel(model_name, device="cuda", compute_type="float16")
                engine = "NVIDIA GPU"
            except Exception:
                model = WhisperModel(model_name, device="cpu", compute_type="int8")
                engine = "CPU"

            chunks = AudioChunkBuffer(maxsize=12)
            capture_errors: queue.Queue = queue.Queue()
            for source, device in devices.items():
                thread = threading.Thread(
                    target=self._capture_source,
                    args=(source, device, started, chunks, capture_errors),
                    name=f"capture-{source}",
                    daemon=True,
                )
                capture_threads.append(thread)
                thread.start()
            source_names = "＋".join(devices)
            self.events.put(("started", f"監聽中 · {engine} · {source_names}", str(store.path)))

            while (
                any(thread.is_alive() for thread in capture_threads)
                or not chunks.empty()
                or not capture_errors.empty()
            ):
                captured = self._next_chunk(chunks, capture_errors)
                if captured is None:
                    if self.stop_event.is_set():
                        break
                    continue
                self.events.put(("level", f"正在辨識{captured.source}聲音…"))
                try:
                    text = transcribe_chunk(model, captured.audio, language)
                except Exception as exc:
                    if engine != "NVIDIA GPU" or not is_cuda_runtime_error(exc):
                        raise
                    self.events.put(("level", "CUDA 元件不完整，正在自動切換至 CPU…"))
                    del model
                    model = WhisperModel(model_name, device="cpu", compute_type="int8")
                    engine = "CPU"
                    text = transcribe_chunk(model, captured.audio, language)
                    self.events.put(("engine", f"已切換至 CPU · {source_names}"))
                if text:
                    line = TranscriptLine(captured.elapsed, text, captured.source)
                    store.append(line)
                    self.events.put(("text", line.render()))
                self.events.put(("level", "等待語音…"))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            self.stop_event.set()
            for thread in capture_threads:
                thread.join(timeout=2)
            self.events.put(("stopped",))


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("聲音筆記 SoundScribe")
        self.geometry("900x650")
        self.minsize(720, 500)
        self.configure(bg="#101827")
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.system_devices: list[dict] = []
        self.microphone_devices: list[dict] = []
        self.current_transcript: Path | None = None
        self._build_ui()
        self.after(100, self._poll_events)
        self.after(150, self.refresh_devices)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#101827")
        style.configure("TLabel", background="#101827", foreground="#dbeafe", font=("Microsoft JhengHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 22, "bold"), foreground="#f8fafc")
        style.configure("Accent.TButton", font=("Microsoft JhengHei UI", 11, "bold"), padding=(18, 10))
        style.configure("TCombobox", padding=7)

        root = ttk.Frame(self, padding=24)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="聲音筆記", style="Title.TLabel").pack(anchor="w")
        ttk.Label(root, text="將電腦正在播放的聲音，即時轉成可搜尋、可匯出的文字。", foreground="#93a4bd").pack(anchor="w", pady=(2, 18))

        settings = ttk.Frame(root)
        settings.pack(fill="x")
        ttk.Label(settings, text="錄音模式").grid(row=0, column=0, sticky="w")
        ttk.Label(settings, text="模型").grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(settings, text="語言").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.mode_box = ttk.Combobox(settings, state="readonly", values=tuple(RECORDING_MODES), width=16)
        self.mode_box.set("兩者都錄")
        self.mode_box.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.mode_box.bind("<<ComboboxSelected>>", self._sync_device_states)
        self.model_box = ttk.Combobox(settings, state="readonly", values=MODELS, width=12)
        self.model_box.set("small")
        self.model_box.grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=(5, 0))
        self.language_box = ttk.Combobox(settings, state="readonly", values=tuple(LANGUAGES), width=17)
        self.language_box.set("繁體中文／中文")
        self.language_box.grid(row=1, column=2, sticky="ew", padx=(12, 0), pady=(5, 0))
        ttk.Label(settings, text="電腦聲音裝置").grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self.system_device_box = ttk.Combobox(settings, state="readonly")
        self.system_device_box.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        ttk.Label(settings, text="麥克風裝置").grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self.microphone_device_box = ttk.Combobox(settings, state="readonly")
        self.microphone_device_box.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        settings.columnconfigure(0, weight=1)

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=16)
        self.start_button = ttk.Button(controls, text="▶ 開始監聽", style="Accent.TButton", command=self.toggle)
        self.start_button.pack(side="left")
        ttk.Button(controls, text="重新整理裝置", command=self.refresh_devices).pack(side="left", padx=8)
        ttk.Button(controls, text="匯出副本", command=self.export).pack(side="left")
        self.status = ttk.Label(controls, text="準備就緒", foreground="#67e8f9")
        self.status.pack(side="right")

        self.text = tk.Text(
            root, wrap="word", bg="#172033", fg="#e5edf8", insertbackground="white",
            relief="flat", padx=18, pady=16, font=("Microsoft JhengHei UI", 12), spacing3=8,
        )
        self.text.pack(fill="both", expand=True)
        self.text.insert("end", "逐字稿會顯示在這裡。播放影片、會議或 Podcast 後按「開始監聽」。\n")
        self.protocol("WM_DELETE_WINDOW", self.close)

    def refresh_devices(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        try:
            self.system_devices, self.microphone_devices = list_audio_devices()
            self.system_device_box["values"] = [item["name"] for item in self.system_devices]
            self.microphone_device_box["values"] = [item["name"] for item in self.microphone_devices]
            if self.system_devices:
                self.system_device_box.current(0)
            if self.microphone_devices:
                self.microphone_device_box.current(0)
            self._sync_device_states()
            self.status.config(
                text=f"找到 {len(self.system_devices)} 個電腦音訊、{len(self.microphone_devices)} 個麥克風"
            )
        except Exception as exc:
            self.status.config(text="缺少音訊套件")
            messagebox.showerror("無法讀取音訊裝置", str(exc))

    def _sync_device_states(self, _event=None) -> None:
        system_state, microphone_state = device_states(self.mode_box.get())
        self.system_device_box.config(state=system_state)
        self.microphone_device_box.config(state=microphone_state)

    def toggle(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.start_button.config(state="disabled", text="正在停止…")
            return
        devices: dict[str, dict] = {}
        for source in active_sources(self.mode_box.get()):
            if source == "電腦":
                index = self.system_device_box.current()
                options = self.system_devices
            else:
                index = self.microphone_device_box.current()
                options = self.microphone_devices
            if index < 0:
                messagebox.showwarning("尚未選擇裝置", f"請先選擇{source}裝置。")
                return
            devices[source] = options[index]
        self.stop_event.clear()
        worker = TranscriptionWorker(self.events, self.stop_event)
        self.worker_thread = threading.Thread(
            target=worker.run,
            args=(devices, self.model_box.get(), LANGUAGES[self.language_box.get()]),
            daemon=True,
        )
        self.worker_thread.start()
        self.start_button.config(state="disabled")

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind in {"status", "level"}:
                    self.status.config(text=event[1])
                elif kind == "started":
                    self.status.config(text=event[1])
                    self.current_transcript = Path(event[2])
                    self.start_button.config(state="normal", text="■ 停止並存檔")
                elif kind == "engine":
                    self.status.config(text=event[1])
                elif kind == "text":
                    self.text.insert("end", event[1] + "\n")
                    self.text.see("end")
                elif kind == "error":
                    messagebox.showerror("監聽中斷", event[1])
                elif kind == "stopped":
                    self.status.config(text="已停止，逐字稿已存檔")
                    self.start_button.config(state="normal", text="▶ 開始監聽")
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def export(self) -> None:
        if not self.current_transcript or not self.current_transcript.exists():
            messagebox.showinfo("尚無逐字稿", "開始監聽並產生文字後即可匯出。")
            return
        target = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("文字檔", "*.txt")])
        if target:
            shutil.copy2(self.current_transcript, target)
            self.status.config(text=f"已匯出：{target}")

    def close(self) -> None:
        self.stop_event.set()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()

