import unittest
from pathlib import Path
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transcript_core
from transcript_core import TranscriptLine, clean_text, format_timestamp, is_speech, to_mono_16k
from main import TranscriptionWorker, is_cuda_runtime_error


class TranscriptCoreTests(unittest.TestCase):
    def test_stereo_to_mono(self):
        result = to_mono_16k(np.array([1.0, -1.0, 0.5, 0.5], dtype=np.float32), 16000, 2)
        np.testing.assert_allclose(result, [0.0, 0.5])

    def test_resample_length(self):
        result = to_mono_16k(np.zeros(48000, dtype=np.float32), 48000, 1)
        self.assertEqual(result.size, 16000)

    def test_silence_gate(self):
        self.assertFalse(is_speech(np.zeros(16000, dtype=np.float32)))
        self.assertTrue(is_speech(np.full(16000, 0.1, dtype=np.float32)))

    def test_formatting(self):
        self.assertEqual(format_timestamp(3661.9), "01:01:01")
        self.assertEqual(clean_text("  你好  \n 世界 "), "你好 世界")

    def test_cuda_runtime_error_detection(self):
        self.assertTrue(is_cuda_runtime_error(RuntimeError("Library cublas64_12.dll is not found")))
        self.assertTrue(is_cuda_runtime_error(RuntimeError("cuDNN failed to initialize")))
        self.assertFalse(is_cuda_runtime_error(RuntimeError("audio device disconnected")))

    def test_recording_modes_select_expected_sources(self):
        self.assertTrue(hasattr(transcript_core, "active_sources"), "缺少錄音模式來源選擇")
        active_sources = transcript_core.active_sources
        self.assertEqual(active_sources("只錄電腦"), ("電腦",))
        self.assertEqual(active_sources("只錄麥克風"), ("麥克風",))
        self.assertEqual(active_sources("兩者都錄"), ("電腦", "麥克風"))

    def test_device_selectors_follow_recording_mode(self):
        self.assertTrue(hasattr(transcript_core, "device_states"), "缺少裝置選單狀態規則")
        device_states = transcript_core.device_states
        self.assertEqual(device_states("只錄電腦"), ("readonly", "disabled"))
        self.assertEqual(device_states("只錄麥克風"), ("disabled", "readonly"))
        self.assertEqual(device_states("兩者都錄"), ("readonly", "readonly"))

    def test_transcript_line_identifies_audio_source(self):
        self.assertIn("source", TranscriptLine.__dataclass_fields__, "逐字稿缺少音訊來源")
        line = TranscriptLine(5.9, "可以聽到嗎？", "麥克風")
        self.assertEqual(line.render(), "[00:00:05] [麥克風] 可以聽到嗎？")

    def test_audio_devices_are_split_into_system_and_microphone(self):
        self.assertTrue(hasattr(transcript_core, "split_audio_devices"), "缺少音訊裝置分類")
        devices = [
            {"index": 1, "name": "Speakers [Loopback]", "maxInputChannels": 2, "isLoopbackDevice": True},
            {"index": 2, "name": "USB Microphone", "maxInputChannels": 1, "isLoopbackDevice": False},
            {"index": 3, "name": "Disabled", "maxInputChannels": 0, "isLoopbackDevice": False},
        ]
        system, microphones = transcript_core.split_audio_devices(devices)
        self.assertEqual([item["index"] for item in system], [1])
        self.assertEqual([item["index"] for item in microphones], [2])

    def test_capture_buffer_keeps_capture_independent_and_orders_by_capture_time(self):
        self.assertTrue(hasattr(transcript_core, "AudioChunkBuffer"), "缺少非同步音訊緩衝")
        buffer = transcript_core.AudioChunkBuffer(maxsize=8)

        def produce():
            buffer.submit("電腦", 10.0, np.array([1.0], dtype=np.float32))
            buffer.submit("麥克風", 5.0, np.array([2.0], dtype=np.float32))

        producer = threading.Thread(target=produce)
        producer.start()
        time.sleep(0.05)  # 模擬辨識模型忙碌時，擷取仍可持續入列。
        producer.join(timeout=0.1)

        self.assertFalse(producer.is_alive())
        self.assertEqual(buffer.get_nowait().source, "麥克風")
        self.assertEqual(buffer.get_nowait().source, "電腦")

    def test_capture_worker_drains_stream_into_buffer_without_transcribing(self):
        self.assertTrue(hasattr(TranscriptionWorker, "_capture_source"), "缺少獨立擷取工作")
        stop_event = threading.Event()
        events = __import__("queue").Queue()
        chunks = transcript_core.AudioChunkBuffer(maxsize=2)
        errors = __import__("queue").Queue()

        class FakeStream:
            def read(self, _frames, exception_on_overflow=False):
                stop_event.set()
                return np.full(8, 5000, dtype=np.int16).tobytes()

            def stop_stream(self):
                pass

            def close(self):
                pass

        class FakeAudio:
            def open(self, **_kwargs):
                return FakeStream()

            def terminate(self):
                pass

        class FakePyAudio:
            paInt16 = 8

            @staticmethod
            def PyAudio():
                return FakeAudio()

        worker = TranscriptionWorker(events, stop_event)
        worker._capture_source(
            "麥克風",
            {"index": 1, "defaultSampleRate": 1, "maxInputChannels": 1},
            time.monotonic(),
            chunks,
            errors,
            FakePyAudio,
        )

        self.assertTrue(errors.empty())
        self.assertEqual(chunks.get_nowait().source, "麥克風")


if __name__ == "__main__":
    unittest.main()

