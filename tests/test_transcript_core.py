import unittest
from pathlib import Path
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transcript_core
from transcript_core import TranscriptLine, clean_text, format_timestamp, is_speech, to_mono_16k
from main import (
    TranscriptionWorker,
    is_cuda_runtime_error,
    list_audio_devices,
    make_soundcard_numpy_compatible,
)


class TranscriptCoreTests(unittest.TestCase):
    def test_soundcard_binary_fromstring_remains_compatible_with_numpy_2_3(self):
        class Backend:
            numpy = np

        class FakeSoundcard:
            mediafoundation = Backend()

        make_soundcard_numpy_compatible(FakeSoundcard)

        result = FakeSoundcard.mediafoundation.numpy.fromstring(
            np.array([0.25, -0.5], dtype=np.float32).tobytes(), dtype=np.float32
        )
        np.testing.assert_allclose(result, [0.25, -0.5])

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

    def test_soundcard_devices_are_listed_as_loopback_and_microphone_sources(self):
        test_case = self

        class FakeDevice:
            def __init__(self, identifier, name):
                self.id = identifier
                self.name = name

        class FakeSoundcard:
            @staticmethod
            def all_speakers():
                return [FakeDevice("speaker-1", "Speakers")]

        class FakeDevices:
            def __init__(self, backends):
                test_case.assertEqual(backends, [FakeMiniaudio.Backend.WASAPI])

            def get_captures(self):
                return [{"id": "mic-1", "name": "USB Microphone"}]

        class FakeMiniaudio:
            class Backend:
                WASAPI = "wasapi"

            Devices = FakeDevices

        system, microphones = list_audio_devices(FakeSoundcard, FakeMiniaudio)

        self.assertEqual(system, [{"id": "speaker-1", "name": "Speakers", "kind": "loopback"}])
        self.assertEqual(microphones, [{"id": "mic-1", "name": "USB Microphone", "kind": "microphone"}])

    def test_capture_worker_drains_soundcard_loopback_into_buffer_without_transcribing(self):
        self.assertTrue(hasattr(TranscriptionWorker, "_capture_source"), "缺少獨立擷取工作")
        stop_event = threading.Event()
        events = __import__("queue").Queue()
        chunks = transcript_core.AudioChunkBuffer(maxsize=2)
        errors = __import__("queue").Queue()
        test_case = self

        class FakeRecorder:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def record(self, numframes):
                test_case.assertEqual(numframes, 2048)
                stop_event.set()
                return np.full((240_000, 1), 0.25, dtype=np.float32)

        class FakeMicrophone:
            def recorder(self, samplerate, channels=None, blocksize=None):
                test_case.assertEqual(samplerate, 48_000)
                test_case.assertIsNone(channels)
                test_case.assertEqual(blocksize, 2048)
                return FakeRecorder()

        class FakeSoundcard:
            @staticmethod
            def get_microphone(identifier, include_loopback=False):
                self.assertEqual(identifier, "speaker-1")
                self.assertTrue(include_loopback)
                return FakeMicrophone()

        worker = TranscriptionWorker(events, stop_event)
        worker._capture_source(
            "電腦",
            {"id": "speaker-1", "name": "Speakers", "kind": "loopback"},
            time.monotonic(),
            chunks,
            errors,
            FakeSoundcard,
        )

        self.assertTrue(errors.empty())
        self.assertEqual(chunks.get_nowait().source, "電腦")

    def test_capture_worker_uses_miniaudio_for_physical_microphone(self):
        import queue

        stop_event = threading.Event()
        chunks = transcript_core.AudioChunkBuffer(maxsize=2)
        errors = queue.Queue()

        class FakeCaptureDevice:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.running = False
                self.stop_callback = None

            def start(self, callback):
                self.running = True
                callback.send(np.full(240_000, 0.25, dtype=np.float32).tobytes())
                stop_event.set()

            def stop(self):
                self.running = False

            def close(self):
                pass

        class FakeMiniaudio:
            class SampleFormat:
                FLOAT32 = "float32"

            class Backend:
                WASAPI = "wasapi"

            CaptureDevice = FakeCaptureDevice

        worker = TranscriptionWorker(queue.Queue(), stop_event)
        worker._capture_source(
            "麥克風",
            {"id": "mic-1", "name": "USB Microphone", "kind": "microphone"},
            time.monotonic(), chunks, errors, (None, FakeMiniaudio),
        )

        self.assertTrue(errors.empty())
        self.assertEqual(chunks.get_nowait().source, "麥克風")

    def test_microphone_disconnect_is_reported_instead_of_waiting_forever(self):
        import queue

        stop_event = threading.Event()
        errors = queue.Queue()

        class DisconnectingDevice:
            def __init__(self, **_kwargs):
                self.running = False
                self.stop_callback = None

            def start(self, _callback):
                self.running = True
                self.running = False
                self.stop_callback()

            def stop(self):
                self.running = False

            def close(self):
                pass

        class FakeMiniaudio:
            class SampleFormat:
                FLOAT32 = "float32"

            class Backend:
                WASAPI = "wasapi"

            CaptureDevice = DisconnectingDevice

        worker = TranscriptionWorker(queue.Queue(), stop_event)
        worker._capture_source(
            "麥克風",
            {"id": "mic-1", "name": "USB Microphone", "kind": "microphone"},
            time.monotonic(), transcript_core.AudioChunkBuffer(), errors,
            (None, FakeMiniaudio),
        )

        source, error = errors.get_nowait()
        self.assertEqual(source, "麥克風")
        self.assertRegex(str(error), "麥克風已中斷")

    def test_stop_request_does_not_drain_queued_transcription_backlog(self):
        self.assertTrue(hasattr(TranscriptionWorker, "_next_chunk"), "停止後仍會處理積壓音訊")
        stop_event = threading.Event()
        stop_event.set()
        worker = TranscriptionWorker(__import__("queue").Queue(), stop_event)
        chunks = transcript_core.AudioChunkBuffer(maxsize=2)
        chunks.submit("電腦", 5.0, np.array([1.0], dtype=np.float32))

        result = worker._next_chunk(chunks, __import__("queue").Queue())

        self.assertIsNone(result)
        self.assertFalse(chunks.empty(), "停止時不應繼續取出排隊片段")

    def test_capture_error_arriving_with_stop_is_not_reported_as_normal_stop(self):
        import queue

        class RacingErrorQueue:
            calls = 0

            def empty(self):
                return True

            def get_nowait(self):
                self.calls += 1
                if self.calls == 1:
                    raise queue.Empty
                return "麥克風", RuntimeError("device disconnected")

        stop_event = threading.Event()
        stop_event.set()
        worker = TranscriptionWorker(queue.Queue(), stop_event)
        chunks = transcript_core.AudioChunkBuffer(maxsize=1)

        with self.assertRaisesRegex(RuntimeError, "麥克風音訊擷取失敗"):
            worker._next_chunk(chunks, RacingErrorQueue())


if __name__ == "__main__":
    unittest.main()

