import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transcript_core
from transcript_core import TranscriptLine, clean_text, format_timestamp, is_speech, to_mono_16k
from main import is_cuda_runtime_error


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


if __name__ == "__main__":
    unittest.main()

