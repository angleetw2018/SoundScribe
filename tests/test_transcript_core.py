import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transcript_core import clean_text, format_timestamp, is_speech, to_mono_16k
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


if __name__ == "__main__":
    unittest.main()

