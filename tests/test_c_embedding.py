"""The pure C embedder must reproduce ONNX Runtime's vectors and token IDs."""

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

SENTENCES = [
    "Cats and dogs are common household pets.",
    "Hello, World!! It's 3:45pm -- let's go.",
    "Café Zoë served crème brûlée; naïve façade.",
    "does bj's take reservations",
    "x" * 150 + " overlong words become unknown",
    "The quick brown fox jumps over the lazy dog. " * 30,
]


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "set RUN_MODEL_TESTS=1")
@unittest.skipUnless(shutil.which("cc") and shutil.which("make"), "needs cc and make")
class CEmbeddingTests(unittest.TestCase):
    def test_c_matches_onnx_runtime(self):
        import compare

        compare.prepare()
        runners = []
        try:
            c = compare.Runner("C", [str(compare.BINARY), str(compare.EXPORT)])
            runners.append(c)
            threaded = compare.Runner("C threaded", [str(compare.BINARY), str(compare.EXPORT), "--threads", "3"])
            runners.append(threaded)
            python = compare.Runner("Python", [sys.executable, str(ROOT / "bench/embed_python.py")])
            runners.append(python)
            for text in SENTENCES:
                expected, actual = python.embed(text), c.embed(text)
                self.assertEqual(actual["ids"], expected["ids"], text)
                self.assertEqual(len(actual["vector"]), 384)
                difference = max(abs(a - b) for a, b in zip(actual["vector"], expected["vector"]))
                self.assertLess(difference, 1e-5, text)
                # Threads split the work by column block; every element is summed in the same order.
                self.assertEqual(threaded.embed(text), actual, text)
            self.assertEqual(len(c.embed("x " * 400)["ids"]), 256)
        finally:
            for runner in runners:
                runner.close()
