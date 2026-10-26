import hashlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_vector_search.embedding import download

from temp_paths import temporary_directory

MODEL_SHA256 = hashlib.sha256(b"model").hexdigest()

class DownloadTests(unittest.TestCase):
    def test_download_is_cached(self):
        with temporary_directory() as directory:
            with patch("mini_vector_search.embedding.urlopen", return_value=io.BytesIO(b"model")) as fetch:
                path = download(Path(directory), "onnx/model.onnx", MODEL_SHA256)
                self.assertEqual(path.read_bytes(), b"model")
                self.assertEqual(download(Path(directory), "onnx/model.onnx", MODEL_SHA256), path)
                self.assertEqual(fetch.call_count, 1)

    def test_failed_download_leaves_no_partial_file(self):
        with temporary_directory() as directory:
            with patch("mini_vector_search.embedding.urlopen", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    download(Path(directory), "model.onnx", MODEL_SHA256)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_checksum_mismatch_leaves_no_file(self):
        with temporary_directory() as directory:
            with patch("mini_vector_search.embedding.urlopen", return_value=io.BytesIO(b"tampered")):
                with self.assertRaises(ValueError):
                    download(Path(directory), "model.onnx", MODEL_SHA256)
            self.assertEqual(list(Path(directory).iterdir()), [])
