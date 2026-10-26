"""Run the model's published ONNX export without PyTorch."""

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSIONS = 384
FILES = {
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
}


def download(directory: Path, name: str, sha256: str) -> Path:
    """Publish only complete, verified downloads; failed attempts can be retried."""
    target = directory / name
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).info("Downloading %s/%s", MODEL, name)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(
                f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", timeout=120
            ) as response:
                shutil.copyfileobj(response, output)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"{name} SHA-256 {digest} does not match pinned {sha256}")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


class Embedder:
    def __init__(self, cache: Path):
        self.cache = cache / REVISION
        self.session = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, DIMENSIONS), dtype=np.float32)
        if self.session is None:
            tokenizer = Tokenizer.from_file(
                str(download(self.cache, "tokenizer.json", FILES["tokenizer.json"]))
            )
            tokenizer.enable_truncation(max_length=256)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            session = ort.InferenceSession(
                str(download(self.cache, "onnx/model.onnx", FILES["onnx/model.onnx"])),
                providers=["CPUExecutionProvider"],
            )
            self.tokenizer, self.session = tokenizer, session
        batches = []
        for start in range(0, len(texts), 32):
            encoded = self.tokenizer.encode_batch(texts[start : start + 32])
            inputs = {
                "input_ids": np.array([item.ids for item in encoded], dtype=np.int64),
                "attention_mask": np.array(
                    [item.attention_mask for item in encoded], dtype=np.int64
                ),
                "token_type_ids": np.array(
                    [item.type_ids for item in encoded], dtype=np.int64
                ),
            }
            tokens = self.session.run(None, inputs)[0]
            mask = inputs["attention_mask"][..., None].astype(np.float32)
            vectors = (tokens * mask).sum(axis=1) / mask.sum(axis=1)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            batches.append(vectors.astype(np.float32))
        return np.concatenate(batches)
