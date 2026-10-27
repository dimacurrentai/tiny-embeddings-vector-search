"""The Python counterpart of embed.c: same line protocol, same model, via ONNX Runtime.

    uv run python bench/embed_python.py [--threads N]

Reads one JSON string per line and writes {"ids": [...], "vector": [...]}.
The model loads before the first reply; --threads limits ONNX Runtime's
intra-op parallelism (default: the runtime's own choice).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_vector_search.embedding import Embedder  # noqa: E402
from mini_vector_search.server import DEFAULT_DATA  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    embedder = Embedder(DEFAULT_DATA / "models", threads=args.threads)
    embedder.encode(["warm up"])
    for line in sys.stdin:
        text = json.loads(line)
        ids = embedder.tokenizer.encode(text).ids
        vector = embedder.encode([text])[0]
        sys.stdout.write(json.dumps({"ids": ids, "vector": [float(f"{x:.9g}") for x in vector]}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
