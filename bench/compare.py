"""Check that embed.c and ONNX Runtime agree, then time both after model load.

    uv run python bench/compare.py [--rounds 5]

Builds bench/build/embed if needed, exports weights if needed, then sends the
CLINC150 fixture sentences plus a few mixed-case and punctuated ones through
four persistent processes: the C embedder on one thread and on all performance
cores, ONNX Runtime with default threads, and ONNX Runtime pinned to one thread. Load time is excluded by timing only
after each process answers its first request.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_vector_search.embedding import REVISION  # noqa: E402
from mini_vector_search.server import DEFAULT_DATA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "bench/build/embed"
EXPORT = DEFAULT_DATA / "models" / REVISION / "c"
FIXTURE = ROOT / "tests/fixtures/clinc150/samples.json"
EXTRA = [
    "Hello, World!! It's 3:45pm -- let's go.",
    "Create a database index to speed up SQL queries.",
    "Café Zoë served crème brûlée; naïve façade, Ærø, Łódź.",
    "The quick brown fox jumps over the lazy dog. " * 20,
    "Unbelievably, pseudoscientific hyperparameterization persisted.",
]


def prepare():
    if not EXPORT.joinpath("weights.bin").is_file():
        subprocess.run([sys.executable, str(ROOT / "bench/export_weights.py")], check=True)
    if not BINARY.is_file() or BINARY.stat().st_mtime < (ROOT / "bench/embed.c").stat().st_mtime:
        subprocess.run(["make", "-s", "-C", str(ROOT / "bench"), "build/embed"], check=True)


class Runner:
    def __init__(self, name, command, stderr=None):
        self.name = name
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=stderr, text=True, cwd=ROOT)

    def embed(self, text):
        self.process.stdin.write(json.dumps(text) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        if "error" in reply:
            raise RuntimeError(f"{self.name}: {reply['error']}")
        return reply

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)


def performance_cores():
    """Threads beyond the performance cores land on efficiency cores and slow the whole request."""
    if platform.system() == "Darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True))
        except (subprocess.CalledProcessError, ValueError):
            pass
    return os.cpu_count() or 1


def runners():
    python = [sys.executable, str(ROOT / "bench/embed_python.py")]
    cores = performance_cores()
    return [
        Runner("C, one thread", [str(BINARY), str(EXPORT)]),
        Runner(f"C, {cores} threads", [str(BINARY), str(EXPORT), "--threads", str(cores)]),
        Runner("ONNX Runtime, default threads", python),
        Runner("ONNX Runtime, one thread", python + ["--threads", "1"]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5, help="timed passes over the sentence set")
    args = parser.parse_args()
    prepare()
    texts = [row["text"] for row in json.loads(FIXTURE.read_text(encoding="utf-8"))] + EXTRA
    active = runners()
    try:
        reference = [active[2].embed(text) for text in texts]  # also absorbs load time
        for runner in active:
            runner.embed(texts[0])
            worst = 0.0
            for text, expected in zip(texts, reference):
                reply = runner.embed(text)
                if reply["ids"] != expected["ids"]:
                    raise SystemExit(f"{runner.name}: token IDs differ for {text!r}\n{reply['ids']}\n{expected['ids']}")
                worst = max(worst, max(abs(a - b) for a, b in zip(reply["vector"], expected["vector"])))
            print(f"{runner.name}: {len(texts)} sentences, token IDs identical, max |difference| {worst:.2e}")
        print()
        print(f"{'runner':32} {'per sentence':>14} {'sentences/s':>12}")
        for runner in active:
            best = float("inf")
            for _ in range(args.rounds):
                start = time.perf_counter()
                for text in texts:
                    runner.embed(text)
                best = min(best, (time.perf_counter() - start) / len(texts))
            print(f"{runner.name:32} {best * 1e6:11.0f} us {1 / best:12.0f}")
        print(f"\nBest of {args.rounds} rounds; sentences average "
              f"{sum(len(r['ids']) for r in reference) / len(reference):.1f} tokens.")
    finally:
        for runner in active:
            runner.close()


if __name__ == "__main__":
    main()
