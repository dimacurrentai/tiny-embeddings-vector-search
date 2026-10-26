"""Temporary localhost server shared by protocol and real-agent tests."""

import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def running_server(directory: Path):
    log_path = directory / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "mini_vector_search.server",
             "--transport", "streamable-http", "--port", "0",
             "--data-dir", str(directory / "index")],
            stdout=log, stderr=log,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                output = log_path.read_text()
                match = re.search(r"Uvicorn running on (http://127\.0\.0\.1:\d+)", output)
                if match:
                    yield match[1] + "/mcp"
                    return
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited during startup:\n{output}")
                time.sleep(0.05)
            raise TimeoutError(f"Server startup timed out:\n{log_path.read_text()}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
