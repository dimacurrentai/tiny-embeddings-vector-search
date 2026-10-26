"""Keep test scratch directories inside the checkout's ignored tmp directory."""

import tempfile
from pathlib import Path

TEMP_ROOT = Path(__file__).resolve().parents[1] / "tmp"


def temporary_directory():
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test-", dir=TEMP_ROOT)
