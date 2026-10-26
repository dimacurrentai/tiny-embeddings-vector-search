"""Reproduce the fixed fixture; no model scores influence selection."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
SOURCE = f"https://raw.githubusercontent.com/clinc/oos-eval/{REVISION}"
DESTINATION = Path(__file__).resolve().parents[1] / "tests/fixtures/clinc150"


def main():
    with urlopen(f"{SOURCE}/data/data_full.json", timeout=30) as response:
        raw = response.read()
    test = json.loads(raw)["test"]
    intents = sorted({intent for _, intent in test})[:20]
    selected = []
    for intent in intents:
        candidates = [(i, text) for i, (text, label) in enumerate(test) if label == intent]
        for index, text in candidates[:10]:
            selected.append({"id": f"test-{index}", "text": text, "intent": intent})
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "samples.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for source, destination in [("LICENSE", "LICENSE"), ("README.md", "UPSTREAM_README.md")]:
        with urlopen(f"{SOURCE}/{source}", timeout=30) as response:
            (DESTINATION / destination).write_bytes(response.read())
    (DESTINATION / "provenance.json").write_text(
        json.dumps({
            "source": f"{SOURCE}/data/data_full.json",
            "revision": REVISION,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "split": "test",
            "selection": "First 20 lexicographically sorted intents; first 10 rows per intent in source order",
            "changes": "Subset and JSON field conversion; original text and intent labels unchanged",
            "license": "CC-BY-3.0",
        }, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(selected)} samples to {DESTINATION}")


if __name__ == "__main__":
    main()
