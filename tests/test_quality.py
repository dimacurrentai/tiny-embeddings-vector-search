"""Small, attributed CLINC150 intent-retrieval regression evaluation."""

import json
import os
import unittest
from collections import Counter
from pathlib import Path

from mini_vector_search.embedding import Embedder
from mini_vector_search.server import DEFAULT_DATA
from mini_vector_search.store import Store

from temp_paths import temporary_directory

FIXTURE = Path(__file__).parent / "fixtures/clinc150/samples.json"


class QualityTests(unittest.TestCase):
    def test_fixture_is_balanced_and_unique(self):
        samples = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(len(samples), 200)
        self.assertEqual(len({row["id"] for row in samples}), 200)
        self.assertEqual(len({row["text"] for row in samples}), 200)
        counts = Counter(row["intent"] for row in samples)
        self.assertEqual(len(counts), 20)
        self.assertEqual(set(counts.values()), {10})

    @unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "set RUN_MODEL_TESTS=1")
    def test_clinc150_subset_retrieval(self):
        samples = json.loads(FIXTURE.read_text(encoding="utf-8"))
        labels = {row["id"]: row["intent"] for row in samples}
        with temporary_directory() as directory:
            store = Store(Path(directory), Embedder(DEFAULT_DATA / "models"))
            for row in samples:
                store.upsert(row["id"], row["text"])
            top1 = hits = relevant_total = reciprocal_total = 0
            for row in samples:
                # Fetch one extra candidate, then exclude the query by ID, not by rank.
                results = [
                    item for item in store.search(row["text"], limit=6)
                    if item["id"] != row["id"]
                ][:5]
                relevant = [labels[item["id"]] == row["intent"] for item in results]
                top1 += relevant[0]
                hits += any(relevant)
                relevant_total += sum(relevant)
                reciprocal_total += next(
                    (1 / rank for rank, match in enumerate(relevant, 1) if match), 0
                )
        count = len(samples)
        metrics = {
            "queries": count,
            "candidates_per_query": count - 1,
            "top1_accuracy": top1 / count,
            "hit_rate_at_5": hits / count,
            "recall_at_5": relevant_total / (count * 9),
            "mrr_at_5": reciprocal_total / count,
        }
        print("\nCLINC150 subset: " + json.dumps(metrics, sort_keys=True), flush=True)
        self.assertGreaterEqual(metrics["top1_accuracy"], 0.70)
        self.assertGreaterEqual(metrics["hit_rate_at_5"], 0.85)
