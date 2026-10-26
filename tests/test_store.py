import unittest
from pathlib import Path

import numpy as np

from mini_vector_search.store import Store

from temp_paths import temporary_directory


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(384, dtype=np.float32)
            vector[0 if "cat" in text else 1] = 1
            vectors.append(vector)
        return np.array(vectors)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.store = Store(self.path, FakeEmbedder())

    def test_ranking_replacement_persistence_and_deletion(self):
        self.store.upsert("cat", "cat on the sofa")
        self.store.upsert("other", "a database")
        results = self.store.search("cat", 2)
        self.assertEqual([r["id"] for r in results], ["cat", "other"])
        self.assertEqual([r["score"] for r in results], [1.0, 0.0])
        self.store.upsert("cat", "a replacement")
        reopened = Store(self.path, FakeEmbedder())
        self.assertEqual(reopened.stats()["count"], 2)
        self.assertEqual(reopened.search("replacement", 1)[0]["text"], "a replacement")
        self.assertEqual(reopened.delete("cat"), {"deleted": True})
        self.assertEqual(reopened.delete("cat"), {"deleted": False})

    def test_empty_collection_does_not_load_model(self):
        store = Store(self.path, object())
        self.assertEqual(store.search("hello"), [])
        self.assertEqual(store.stats()["count"], 0)

    def test_invalid_inputs_do_not_write(self):
        for id, text in [("", "text"), ("id", " \n")]:
            with self.assertRaises(ValueError):
                self.store.upsert(id, text)
        for limit in [0, 101, True, 1.5]:
            with self.assertRaises(ValueError):
                self.store.search("hello", limit)
        with self.assertRaises(ValueError):
            self.store.search(" ")
        self.assertEqual(self.store.stats()["count"], 0)

    def test_sql_characters_are_plain_data(self):
        id = "'; DROP TABLE documents; --"
        self.store.upsert(id, "cat")
        self.assertEqual(self.store.search("cat")[0]["id"], id)
        self.assertTrue(self.store.delete(id)["deleted"])
