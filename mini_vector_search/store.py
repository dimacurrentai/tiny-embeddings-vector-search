"""SQLite persistence and exact cosine search for small collections."""

import sqlite3
from contextlib import closing
from pathlib import Path
from threading import RLock

import numpy as np

from .embedding import DIMENSIONS, MODEL, REVISION, Embedder


def nonblank(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


class Store:
    def __init__(self, directory: Path, embedder=None):
        directory.mkdir(parents=True, exist_ok=True)
        self.database = directory / "documents.sqlite3"
        self.embedder = embedder if embedder is not None else Embedder(directory / "models")
        self.lock = RLock()
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS documents "
                "(id TEXT PRIMARY KEY, text TEXT NOT NULL, vector BLOB NOT NULL)"
            )

    def connect(self):
        # Callers close explicitly: sqlite's context manager only manages transactions.
        return closing(sqlite3.connect(self.database, timeout=30))

    def upsert(self, id: str, text: str) -> dict:
        nonblank(id, "id")
        nonblank(text, "text")
        with self.lock:
            vector = self.embedder.encode([text])[0].astype("<f4")
            with self.connect() as connection, connection:
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET text=excluded.text, vector=excluded.vector",
                    (id, text, vector.tobytes()),
                )
        return {"id": id}

    def search(self, query: str, limit: int = 5) -> list[dict]:
        nonblank(query, "query")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        with self.lock:
            with self.connect() as connection:
                rows = connection.execute("SELECT id, text, vector FROM documents ORDER BY id").fetchall()
            if not rows:
                return []
            query_vector = self.embedder.encode([query])[0]
            vectors = np.stack([np.frombuffer(row[2], dtype="<f4") for row in rows])
            scores = np.clip(vectors @ query_vector, -1, 1)
            best = np.argsort(-scores, kind="stable")[:limit]
            return [
                {"id": rows[i][0], "text": rows[i][1], "score": float(scores[i])}
                for i in best
            ]

    def delete(self, id: str) -> dict:
        nonblank(id, "id")
        with self.lock, self.connect() as connection, connection:
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (id,))
            return {"deleted": cursor.rowcount == 1}

    def stats(self) -> dict:
        with self.lock, self.connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return {"count": count, "model": MODEL, "revision": REVISION, "dimensions": DIMENSIONS}
