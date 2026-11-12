"""SQLite persistence for readers created in the app and for every interaction.

The event log doubles as training data: each row is (reader, story, action,
time), which is exactly what a learned ranker needs once there is enough of it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from .profiles import ACTIONS, Event, Profile

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    languages   TEXT NOT NULL,
    categories  TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    reader_id   TEXT NOT NULL,
    article_id  TEXT NOT NULL,
    action      TEXT NOT NULL,
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_reader ON events (reader_id, id);
"""

#: Keep at most this many impressions per reader; older ones are pruned.
MAX_VIEWS_PER_READER = 5000


class Store:
    """Thread-safe store. Use ``":memory:"`` for an ephemeral one."""

    def __init__(self, path: str | Path = ":memory:"):
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._tx() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    # Readers
    # ------------------------------------------------------------------ #

    def create_reader(self, name: str, languages: Iterable[str], categories: dict[str, float]) -> Profile:
        reader_id = "r_" + uuid.uuid4().hex[:10]
        langs = tuple(dict.fromkeys(str(x).strip().lower() for x in languages if str(x).strip()))
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO readers (id, name, languages, categories, created_at) VALUES (?,?,?,?,?)",
                (reader_id, name.strip() or "Reader", json.dumps(langs), json.dumps(categories), time.time()),
            )
        return Profile(
            id=reader_id, name=name.strip() or "Reader", languages=langs, categories=dict(categories)
        )

    def update_reader(
        self,
        reader_id: str,
        name: str | None = None,
        languages: Iterable[str] | None = None,
        categories: dict[str, float] | None = None,
    ) -> bool:
        current = self.get_reader(reader_id)
        if current is None:
            return False
        new_name = current.name if name is None else (name.strip() or current.name)
        new_languages = (
            current.languages
            if languages is None
            else tuple(dict.fromkeys(str(x).strip().lower() for x in languages if str(x).strip()))
        )
        new_categories = current.categories if categories is None else dict(categories)
        with self._tx() as conn:
            conn.execute(
                "UPDATE readers SET name=?, languages=?, categories=? WHERE id=?",
                (new_name, json.dumps(new_languages), json.dumps(new_categories), reader_id),
            )
        return True

    def delete_reader(self, reader_id: str) -> bool:
        with self._tx() as conn:
            conn.execute("DELETE FROM events WHERE reader_id=?", (reader_id,))
            cursor = conn.execute("DELETE FROM readers WHERE id=?", (reader_id,))
        return cursor.rowcount > 0

    def get_reader(self, reader_id: str) -> Profile | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM readers WHERE id=?", (reader_id,)).fetchone()
        return self._profile(row) if row else None

    def list_readers(self) -> list[Profile]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM readers ORDER BY created_at DESC").fetchall()
        return [self._profile(row) for row in rows]

    @staticmethod
    def _profile(row: sqlite3.Row) -> Profile:
        return Profile(
            id=row["id"],
            name=row["name"],
            kind="reader",
            languages=tuple(json.loads(row["languages"])),
            categories={str(k): float(v) for k, v in json.loads(row["categories"]).items()},
        )

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    def add_events(self, reader_id: str, events: Iterable[tuple[str, str]], ts: float | None = None) -> int:
        """Append (article_id, action) pairs. Unknown actions raise ValueError."""
        now = time.time() if ts is None else ts
        rows = []
        for article_id, action in events:
            if action not in ACTIONS:
                raise ValueError(f"unknown action '{action}'")
            rows.append((reader_id, str(article_id), action, now))
        if not rows:
            return 0
        with self._tx() as conn:
            conn.executemany("INSERT INTO events (reader_id, article_id, action, ts) VALUES (?,?,?,?)", rows)
            conn.execute(
                """DELETE FROM events WHERE reader_id=? AND action='view' AND id NOT IN (
                       SELECT id FROM events WHERE reader_id=? AND action='view'
                       ORDER BY id DESC LIMIT ?)""",
                (reader_id, reader_id, MAX_VIEWS_PER_READER),
            )
        return len(rows)

    def remove_events(self, reader_id: str, article_id: str, action: str) -> int:
        """Undo an action (for example un-like)."""
        with self._tx() as conn:
            cursor = conn.execute(
                "DELETE FROM events WHERE reader_id=? AND article_id=? AND action=?",
                (reader_id, article_id, action),
            )
        return cursor.rowcount

    def events(self, reader_id: str) -> list[Event]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT article_id, action, ts FROM events WHERE reader_id=? ORDER BY id", (reader_id,)
            ).fetchall()
        return [Event(row["article_id"], row["action"], row["ts"]) for row in rows]

    def clear_events(self, reader_id: str) -> int:
        with self._tx() as conn:
            cursor = conn.execute("DELETE FROM events WHERE reader_id=?", (reader_id,))
        return cursor.rowcount

    def event_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT action, COUNT(*) AS n FROM events GROUP BY action").fetchall()
        return {row["action"]: row["n"] for row in rows}
