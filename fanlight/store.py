"""SQLite persistence for readers created in the app and for every interaction.

The event log doubles as training data: each row is (reader, story, action,
time), which is exactly what a learned ranker needs once there is enough of it.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .ingest import parse_datetime
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
CREATE TABLE IF NOT EXISTS live_articles (
    id          TEXT PRIMARY KEY,
    feed        TEXT NOT NULL,
    published   REAL NOT NULL,
    fetched     REAL NOT NULL,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS live_by_published ON live_articles (published);
CREATE TABLE IF NOT EXISTS feed_state (
    feed          TEXT PRIMARY KEY,
    etag          TEXT,
    modified      TEXT,
    last_attempt  REAL,
    last_success  REAL,
    last_error    TEXT,
    items         INTEGER NOT NULL DEFAULT 0,
    new_items     INTEGER NOT NULL DEFAULT 0
);
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
        # The id doubles as the reader's key in public mode, so it must be unguessable.
        reader_id = "r_" + secrets.token_urlsafe(16)
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

    # ------------------------------------------------------------------ #
    # Live articles
    # ------------------------------------------------------------------ #

    def add_live_articles(self, rows: Iterable[dict[str, Any]], fetched: float | None = None) -> int:
        """Store feed rows; returns how many were new. Existing stories keep their first copy."""
        now = time.time() if fetched is None else fetched
        values = []
        for row in rows:
            published = _timestamp(row.get("date"), now)
            values.append(
                (row["_id"], row.get("feed", ""), published, now, json.dumps(row, ensure_ascii=False))
            )
        if not values:
            return 0
        with self._tx() as conn:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO live_articles (id, feed, published, fetched, data) VALUES (?,?,?,?,?)",
                values,
            )
            return conn.total_changes - before

    def live_articles(self, since: float = 0.0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM live_articles WHERE published >= ? ORDER BY published", (since,)
            ).fetchall()
        return [json.loads(row["data"]) for row in rows]

    def live_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM live_articles").fetchone()[0])

    def prune_live(self, before: float) -> int:
        with self._tx() as conn:
            cursor = conn.execute("DELETE FROM live_articles WHERE published < ?", (before,))
        return cursor.rowcount

    def feed_states(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM feed_state").fetchall()
        return {row["feed"]: dict(row) for row in rows}

    def record_fetch(
        self,
        feed: str,
        ok: bool,
        attempted: float,
        etag: str | None = None,
        modified: str | None = None,
        error: str | None = None,
        items: int = 0,
        new_items: int = 0,
        not_modified: bool = False,
    ) -> None:
        with self._tx() as conn:
            conn.execute("INSERT OR IGNORE INTO feed_state (feed) VALUES (?)", (feed,))
            if ok:
                conn.execute(
                    """UPDATE feed_state SET last_attempt=?, last_success=?, last_error=NULL,
                       etag=COALESCE(?, etag), modified=COALESCE(?, modified),
                       items=CASE WHEN ? THEN items ELSE ? END, new_items=? WHERE feed=?""",
                    (attempted, attempted, etag, modified, not_modified, items, new_items, feed),
                )
            else:
                conn.execute(
                    "UPDATE feed_state SET last_attempt=?, last_error=?, new_items=0 WHERE feed=?",
                    (attempted, error, feed),
                )

    def event_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT action, COUNT(*) AS n FROM events GROUP BY action").fetchall()
        return {row["action"]: row["n"] for row in rows}


def _timestamp(value: Any, default: float) -> float:
    parsed = parse_datetime(value)
    return parsed.timestamp() if parsed else default
