"""Live mode: keep a rolling window of feed stories and refresh it on a timer."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .feeds import FeedConfig, fetch_all, reference_rows
from .ingest import RawData
from .store import Store

log = logging.getLogger("fanlight")


@dataclass
class RefreshReport:
    started: float
    seconds: float
    feeds: int
    ok: int
    not_modified: int
    failed: dict[str, str] = field(default_factory=dict)
    fetched: int = 0
    new: int = 0
    pruned: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def summary(self) -> str:
        text = (
            f"{self.ok}/{self.feeds} feeds ok ({self.not_modified} unchanged), "
            f"{self.new} new of {self.fetched} stories, {self.pruned} expired, {self.seconds:.1f}s"
        )
        if self.failed:
            text += "; failed: " + ", ".join(f"{k} ({v})" for k, v in sorted(self.failed.items()))
        return text


class LiveNews:
    """Feeds plus the store that holds what they have published recently."""

    def __init__(
        self,
        store: Store,
        feeds: list[FeedConfig],
        retention_days: float = 14.0,
        client_factory: Callable[[], httpx.Client] | None = None,
    ):
        self.store = store
        self.feeds = feeds
        self.retention_days = retention_days
        self.client_factory = client_factory
        self.last_report: RefreshReport | None = None
        self._refresh_lock = threading.Lock()

    def refresh(self) -> RefreshReport:
        """Fetch every feed once, store new stories, drop expired ones."""
        with self._refresh_lock:
            started = time.time()
            client = self.client_factory() if self.client_factory else None
            try:
                results = fetch_all(self.feeds, self.store.feed_states(), client=client)
            finally:
                if client is not None:
                    client.close()
            report = RefreshReport(started=started, seconds=0.0, feeds=len(results), ok=0, not_modified=0)
            for result in results:
                if not result.ok:
                    report.failed[result.feed] = result.error or "failed"
                    self.store.record_fetch(result.feed, False, started, error=result.error)
                    continue
                report.ok += 1
                if result.not_modified:
                    report.not_modified += 1
                    self.store.record_fetch(result.feed, True, started, not_modified=True)
                    continue
                new = self.store.add_live_articles(result.rows, fetched=started)
                report.fetched += len(result.rows)
                report.new += new
                self.store.record_fetch(
                    result.feed,
                    True,
                    started,
                    etag=result.etag,
                    modified=result.modified,
                    items=len(result.rows),
                    new_items=new,
                )
            report.pruned = self.store.prune_live(time.time() - self.retention_days * 86_400.0)
            report.seconds = time.time() - started
            self.last_report = report
            log.info("Feed refresh: %s", report.summary())
            return report

    def raw(self) -> RawData:
        """Everything in the retention window, in the dataset's raw shape."""
        since = time.time() - self.retention_days * 86_400.0
        return {"articles": self.store.live_articles(since), **reference_rows(self.feeds)}

    def status(self) -> dict[str, Any]:
        states = self.store.feed_states()
        now = time.time()
        rows = []
        for feed in self.feeds:
            state = states.get(feed.id, {})
            rows.append(
                {
                    "id": feed.id,
                    "outlet": feed.outlet,
                    "language": feed.language,
                    "category": feed.category,
                    "enabled": feed.enabled,
                    "ok": bool(state.get("last_success")) and not state.get("last_error"),
                    "last_success": state.get("last_success"),
                    "last_error": state.get("last_error"),
                    "items": state.get("items", 0),
                    "new_items": state.get("new_items", 0),
                }
            )
        last = self.last_report
        return {
            "feeds": rows,
            "stories": self.store.live_count(),
            "last_refresh": last.started
            if last
            else max((r["last_success"] or 0 for r in rows), default=0) or None,
            "minutes_since_refresh": round((now - last.started) / 60.0, 1) if last else None,
        }


class Refresher(threading.Thread):
    """Background thread: refresh feeds, then hand fresh data to ``on_update``."""

    def __init__(self, live: LiveNews, on_update: Callable[[RawData], None], interval_minutes: float = 15.0):
        super().__init__(name="fanlight-refresher", daemon=True)
        self.live = live
        self.on_update = on_update
        self.interval = max(interval_minutes, 1.0) * 60.0
        self._stop = threading.Event()
        self.rebuilds = 0

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            self.tick()

    def tick(self) -> RefreshReport | None:
        try:
            report = self.live.refresh()
            if report.new or report.pruned:
                self.on_update(self.live.raw())
                self.rebuilds += 1
            return report
        except Exception:  # keep the thread alive whatever one refresh does
            log.exception("Feed refresh failed")
            return None

    def stop(self) -> None:
        self._stop.set()
