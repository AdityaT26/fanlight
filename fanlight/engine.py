"""The engine: one object that wires data, index, profiles, cohorts and storage.

Both the CLI and the web API talk to this class and nothing below it, so the
two front ends cannot drift apart.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import pickle
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from . import __version__
from .cohorts import NEW_READER, fit_cohorts
from .config import SIGNALS, RankingConfig, Settings
from .feeds import load_feeds
from .index import INDEX_VERSION, CorpusIndex, build_index, resolve_as_of
from .ingest import COLLECTIONS, RawData, build_corpus, load_json_dir, load_mongo
from .live import LiveNews
from .profiles import Profile, ProfileState, build_state, dataset_profiles, public_id, top_interests
from .ranker import RankedFeed, RankedItem, Ranker
from .store import Store

log = logging.getLogger("fanlight")

SIGNAL_HELP: dict[str, str] = {
    "category": "How well the story's category matches what this reader follows and reads.",
    "tags": "Overlap between the story's topic tags and tags the reader engages with.",
    "content": "Text similarity to stories the reader opened, liked or saved.",
    "freshness": "How recently the story was published.",
    "source": "Reach of the outlet, plus outlets this reader tends to read.",
    "popularity": "How many outlets reported the story and how readers engaged with it.",
}


class UnknownReader(KeyError):
    """Raised when a reader id matches neither an imported nor a created reader."""


def _locked(method):
    """Run an engine method under the engine lock, so a live rebuild never swaps
    the index out from under a request that is halfway through using it."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class Engine:
    def __init__(
        self,
        index: CorpusIndex,
        profiles: Iterable[Profile] = (),
        store: Store | None = None,
        settings: Settings | None = None,
        config: RankingConfig | None = None,
        origin: str = "memory",
        live: LiveNews | None = None,
    ):
        self.settings = settings or Settings()
        self.config = config or RankingConfig()
        self.store = store or Store(":memory:")
        self.origin = origin
        self.live = live
        self.edition = 0
        self._lock = threading.RLock()
        self._dataset = {p.id: p for p in profiles}
        self._install(index)

    def _install(self, index: CorpusIndex) -> None:
        """Make ``index`` the one every request uses (callers hold the lock or are __init__)."""
        self.index = index
        self.ranker = Ranker(index, self.config)
        self._category_counts = Counter(key for a in index.corpus.articles for key in a.categories)
        cohort_profiles = list(self._dataset.values()) + [
            p for p in self.store.list_readers() if p.categories
        ]
        self.cohorts, self._cohort_of = fit_cohorts(index, cohort_profiles, k=self.settings.cohorts)
        self.edition += 1
        self.edition_built = time.time()

    def rebuild(self, raw: RawData, as_of: str = "now") -> None:
        """Build a new index from ``raw`` and swap it in. Requests keep being served
        from the old index while the new one builds; the swap itself is instant."""
        started = time.perf_counter()
        index = build_index(build_corpus(raw), as_of=as_of)
        with self._lock:
            self._install(index)
        log.info(
            "Rebuilt index: %d stories in %.1fs (edition %d)",
            index.size,
            time.perf_counter() - started,
            self.edition,
        )

    def _tick_clock(self) -> None:
        """Live news is ranked against the wall clock, not the newest story."""
        if self.live is not None:
            self.index.as_of = time.time()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_raw(cls, raw: RawData, as_of: str = "latest", **kwargs: Any) -> Engine:
        corpus = build_corpus(raw)
        profiles = dataset_profiles(raw, corpus)
        return cls(build_index(corpus, as_of=as_of), profiles, **kwargs)

    @classmethod
    def load(
        cls,
        settings: Settings | None = None,
        demo: bool = False,
        source: str = "json",
        rebuild: bool = False,
        store: Store | None = None,
    ) -> Engine:
        """Build (or load from cache) the engine for the configured data source."""
        settings = settings or Settings.from_env()
        store = store or Store(settings.db_path)
        if demo:
            from .synth import make_raw

            raw = make_raw(n_articles=900, n_users=60)
            return cls.from_raw(raw, as_of=settings.as_of, store=store, settings=settings, origin="demo")
        if source == "live":
            live = LiveNews(store, load_feeds(settings.feeds_path), retention_days=settings.retention_days)
            if store.live_count() == 0:
                log.info("No stored stories yet: fetching %d feeds", len(live.feeds))
                live.refresh()
            raw = live.raw()
            if not raw["articles"]:
                failed = live.last_report.failed if live.last_report else {}
                detail = (
                    f" Errors: {'; '.join(f'{k}: {v}' for k, v in list(failed.items())[:3])}"
                    if failed
                    else ""
                )
                raise RuntimeError(
                    f"No live stories: none of the {len(live.feeds)} feeds returned anything recent. "
                    f"Check the internet connection and the feed list (FANLIGHT_FEEDS).{detail}"
                )
            index = build_index(build_corpus(raw), as_of="now")
            return cls(index, (), store=store, settings=settings, origin="live", live=live)
        if source == "mongo":
            if not settings.mongo_uri:
                raise RuntimeError("Set FANLIGHT_MONGO_URI to read from MongoDB")
            raw = load_mongo(settings.mongo_uri, settings.mongo_db)
            return cls.from_raw(raw, as_of=settings.as_of, store=store, settings=settings, origin="mongo")

        cache = settings.state_dir / "cache" / f"index-{_fingerprint(settings.data_dir)}.pkl"
        if cache.is_file() and not rebuild:
            try:
                with cache.open("rb") as handle:
                    payload = pickle.load(handle)
                index, profiles = payload["index"], payload["profiles"]
                index.as_of = resolve_as_of(settings.as_of, index.published)
                log.info("Loaded cached index (%d articles)", index.size)
                return cls(index, profiles, store=store, settings=settings, origin="json")
            except Exception as exc:  # stale or corrupt cache: rebuild
                log.warning("Ignoring unreadable index cache (%s)", exc)

        started = time.perf_counter()
        raw = load_json_dir(settings.data_dir)
        corpus = build_corpus(raw)
        profiles = dataset_profiles(raw, corpus)
        index = build_index(corpus, as_of=settings.as_of)
        log.info("Built index for %d articles in %.1fs", index.size, time.perf_counter() - started)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump({"index": index, "profiles": profiles}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(cache)
        except OSError as exc:
            log.warning("Could not write index cache: %s", exc)
        return cls(index, profiles, store=store, settings=settings, origin="json")

    # ------------------------------------------------------------------ #
    # Readers
    # ------------------------------------------------------------------ #

    def profile(self, reader_id: str) -> Profile:
        """Stated preferences plus the full interaction history for a reader."""
        base = self._dataset.get(reader_id) or self._dataset.get(public_id(reader_id))
        if base is None:
            base = self.store.get_reader(reader_id)
        if base is None:
            raise UnknownReader(reader_id)
        return replace(base, events=self.store.events(base.id))

    @_locked
    def create_reader(self, name: str, languages: Iterable[str], categories: Iterable[str]) -> Profile:
        return self.store.create_reader(
            name, self._languages(languages), dict.fromkeys(self._categories(categories), 1.0)
        )

    @_locked
    def update_reader(
        self,
        reader_id: str,
        name: str | None = None,
        languages: Iterable[str] | None = None,
        categories: Iterable[str] | None = None,
    ) -> Profile:
        if reader_id in self._dataset:
            raise PermissionError("Imported readers are read-only; create a reader to edit interests")
        ok = self.store.update_reader(
            reader_id,
            name=name,
            languages=None if languages is None else self._languages(languages),
            categories=None if categories is None else dict.fromkeys(self._categories(categories), 1.0),
        )
        if not ok:
            raise UnknownReader(reader_id)
        return self.profile(reader_id)

    @_locked
    def delete_reader(self, reader_id: str) -> None:
        if reader_id in self._dataset:
            raise PermissionError("Imported readers cannot be deleted")
        if not self.store.delete_reader(reader_id):
            raise UnknownReader(reader_id)

    def _languages(self, languages: Iterable[str]) -> list[str]:
        clean = [str(x).strip().lower() for x in languages]
        unknown = [x for x in clean if x not in self.index.languages]
        if unknown:
            raise ValueError(f"unknown language(s): {', '.join(unknown)}")
        return clean

    def _categories(self, categories: Iterable[str]) -> list[str]:
        clean = list(dict.fromkeys(str(x) for x in categories))
        unknown = [x for x in clean if x not in self.index.corpus.categories]
        if unknown:
            raise ValueError(f"unknown category key(s): {', '.join(unknown)}")
        return clean

    @_locked
    def record(self, reader_id: str, events: Iterable[tuple[str, str]]) -> int:
        """Log interactions. Events for stories not in the index are dropped."""
        profile = self.profile(reader_id)
        known = [(a, action) for a, action in events if a in self.index.position]
        return self.store.add_events(profile.id, known)

    @_locked
    def undo(self, reader_id: str, article_id: str, action: str) -> int:
        return self.store.remove_events(self.profile(reader_id).id, article_id, action)

    @_locked
    def reset_history(self, reader_id: str) -> int:
        return self.store.clear_events(self.profile(reader_id).id)

    @_locked
    def list_readers(self, limit: int = 60, ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Created readers first, then imported ones (those with interests first).

        With ``ids``, only those readers (unknown ids are skipped): this is how a
        browser in public mode lists the readers it created and nobody else's.
        """
        if ids is not None:
            cards = []
            for reader_id in dict.fromkeys(ids):
                try:
                    cards.append(self._reader_card(self.profile(reader_id)))
                except UnknownReader:
                    continue
            return cards
        created = [self._reader_card(p) for p in self.store.list_readers()]
        imported = sorted(self._dataset.values(), key=lambda p: (not p.categories, p.id))
        return created + [self._reader_card(p) for p in imported[: max(limit - len(created), 0)]]

    def _reader_card(self, profile: Profile) -> dict[str, Any]:
        names = self.index.corpus.categories
        top = sorted(profile.categories.items(), key=lambda kv: -kv[1])[:3]
        return {
            "id": profile.id,
            "name": profile.name,
            "kind": profile.kind,
            "languages": list(profile.languages),
            "cohort": self._cohort_of.get(profile.id) or self._stated_cohort(profile),
            "follows": [names[k].name for k, _ in top if k in names],
            "interest_count": len(profile.categories),
        }

    def _stated_cohort(self, profile: Profile) -> str:
        positions = {key: j for j, key in enumerate(self.index.category_keys)}
        vector = np.zeros(len(positions))
        for key, weight in profile.categories.items():
            if key in positions:
                vector[positions[key]] = weight
        return self.cohorts.assign(vector)

    @_locked
    def describe_reader(
        self, reader_id: str, state: ProfileState | None = None, profile: Profile | None = None
    ) -> dict[str, Any]:
        profile = profile or self.profile(reader_id)
        state = state or build_state(self.index, profile, self.config)
        actions = Counter(e.action for e in profile.events if e.article_id in self.index.position)
        recent, seen = [], set()
        for event in reversed(profile.events):
            if event.action == "view" or event.article_id in seen:
                continue
            i = self.index.position.get(event.article_id)
            if i is None:
                continue
            seen.add(event.article_id)
            recent.append(
                {"id": event.article_id, "title": self.index.article(i).title, "action": event.action}
            )
            if len(recent) == 8:
                break
        cohort = self.cohorts.assign(state.category_affinity) if state.has_category_signal else NEW_READER
        return {
            **self._reader_card(profile),
            "cohort": cohort,
            "editable": profile.kind == "reader",
            "stated": sorted(profile.categories),
            "interests": top_interests(self.index, state, profile),
            "activity": {
                name: int(actions.get(name, 0)) for name in ("view", "open", "like", "save", "hide")
            },
            "recent": recent,
        }

    # ------------------------------------------------------------------ #
    # Ranking
    # ------------------------------------------------------------------ #

    @_locked
    def rank(
        self,
        reader_id: str,
        k: int = 20,
        overrides: Mapping[str, Any] | None = None,
        categories: Iterable[str] | None = None,
        now: float | None = None,
    ) -> tuple[Profile, ProfileState, RankedFeed]:
        self._tick_clock()
        profile = self.profile(reader_id)
        config = self.config.with_overrides(overrides)
        state = build_state(self.index, profile, config, now=now)
        wanted = self._categories(categories) if categories else None
        return profile, state, self.ranker.rank(profile, k=k, config=config, categories=wanted, state=state)

    @_locked
    def feed(
        self,
        reader_id: str,
        k: int = 20,
        overrides: Mapping[str, Any] | None = None,
        categories: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """A JSON-ready feed: stories, explanations, reader summary and feed shape."""
        profile, state, ranked = self.rank(reader_id, k=k, overrides=overrides, categories=categories)
        marks: dict[str, set[str]] = {"like": set(), "save": set()}
        for event in profile.events:
            if event.action in marks:
                marks[event.action].add(event.article_id)
        return {
            "reader": self.describe_reader(profile.id, state, profile),
            "items": [self._item(item, marks) for item in ranked.items],
            "eligible": ranked.eligible,
            "active_signals": ranked.active_signals,
            "weights": ranked.weights,
            "shape": ranked.diagnostics,
            "config": self.config.with_overrides(overrides).public_dict(),
            "elapsed_ms": ranked.elapsed_ms,
        }

    def _item(self, item: RankedItem, marks: Mapping[str, set[str]]) -> dict[str, Any]:
        article = self.article_json(item.index)
        return {
            **article,
            "rank": item.rank,
            "base_rank": item.base_rank,
            "match": round(item.score * 100),
            "signals": item.signals,
            "contributions": item.contributions,
            "reasons": item.reasons,
            "discovery": item.discovery,
            "related": [
                {
                    "id": self.index.article(j).id,
                    "title": self.index.article(j).title,
                    "url": self.index.article(j).url,
                    "source": self._source_name(self.index.article(j).source),
                    "sentiment": self.index.article(j).sentiment,
                }
                for j in item.related[:5]
            ],
            "liked": article["id"] in marks["like"],
            "saved": article["id"] in marks["save"],
        }

    def _source_name(self, key: str | None) -> str | None:
        if not key:
            return None
        source = self.index.corpus.sources.get(key)
        return source.name if source else key

    @_locked
    def article_json(self, i: int) -> dict[str, Any]:
        article = self.index.article(i)
        corpus = self.index.corpus
        age_hours = max(self.index.as_of - self.index.published[i], 0.0) / 3600.0
        return {
            "id": article.id,
            "title": article.title,
            "summary": article.summary,
            "url": article.url,
            "image": article.image,
            "language": article.language,
            "published": article.published.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "age_hours": round(float(age_hours), 1),
            "source": self._source_name(article.source),
            "categories": [{"key": key, "name": corpus.categories[key].name} for key in article.categories],
            "tags": [{"key": key, "name": corpus.tags[key].name} for key in article.tags],
            "sentiment": article.sentiment,
            "coverage": int(self.index.coverage[i]),
        }

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    @_locked
    def meta(self) -> dict[str, Any]:
        self._tick_clock()
        corpus = self.index.corpus
        stats = self.index.stats()
        return {
            "version": __version__,
            "origin": self.origin,
            "as_of": datetime.fromtimestamp(self.index.as_of, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "stats": stats,
            "ingest": corpus.report.as_dict(),
            "languages": self.index.languages,
            "categories": [
                {"key": key, "name": corpus.categories[key].name, "count": count}
                for key, count in self._category_counts.most_common()
            ],
            "signals": [{"key": s, "help": SIGNAL_HELP[s]} for s in SIGNALS],
            "defaults": self.config.public_dict(),
            "cohorts": [
                {"label": label, "size": size}
                for label, size in zip(self.cohorts.labels, self.cohorts.sizes, strict=False)
            ],
            "readers": {"imported": len(self._dataset), "created": len(self.store.list_readers())},
            "events": self.store.event_counts(),
            "public": self.settings.public,
            "edition": self.edition,
            "edition_built": datetime.fromtimestamp(self.edition_built, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "live": self.live.status() if self.live is not None else None,
        }


def _fingerprint(data_dir: Path) -> str:
    """Cache key: code/library versions plus name, size and mtime of each data file."""
    digest = hashlib.sha1()
    digest.update(f"{INDEX_VERSION}|{__version__}|{sklearn.__version__}|{np.__version__}".encode())
    for name in COLLECTIONS:
        for suffix in (".json", ".jsonl"):
            path = Path(data_dir) / f"{name}{suffix}"
            if path.is_file():
                stat = path.stat()
                digest.update(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]
