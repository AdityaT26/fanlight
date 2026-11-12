"""Live news from RSS and Atom feeds.

Feeds are fetched with conditional requests (ETag / Last-Modified), parsed with
feedparser, and turned into rows of the same shape as the dataset export, so
everything downstream (ingest, index, ranking) is shared with the static mode.

Only what a feed reader shows is kept: the headline, the feed's own summary,
its image and the link back to the publisher.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import httpx

from . import __version__
from .ingest import clean_text, slugify

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 in CI
    import tomli as tomllib

log = logging.getLogger("fanlight")

DEFAULT_FEEDS = Path(__file__).with_name("feeds.toml")
USER_AGENT = f"Fanlight/{__version__} (+https://github.com/AdityaT26/fanlight; RSS reader)"

#: Ignore entries published more than this long before the fetch.
MAX_ENTRY_AGE_DAYS = 14.0
#: Dates this far in the future are publisher clock or timezone errors.
FUTURE_TOLERANCE = 3600.0
#: Keep this much of a summary for the text index (only an excerpt is shown).
MAX_BODY_CHARS = 2500

#: Live taxonomy. Keys are category names; values are words that, found in a
#: story's own tags or URL path, move it into that category.
CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "India": ("india", "national", "nation", "desh", "states", "cities", "city"),
    "World": ("world", "international", "global", "videsh", "abroad"),
    "Business": (
        "business",
        "economy",
        "markets",
        "market",
        "money",
        "finance",
        "companies",
        "budget",
        "industry",
    ),
    "Technology": (
        "technology",
        "tech",
        "gadgets",
        "gadget",
        "mobile",
        "internet",
        "ai",
        "science-technology",
    ),
    "Science": ("science", "environment", "space", "climate", "energy"),
    "Health": ("health", "fitness", "wellness", "covid"),
    "Sport": (
        "sport",
        "sports",
        "football",
        "tennis",
        "hockey",
        "olympics",
        "khel",
        "badminton",
        "chess",
        "athletics",
    ),
    "Cricket": ("cricket", "ipl"),
    "Entertainment": (
        "entertainment",
        "film",
        "films",
        "movies",
        "movie",
        "bollywood",
        "music",
        "tv",
        "television",
        "celebrity",
        "culture",
        "manoranjan",
        "web-series",
        "ott",
        "hollywood",
    ),
    "Lifestyle": (
        "lifestyle",
        "life-style",
        "food",
        "travel",
        "fashion",
        "relationship",
        "religion",
        "astrology",
        "jyotish",
    ),
    "Education": ("education", "jobs", "career", "exam", "exams"),
    "Top News": (),
}
_HINT_INDEX = {word: name for name, words in CATEGORY_HINTS.items() for word in words}
#: The path segments are matched most specific first, so "india/cricket/..." is Cricket.
_PRIORITY = {"Cricket": 0, "Technology": 1, "Science": 1, "Health": 1, "Entertainment": 1, "Education": 1}
_TRACKING = re.compile(r"^(utm_|at_|fbclid|gclid|ref$|src$|cmpid|ito$|ncid$)")
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeedConfig:
    id: str
    url: str
    outlet: str
    language: str = "english"
    category: str = "Top News"
    reach: int = 2
    enabled: bool = True


def load_feeds(path: str | Path | None = None) -> list[FeedConfig]:
    """Read a feeds TOML file (the packaged default when ``path`` is None)."""
    file = Path(path) if path else DEFAULT_FEEDS
    with file.open("rb") as handle:
        data = tomllib.load(handle)
    feeds: list[FeedConfig] = []
    seen: set[str] = set()
    for number, row in enumerate(data.get("feed", []), start=1):
        missing = [key for key in ("id", "url", "outlet") if not row.get(key)]
        if missing:
            raise ValueError(f"{file}: feed #{number} is missing {', '.join(missing)}")
        if row["id"] in seen:
            raise ValueError(f"{file}: feed id {row['id']!r} is used twice")
        if not str(row["url"]).startswith(("http://", "https://")):
            raise ValueError(f"{file}: feed {row['id']!r} needs an http(s) url")
        category = row.get("category", "Top News")
        if category not in CATEGORY_HINTS:
            raise ValueError(
                f"{file}: feed {row['id']!r} has unknown category {category!r}; use one of {', '.join(CATEGORY_HINTS)}"
            )
        reach = int(row.get("reach", 2))
        if reach not in (1, 2, 3):
            raise ValueError(f"{file}: feed {row['id']!r} reach must be 1, 2 or 3")
        seen.add(row["id"])
        feeds.append(
            FeedConfig(
                id=str(row["id"]),
                url=str(row["url"]),
                outlet=str(row["outlet"]),
                language=str(row.get("language", "english")).lower(),
                category=category,
                reach=reach,
                enabled=bool(row.get("enabled", True)),
            )
        )
    if not any(f.enabled for f in feeds):
        raise ValueError(f"{file}: no enabled feeds")
    return feeds


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def canonical_url(url: str) -> str:
    """Drop tracking parameters and fragments so one story has one URL."""
    parts = urlsplit(url.strip())
    query = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _TRACKING.match(k.lower())
    ]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, urlencode(query), ""))


def article_id(url: str) -> str:
    return "l" + hashlib.sha1(canonical_url(url).encode("utf-8")).hexdigest()[:23]


def infer_category(default: str, tags: Iterable[str], url: str) -> str:
    """Refine a feed's default category from the entry's tags and URL path."""
    candidates: list[str] = []
    words = [slugify(tag) for tag in tags]
    words += urlsplit(url).path.lower().split("/")[:-1]  # the last path segment is the story slug
    for word in words:
        if word in _HINT_INDEX:
            candidates.append(_HINT_INDEX[word])
        else:  # "science-environment", "india-news": match the parts, ignoring short ones
            candidates.extend(_HINT_INDEX[w] for w in word.split("-") if w in _HINT_INDEX and len(w) > 3)
    if not candidates:
        return default
    return min(candidates, key=lambda name: (_PRIORITY.get(name, 2), candidates.index(name)))


def _entry_time(entry: Mapping[str, Any]) -> float | None:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        value = entry.get(key)
        if value:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc).timestamp()
            except (TypeError, ValueError):
                continue
    return None


def _entry_image(entry: Mapping[str, Any]) -> str | None:
    candidates: list[str] = []
    for key in ("media_content", "media_thumbnail"):
        items = sorted(entry.get(key) or [], key=lambda m: -int(str(m.get("width") or 0) or 0))
        candidates.extend(m.get("url", "") for m in items if str(m.get("medium", "image")) == "image")
    for link in entry.get("links") or []:
        if link.get("rel") == "enclosure" and str(link.get("type", "")).startswith("image/"):
            candidates.append(link.get("href", ""))
    image = entry.get("image")
    if isinstance(image, Mapping):
        candidates.append(image.get("href", ""))
    summary = entry.get("summary") or ""
    match = re.search(r"<img[^>]+src=[\"']([^\"']+)", summary)
    if match:
        candidates.append(match.group(1))
    for url in candidates:
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return url
    return None


def parse_feed(content: bytes, feed: FeedConfig, fetched_at: float | None = None) -> list[dict[str, Any]]:
    """Turn feed bytes into article rows (the dataset's raw shape)."""
    fetched_at = time.time() if fetched_at is None else fetched_at
    parsed = feedparser.parse(content)
    if not parsed.get("version") or (parsed.bozo and not parsed.entries):
        # An HTML error page "parses" as an empty feed; treat it as the failure it is.
        reason = parsed.get("bozo_exception") or "no RSS or Atom document found"
        raise ValueError(f"not a feed: {reason}")
    rows: list[dict[str, Any]] = []
    oldest = fetched_at - MAX_ENTRY_AGE_DAYS * 86_400.0
    for entry in parsed.entries:
        link = entry.get("link") or ""
        title = clean_text(entry.get("title"))
        if not link.startswith(("http://", "https://")) or not title:
            continue
        published = _entry_time(entry)
        if published is None or published > fetched_at + FUTURE_TOLERANCE:
            published = fetched_at
        if published < oldest:
            continue
        summary = clean_text(entry.get("summary") or entry.get("description"))
        if summary == title:
            summary = ""
        content_text = ""
        for block in entry.get("content") or []:
            content_text = clean_text(block.get("value"))
            if content_text:
                break
        tags = [t.get("term", "") for t in entry.get("tags") or [] if t.get("term")]
        category = infer_category(feed.category, tags, link)
        stamp = datetime.fromtimestamp(published, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        rows.append(
            {
                "_id": article_id(link),
                "title": title,
                "body": (summary + " " + content_text).strip()[:MAX_BODY_CHARS],
                "body_short": summary,
                "link": canonical_url(link),
                "date": stamp,
                "language": feed.language,
                "category": [slugify(category)],
                "tags": [],
                "source": feed_source_id(feed),
                "status": "published",
                "thumbnail_url": _entry_image(entry),
                "author_name": clean_text(entry.get("author")) or None,
                "feed": feed.id,
            }
        )
    return rows


def feed_source_id(feed: FeedConfig) -> str:
    return "src-" + slugify(feed.outlet)


#: Outlet reach (1-3) mapped onto the popularity-rank scale the index expects.
REACH_PRIORITY = {1: 300, 2: 5000, 3: 80000}


def reference_rows(feeds: Iterable[FeedConfig]) -> dict[str, list[dict[str, Any]]]:
    """Category and outlet rows to go with live articles."""
    categories = [{"_id": slugify(name), "name": name, "status": "active"} for name in CATEGORY_HINTS]
    outlets: dict[str, dict[str, Any]] = {}
    for feed in feeds:
        sid = feed_source_id(feed)
        priority = REACH_PRIORITY[feed.reach]
        if sid not in outlets or priority < outlets[sid]["source_priority"]:
            outlets[sid] = {
                "_id": sid,
                "source_id": slugify(feed.outlet).replace("-", "_"),
                "name": feed.outlet,
                "source_url": f"{urlsplit(feed.url).scheme}://{urlsplit(feed.url).netloc}",
                "source_priority": priority,
            }
    return {"article_categories": categories, "sources": list(outlets.values())}


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


@dataclass
class FetchResult:
    feed: str
    ok: bool
    status: int | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    etag: str | None = None
    modified: str | None = None
    error: str | None = None
    seconds: float = 0.0

    @property
    def not_modified(self) -> bool:
        return self.status == 304


def fetch_feed(
    feed: FeedConfig,
    client: httpx.Client,
    etag: str | None = None,
    modified: str | None = None,
) -> FetchResult:
    """Fetch one feed. Never raises; failures come back as ``ok=False``."""
    started = time.perf_counter()
    headers = {"Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.5"}
    if etag:
        headers["If-None-Match"] = etag
    if modified:
        headers["If-Modified-Since"] = modified
    try:
        response = client.get(feed.url, headers=headers)
        if response.status_code == 304:
            return FetchResult(
                feed.id, True, 304, etag=etag, modified=modified, seconds=time.perf_counter() - started
            )
        if response.status_code >= 400:
            return FetchResult(
                feed.id,
                False,
                response.status_code,
                error=f"HTTP {response.status_code}",
                seconds=time.perf_counter() - started,
            )
        rows = parse_feed(response.content, feed)
        return FetchResult(
            feed.id,
            True,
            response.status_code,
            rows=rows,
            etag=response.headers.get("ETag"),
            modified=response.headers.get("Last-Modified"),
            seconds=time.perf_counter() - started,
        )
    except (httpx.HTTPError, ValueError) as exc:
        return FetchResult(
            feed.id,
            False,
            None,
            error=f"{type(exc).__name__}: {exc}"[:300],
            seconds=time.perf_counter() - started,
        )


def make_client(timeout: float = 15.0) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(timeout, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        limits=httpx.Limits(max_connections=8),
    )


def fetch_all(
    feeds: Iterable[FeedConfig],
    states: Mapping[str, Mapping[str, Any]] | None = None,
    client: httpx.Client | None = None,
    workers: int = 8,
) -> list[FetchResult]:
    """Fetch every enabled feed concurrently."""
    states = states or {}
    own_client = client is None
    client = client or make_client()
    try:
        enabled = [f for f in feeds if f.enabled]
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(enabled)))) as pool:
            return list(
                pool.map(
                    lambda f: fetch_feed(
                        f,
                        client,
                        (states.get(f.id) or {}).get("etag"),
                        (states.get(f.id) or {}).get("modified"),
                    ),
                    enabled,
                )
            )
    finally:
        if own_client:
            client.close()
