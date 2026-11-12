"""Load raw exports (JSON files or MongoDB) and normalise them into a Corpus.

The production data is a MongoDB export with the usual inconsistencies: ids as
``{"$oid": ...}`` or bare strings, three overlapping tag collections, dates in
several formats, soft-deleted rows and re-ingested duplicates. Everything
messy is handled here so the rest of the engine sees clean, typed objects.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Article, Category, Corpus, Engagement, IngestReport, Source, Tag

RawData = dict[str, list[dict[str, Any]]]

#: Collections read from disk or MongoDB. Only ``articles`` is required.
COLLECTIONS: tuple[str, ...] = (
    "articles",
    "article_categories",
    "article_tags",
    "tags",
    "tag",
    "sources",
    "source",
    "user_preferences",
)

_DATE_FORMATS = (
    "%d/%m/%Y %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y",
    "%Y-%m-%d",
)
_FRACTION = re.compile(r"\.(\d+)")
_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_SLUG = re.compile(r"[^a-z0-9ऀ-ॿ]+")
_VIDEO = re.compile(r"\.(mp4|m3u8|webm|mov)(\?|$)", re.IGNORECASE)
_SENTIMENTS = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "natural": "neutral",  # typo present in the production data
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # JSON Lines (one document per line), as written by ``mongoexport``.
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = [data]
    return [row for row in data if isinstance(row, dict)]


def load_json_dir(data_dir: str | Path) -> RawData:
    """Read every known collection from ``data_dir`` (``<name>.json`` or ``.jsonl``)."""
    root = Path(data_dir)
    raw: RawData = {}
    for name in COLLECTIONS:
        for suffix in (".json", ".jsonl"):
            path = root / f"{name}{suffix}"
            if path.is_file():
                raw[name] = _read_json(path)
                break
        else:
            raw[name] = []
    if not raw["articles"]:
        raise FileNotFoundError(
            f"No articles found in '{root}'. Expected '{root / 'articles.json'}'. "
            "Put the dataset there, set FANLIGHT_DATA_DIR, or run with --demo."
        )
    return raw


def load_mongo(uri: str, database: str = "unbiasly") -> RawData:
    """Read every known collection from MongoDB. Requires the ``mongo`` extra."""
    try:
        import pymongo
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("MongoDB support needs pymongo: pip install 'fanlight[mongo]'") from exc
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=10_000)
    try:
        db = client[database]
        raw = {name: list(db[name].find({})) for name in COLLECTIONS}
    finally:
        client.close()
    if not raw["articles"]:
        raise RuntimeError(f"Database '{database}' has no articles")
    return raw


# --------------------------------------------------------------------------- #
# Small normalisers
# --------------------------------------------------------------------------- #


def oid(value: Any) -> str | None:
    """Return a database id as a string, whatever shape it arrived in."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("$oid")
        if value is None:
            return None
    text = str(value).strip()
    return text or None


def slugify(text: str) -> str:
    """Canonical key for names: lower-case, '&' -> 'and', runs of junk -> '-'."""
    lowered = text.strip().lower().replace("&", " and ")
    return _SLUG.sub("-", lowered).strip("-")


def parse_datetime(value: Any) -> datetime | None:
    """Parse the date shapes seen in the exports. Naive values are taken as UTC."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return parse_datetime(value.get("$date", value.get("$numberLong")))
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        seconds = value / 1000.0 if abs(value) > 1e11 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return parse_datetime(int(text))
    iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    # Python 3.10 only accepts 3 or 6 fractional digits.
    iso = _FRACTION.sub(lambda m: "." + (m.group(1) + "000000")[:6], iso, count=1)
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        parsed = None
        for fmt in _DATE_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def clean_text(value: Any) -> str:
    """Strip markup and collapse whitespace."""
    if not isinstance(value, str):
        return ""
    return _SPACE.sub(" ", html.unescape(_TAGS.sub(" ", value))).strip()


def _excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:- ")
    return cut + "…"


def _int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _titlecase(name: str) -> str:
    return " ".join(word if word.isupper() else word[:1].upper() + word[1:] for word in name.split())


# --------------------------------------------------------------------------- #
# Reference tables
# --------------------------------------------------------------------------- #


def _build_categories(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Category], dict[str, str]]:
    categories: dict[str, Category] = {}
    aliases: dict[str, str] = {}
    for row in rows:
        raw_id, name = oid(row.get("_id")), clean_text(row.get("name"))
        if not raw_id or not name:
            continue
        key = slugify(name)
        active = str(row.get("status", "active")).lower() == "active"
        known = categories.get(key)
        if known is None:
            categories[key] = Category(key=key, name=name, active=active)
        elif active and not known.active:
            categories[key] = Category(key=key, name=name, active=True)
        aliases[raw_id] = key
    return categories, aliases


def _build_tags(*collections: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Tag], dict[str, str]]:
    tags: dict[str, Tag] = {}
    aliases: dict[str, str] = {}
    for rows in collections:
        for row in rows:
            raw_id, name = oid(row.get("_id")), clean_text(row.get("name"))
            if not raw_id or not name:
                continue
            key = slugify(name)
            tags.setdefault(key, Tag(key=key, name=_titlecase(name)))
            aliases[raw_id] = key
    return tags, aliases


def _build_sources(*collections: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Source], dict[str, str]]:
    sources: dict[str, Source] = {}
    aliases: dict[str, str] = {}
    for rows in collections:
        for row in rows:
            raw_id = oid(row.get("_id"))
            slug = clean_text(row.get("source_id")) or raw_id
            if not raw_id or not slug:
                continue
            aliases[raw_id] = slug
            if slug in sources:
                continue
            # 'name:' is a typo that exists in the production collection.
            name = clean_text(row.get("name") or row.get("name:"))
            priority = row.get("source_priority")
            sources[slug] = Source(
                key=slug,
                name=name or _titlecase(slug.replace("_", " ")),
                url=row.get("source_url") or None,
                icon=row.get("source_icon") or None,
                priority=int(priority) if isinstance(priority, (int, float)) and priority > 0 else None,
            )
    return sources, aliases


# --------------------------------------------------------------------------- #
# Articles
# --------------------------------------------------------------------------- #


def _image(row: Mapping[str, Any]) -> str | None:
    url = row.get("thumbnail_url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None
    if _VIDEO.search(url):
        return None
    return url


def _quality(row: Mapping[str, Any]) -> tuple[int, float, int]:
    """Which of two duplicate rows to keep: engagement data, then recency, then length."""
    updated = parse_datetime(row.get("updatedAt") or row.get("updated_at"))
    return (
        1 if "views" in row else 0,
        updated.timestamp() if updated else 0.0,
        len(row.get("body") or ""),
    )


def build_corpus(raw: RawData) -> Corpus:
    """Normalise raw collections into a clean, de-duplicated :class:`Corpus`."""
    report = IngestReport(raw_articles=len(raw.get("articles", [])))
    categories, category_aliases = _build_categories(raw.get("article_categories", []))
    tags, tag_aliases = _build_tags(raw.get("tag", []), raw.get("tags", []), raw.get("article_tags", []))
    sources, source_aliases = _build_sources(raw.get("sources", []), raw.get("source", []))

    kept_rows: list[Mapping[str, Any]] = []
    kept_meta: list[tuple[str, datetime]] = []
    seen: dict[tuple[str, str], int] = {}

    for row in raw.get("articles", []):
        if row.get("is_deleted") is True:
            report.dropped_deleted += 1
            continue
        if str(row.get("status", "published")).lower() != "published":
            report.dropped_unpublished += 1
            continue
        title = clean_text(row.get("title"))
        if not title:
            report.dropped_no_title += 1
            continue
        published = parse_datetime(row.get("date")) or parse_datetime(
            row.get("updatedAt") or row.get("updated_at")
        )
        if published is None:
            report.dropped_no_date += 1
            continue
        article_id = oid(row.get("_id")) or clean_text(row.get("article_id"))
        if not article_id:
            report.dropped_no_title += 1
            continue
        language = clean_text(row.get("language")).lower() or "english"

        keys = [("title", f"{language}|{title.casefold()}")]
        if row.get("article_id"):
            keys.append(("article_id", str(row["article_id"])))
        if row.get("link"):
            keys.append(("link", str(row["link"]).strip()))
        duplicate_of = next((seen[k] for k in keys if k in seen), None)
        if duplicate_of is not None:
            report.dropped_duplicate += 1
            if _quality(row) > _quality(kept_rows[duplicate_of]):
                kept_rows[duplicate_of] = row
                kept_meta[duplicate_of] = (article_id, published)
            for k in keys:
                seen.setdefault(k, duplicate_of)
            continue
        for k in keys:
            seen[k] = len(kept_rows)
        kept_rows.append(row)
        kept_meta.append((article_id, published))

    articles: list[Article] = []
    documents: list[str] = []
    for row, (article_id, published) in zip(kept_rows, kept_meta, strict=False):
        title = clean_text(row.get("title"))
        body = clean_text(row.get("body"))
        summary = clean_text(row.get("body_short")) or body

        article_categories: list[str] = []
        for ref in row.get("category") or []:
            key = category_aliases.get(oid(ref) or "")
            if key is None:
                report.unknown_category_refs += 1
            elif key not in article_categories:
                article_categories.append(key)

        article_tags: list[str] = []
        for ref in row.get("tags") or []:
            key = tag_aliases.get(oid(ref) or "")
            if key is None:
                report.unknown_tag_refs += 1
            elif key not in article_tags:
                article_tags.append(key)

        source = None
        source_ref = oid(row.get("source"))
        if source_ref:
            source = source_aliases.get(source_ref)
            if source is None:
                report.unknown_source_refs += 1

        sentiment = _SENTIMENTS.get(str(row.get("sentiment") or "").strip().lower())
        articles.append(
            Article(
                id=article_id,
                title=title,
                summary=_excerpt(summary, 420),
                url=str(row.get("link") or "").strip(),
                language=clean_text(row.get("language")).lower() or "english",
                published=published,
                categories=tuple(article_categories),
                tags=tuple(article_tags),
                source=source,
                image=_image(row),
                sentiment=sentiment,
                author=clean_text(row.get("author_name")) or None,
                engagement=Engagement(
                    views=_int(row.get("views")),
                    likes=_int(row.get("likes")),
                    dislikes=_int(row.get("dislikes")),
                    bookmarks=_int(row.get("bookmarks")),
                    shares=_int(row.get("share")),
                ),
            )
        )
        # The headline carries the most signal per word, so it counts three times.
        documents.append(f"{title}. {title}. {title}. {summary} {body[:2500]}")

    report.kept = len(articles)
    return Corpus(
        articles=articles,
        categories=categories,
        tags=tags,
        sources=sources,
        category_aliases=category_aliases,
        source_aliases=source_aliases,
        tag_aliases=tag_aliases,
        documents=documents,
        report=report,
    )
