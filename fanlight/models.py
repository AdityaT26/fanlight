"""Typed domain objects shared across the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Category:
    key: str
    name: str
    active: bool = True


@dataclass(frozen=True)
class Tag:
    key: str
    name: str


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    url: str | None = None
    icon: str | None = None
    #: Global popularity rank of the outlet (lower is more prominent).
    priority: int | None = None


@dataclass(frozen=True)
class Engagement:
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    bookmarks: int = 0
    shares: int = 0

    @property
    def score(self) -> float:
        """Net engagement, weighting deliberate actions over passive views."""
        raw = 0.2 * self.views + self.likes + 2.0 * self.bookmarks + 3.0 * self.shares - 2.0 * self.dislikes
        return max(raw, 0.0)


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    summary: str
    url: str
    language: str
    published: datetime
    categories: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    source: str | None = None
    image: str | None = None
    sentiment: str | None = None
    author: str | None = None
    engagement: Engagement = field(default_factory=Engagement)


@dataclass
class IngestReport:
    """What happened to the raw records on the way in."""

    raw_articles: int = 0
    kept: int = 0
    dropped_deleted: int = 0
    dropped_unpublished: int = 0
    dropped_no_title: int = 0
    dropped_no_date: int = 0
    dropped_duplicate: int = 0
    unknown_category_refs: int = 0
    unknown_tag_refs: int = 0
    unknown_source_refs: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class Corpus:
    articles: list[Article]
    categories: dict[str, Category]
    tags: dict[str, Tag]
    sources: dict[str, Source]
    #: Raw database id -> canonical key, for resolving references in user data.
    category_aliases: dict[str, str] = field(default_factory=dict)
    source_aliases: dict[str, str] = field(default_factory=dict)
    tag_aliases: dict[str, str] = field(default_factory=dict)
    #: Text used to build the content index, parallel to ``articles``.
    documents: list[str] = field(default_factory=list)
    report: IngestReport = field(default_factory=IngestReport)
