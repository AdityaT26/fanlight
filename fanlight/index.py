"""The corpus index: everything about articles that does not depend on the reader.

Built once (and cached on disk), it holds the TF-IDF matrix, category/tag
membership, per-article source quality and popularity, and story clusters
(the same event reported by several outlets).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from .ingest import parse_datetime
from .models import Article, Corpus
from .text import make_vectorizer

#: Bump when the index layout or build logic changes, to invalidate caches.
INDEX_VERSION = 3

#: Articles from different outlets are the same story above this cosine similarity...
STORY_SIMILARITY = 0.5
#: ...but one outlet's own articles must be near-identical (templated pages and
#: shared boilerplate make an outlet's unrelated articles look alike).
SAME_SOURCE_SIMILARITY = 0.85
#: Both only apply to same-language articles published within this many hours.
STORY_WINDOW_HOURS = 72.0
#: Source-quality score for articles whose outlet is unknown.
UNKNOWN_SOURCE_QUALITY = 0.3


@dataclass
class CorpusIndex:
    corpus: Corpus
    as_of: float
    vectorizer: Any
    matrix: sparse.csr_matrix
    category_keys: list[str]
    tag_keys: list[str]
    source_keys: list[str]
    languages: list[str]
    category_matrix: np.ndarray
    tag_matrix: np.ndarray
    language_id: np.ndarray
    source_id: np.ndarray
    published: np.ndarray
    source_quality: np.ndarray
    popularity: np.ndarray
    story: np.ndarray
    coverage: np.ndarray
    position: dict[str, int] = field(default_factory=dict)
    build_seconds: float = 0.0

    @property
    def size(self) -> int:
        return len(self.corpus.articles)

    def article(self, i: int) -> Article:
        return self.corpus.articles[i]

    def story_members(self, i: int) -> np.ndarray:
        """Indices of every article reporting the same story as ``i``."""
        return np.flatnonzero(self.story == self.story[i])

    def stats(self) -> dict[str, Any]:
        published = self.published
        return {
            "articles": self.size,
            "stories": len(np.unique(self.story)),
            "categories": len(self.category_keys),
            "tags": len(self.tag_keys),
            "sources": len(np.unique(self.source_id[self.source_id >= 0])),
            "languages": {name: int((self.language_id == j).sum()) for j, name in enumerate(self.languages)},
            "vocabulary": int(self.matrix.shape[1]),
            "oldest": float(published.min()) if self.size else None,
            "newest": float(published.max()) if self.size else None,
            "as_of": self.as_of,
        }


def resolve_as_of(setting: str, published: np.ndarray) -> float:
    """``latest`` -> newest story, ``now`` -> wall clock, otherwise an ISO date."""
    choice = (setting or "latest").strip().lower()
    if choice == "latest":
        return float(published.max()) if len(published) else time.time()
    if choice == "now":
        return time.time()
    parsed = parse_datetime(setting)
    if parsed is None:
        raise ValueError(f"FANLIGHT_AS_OF must be 'latest', 'now' or an ISO date, got {setting!r}")
    return parsed.timestamp()


def cluster_stories(
    matrix: sparse.csr_matrix,
    published: np.ndarray,
    language_id: np.ndarray,
    source_id: np.ndarray,
    threshold: float = STORY_SIMILARITY,
    same_source_threshold: float = SAME_SOURCE_SIMILARITY,
    window_hours: float = STORY_WINDOW_HOURS,
    chunk: int = 512,
) -> np.ndarray:
    """Group near-identical reports with union-find over thresholded cosine pairs.

    Rows of ``matrix`` are L2-normalised, so a dot product is a cosine. Only
    pairs in the same language and within ``window_hours`` are linked, which
    keeps unrelated recurring headlines (daily market wraps, horoscopes) apart.
    Pairs from one outlet need ``same_source_threshold`` instead of ``threshold``.
    """
    n = matrix.shape[0]
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    transposed = matrix.T.tocsr()
    window = window_hours * 3600.0
    for start in range(0, n, chunk):
        block = (matrix[start : start + chunk] @ transposed).tocoo()
        rows = block.row + start
        keep = (block.data >= min(threshold, same_source_threshold)) & (rows < block.col)
        rows, cols, sims = rows[keep], block.col[keep], block.data[keep]
        one_outlet = (source_id[rows] == source_id[cols]) & (source_id[rows] >= 0)
        same = (
            (sims >= np.where(one_outlet, same_source_threshold, threshold))
            & (language_id[rows] == language_id[cols])
            & (np.abs(published[rows] - published[cols]) <= window)
        )
        for a, b in zip(rows[same].tolist(), cols[same].tolist(), strict=False):
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32)


def _source_quality(corpus: Corpus, source_keys: list[str], source_id: np.ndarray) -> np.ndarray:
    """Map outlet popularity rank (log scale) to 0.05..1; unknown outlets get 0.3."""
    ranks = np.array([corpus.sources[key].priority or np.nan for key in source_keys], dtype=np.float64)
    per_source = np.full(len(source_keys), UNKNOWN_SOURCE_QUALITY, dtype=np.float32)
    known = ~np.isnan(ranks)
    if known.any():
        logs = np.log10(ranks[known])
        low, high = float(logs.min()), float(np.percentile(logs, 95))
        span = max(high - low, 1e-9)
        per_source[known] = np.clip(1.0 - (logs - low) / span, 0.05, 1.0)
    quality = np.full(len(source_id), UNKNOWN_SOURCE_QUALITY, dtype=np.float32)
    has_source = source_id >= 0
    quality[has_source] = per_source[source_id[has_source]]
    return quality


def _popularity(corpus: Corpus, coverage: np.ndarray) -> np.ndarray:
    """Blend reader engagement with how many outlets picked the story up."""
    engagement = np.array([a.engagement.score for a in corpus.articles], dtype=np.float64)
    coverage_part = np.clip(np.log(coverage) / np.log(6.0), 0.0, 1.0)
    if engagement.max(initial=0.0) <= 0:
        return coverage_part.astype(np.float32)
    ceiling = max(float(np.percentile(engagement[engagement > 0], 99)), 1.0)
    engagement_part = np.clip(np.log1p(engagement) / np.log1p(ceiling), 0.0, 1.0)
    return (0.6 * coverage_part + 0.4 * engagement_part).astype(np.float32)


def build_index(corpus: Corpus, as_of: str = "latest") -> CorpusIndex:
    """Vectorise the corpus and precompute every reader-independent signal."""
    started = time.perf_counter()
    articles = corpus.articles
    n = len(articles)
    if n == 0:
        raise ValueError("cannot index an empty corpus")
    if len(corpus.documents) != n:
        raise ValueError("corpus.documents must be parallel to corpus.articles")

    vectorizer = make_vectorizer(n)
    matrix = vectorizer.fit_transform(corpus.documents).tocsr()
    matrix.sort_indices()

    category_keys = sorted({key for a in articles for key in a.categories})
    tag_keys = sorted({key for a in articles for key in a.tags})
    source_keys = sorted({a.source for a in articles if a.source})
    languages = sorted({a.language for a in articles})
    category_pos = {key: j for j, key in enumerate(category_keys)}
    tag_pos = {key: j for j, key in enumerate(tag_keys)}
    source_pos = {key: j for j, key in enumerate(source_keys)}
    language_pos = {key: j for j, key in enumerate(languages)}

    category_matrix = np.zeros((n, len(category_keys)), dtype=np.float32)
    tag_matrix = np.zeros((n, len(tag_keys)), dtype=np.float32)
    language_id = np.zeros(n, dtype=np.int32)
    source_id = np.full(n, -1, dtype=np.int32)
    published = np.zeros(n, dtype=np.float64)
    for i, article in enumerate(articles):
        for key in article.categories:
            category_matrix[i, category_pos[key]] = 1.0
        for key in article.tags:
            tag_matrix[i, tag_pos[key]] = 1.0
        language_id[i] = language_pos[article.language]
        if article.source:
            source_id[i] = source_pos[article.source]
        published[i] = article.published.timestamp()

    story = cluster_stories(matrix, published, language_id, source_id)
    # Coverage = number of distinct outlets reporting the story.
    coverage = np.ones(n, dtype=np.int32)
    order = np.argsort(story, kind="stable")
    boundaries = np.flatnonzero(np.diff(story[order])) + 1
    for members in np.split(order, boundaries):
        if len(members) > 1:
            outlets = {int(s) for s in source_id[members] if s >= 0}
            coverage[members] = max(len(outlets), 1)

    corpus.documents = []  # only needed for vectorising; keep the cache small
    return CorpusIndex(
        corpus=corpus,
        as_of=resolve_as_of(as_of, published),
        vectorizer=vectorizer,
        matrix=matrix,
        category_keys=category_keys,
        tag_keys=tag_keys,
        source_keys=source_keys,
        languages=languages,
        category_matrix=category_matrix,
        tag_matrix=tag_matrix,
        language_id=language_id,
        source_id=source_id,
        published=published,
        source_quality=_source_quality(corpus, source_keys, source_id),
        popularity=_popularity(corpus, coverage),
        story=story,
        coverage=coverage,
        position={a.id: i for i, a in enumerate(articles)},
        build_seconds=time.perf_counter() - started,
    )
