"""Reader profiles: stated preferences plus behaviour, turned into affinities.

A :class:`Profile` is plain data (what the reader told us and what they did).
:func:`build_state` projects it onto a :class:`~fanlight.index.CorpusIndex` to
get the numeric affinities the ranker consumes.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import RankingConfig
from .index import CorpusIndex
from .ingest import RawData, oid, parse_datetime
from .models import Corpus

#: Evidence weight of each action. ``view`` is an impression: exposure, no vote.
ACTIONS: dict[str, float] = {
    "view": 0.0,
    "open": 1.0,
    "like": 2.0,
    "save": 2.5,
    "hide": -1.0,
}

#: Scale of negative evidence relative to positive. At 1.0 a "hide" cancels an
#: "open": hiding one story says "not this one" more than "never this category".
NEGATIVE_STRENGTH = 1.0
#: Evidence worth of a stated interest (one "open" is worth 1 before damping).
EXPLICIT_STRENGTH = 1.5
#: Evidence scale at which affinity reaches 1 - 1/e (about 0.63).
SATURATION = 1.5
#: Positive stories needed before the content signal is trusted halfway.
CONTENT_HALF_CONFIDENCE = 3.0
#: A category counts as "avoided" in summaries below this affinity.
AVOIDED_BELOW = -0.3
#: How strongly hidden stories pull the content vector away from similar text.
NEGATIVE_CONTENT = 0.5


@dataclass(frozen=True)
class Event:
    article_id: str
    action: str
    ts: float


@dataclass
class Profile:
    id: str
    name: str = ""
    kind: str = "reader"  # "reader" (created in the app) or "dataset" (imported)
    languages: tuple[str, ...] = ()
    #: Stated interests, key -> strength in (0, 1].
    categories: dict[str, float] = field(default_factory=dict)
    tags: dict[str, float] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    events: list[Event] = field(default_factory=list)


@dataclass
class ProfileState:
    """A profile projected onto an index. All arrays align with index columns."""

    category_affinity: np.ndarray
    tag_affinity: np.ndarray
    source_affinity: np.ndarray
    category_exposure: np.ndarray
    content_vector: np.ndarray | None
    views: dict[int, int]
    read: set[int]
    hidden: set[int]
    learned_categories: np.ndarray
    positives: int
    negatives: int

    @property
    def content_confidence(self) -> float:
        """0..1: how much the taste vector can be trusted, given the evidence behind it."""
        if self.content_vector is None:
            return 0.0
        return self.positives / (self.positives + CONTENT_HALF_CONFIDENCE)

    @property
    def has_category_signal(self) -> bool:
        return bool(np.any(self.category_affinity != 0))

    @property
    def has_tag_signal(self) -> bool:
        return bool(np.any(self.tag_affinity != 0))

    @property
    def has_source_signal(self) -> bool:
        return bool(np.any(self.source_affinity > 0))


def _affinity(explicit: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Combine stated interest with log-damped behaviour into [-1, 1].

    Evidence is ``1.5 * stated + log(1 + positive) - log(1 + negative)`` and
    affinity is ``sign(e) * (1 - exp(-|e| / tau))``. Two properties matter:

    * Log damping: the 30th click on a category adds far less than the first,
      so one heavy interest cannot drown out everything else.
    * Each category saturates on its own. Reading a lot of Technology never
      lowers a reader's affinity for Politics; only hiding things does.
    """
    evidence = EXPLICIT_STRENGTH * explicit + np.log1p(positive) - NEGATIVE_STRENGTH * np.log1p(negative)
    affinity = np.sign(evidence) * (1.0 - np.exp(-np.abs(evidence) / SATURATION))
    return affinity.astype(np.float32)


def build_state(
    index: CorpusIndex,
    profile: Profile,
    config: RankingConfig | None = None,
    now: float | None = None,
) -> ProfileState:
    """Project ``profile`` onto ``index``. ``now`` is the clock for event decay."""
    config = config or RankingConfig()
    now = time.time() if now is None else now
    half_life = config.event_half_life_days * 86_400.0

    n_categories, n_tags, n_sources = (
        len(index.category_keys),
        len(index.tag_keys),
        len(index.source_keys),
    )
    category_pos = {key: j for j, key in enumerate(index.category_keys)}
    tag_pos = {key: j for j, key in enumerate(index.tag_keys)}
    source_pos = {key: j for j, key in enumerate(index.source_keys)}

    explicit_categories = np.zeros(n_categories, dtype=np.float64)
    for key, weight in profile.categories.items():
        if key in category_pos:
            explicit_categories[category_pos[key]] = min(max(float(weight), 0.0), 1.0)
    explicit_tags = np.zeros(n_tags, dtype=np.float64)
    for key, weight in profile.tags.items():
        if key in tag_pos:
            explicit_tags[tag_pos[key]] = min(max(float(weight), 0.0), 1.0)
    explicit_sources = np.zeros(n_sources, dtype=np.float64)
    for key in profile.sources:
        if key in source_pos:
            explicit_sources[source_pos[key]] = 1.0

    pos_cat, neg_cat = np.zeros(n_categories), np.zeros(n_categories)
    pos_tag, neg_tag = np.zeros(n_tags), np.zeros(n_tags)
    pos_src = np.zeros(n_sources)
    exposure = np.zeros(n_categories)
    views: dict[int, int] = {}
    read: set[int] = set()
    hidden: set[int] = set()
    pos_rows: dict[int, float] = {}
    neg_rows: dict[int, float] = {}
    counted: set[tuple[int, str]] = set()

    for event in profile.events:
        i = index.position.get(event.article_id)
        weight = ACTIONS.get(event.action)
        if i is None or weight is None:
            continue
        if event.action == "view":
            views[i] = views.get(i, 0) + 1
            exposure += index.category_matrix[i]
            continue
        if (i, event.action) in counted:  # repeating an action adds no evidence
            continue
        counted.add((i, event.action))
        evidence = abs(weight) * 0.5 ** (max(now - event.ts, 0.0) / half_life)
        if weight > 0:
            read.add(i)
            pos_cat += evidence * index.category_matrix[i]
            pos_tag += evidence * index.tag_matrix[i]
            if index.source_id[i] >= 0:
                pos_src[index.source_id[i]] += evidence
            pos_rows[i] = pos_rows.get(i, 0.0) + evidence
        else:
            hidden.update(int(j) for j in index.story_members(i))
            neg_cat += evidence * index.category_matrix[i]
            neg_tag += evidence * index.tag_matrix[i]
            neg_rows[i] = neg_rows.get(i, 0.0) + evidence

    content_vector = None
    if pos_rows:
        rows = np.fromiter(pos_rows.keys(), dtype=np.int64)
        weights = np.fromiter(pos_rows.values(), dtype=np.float64)
        vector = np.asarray(index.matrix[rows].T @ (weights / weights.sum())).ravel()
        if neg_rows:
            rows = np.fromiter(neg_rows.keys(), dtype=np.int64)
            weights = np.fromiter(neg_rows.values(), dtype=np.float64)
            vector -= NEGATIVE_CONTENT * np.asarray(index.matrix[rows].T @ (weights / weights.sum())).ravel()
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            content_vector = (vector / norm).astype(np.float32)

    return ProfileState(
        category_affinity=_affinity(explicit_categories, pos_cat, neg_cat),
        tag_affinity=_affinity(explicit_tags, pos_tag, neg_tag),
        source_affinity=np.clip(_affinity(explicit_sources, pos_src, np.zeros(n_sources)), 0.0, 1.0),
        category_exposure=exposure,
        content_vector=content_vector,
        views=views,
        read=read,
        hidden=hidden,
        learned_categories=pos_cat,
        positives=len(pos_rows),
        negatives=len(neg_rows),
    )


def top_interests(
    index: CorpusIndex, state: ProfileState, profile: Profile, limit: int = 8
) -> dict[str, list[dict[str, Any]]]:
    """Human-readable view of what the engine believes the reader likes."""
    corpus = index.corpus

    def rows(keys: list[str], affinity: np.ndarray, names: Mapping[str, Any], stated: Mapping) -> list:
        order = np.argsort(-affinity, kind="stable")
        out = []
        for j in order[:limit]:
            if affinity[j] <= 0:
                break
            key = keys[j]
            out.append(
                {
                    "key": key,
                    "name": names[key].name if key in names else key,
                    "affinity": round(float(affinity[j]), 3),
                    "stated": key in stated,
                }
            )
        return out

    avoided = [
        {"key": index.category_keys[j], "name": corpus.categories[index.category_keys[j]].name}
        for j in np.flatnonzero(state.category_affinity <= AVOIDED_BELOW)
    ]
    return {
        "categories": rows(
            index.category_keys, state.category_affinity, corpus.categories, profile.categories
        ),
        "tags": rows(index.tag_keys, state.tag_affinity, corpus.tags, profile.tags),
        "avoided": avoided,
    }


# --------------------------------------------------------------------------- #
# Profiles from the imported user_preferences collection
# --------------------------------------------------------------------------- #


def public_id(raw_user_id: str) -> str:
    """Stable, non-reversible handle for an imported user."""
    return "u_" + hashlib.sha1(raw_user_id.encode("utf-8")).hexdigest()[:10]


def dataset_profiles(raw: RawData, corpus: Corpus) -> list[Profile]:
    """Build one profile per user from ``user_preferences``.

    Users can have several preference documents; the most recently updated one
    wins. A category repeated in the list counts as a stronger interest.
    """
    latest: dict[str, tuple[float, Mapping[str, Any]]] = {}
    for row in raw.get("user_preferences", []):
        user_id = oid(row.get("user_id"))
        if not user_id:
            continue
        updated = parse_datetime(row.get("updatedAt") or row.get("createdAt"))
        stamp = updated.timestamp() if updated else 0.0
        if user_id not in latest or stamp >= latest[user_id][0]:
            latest[user_id] = (stamp, row)

    profiles: list[Profile] = []
    for user_id, (_, row) in latest.items():
        counts = Counter(
            key
            for key in (
                corpus.category_aliases.get(oid(ref) or "") for ref in _as_list(row.get("article_category"))
            )
            if key
        )
        categories: dict[str, float] = {}
        if counts:
            top = math.log1p(max(counts.values()))
            # Floor at 0.35: a category picked once still clearly beats "not picked".
            categories = {key: round(max(math.log1p(count) / top, 0.35), 4) for key, count in counts.items()}
        tag_keys = {
            key
            for key in (corpus.tag_aliases.get(oid(ref) or "") for ref in _as_list(row.get("article_tag")))
            if key
        }
        sources = tuple(
            sorted(
                {
                    key
                    for key in (
                        corpus.source_aliases.get(oid(ref) or "") for ref in _as_list(row.get("source"))
                    )
                    if key
                }
            )
        )
        languages = _languages(row.get("language"))
        pid = public_id(user_id)
        profiles.append(
            Profile(
                id=pid,
                name=f"Reader {pid[2:6].upper()}",
                kind="dataset",
                languages=languages,
                categories=categories,
                tags=dict.fromkeys(sorted(tag_keys), 1.0),
                sources=sources,
            )
        )
    profiles.sort(key=lambda p: p.id)
    return profiles


def _as_list(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _languages(value: Any) -> tuple[str, ...]:
    out = []
    for item in _as_list(value):
        text = str(item).strip().lower()
        if text and text not in out:
            out.append(text)
    return tuple(out) or ("english",)
