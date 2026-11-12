"""Score fusion, diversity-aware re-ranking and discovery slots.

Ranking happens in three explainable steps:

1. **Signals.** Six signals, each in ``[0, 1]`` (category and tags can go
   negative for things the reader hid), are computed for every article.
2. **Fusion.** A weighted average over the signals that carry information for
   this reader. A signal with nothing to say (for example ``content`` before
   the first click) is left out and its weight is shared by the others, so
   scores stay comparable between new and established readers.
3. **Selection.** A greedy pass fills the feed one slot at a time. Regular
   slots trade score against redundancy with what is already above (similar
   text, or the same category or outlet in the last few slots). A fixed share
   of slots is reserved for *discovery*: the best general-interest story from
   a category the reader has not engaged with, rotating towards categories
   they have seen least. Without those slots a personalised feed only ever
   confirms what it already believes. One article per story survives.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import SIGNALS, RankingConfig
from .index import CorpusIndex
from .profiles import Profile, ProfileState, build_state

#: Redundancy of repeating a category / an outlet across the recent slots,
#: relative to repeating the text itself (1.0).
SAME_CATEGORY = 0.7
SAME_SOURCE = 1.0
#: ``diversity`` = 1 maps to this trade-off weight in the greedy objective.
MAX_LAMBDA = 0.5
#: Freshness mixes a fast decay (the configured half-life) with a slow one, so
#: last week still beats last month once both are "old".
SLOW_DECAY_SHARE = 0.3
SLOW_DECAY_FACTOR = 10.0
#: Unanswered impressions stop counting against a story after this many.
MAX_SEEN = 6
#: A story is a discovery candidate if the reader's affinity for its best
#: category is below this (and not negative).
DISCOVERY_BELOW = 0.25
#: How many discovery candidates the selector considers.
DISCOVERY_POOL = 120
#: Discount for a discovery pick from a category already used by an earlier one.
DISCOVERY_REPEAT = 0.25
#: Outlet affinity at which a story is explained as "from an outlet you read"
#: (about three opens from that outlet).
FAVOURITE_OUTLET = 0.65
#: A discovery pick must not be this similar to something already in the feed.
DISCOVERY_MAX_SIMILARITY = 0.5


@dataclass
class RankedItem:
    index: int
    rank: int
    base_rank: int
    score: float
    signals: dict[str, float]
    contributions: dict[str, float]
    reasons: list[str]
    discovery: bool = False
    related: list[int] = field(default_factory=list)


@dataclass
class RankedFeed:
    items: list[RankedItem]
    eligible: int
    active_signals: list[str]
    weights: dict[str, float]
    diagnostics: dict[str, Any]
    elapsed_ms: float


@dataclass
class ScoreTable:
    """Per-article signals and fused score for one reader."""

    signals: dict[str, np.ndarray]
    weights: dict[str, float]
    active: list[str]
    score: np.ndarray
    eligible: np.ndarray
    best_category: np.ndarray
    #: General appeal times how little the reader has seen of the category.
    discovery: np.ndarray
    discoverable: np.ndarray


def _row_max(membership: np.ndarray, affinity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: max affinity over the columns it belongs to (0 if none) and its argmax."""
    if membership.shape[1] == 0:
        zeros = np.zeros(membership.shape[0], dtype=np.float32)
        return zeros, np.full(membership.shape[0], -1, dtype=np.int64)
    masked = np.where(membership > 0, affinity[None, :], -np.inf)
    arg = masked.argmax(axis=1)
    best = masked[np.arange(len(arg)), arg]
    empty = ~np.isfinite(best)
    best = np.where(empty, 0.0, best)
    arg = np.where(empty, -1, arg)
    return best.astype(np.float32), arg


def discovery_slots(k: int, share: float) -> set[int]:
    """0-based feed positions reserved for discovery, spread evenly.

    The first one sits half a period in, so a short feed still gets one and
    the top slot always stays with the reader's known interests.
    """
    if share <= 0 or k <= 1:
        return set()
    period = 1.0 / share
    slots, position = set(), period / 2.0
    while position < k:
        slots.add(int(position))
        position += period
    slots.discard(0)
    return slots


class Ranker:
    """Ranks the articles of one index for any profile."""

    def __init__(self, index: CorpusIndex, config: RankingConfig | None = None):
        self.index = index
        self.config = config or RankingConfig()
        self._prominent_cut = float(np.percentile(index.source_quality, 75)) if index.size else 1.0

    # ------------------------------------------------------------------ #
    # Steps 1 + 2: signals and fusion
    # ------------------------------------------------------------------ #

    def score(
        self,
        profile: Profile,
        state: ProfileState,
        config: RankingConfig,
        categories: Iterable[str] | None = None,
        min_results: int = 1,
    ) -> ScoreTable:
        index = self.index
        n = index.size
        membership = index.category_matrix
        counts = membership.sum(axis=1)

        # --- category: mostly the best-matching category, a little the average
        cat_max, best_category = _row_max(membership, state.category_affinity)
        cat_mean = (membership @ state.category_affinity) / np.maximum(counts, 1.0)
        category = 0.7 * cat_max + 0.3 * cat_mean

        # --- tags
        tags, _ = _row_max(index.tag_matrix, state.tag_affinity)

        # --- eligibility (needed before content calibration)
        eligible = index.published <= index.as_of + 1e-6
        if profile.languages:
            allowed = [j for j, name in enumerate(index.languages) if name in profile.languages]
            eligible &= np.isin(index.language_id, allowed)
        if categories:
            wanted_keys = set(categories)
            wanted = [j for j, key in enumerate(index.category_keys) if key in wanted_keys]
            eligible &= membership[:, wanted].sum(axis=1) > 0
        if state.hidden:
            eligible[list(state.hidden)] = False
        if config.hide_read and state.read:
            eligible[list(state.read)] = False
        if config.max_age_days is not None:
            recent = eligible & (index.published >= index.as_of - config.max_age_days * 86_400.0)
            # A quiet corpus (or a narrow filter) should not produce an empty feed.
            if int(recent.sum()) >= min_results:
                eligible = recent

        # --- content: cosine to the reader's taste vector, calibrated so the
        # closest unread stories approach 1 regardless of how sparse the text
        # is, then scaled by how much evidence stands behind the vector
        content = np.zeros(n, dtype=np.float32)
        if state.content_vector is not None:
            cosine = np.asarray(index.matrix @ state.content_vector).ravel()
            pool = cosine[eligible] if eligible.any() else cosine
            scale = float(np.percentile(pool, 99.5)) if len(pool) else 0.0
            if scale > 1e-6:
                content = (state.content_confidence * np.clip(cosine / scale, 0.0, 1.0)).astype(np.float32)

        # --- freshness: two-speed exponential decay from the index's reference time
        age_hours = np.maximum(index.as_of - index.published, 0.0) / 3600.0
        fast = np.power(0.5, age_hours / config.half_life_hours)
        slow = np.power(0.5, age_hours / (SLOW_DECAY_FACTOR * config.half_life_hours))
        freshness = ((1.0 - SLOW_DECAY_SHARE) * fast + SLOW_DECAY_SHARE * slow).astype(np.float32)

        # --- source: outlet reach, blended with outlets the reader favours
        source = index.source_quality.copy()
        if state.has_source_signal:
            personal = np.zeros(n, dtype=np.float32)
            known = index.source_id >= 0
            personal[known] = state.source_affinity[index.source_id[known]]
            source = 0.5 * source + 0.5 * personal

        signals = {
            "category": category.astype(np.float32),
            "tags": tags,
            "content": content,
            "freshness": freshness,
            "source": source.astype(np.float32),
            "popularity": index.popularity,
        }
        informative = {
            "category": state.has_category_signal,
            "tags": state.has_tag_signal,
            "content": state.content_vector is not None,
            "freshness": True,
            "source": True,
            "popularity": bool(np.any(index.popularity > 0)),
        }
        raw = config.weights.as_dict()
        active = [s for s in SIGNALS if informative[s] and raw[s] > 0]
        total = sum(raw[s] for s in active)
        weights = {s: (raw[s] / total if s in active and total > 0 else 0.0) for s in SIGNALS}

        score = np.zeros(n, dtype=np.float64)
        for name in active:
            score += weights[name] * signals[name]
        fatigue = np.ones(n, dtype=np.float64)
        for i, count in state.views.items():
            if i not in state.read:
                fatigue[i] = config.seen_decay ** min(count, MAX_SEEN)
        score = np.clip(score * fatigue, 0.0, 1.0)

        # --- discovery: general appeal (no personal signals) of stories from
        # categories the reader has not engaged with, favouring the least seen
        general = {s: raw[s] for s in ("freshness", "source", "popularity")}
        general_total = sum(general.values()) or 1.0
        appeal = sum(general[s] * signals[s] for s in general) / general_total
        openness = 1.0 / (1.0 + state.category_exposure / 2.0)
        category_openness = np.where(counts > 0, (membership @ openness) / np.maximum(counts, 1.0), 0.5)
        discovery = appeal * category_openness * fatigue
        if state.has_category_signal:
            discoverable = eligible & (cat_max < DISCOVERY_BELOW) & (cat_max >= 0.0)
        else:
            discoverable = np.zeros(n, dtype=bool)

        return ScoreTable(signals, weights, active, score, eligible, best_category, discovery, discoverable)

    # ------------------------------------------------------------------ #
    # Step 3: greedy selection
    # ------------------------------------------------------------------ #

    def _select(
        self,
        pool: np.ndarray,
        scores: np.ndarray,
        discovery: np.ndarray,
        discoverable: np.ndarray,
        k: int,
        config: RankingConfig,
    ) -> list[tuple[int, bool]]:
        """Greedy pick of ``k`` pool positions; returns (position, is_discovery)."""
        index = self.index
        k = min(k, len(pool))
        if k == 0:
            return []
        trade = MAX_LAMBDA * config.diversity
        reserved = discovery_slots(k, config.discovery) if discoverable.any() else set()
        if trade <= 0 and not reserved:
            return [(int(p), False) for p in np.argsort(-scores, kind="stable")[:k]]

        vectors = index.matrix[pool]
        similarity = np.clip((vectors @ vectors.T).toarray(), 0.0, 1.0)
        categories = index.category_matrix[pool]
        shares_category = ((categories @ categories.T) > 0).astype(np.float64)
        sources = index.source_id[pool]
        same_source = ((sources[:, None] == sources[None, :]) & (sources[:, None] >= 0)).astype(np.float64)

        window = config.repeat_window
        decay = np.linspace(1.0, 1.0 / window, window)  # the slot just above matters most
        chosen: list[tuple[int, bool]] = []
        available = np.ones(len(pool), dtype=bool)
        text_redundancy = np.zeros(len(pool))
        discovered = np.zeros(len(pool))  # shares a category with an earlier discovery
        for slot in range(k):
            pick = -1
            if slot in reserved:
                usable = available & discoverable & (text_redundancy < DISCOVERY_MAX_SIMILARITY)
                if usable.any():
                    value = discovery * np.where(discovered > 0, DISCOVERY_REPEAT, 1.0)
                    pick = int(np.where(usable, value, -np.inf).argmax())
                    discovered = np.maximum(discovered, shares_category[pick])
            is_discovery = pick >= 0
            if pick < 0:
                redundancy = text_redundancy
                recent = [p for p, _ in chosen[-window:]][::-1]
                if recent and trade > 0:
                    weights = decay[: len(recent)][:, None] / decay.sum()
                    redundancy = np.maximum.reduce(
                        [
                            text_redundancy,
                            SAME_CATEGORY * (weights * shares_category[recent]).sum(axis=0),
                            SAME_SOURCE * (weights * same_source[recent]).sum(axis=0),
                        ]
                    )
                objective = (1.0 - trade) * scores - trade * redundancy
                pick = int(np.where(available, objective, -np.inf).argmax())
            chosen.append((pick, is_discovery))
            available[pick] = False
            text_redundancy = np.maximum(text_redundancy, similarity[pick])
        return chosen

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def rank(
        self,
        profile: Profile,
        k: int = 20,
        config: RankingConfig | None = None,
        categories: Iterable[str] | None = None,
        now: float | None = None,
        state: ProfileState | None = None,
    ) -> RankedFeed:
        """Return the top ``k`` stories for ``profile``."""
        started = time.perf_counter()
        config = config or self.config
        index = self.index
        state = state or build_state(index, profile, config, now=now)
        table = self.score(profile, state, config, categories, min_results=k)

        candidates = np.flatnonzero(table.eligible)
        order = candidates[np.argsort(-table.score[candidates], kind="stable")]
        # One article per story: the best-scoring report represents the story.
        _, first = np.unique(index.story[order], return_index=True)
        order = order[np.sort(first)]
        base_rank = {int(i): r for r, i in enumerate(order, start=1)}

        pool = order[: max(config.pool_size, k)]
        if config.discovery > 0 and not categories:
            extra = order[table.discoverable[order]]
            extra = extra[np.argsort(-table.discovery[extra], kind="stable")][:DISCOVERY_POOL]
            pool = np.concatenate([pool, extra[~np.isin(extra, pool)]])
            discoverable = table.discoverable[pool]
        else:
            discoverable = np.zeros(len(pool), dtype=bool)

        picks = self._select(pool, table.score[pool], table.discovery[pool], discoverable, k, config)
        items = []
        for rank, (position, is_discovery) in enumerate(picks, start=1):
            i = int(pool[position])
            signals = {s: round(float(table.signals[s][i]), 4) for s in SIGNALS}
            contributions = {s: round(table.weights[s] * float(table.signals[s][i]), 4) for s in table.active}
            items.append(
                RankedItem(
                    index=i,
                    rank=rank,
                    base_rank=base_rank[i],
                    score=round(float(table.score[i]), 4),
                    signals=signals,
                    contributions=contributions,
                    reasons=self._reasons(i, table, state, profile, is_discovery),
                    discovery=is_discovery,
                    related=[int(j) for j in index.story_members(i) if j != i],
                )
            )

        return RankedFeed(
            items=items,
            eligible=len(order),
            active_signals=table.active,
            weights={s: round(w, 4) for s, w in table.weights.items()},
            diagnostics=self.diagnostics([item.index for item in items]),
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )

    def diagnostics(self, selected: list[int]) -> dict[str, Any]:
        """Shape of a feed: variety of categories and outlets, age, redundancy."""
        index = self.index
        if not selected:
            return {
                "categories": [],
                "sources": 0,
                "median_age_hours": None,
                "similarity": None,
                "sentiment": {},
            }
        corpus = index.corpus
        mix: Counter[str] = Counter()
        for i in selected:
            article = index.article(i)
            mix[article.categories[0] if article.categories else ""] += 1
        vectors = index.matrix[selected]
        sims = (vectors @ vectors.T).toarray()
        n = len(selected)
        off_diagonal = (sims.sum() - np.trace(sims)) / (n * (n - 1)) if n > 1 else 0.0
        ages = (index.as_of - index.published[selected]) / 3600.0
        sentiment = Counter(index.article(i).sentiment or "unknown" for i in selected)
        return {
            "categories": [
                {
                    "key": key,
                    "name": corpus.categories[key].name if key in corpus.categories else "General",
                    "count": count,
                }
                for key, count in mix.most_common()
            ],
            "sources": len({index.article(i).source for i in selected if index.article(i).source}),
            "median_age_hours": round(float(np.median(ages)), 1),
            "similarity": round(float(off_diagonal), 4),
            "sentiment": dict(sentiment),
        }

    # ------------------------------------------------------------------ #
    # Explanations
    # ------------------------------------------------------------------ #

    def _reasons(
        self,
        i: int,
        table: ScoreTable,
        state: ProfileState,
        profile: Profile,
        is_discovery: bool,
    ) -> list[str]:
        """Up to three plain-language reasons, strongest contribution first."""
        index = self.index
        corpus = index.corpus
        article = index.article(i)
        value = {s: float(table.signals[s][i]) for s in SIGNALS}
        notes: dict[str, str] = {}

        j = int(table.best_category[i])
        if j >= 0 and value["category"] >= 0.3:
            key = index.category_keys[j]
            name = corpus.categories[key].name if key in corpus.categories else key
            learned = state.learned_categories[j] > 0
            if key in profile.categories and not learned:
                notes["category"] = f"You follow {name}"
            else:
                notes["category"] = f"You often read {name}"
        if value["tags"] >= 0.3 and article.tags:
            tag_pos = {key: t for t, key in enumerate(index.tag_keys)}
            best = max(article.tags, key=lambda key: state.tag_affinity[tag_pos[key]])
            notes["tags"] = f"Tagged {corpus.tags[best].name}, a topic you read"
        if value["content"] >= 0.45:
            notes["content"] = "Close to stories you opened"
        age_hours = max(index.as_of - index.published[i], 0.0) / 3600.0
        if age_hours < 1:
            notes["freshness"] = "Published in the last hour"
        elif age_hours <= 24:
            hours = round(age_hours)
            notes["freshness"] = f"Published {hours} hour{'s' if hours != 1 else ''} ago"
        if index.coverage[i] >= 2:
            notes["popularity"] = f"Reported by {int(index.coverage[i])} outlets"
        elif value["popularity"] >= 0.5:
            notes["popularity"] = "Readers engaged with this story"
        source_j = int(index.source_id[i])
        if source_j >= 0 and state.source_affinity[source_j] >= FAVOURITE_OUTLET:
            notes["source"] = "From an outlet you read"
        elif value["source"] >= 0.75 and index.source_quality[i] > self._prominent_cut:
            # Only worth saying when it sets the story apart: in a feed of major
            # outlets every story would qualify.
            notes["source"] = "From a widely read outlet"

        ranked = sorted(
            notes,
            key=lambda s: table.weights.get(s, 0.0) * abs(value[s]),
            reverse=True,
        )
        reasons = [notes[s] for s in ranked]
        if is_discovery:
            reasons.insert(0, "Something outside your usual reading")
        return reasons[:3]
