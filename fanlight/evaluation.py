"""Offline evaluation by simulated readers.

There is no click log for the dataset, so ranking quality cannot be measured
against real behaviour. Instead each simulated reader gets a *hidden* taste
(a few categories with graded strengths) and tells the system about only one
of them. Round after round the reader scans the feed, opens what matches the
hidden taste and occasionally hides what does not; the ranker sees nothing
but those interactions.

What this measures: whether the system learns unstated interests from
feedback, and what diversity and freshness it trades for that. What it does
not measure: real click-through. Ground truth is defined on the dataset's
editorial category labels, never on the text features the ranker uses, so the
content signal gets no free credit.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .config import RankingConfig
from .index import CorpusIndex
from .profiles import Event, Profile
from .ranker import Ranker

Policy = Callable[[Profile, int], list[int]]

#: Stories older than this are not news any more: they earn no relevance.
NEWS_WINDOW_DAYS = 14.0
#: Chance the reader looks at slot p is ``ATTENTION ** (p - 1)``.
ATTENTION = 0.95
#: Chance an examined story is opened, per unit of hidden relevance.
OPEN_RATE = 0.9
#: Chance an examined, irrelevant story is hidden.
HIDE_RATE = 0.1


@dataclass
class Persona:
    language: str
    taste: dict[str, float]  # hidden: category key -> relevance in (0, 1]
    stated: str  # the single category the reader tells the system about


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(pos + 2) for pos, g in enumerate(gains))


def make_personas(index: CorpusIndex, n: int, seed: int = 0, min_articles: int = 25) -> list[Persona]:
    """Sample readers whose hidden tastes are supported by enough recent articles."""
    rng = random.Random(seed)
    recent = index.published >= index.as_of - NEWS_WINDOW_DAYS * 86_400.0
    personas: list[Persona] = []
    by_language: dict[str, list[str]] = {}
    for j, language in enumerate(index.languages):
        mask = recent & (index.language_id == j)
        counts = index.category_matrix[mask].sum(axis=0)
        usable = [index.category_keys[c] for c in np.flatnonzero(counts >= min_articles)]
        if len(usable) >= 4:
            by_language[language] = usable
    if not by_language:
        raise ValueError("corpus too small to simulate readers")
    languages = sorted(by_language)
    shares = [float((index.language_id == index.languages.index(name)).sum()) for name in languages]
    for _ in range(n):
        language = rng.choices(languages, weights=shares)[0]
        picked = rng.sample(by_language[language], rng.randint(2, min(4, len(by_language[language]))))
        strengths = [1.0, 0.7, 0.5, 0.35][: len(picked)]
        personas.append(Persona(language, dict(zip(picked, strengths, strict=False)), stated=picked[0]))
    return personas


def relevance(index: CorpusIndex, persona: Persona) -> np.ndarray:
    """Hidden graded relevance of every article for a persona.

    Zero for other languages and for anything outside the news window.
    """
    gains = np.zeros(index.size, dtype=np.float64)
    for key, strength in persona.taste.items():
        column = index.category_keys.index(key)
        gains = np.maximum(gains, strength * index.category_matrix[:, column])
    language = index.languages.index(persona.language)
    gains[index.language_id != language] = 0.0
    gains[index.published < index.as_of - NEWS_WINDOW_DAYS * 86_400.0] = 0.0
    return gains


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #


def _unread(index: CorpusIndex, profile: Profile) -> np.ndarray:
    mask = np.ones(index.size, dtype=bool)
    for event in profile.events:
        if event.action != "view":
            i = index.position.get(event.article_id)
            if i is not None:
                mask[i] = False
    return mask


def fanlight_policy(index: CorpusIndex, config: RankingConfig, now: float) -> Policy:
    ranker = Ranker(index, config)

    def policy(profile: Profile, k: int) -> list[int]:
        return [item.index for item in ranker.rank(profile, k=k, now=now).items]

    return policy


def v1_policy(index: CorpusIndex) -> Policy:
    """The heuristic that v1's XGBoost model was fitted to (its expected label).

    v1 trained a regressor on synthetic labels drawn from a hand-written
    formula, so in expectation the model reproduced the formula: language
    match and stated-category overlap, times a linear 180-day freshness ramp,
    times how common the article's categories are in the corpus. It never used
    behaviour, so it cannot learn.
    """
    frequency = index.category_matrix.sum(axis=0)
    frequency = frequency / max(float(frequency.max()), 1.0)
    counts = np.maximum(index.category_matrix.sum(axis=1), 1.0)
    category_frequency = (index.category_matrix @ frequency) / counts
    age_days = (index.published.max() - index.published) / 86_400.0
    freshness = np.clip(0.9 - (age_days / 180.0) * 0.6, 0.3, 0.9)

    def policy(profile: Profile, k: int) -> list[int]:
        stated = np.array([key in profile.categories for key in index.category_keys], dtype=np.float64)
        overlap = index.category_matrix @ stated
        match = np.isin(index.language_id, [index.languages.index(x) for x in profile.languages])
        engaged = 0.8 * (0.5 + 0.5 * np.minimum(overlap, 5) / 5.0)
        engagement = np.where(match & (overlap > 0), engaged, 0.2)
        score = engagement * freshness * category_frequency
        score[~_unread(index, profile)] = -np.inf
        return np.argsort(-score, kind="stable")[:k].tolist()

    return policy


def recency_policy(index: CorpusIndex) -> Policy:
    def policy(profile: Profile, k: int) -> list[int]:
        allowed = np.isin(index.language_id, [index.languages.index(x) for x in profile.languages])
        score = np.where(allowed & _unread(index, profile), index.published, -np.inf)
        return np.argsort(-score, kind="stable")[:k].tolist()

    return policy


def random_policy(index: CorpusIndex, seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)

    def policy(profile: Profile, k: int) -> list[int]:
        allowed = np.isin(index.language_id, [index.languages.index(x) for x in profile.languages])
        allowed &= index.published >= index.as_of - NEWS_WINDOW_DAYS * 86_400.0
        candidates = np.flatnonzero(allowed & _unread(index, profile))
        return rng.choice(candidates, size=min(k, len(candidates)), replace=False).tolist()

    return policy


def default_policies(
    index: CorpusIndex, config: RankingConfig | None = None, now: float = 0.0
) -> dict[str, Policy]:
    config = config or RankingConfig()
    return {
        "fanlight": fanlight_policy(index, config, now),
        "fanlight, no variety or discovery": fanlight_policy(
            index, replace(config, diversity=0.0, discovery=0.0), now
        ),
        "v1 label formula": v1_policy(index),
        "newest first": recency_policy(index),
        "random": random_policy(index),
    }


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def _feed_metrics(
    index: CorpusIndex, feed: list[int], gains: np.ndarray, unread: np.ndarray, persona: Persona
) -> dict[str, float]:
    got = [float(gains[i]) for i in feed]
    # Best achievable feed right now: the highest-gain stories not yet read.
    ideal = _dcg(sorted(gains[unread].tolist(), reverse=True)[: len(feed)])
    categories = Counter(index.article(i).categories[0] if index.article(i).categories else "" for i in feed)
    total = sum(categories.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in categories.values()) if total else 0.0
    shown = {key for i in feed for key in index.article(i).categories}
    return {
        "ndcg": _dcg(got) / ideal if ideal > 0 else 0.0,
        "precision": sum(g > 0 for g in got) / max(len(feed), 1),
        "interest_recall": len(shown & set(persona.taste)) / len(persona.taste),
        "category_entropy": entropy,
        "sources": float(len({index.article(i).source for i in feed if index.article(i).source})),
        "median_age_hours": float(np.median((index.as_of - index.published[feed]) / 3600.0)) if feed else 0.0,
    }


def simulate(
    index: CorpusIndex,
    n_readers: int = 40,
    rounds: int = 6,
    feed_size: int = 20,
    seed: int = 0,
    policies: dict[str, Policy] | None = None,
    config: RankingConfig | None = None,
) -> dict[str, Any]:
    """Run every policy against the same simulated readers.

    Returns per-policy metrics averaged over readers for each round; round 0
    is the feed before any interaction.
    """
    started = time.perf_counter()
    now = index.as_of  # simulated interactions happen "at" the corpus time
    policies = policies or default_policies(index, config, now)
    personas = make_personas(index, n_readers, seed)

    results: dict[str, Any] = {}
    for name, policy in policies.items():
        per_round: list[dict[str, list[float]]] = [dict() for _ in range(rounds + 1)]
        for p, persona in enumerate(personas):
            rng = random.Random(f"{seed}:{p}")  # same reader behaviour for every policy
            gains = relevance(index, persona)
            profile = Profile(id=f"sim-{p}", languages=(persona.language,), categories={persona.stated: 1.0})
            for step in range(rounds + 1):
                feed = policy(profile, feed_size)
                metrics = _feed_metrics(index, feed, gains, _unread(index, profile), persona)
                for metric, value in metrics.items():
                    per_round[step].setdefault(metric, []).append(value)
                if step == rounds:
                    break
                opened = 0
                for position, i in enumerate(feed):
                    if rng.random() > ATTENTION**position:
                        continue
                    article_id = index.article(i).id
                    profile.events.append(Event(article_id, "view", now))
                    if rng.random() < OPEN_RATE * gains[i]:
                        profile.events.append(Event(article_id, "open", now))
                        opened += 1
                    elif gains[i] == 0 and rng.random() < HIDE_RATE:
                        profile.events.append(Event(article_id, "hide", now))
                per_round[step].setdefault("opens", []).append(float(opened))
        results[name] = [
            {metric: round(float(np.mean(values)), 4) for metric, values in row.items()} for row in per_round
        ]

    return {
        "readers": len(personas),
        "rounds": rounds,
        "feed_size": feed_size,
        "seed": seed,
        "policies": results,
        "seconds": round(time.perf_counter() - started, 1),
    }


def representation_check(
    index: CorpusIndex, sample: int = 1500, neighbours: int = 10, seed: int = 0
) -> dict[str, Any]:
    """Does the text index capture topic? k-NN category agreement on real labels.

    For a sample of articles, look up their nearest neighbours by TF-IDF cosine
    and count how often a neighbour shares a category. The baseline is the
    agreement expected from picking articles at random.
    """
    rng = np.random.default_rng(seed)
    labelled = np.flatnonzero(index.category_matrix.sum(axis=1) > 0)
    if len(labelled) < neighbours + 2:
        return {"sample": 0, "agreement": None, "chance": None}
    rows = rng.choice(labelled, size=min(sample, len(labelled)), replace=False)
    sims = (index.matrix[rows] @ index.matrix.T).toarray()
    sims[np.arange(len(rows)), rows] = -1.0
    other_language = index.language_id[rows][:, None] != index.language_id[None, :]
    sims[other_language] = -1.0
    nearest = np.argpartition(-sims, neighbours, axis=1)[:, :neighbours]
    shared = np.einsum("rc,rnc->rn", index.category_matrix[rows], index.category_matrix[nearest]) > 0
    share = index.category_matrix[labelled].mean(axis=0)
    chance = float((index.category_matrix[rows] @ share).mean())
    return {
        "sample": len(rows),
        "neighbours": neighbours,
        "agreement": round(float(shared.mean()), 4),
        "chance": round(chance, 4),
    }


def format_report(report: dict[str, Any]) -> str:
    """Plain-text table: first vs. last round for each policy."""
    last = report["rounds"]
    lines = [
        f"Simulated readers: {report['readers']}, feed size {report['feed_size']}, "
        f"{last} rounds of feedback (seed {report['seed']})",
        "",
        f"{'policy':<34}{'nDCG start':>11}{'nDCG end':>10}{'opens/reader':>14}{'interests found':>17}"
        f"{'cat. entropy':>14}{'outlets':>9}{'median age h':>14}",
    ]
    for name, rows in report["policies"].items():
        first, final = rows[0], rows[last]
        opens = sum(row.get("opens", 0.0) for row in rows)
        lines.append(
            f"{name:<34}{first['ndcg']:>11.3f}{final['ndcg']:>10.3f}{opens:>14.1f}"
            f"{final['interest_recall']:>17.2f}{final['category_entropy']:>14.2f}"
            f"{final['sources']:>9.1f}{final['median_age_hours']:>14.1f}"
        )
    return "\n".join(lines)
