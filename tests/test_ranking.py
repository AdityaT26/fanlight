from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import key_of

from fanlight.config import SIGNALS, RankingConfig
from fanlight.profiles import Event, Profile, build_state
from fanlight.ranker import Ranker, discovery_slots


def reader(index, *names, language="english", **kwargs) -> Profile:
    return Profile(id="t", languages=(language,), categories={key_of(index, n): 1.0 for n in names}, **kwargs)


def first_category(index, item) -> str:
    return index.article(item.index).categories[0]


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #


def test_affinity_saturates_and_is_independent_per_category(index):
    sports = key_of(index, "Sports")
    politics = key_of(index, "Politics")
    sports_ids = [a.id for a in index.corpus.articles if sports in a.categories][:30]
    base = reader(index, "Politics")
    heavy = reader(index, "Politics", events=[Event(i, "open", index.as_of) for i in sports_ids])
    col = {k: j for j, k in enumerate(index.category_keys)}
    s0, s1 = build_state(index, base, now=index.as_of), build_state(index, heavy, now=index.as_of)
    # Reading a lot of sport does not lower the stated politics interest.
    assert s1.category_affinity[col[politics]] == pytest.approx(s0.category_affinity[col[politics]])
    assert 0.85 < s1.category_affinity[col[sports]] < 1.0
    assert np.all(np.abs(s1.category_affinity) <= 1.0)


def test_hide_makes_affinity_negative_and_hides_the_whole_story(index):
    echoes = [i for i, a in enumerate(index.corpus.articles) if len(index.story_members(i)) > 1]
    i = echoes[0]
    article = index.article(i)
    profile = Profile(id="t", languages=(article.language,), events=[Event(article.id, "hide", index.as_of)])
    state = build_state(index, profile, now=index.as_of)
    assert set(index.story_members(i)) <= state.hidden
    col = index.category_keys.index(article.categories[0])
    assert state.category_affinity[col] < 0


def test_old_events_decay(index):
    article = index.article(0)
    week = 7 * 86_400.0
    fresh = Profile(id="t", events=[Event(article.id, "open", index.as_of)])
    stale = Profile(id="t", events=[Event(article.id, "open", index.as_of - 20 * week)])
    col = index.category_keys.index(article.categories[0])
    a_fresh = build_state(index, fresh, now=index.as_of).category_affinity[col]
    a_stale = build_state(index, stale, now=index.as_of).category_affinity[col]
    assert a_fresh > a_stale > 0


def test_repeated_actions_count_once_and_unknown_ids_are_ignored(index):
    article = index.article(0)
    once = Profile(id="t", events=[Event(article.id, "like", index.as_of)])
    many = Profile(
        id="t", events=[Event(article.id, "like", index.as_of)] * 5 + [Event("nope", "like", index.as_of)]
    )
    a = build_state(index, once, now=index.as_of).category_affinity
    b = build_state(index, many, now=index.as_of).category_affinity
    assert np.allclose(a, b)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def test_feed_respects_language_and_stated_interest(index):
    ranker = Ranker(index)
    feed = ranker.rank(reader(index, "Technology"), k=20)
    assert len(feed.items) == 20
    assert all(index.article(it.index).language == "english" for it in feed.items)
    tech = key_of(index, "Technology")
    regular = [it for it in feed.items if not it.discovery]
    share = sum(first_category(index, it) == tech for it in regular) / len(regular)
    assert share >= 0.5
    assert feed.items[0].reasons[0] == "You follow Technology"


def test_hindi_reader_gets_hindi(index):
    feed = Ranker(index).rank(reader(index, "Politics", language="hindi"), k=10)
    assert feed.items and all(index.article(it.index).language == "hindi" for it in feed.items)


def test_one_article_per_story(index):
    feed = Ranker(index).rank(Profile(id="t", languages=("english",)), k=60)
    stories = [index.story[it.index] for it in feed.items]
    assert len(stories) == len(set(stories))


def test_scores_and_contributions_are_consistent(index):
    feed = Ranker(index).rank(reader(index, "Business"), k=15)
    assert sum(feed.weights.values()) == pytest.approx(1.0, abs=1e-3)
    for item in feed.items:
        assert 0.0 <= item.score <= 1.0
        assert sum(item.contributions.values()) == pytest.approx(item.score, abs=2e-3)
        assert set(item.signals) == set(SIGNALS)
    # Uninformative signals sit out for a reader with no history.
    assert "content" not in feed.active_signals and "tags" not in feed.active_signals


def test_cold_start_reader_still_gets_a_feed(index):
    feed = Ranker(index).rank(Profile(id="t"), k=10)
    assert len(feed.items) == 10
    assert "category" not in feed.active_signals
    assert not any(it.discovery for it in feed.items)


def test_learning_from_clicks(index):
    """Opening sports stories should pull sport up for a reader who only stated politics."""
    ranker = Ranker(index)
    sports = key_of(index, "Sports")
    profile = reader(index, "Politics")
    before = ranker.rank(profile, k=20)
    count_before = sum(first_category(index, it) == sports for it in before.items)
    opened = [a.id for a in index.corpus.articles if sports in a.categories and a.language == "english"][:6]
    profile.events = [Event(i, "open", index.as_of) for i in opened]
    after = ranker.rank(profile, k=20)
    count_after = sum(first_category(index, it) == sports for it in after.items)
    assert count_after > count_before
    assert "content" in after.active_signals
    assert not {index.article(it.index).id for it in after.items} & set(opened)  # read stories leave


def test_hide_read_can_be_turned_off(index):
    article = next(
        a
        for i, a in enumerate(index.corpus.articles)
        if a.language == "english" and len(index.story_members(i)) == 1
    )
    profile = Profile(id="t", languages=("english",), events=[Event(article.id, "open", index.as_of)])
    shown = Ranker(index).rank(
        profile, k=index.size, config=RankingConfig(hide_read=False, max_age_days=None)
    )
    assert article.id in {index.article(it.index).id for it in shown.items}


def test_impressions_without_clicks_sink_a_story(index):
    ranker = Ranker(index)
    profile = reader(index, "Science")
    top = ranker.rank(profile, k=1).items[0]
    profile.events = [Event(index.article(top.index).id, "view", index.as_of)] * 4
    again = ranker.rank(profile, k=1).items[0]
    assert again.index != top.index


def test_diversity_spreads_categories_and_outlets(index):
    ranker = Ranker(index)
    profile = reader(index, "Politics", "Business")
    flat = ranker.rank(profile, k=20, config=RankingConfig(diversity=0.0, discovery=0.0))
    varied = ranker.rank(profile, k=20, config=RankingConfig(diversity=1.0, discovery=0.0))

    def longest_run(items):
        best = run = 1
        for a, b in itertools.pairwise(items):
            run = run + 1 if index.source_id[a.index] == index.source_id[b.index] else 1
            best = max(best, run)
        return best

    assert varied.diagnostics["similarity"] <= flat.diagnostics["similarity"]
    assert longest_run(varied.items) <= longest_run(flat.items)
    assert [it.base_rank for it in flat.items] == list(range(1, 21))


def test_discovery_slots_bring_in_unread_categories(index):
    profile = reader(index, "Sports")
    sports = key_of(index, "Sports")
    feed = Ranker(index).rank(profile, k=20, config=RankingConfig(discovery=0.2))
    picks = [it for it in feed.items if it.discovery]
    assert len(picks) == len(discovery_slots(20, 0.2)) == 4
    assert all(sports not in index.article(it.index).categories for it in picks)
    assert len({first_category(index, it) for it in picks}) >= 3  # rotates categories
    assert all(it.reasons[0].startswith("Something outside") for it in picks)
    assert not any(
        it.discovery for it in Ranker(index).rank(profile, k=20, config=RankingConfig(discovery=0)).items
    )


def test_discovery_slot_positions():
    assert discovery_slots(20, 0.1) == {5, 15}
    assert discovery_slots(20, 0.5) == {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}
    assert discovery_slots(1, 0.5) == set()
    assert discovery_slots(20, 0) == set()


def test_category_filter_and_age_limit(index):
    ranker = Ranker(index)
    science = key_of(index, "Science")
    feed = ranker.rank(reader(index, "Politics"), k=10, categories=[science])
    assert feed.items and all(science in index.article(it.index).categories for it in feed.items)
    recent = ranker.rank(Profile(id="t"), k=5, config=RankingConfig(max_age_days=1.0))
    assert all(index.as_of - index.published[it.index] <= 86_400 for it in recent.items)
    # A limit that would leave too few stories is lifted rather than returning nothing.
    tiny = ranker.rank(Profile(id="t"), k=50, config=RankingConfig(max_age_days=0.01))
    assert len(tiny.items) == 50


def test_freshness_prefers_newer_stories(index):
    feed = Ranker(index).rank(Profile(id="t"), k=index.size, config=RankingConfig(max_age_days=None))
    fresh = {it.index: it.signals["freshness"] for it in feed.items}
    ordered = sorted(fresh, key=lambda i: index.published[i])
    values = [fresh[i] for i in ordered]
    assert all(a <= b + 1e-6 for a, b in itertools.pairwise(values))


def test_weight_overrides_change_the_order(index):
    ranker = Ranker(index)
    profile = reader(index, "Entertainment")
    default = ranker.rank(profile, k=10)
    fresh_only = ranker.rank(
        profile,
        k=10,
        config=RankingConfig().with_overrides({"weights": {"category": 0, "source": 0, "popularity": 0}}),
    )
    assert [i.index for i in default.items] != [i.index for i in fresh_only.items]
    assert fresh_only.active_signals == ["freshness"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"diversity": 2},
        {"weights": {"bogus": 1}},
        {"weights": {"category": -1}},
        {"nope": 1},
        {"max_age_days": 0},
    ],
)
def test_invalid_overrides_are_rejected(overrides):
    with pytest.raises(ValueError):
        RankingConfig().with_overrides(overrides)
