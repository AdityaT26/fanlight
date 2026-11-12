from __future__ import annotations

import copy

import numpy as np
import pytest

from fanlight.evaluation import representation_check
from fanlight.index import build_index, cluster_stories, resolve_as_of
from fanlight.ingest import build_corpus
from fanlight.text import tokenize


def test_tokenizer_keeps_devanagari_words_whole():
    tokens = tokenize("मुख्यमंत्री ने बजट प्रस्तावों का बचाव किया।")
    assert "मुख्यमंत्री" in tokens
    assert "प्रस्तावों" in tokens
    assert "ने" not in tokens and "का" not in tokens  # stopwords
    assert all("।" not in t for t in tokens)


def test_tokenizer_english():
    assert tokenize("The RBI holds rates, says it's watching inflation!") == [
        "rbi",
        "holds",
        "rates",
        "watching",
        "inflation",
    ]


def test_index_shapes(index):
    n = index.size
    assert index.matrix.shape[0] == n
    assert index.category_matrix.shape == (n, len(index.category_keys))
    assert index.tag_matrix.shape == (n, len(index.tag_keys))
    assert len(index.story) == len(index.coverage) == len(index.popularity) == n
    assert np.all((index.source_quality >= 0) & (index.source_quality <= 1))
    assert np.all((index.popularity >= 0) & (index.popularity <= 1))
    norms = np.sqrt(np.asarray(index.matrix.multiply(index.matrix).sum(axis=1)).ravel())
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-4)
    assert index.as_of == pytest.approx(index.published.max())


def test_source_quality_follows_outlet_rank(index):
    sources = index.corpus.sources
    prominent = index.source_keys.index("harbour_ledger")  # priority 420
    obscure = index.source_keys.index("the_tidewater")  # priority 730000
    q_prominent = index.source_quality[index.source_id == prominent]
    q_obscure = index.source_quality[index.source_id == obscure]
    assert sources["harbour_ledger"].priority < sources["the_tidewater"].priority
    assert q_prominent.size and q_obscure.size
    assert q_prominent.mean() > q_obscure.mean()


def test_stories_group_echoed_reports_across_outlets(index):
    articles = index.corpus.articles
    echoes = [i for i, a in enumerate(articles) if a.title.endswith(", sources say")]
    assert echoes, "fixture should contain re-reported stories"
    grouped = sum(1 for i in echoes if len(index.story_members(i)) > 1)
    assert grouped / len(echoes) > 0.8
    for i in echoes:
        members = index.story_members(i)
        assert {index.article(j).language for j in members} == {index.article(i).language}
        assert index.coverage[i] == len({index.source_id[j] for j in members})


def test_cluster_requires_same_language_and_time_window():
    from scipy import sparse

    rows = sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    published = np.array([0.0, 3600.0, 0.0, 400 * 3600.0])
    language = np.array([0, 0, 1, 0])
    source = np.array([0, 1, 2, 3])
    labels = cluster_stories(rows, published, language, source)
    assert labels[0] == labels[1]  # same language, one hour apart
    assert labels[2] != labels[0]  # other language
    assert labels[3] != labels[0]  # outside the window


def test_cluster_is_stricter_within_one_outlet():
    from scipy import sparse

    a = np.array([1.0, 0.0])
    b = np.array([0.7, 0.714])  # cosine ~0.70 with a
    rows = sparse.csr_matrix(np.vstack([a, b, a, b]).astype(np.float32))
    zeros = np.zeros(4)
    same = cluster_stories(rows[:2], zeros[:2], np.zeros(2, int), np.array([5, 5]))
    different = cluster_stories(rows[2:], zeros[2:], np.zeros(2, int), np.array([5, 6]))
    assert same[0] != same[1]
    assert different[0] == different[1]


def test_resolve_as_of():
    published = np.array([100.0, 200.0])
    assert resolve_as_of("latest", published) == 200.0
    assert resolve_as_of("2024-01-01T00:00:00Z", published) == 1704067200.0
    with pytest.raises(ValueError):
        resolve_as_of("someday", published)


def test_empty_corpus_is_rejected(raw):
    small = copy.deepcopy(raw)
    small["articles"] = []
    with pytest.raises(ValueError):
        build_index(build_corpus(small))


def test_text_index_captures_topic(index):
    check = representation_check(index, sample=300)
    assert check["agreement"] > 3 * check["chance"]
