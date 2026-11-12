from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from fanlight.ingest import build_corpus, clean_text, load_json_dir, oid, parse_datetime, slugify
from fanlight.synth import make_raw


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-08-04T12:34:39.047Z", datetime(2024, 8, 4, 12, 34, 39, 47000, tzinfo=timezone.utc)),
        (
            {"$date": "2024-08-21T02:35:52.651Z"},
            datetime(2024, 8, 21, 2, 35, 52, 651000, tzinfo=timezone.utc),
        ),
        ("2024-07-23T17:52:20.486000", datetime(2024, 7, 23, 17, 52, 20, 486000, tzinfo=timezone.utc)),
        ("05/08/2024 21:16:18", datetime(2024, 8, 5, 21, 16, 18, tzinfo=timezone.utc)),
        ("2024-08-04T12:34:39.5+05:30", datetime(2024, 8, 4, 7, 4, 39, 500000, tzinfo=timezone.utc)),
        (1722774879000, datetime(2024, 8, 4, 12, 34, 39, tzinfo=timezone.utc)),
        ({"$date": {"$numberLong": "1722774879000"}}, datetime(2024, 8, 4, 12, 34, 39, tzinfo=timezone.utc)),
    ],
)
def test_parse_datetime_handles_every_shape_in_the_export(value, expected):
    assert parse_datetime(value) == expected


@pytest.mark.parametrize("value", [None, "", "not a date", {}, "32/13/2024 99:99:99"])
def test_parse_datetime_returns_none_for_garbage(value):
    assert parse_datetime(value) is None


def test_oid_and_slug():
    assert oid({"$oid": "abc"}) == "abc"
    assert oid(" abc ") == "abc"
    assert oid(None) is None and oid({}) is None
    assert slugify("Culture & Lifestyle") == slugify("Culture and Lifestyle") == "culture-and-lifestyle"
    assert slugify("Science ") == "science"


def test_clean_text_strips_markup():
    assert clean_text("<p>Hello&nbsp;<b>world</b></p>\n\n") == "Hello world"
    assert clean_text(None) == ""


def test_build_corpus_drops_deleted_drafts_and_duplicates():
    raw = make_raw(n_articles=200, n_users=5, seed=3)
    corpus = build_corpus(raw)
    report = corpus.report
    assert report.raw_articles == len(raw["articles"])
    assert report.dropped_deleted == 1
    assert report.dropped_unpublished == 1
    assert report.dropped_duplicate >= 1
    assert report.kept == len(corpus.articles)
    titles = [a.title for a in corpus.articles]
    assert "Withdrawn story" not in titles and "Unpublished draft" not in titles
    ids = [a.id for a in corpus.articles]
    assert len(ids) == len(set(ids))


def test_build_corpus_resolves_references_and_merges_aliases():
    corpus = build_corpus(make_raw(n_articles=200, n_users=5, seed=3))
    # "Health and Wellness" (inactive duplicate) merges into the active "Health & Wellness".
    assert "health-and-wellness" in corpus.categories
    assert corpus.categories["health-and-wellness"].active
    # Tags from both collections resolve, whatever the casing or id shape.
    assert any(a.tags for a in corpus.articles)
    assert all(t in corpus.tags for a in corpus.articles for t in a.tags)
    # The 'name:' typo in the sources collection still yields a name.
    assert corpus.sources["civic_wire"].name == "Civic Wire"
    # The day-first legacy date parsed; the unknown category reference was counted.
    assert any(a.title.startswith("Observatory logs") for a in corpus.articles)
    assert corpus.report.unknown_category_refs == 1


def test_load_json_dir_reads_json_and_jsonl(tmp_path):
    rows = [{"_id": "a1", "title": "One", "date": "2024-01-01", "status": "published"}]
    (tmp_path / "articles.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "article_categories.json").write_text(json.dumps({"_id": "c", "name": "X"}), encoding="utf-8")
    raw = load_json_dir(tmp_path)
    assert raw["articles"] == rows
    assert raw["article_categories"] == [{"_id": "c", "name": "X"}]
    assert raw["tags"] == []


def test_load_json_dir_explains_a_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="--demo"):
        load_json_dir(tmp_path)
