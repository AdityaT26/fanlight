from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from feedserver import FeedServer, render

from fanlight import cli
from fanlight.api import create_app
from fanlight.config import Settings
from fanlight.engine import Engine
from fanlight.feeds import (
    CATEGORY_HINTS,
    FeedConfig,
    article_id,
    canonical_url,
    infer_category,
    load_feeds,
    parse_feed,
)
from fanlight.live import LiveNews, Refresher
from fanlight.store import Store

WORLD = FeedConfig("world", "https://wire.example/world", "Meridian Wire", "english", "World", 1)
TECH = FeedConfig("tech", "https://circuit.example/", "Circuit Notes", "english", "Technology", 2)
HINDI = FeedConfig("hindi", "https://darpan.example/", "Samachar Darpan", "hindi", "Top News", 1)


@pytest.fixture()
def server():
    with FeedServer() as running:
        yield running


@pytest.fixture()
def live_settings(server, tmp_path):
    feeds = server.feeds_toml(tmp_path / "feeds.toml")
    return Settings(state_dir=tmp_path / "var", feeds_path=feeds, cohorts=3)


# --------------------------------------------------------------------------- #
# Feed list
# --------------------------------------------------------------------------- #


def test_packaged_feed_list_is_valid():
    feeds = load_feeds()
    assert len(feeds) >= 15
    assert len({f.id for f in feeds}) == len(feeds)
    assert {f.language for f in feeds} == {"english", "hindi"}
    assert all(f.category in CATEGORY_HINTS and f.url.startswith("https://") for f in feeds)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            '[[feed]]\nid = "a"\nurl = "https://x.example/rss"\noutlet = "X"\ncategory = "Gossip"\n',
            "unknown category",
        ),
        (
            '[[feed]]\nid = "a"\nurl = "https://x.example/rss"\noutlet = "X"\n[[feed]]\nid = "a"\nurl = "https://y.example/rss"\noutlet = "Y"\n',
            "used twice",
        ),
        ('[[feed]]\nid = "a"\noutlet = "X"\n', "missing url"),
        ('[[feed]]\nid = "a"\nurl = "ftp://x.example/rss"\noutlet = "X"\n', "http"),
        ('[[feed]]\nid = "a"\nurl = "https://x.example/rss"\noutlet = "X"\nreach = 7\n', "reach"),
        ('[[feed]]\nid = "a"\nurl = "https://x.example/rss"\noutlet = "X"\nenabled = false\n', "no enabled"),
    ],
)
def test_feed_list_validation(tmp_path, body, message):
    path = tmp_path / "feeds.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_feeds(path)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_canonical_url_drops_tracking_but_keeps_real_parameters():
    a = "https://Wire.example/news/a?at_medium=RSS&at_campaign=rss&id=7#top"
    b = "https://wire.example/news/a?utm_source=feed&id=7"
    assert canonical_url(a) == canonical_url(b) == "https://wire.example/news/a?id=7"
    assert article_id(a) == article_id(b)
    assert article_id(a) != article_id("https://wire.example/news/a?id=8")


@pytest.mark.parametrize(
    ("default", "tags", "url", "expected"),
    [
        ("Top News", [], "https://darpan.example/sports/cricket/story/x", "Cricket"),
        ("India", [], "https://paper.example/article/sports/football/x", "Sport"),
        ("Top News", [], "https://darpan.example/india/news/story/x", "India"),
        ("World", ["Gadgets"], "https://circuit.example/2026/11/x", "Technology"),
        ("World", [], "https://wire.example/news/articles/x", "World"),
        ("Top News", ["Bollywood"], "https://x.example/a", "Entertainment"),
        ("Top News", [], "https://x.example/entertainment/web-series/x", "Entertainment"),
        ("Business", [], "https://x.example/cricket-is-the-story-slug", "Business"),  # slug is not a section
    ],
)
def test_category_inference(default, tags, url, expected):
    assert infer_category(default, tags, url) == expected


def test_parse_rss():
    now = time.time()
    rows = parse_feed(render("world.xml"), WORLD, fetched_at=now)
    titles = [r["title"] for r in rows]
    assert "An item without a link is skipped" not in titles
    assert "A story from last month is too old to keep" not in titles
    harbour = [r for r in rows if r["title"].startswith("Harbour talks")]
    assert len(harbour) == 2 and harbour[0]["_id"] == harbour[1]["_id"]  # duplicate by URL
    first = harbour[0]
    assert first["link"] == "https://wire.example/news/articles/harbour-talks?id=7"
    assert first["body_short"].startswith("Negotiators from both coastal states agreed")
    assert "<" not in first["body_short"]
    assert first["thumbnail_url"] == "https://img.example/large/harbour.jpg"  # widest thumbnail
    assert first["category"] == ["world"] and first["language"] == "english"
    ferry = next(r for r in rows if r["title"].startswith("Port city"))
    assert abs(_ts(ferry["date"]) - now) < 5  # a future date is clamped to the fetch time
    glacier = next(r for r in rows if r["title"].startswith("Glacier"))
    assert glacier["category"] == ["science"] and "eleven weeks" in glacier["body"]
    final = next(r for r in rows if r["title"].startswith("Underdogs"))
    assert final["category"] == ["sport"]


def test_parse_atom_and_hindi():
    tech = parse_feed(render("tech.atom"), TECH)
    assert [r["title"] for r in tech] == [
        "Battery start-up shows a cell that charges in nine minutes",
        "Open source maintainers vote on a new release cadence",
    ]
    assert tech[0]["thumbnail_url"] == "https://img.example/cell.png"
    assert tech[0]["author_name"] == "R. Iyer" and tech[0]["category"] == ["technology"]
    assert tech[0]["body_short"] == "A lab prototype reached 80 percent charge in nine minutes."

    hindi = parse_feed(render("hindi.xml"), HINDI)
    by_title = {r["title"]: r for r in hindi}
    metro = by_title["नई मेट्रो लाइन पर परीक्षण शुरू, अगले महीने से यात्री सेवा"]
    assert metro["category"] == ["india"] and metro["thumbnail_url"] == "https://img.example/metro.jpg"
    century = by_title["युवा बल्लेबाज़ ने पहले ही मैच में शतक जड़ा"]
    assert (
        century["category"] == ["cricket"] and century["thumbnail_url"] == "https://img.example/century.jpg"
    )
    assert all(r["language"] == "hindi" for r in hindi)


def test_parse_rejects_non_feeds():
    with pytest.raises(ValueError, match="not a feed"):
        parse_feed(b"<html><body>404</body></html>", WORLD)


def _ts(value: str) -> float:
    from fanlight.ingest import parse_datetime

    return parse_datetime(value).timestamp()


# --------------------------------------------------------------------------- #
# Fetching and storage
# --------------------------------------------------------------------------- #


def test_refresh_fetches_stores_and_uses_conditional_requests(server, tmp_path):
    store = Store(tmp_path / "s.db")
    live = LiveNews(store, load_feeds(server.feeds_toml(tmp_path / "feeds.toml")))
    first = live.refresh()
    assert first.feeds == 5 and first.ok == 3
    assert set(first.failed) == {"broken", "gone"}
    assert "HTTP 404" in first.failed["gone"] and "not a feed" in first.failed["broken"]
    assert first.new == store.live_count() == 9  # 5 world (one duplicate) + 2 tech + 3 hindi, minus 1

    second = live.refresh()
    assert second.new == 0 and second.not_modified == 2  # world and hindi send ETags
    states = store.feed_states()
    assert states["world"]["etag"] == '"world-v1"' and states["world"]["items"] == 5
    assert states["gone"]["last_error"] and not states["gone"]["last_success"]

    status = live.status()
    assert status["stories"] == 9
    assert {f["id"]: f["ok"] for f in status["feeds"]} == {
        "world": True,
        "tech": True,
        "hindi": True,
        "broken": False,
        "gone": False,
    }
    store.close()


def test_retention_drops_old_stories(tmp_path):
    store = Store(tmp_path / "s.db")
    old = {"_id": "lold", "title": "Old", "date": "2020-01-01T00:00:00Z", "feed": "x"}
    new = {"_id": "lnew", "title": "New", "date": "2099-01-01T00:00:00Z", "feed": "x"}
    assert store.add_live_articles([old, new]) == 2
    assert store.add_live_articles([new]) == 0
    assert store.prune_live(time.time()) == 1
    assert [r["_id"] for r in store.live_articles()] == ["lnew"]
    store.close()


# --------------------------------------------------------------------------- #
# Engine in live mode
# --------------------------------------------------------------------------- #


def test_live_engine_ranks_fresh_news_and_picks_up_new_stories(server, live_settings):
    engine = Engine.load(live_settings, source="live")
    assert engine.origin == "live" and engine.index.size == 9
    reader = engine.create_reader("Mira", ["english"], ["technology"])
    feed = engine.feed(reader.id, k=5)
    assert feed["items"][0]["title"].startswith("Battery start-up")
    assert feed["items"][0]["image"] == "https://img.example/cell.png"
    assert abs(engine.index.as_of - time.time()) < 5  # ranked against the wall clock
    assert engine.meta()["live"]["stories"] == 9

    # A publisher adds a story; the refresher picks it up and swaps the index in.
    xml = render("tech.atom").replace(
        b"<entry>",
        b"<entry><title>Chip makers agree on an open interconnect</title>"
        b'<link rel="alternate" href="https://circuit.example/2026/11/interconnect"/><id>urn:x:3</id>'
        b"<updated>" + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()).encode() + b"</updated>"
        b"<summary>Seven companies signed up.</summary></entry><entry>",
        1,
    )
    server.routes["/tech.atom"] = (200, xml, None)
    edition = engine.edition
    report = Refresher(engine.live, engine.rebuild, 15).tick()
    assert report.new == 1 and engine.edition == edition + 1 and engine.index.size == 10
    titles = [i["title"] for i in engine.feed(reader.id, k=10)["items"]]
    assert "Chip makers agree on an open interconnect" in titles
    # Nothing new: no rebuild.
    assert Refresher(engine.live, engine.rebuild, 15).tick().new == 0 and engine.edition == edition + 1


def test_live_engine_explains_when_no_feed_works(tmp_path):
    feeds = tmp_path / "feeds.toml"
    feeds.write_text('[[feed]]\nid = "x"\nurl = "http://127.0.0.1:9/feed"\noutlet = "X"\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="No live stories"):
        Engine.load(Settings(state_dir=tmp_path, feeds_path=feeds), source="live")


def test_rebuild_is_safe_under_concurrent_requests(server, live_settings):
    engine = Engine.load(live_settings, source="live")
    reader = engine.create_reader("Load", ["english", "hindi"], ["world"])
    errors: list[BaseException] = []

    def hammer():
        try:
            for _ in range(15):
                assert engine.feed(reader.id, k=5)["items"]
        except BaseException as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(5):
        engine.rebuild(engine.live.raw())
    for t in threads:
        t.join()
    assert not errors


# --------------------------------------------------------------------------- #
# API and CLI
# --------------------------------------------------------------------------- #


def test_public_mode_keeps_readers_private(server, live_settings):
    engine = Engine.load(replace(live_settings, public=True), source="live")
    with TestClient(create_app(engine)) as http:
        mine = http.post(
            "/api/readers", json={"name": "Asha", "languages": ["hindi"], "categories": ["cricket"]}
        ).json()
        other = http.post("/api/readers", json={"name": "Ben", "languages": ["english"]}).json()
        assert len(mine["id"]) > 20  # the id is the key, so it must be long
        assert http.get("/api/readers").json() == []
        listed = http.get("/api/readers", params={"ids": f"{mine['id']},nope"}).json()
        assert [r["id"] for r in listed] == [mine["id"]] and other["id"] not in str(listed)
        feed = http.get(f"/api/readers/{mine['id']}/feed").json()
        assert feed["items"][0]["title"] == "युवा बल्लेबाज़ ने पहले ही मैच में शतक जड़ा"
        meta = http.get("/api/meta").json()
        assert meta["public"] is True and meta["origin"] == "live" and meta["live"]["stories"] == 9
        assert {f["id"] for f in http.get("/api/feeds").json()["feeds"]} == {
            "world",
            "tech",
            "hindi",
            "broken",
            "gone",
        }


def test_feeds_endpoint_needs_live_mode(engine):
    with TestClient(create_app(engine)) as http:
        assert http.get("/api/feeds").status_code == 404
        assert http.get("/api/meta").json()["live"] is None


def test_cli_fetch_and_feeds(server, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FANLIGHT_STATE_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("FANLIGHT_FEEDS", str(server.feeds_toml(tmp_path / "feeds.toml")))
    assert cli.main(["fetch"]) == 0
    out = capsys.readouterr().out
    assert "3/5 feeds ok" in out and "9 new" in out and "failed: broken" in out
    assert cli.main(["feeds", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["stories"] == 9
    assert cli.main(["feeds"]) == 0
    assert "failing: HTTP 404" in capsys.readouterr().out
    assert cli.main(["readers", "--source", "live"]) == 0
