from __future__ import annotations

import json

import pytest
from conftest import key_of
from fastapi.testclient import TestClient

from fanlight import cli
from fanlight.api import create_app
from fanlight.config import Settings, read_dotenv
from fanlight.engine import Engine, UnknownReader
from fanlight.evaluation import simulate
from fanlight.store import Store

# --------------------------------------------------------------------------- #
# Engine and store
# --------------------------------------------------------------------------- #


def test_imported_readers_have_cohorts_and_stable_ids(engine, profiles):
    readers = engine.list_readers(100)
    assert len(readers) == len(profiles)
    assert all(r["id"].startswith("u_") for r in readers)
    with_interests = [r for r in readers if r["interest_count"]]
    assert with_interests and all(r["cohort"] != "New readers" for r in with_interests)
    assert any(r["cohort"] == "New readers" for r in readers)  # the cold-start user
    # The newest preference document wins for users with several.
    assert all(r["languages"] for r in readers)


def test_reader_lifecycle_and_feedback(engine, index):
    reader = engine.create_reader("Mira", ["english"], [key_of(index, "Science")])
    assert engine.list_readers()[0]["id"] == reader.id
    feed = engine.feed(reader.id, k=8)
    first = feed["items"][0]
    assert first["reasons"] and 0 <= first["match"] <= 100

    assert engine.record(reader.id, [(first["id"], "like"), ("not-a-story", "open")]) == 1
    after = engine.feed(reader.id, k=8)
    assert first["id"] not in {i["id"] for i in after["items"]}
    assert after["reader"]["activity"]["like"] == 1
    assert engine.undo(reader.id, first["id"], "like") == 1

    updated = engine.update_reader(reader.id, name="Mira K", categories=[key_of(index, "Sports")])
    assert updated.name == "Mira K" and list(updated.categories) == [key_of(index, "Sports")]
    engine.delete_reader(reader.id)
    with pytest.raises(UnknownReader):
        engine.profile(reader.id)


def test_imported_readers_are_read_only_but_can_interact(engine):
    reader_id = engine.list_readers()[0]["id"]
    with pytest.raises(PermissionError):
        engine.update_reader(reader_id, name="x")
    with pytest.raises(PermissionError):
        engine.delete_reader(reader_id)
    story = engine.feed(reader_id, k=1)["items"][0]["id"]
    assert engine.record(reader_id, [(story, "save")]) == 1
    assert engine.feed(reader_id, k=3)["items"][0]["id"] != story
    assert engine.reset_history(reader_id) == 1


def test_validation(engine):
    with pytest.raises(ValueError):
        engine.create_reader("x", ["klingon"], [])
    with pytest.raises(ValueError):
        engine.create_reader("x", ["english"], ["not-a-category"])
    with pytest.raises(ValueError):
        engine.store.add_events("r", [("a", "explode")])


def test_store_prunes_old_impressions(tmp_path, monkeypatch):
    import fanlight.store as store_module

    monkeypatch.setattr(store_module, "MAX_VIEWS_PER_READER", 5)
    store = Store(tmp_path / "s.db")
    store.add_events("r", [(f"a{i}", "view") for i in range(12)] + [("a0", "like")])
    events = store.events("r")
    assert sum(e.action == "view" for e in events) == 5
    assert any(e.action == "like" for e in events)
    store.close()


def test_store_persists_across_connections(tmp_path):
    first = Store(tmp_path / "s.db")
    profile = first.create_reader("Ana", ["hindi"], {"politics": 1.0})
    first.add_events(profile.id, [("x", "open")])
    first.close()
    second = Store(tmp_path / "s.db")
    assert second.get_reader(profile.id).languages == ("hindi",)
    assert [e.action for e in second.events(profile.id)] == ["open"]
    second.close()


def test_engine_load_caches_the_index(tmp_path):
    data = tmp_path / "data"
    assert cli.main(["demo-data", str(data), "--articles", "150", "--users", "8"]) == 0
    settings = Settings(data_dir=data, state_dir=tmp_path / "var", cohorts=3)
    built = Engine.load(settings)
    cache = list((tmp_path / "var" / "cache").glob("index-*.pkl"))
    assert len(cache) == 1
    reloaded = Engine.load(settings)
    assert reloaded.index.size == built.index.size
    assert reloaded.index.position == built.index.position


def test_settings_read_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nFANLIGHT_DATA_DIR="../data"\nexport FANLIGHT_COHORTS=5\nBROKEN LINE\n', encoding="utf-8"
    )
    values = read_dotenv(env_file)
    assert values == {"FANLIGHT_DATA_DIR": "../data", "FANLIGHT_COHORTS": "5"}
    settings = Settings.from_env(values)
    assert str(settings.data_dir) == "../data" and settings.cohorts == 5


def test_simulation_v2_finds_unstated_interests(index):
    """On this small, clean corpus v1's stated-category-only feed is close to optimal
    for clicks; the real dataset is where it falls behind (see `fanlight evaluate`).
    What Fanlight must show anywhere: it discovers interests the reader never stated."""
    report = simulate(index, n_readers=12, rounds=3, feed_size=10, seed=1)
    policies = report["policies"]
    v2 = policies["fanlight"]
    assert v2[-1]["interest_recall"] > policies["v1 label formula"][-1]["interest_recall"]
    assert v2[-1]["ndcg"] > policies["random"][-1]["ndcg"]
    assert v2[-1]["ndcg"] > policies["newest first"][-1]["ndcg"]
    assert v2[-1]["category_entropy"] > policies["fanlight, no variety or discovery"][-1]["category_entropy"]


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(engine):
    with TestClient(create_app(engine)) as http:
        yield http


def test_api_meta_and_static(client):
    meta = client.get("/api/meta").json()
    assert meta["stats"]["articles"] > 400
    assert {s["key"] for s in meta["signals"]} == {
        "category",
        "tags",
        "content",
        "freshness",
        "source",
        "popularity",
    }
    assert client.get("/healthz").json()["ok"] is True
    page = client.get("/")
    assert page.status_code == 200 and "Fanlight" in page.text
    for asset in ("/static/app.js", "/static/styles.css", "/static/icon.svg", "/manifest.webmanifest"):
        assert client.get(asset).status_code == 200, asset


def test_api_reader_flow(client, index):
    created = client.post(
        "/api/readers",
        json={"name": "Dev", "languages": ["english"], "categories": [key_of(index, "Technology")]},
    )
    assert created.status_code == 201
    rid = created.json()["id"]
    feed = client.get(
        f"/api/readers/{rid}/feed", params={"k": 5, "diversity": 0.8, "w_freshness": 0.5}
    ).json()
    assert len(feed["items"]) == 5 and feed["config"]["diversity"] == 0.8
    story = feed["items"][0]["id"]

    assert client.post(
        f"/api/readers/{rid}/events", json={"events": [{"article_id": story, "action": "like"}]}
    ).json() == {"stored": 1}
    again = client.get(f"/api/readers/{rid}/feed", params={"hide_read": False, "k": 50}).json()
    assert next(i for i in again["items"] if i["id"] == story)["liked"] is True
    assert client.request(
        "DELETE", f"/api/readers/{rid}/events", json={"article_id": story, "action": "like"}
    ).json() == {"removed": 1}

    assert client.patch(f"/api/readers/{rid}", json={"name": "Dev 2"}).json()["name"] == "Dev 2"
    assert client.get(f"/api/articles/{story}").json()["id"] == story
    assert client.delete(f"/api/readers/{rid}").status_code == 204
    assert client.get(f"/api/readers/{rid}").status_code == 404


def test_api_rejects_bad_input(client):
    rid = client.get("/api/readers").json()[0]["id"]
    assert client.get("/api/readers/nobody/feed").status_code == 404
    assert client.get(f"/api/readers/{rid}/feed", params={"diversity": 3}).status_code == 422
    assert client.get(f"/api/readers/{rid}/feed", params={"category": "nope"}).status_code == 422
    assert (
        client.post(
            f"/api/readers/{rid}/events", json={"events": [{"article_id": "x", "action": "explode"}]}
        ).status_code
        == 422
    )
    assert client.post("/api/readers", json={"languages": ["klingon"]}).status_code == 422
    assert client.patch(f"/api/readers/{rid}", json={"name": "x"}).status_code == 403
    assert client.get("/api/articles/missing").status_code == 404


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    assert cli.main(["demo-data", str(data), "--articles", "200", "--users", "10"]) == 0
    monkeypatch.setenv("FANLIGHT_DATA_DIR", str(data))
    monkeypatch.setenv("FANLIGHT_STATE_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("FANLIGHT_COHORTS", "3")
    capsys.readouterr()

    assert cli.main(["index"]) == 0
    assert "Articles kept" in capsys.readouterr().out

    assert cli.main(["readers", "--json"]) == 0
    readers = json.loads(capsys.readouterr().out)
    rid = readers[0]["id"]

    assert cli.main(["feed", rid, "-k", "5", "--no-colour"]) == 0
    out = capsys.readouterr().out
    assert " 1. [" in out and "eligible stories" in out

    assert cli.main(["create-reader", "Zoe", "--category", "science"]) == 0
    new_id = capsys.readouterr().out.strip()
    assert new_id.startswith("r_")
    assert cli.main(["feed", new_id, "--json", "-k", "3"]) == 0
    assert len(json.loads(capsys.readouterr().out)["items"]) == 3

    assert cli.main(["feed", "ghost"]) == 2
    assert "ghost" in capsys.readouterr().err


def test_cli_missing_data_is_a_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FANLIGHT_DATA_DIR", str(tmp_path / "nothing"))
    monkeypatch.setenv("FANLIGHT_STATE_DIR", str(tmp_path / "var"))
    assert cli.main(["index"]) == 2
    assert "--demo" in capsys.readouterr().err
