from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from fanlight import cli
from fanlight.evaluation import (
    Persona,
    _feed_metrics,
    default_policies,
    relevance,
    simulate,
    simulate_seeds,
    v1_model_policy,
)
from fanlight.profiles import Profile


def _stories_in(index, category: str, n: int) -> list[int]:
    column = index.category_keys.index(category)
    english = index.languages.index("english")
    rows = np.flatnonzero((index.category_matrix[:, column] > 0) & (index.language_id == english))
    return rows[:n].tolist()


def test_unstated_found_ignores_the_stated_interest(index):
    stated, other = index.category_keys[0], index.category_keys[1]
    persona = Persona("english", {stated: 1.0, other: 0.7}, stated=stated)
    gains = relevance(index, persona)
    unread = np.ones(index.size, dtype=bool)
    only_stated = _feed_metrics(index, _stories_in(index, stated, 5), gains, unread, persona)
    assert only_stated["unstated_found"] == 0.0
    assert only_stated["interest_recall"] == 0.5  # the old, looser measure counts the stated one
    both = _feed_metrics(
        index, _stories_in(index, stated, 3) + _stories_in(index, other, 2), gains, unread, persona
    )
    assert both["unstated_found"] == 1.0


def test_simulation_is_reproducible_for_a_seed(index):
    first = simulate(index, n_readers=6, rounds=2, feed_size=8, seed=3)
    again = simulate(index, n_readers=6, rounds=2, feed_size=8, seed=3)
    other = simulate(index, n_readers=6, rounds=2, feed_size=8, seed=4)
    assert first["policies"] == again["policies"]
    assert first["policies"] != other["policies"]


def test_seed_summary(index):
    report = simulate_seeds(
        index,
        [0, 1, 2],
        policy_factory=lambda: default_policies(index, now=index.as_of),
        n_readers=5,
        rounds=2,
        feed_size=8,
    )
    assert report["seeds"] == [0, 1, 2] and len(report["runs"]) == 3
    stats = report["summary"]["fanlight"]["nDCG end"]
    finals = [run["policies"]["fanlight"][-1]["ndcg"] for run in report["runs"]]
    assert stats["mean"] == pytest.approx(np.mean(finals), abs=1e-4)
    assert stats["sd"] == pytest.approx(np.std(finals, ddof=1), abs=1e-4)
    assert stats["min"] <= stats["mean"] <= stats["max"]
    with pytest.raises(ValueError):
        simulate_seeds(index, [])


class _StandInModel:
    """Scores like v1's model in spirit: stated-category overlap first, then freshness."""

    def predict(self, features):
        return features[:, 2] * 100 - features[:, 3]


def test_v1_model_policy_uses_v1_features(index):
    stated = index.category_keys[0]
    policy = v1_model_policy(index, _StandInModel())
    picks = policy(Profile(id="v1", languages=("english",), categories={stated: 1.0}), 10)
    column = index.category_keys.index(stated)
    assert len(picks) == 10 and all(index.category_matrix[i, column] > 0 for i in picks)


def test_cli_evaluate_with_v1_model_and_seeds(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    assert cli.main(["demo-data", str(data), "--articles", "700", "--users", "8"]) == 0
    monkeypatch.setenv("FANLIGHT_DATA_DIR", str(data))
    monkeypatch.setenv("FANLIGHT_STATE_DIR", str(tmp_path / "var"))
    model = tmp_path / "trained_model.pkl"
    model.write_bytes(pickle.dumps(_StandInModel()))
    capsys.readouterr()

    common = ["evaluate", "--readers", "4", "--rounds", "2", "-k", "6", "--v1-model", str(model)]
    assert cli.main([*common, "--seeds", "2", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["seeds"] == [0, 1]
    assert set(report["summary"]) == {
        "fanlight",
        "fanlight, no variety or discovery",
        "v1 trained model",
        "newest first",
        "random",
    }
    assert cli.main([*common, "--seeds", "2"]) == 0
    assert "2 seeds (0, 1)" in capsys.readouterr().out
    assert cli.main(common[:-2]) == 0  # without the model, the label formula stands in
    out = capsys.readouterr().out
    assert "v1 label formula" in out and "unstated found" in out
