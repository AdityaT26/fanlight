"""Score the simulated readers with v1's actual trained XGBoost model.

Not part of the package: it needs the v1 artifact and xgboost.

    pip install "xgboost==2.1.3"
    python scripts/compare_v1_model.py path/to/trained_model.pkl

v1 built four features per (reader, story): reader language (1 english, 2 other),
language match, number of stated categories the story is in, and days since the
story was updated. v1 measured age against the wall clock, which by now puts every
story far outside the range the model was trained on; here age is measured from the
newest story in the dataset instead, which is the kindest reading of v1.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np

from fanlight.config import Settings
from fanlight.engine import Engine
from fanlight.evaluation import _unread, default_policies, format_report, simulate
from fanlight.store import Store


def v1_model_policy(index, model):
    age_days = np.floor((index.as_of - index.published) / 86_400.0)

    def policy(profile, k):
        language = profile.languages[0] if profile.languages else "english"
        stated = np.array([key in profile.categories for key in index.category_keys], dtype=float)
        features = np.column_stack(
            [
                np.full(index.size, 1.0 if language == "english" else 2.0),
                (index.language_id == index.languages.index(language)).astype(float)
                if language in index.languages
                else np.zeros(index.size),
                index.category_matrix @ stated,
                age_days,
            ]
        )
        score = model.predict(features).astype(float)
        score[~_unread(index, profile)] = -np.inf
        return np.argsort(-score, kind="stable")[:k].tolist()

    return policy


def main(path: str) -> None:
    with open(path, "rb") as handle:
        model = pickle.load(handle)  # only unpickle files you trust
    engine = Engine.load(Settings.from_env(), store=Store(":memory:"))
    policies = default_policies(engine.index, now=engine.index.as_of)
    policies["v1 trained model"] = v1_model_policy(engine.index, model)
    print(format_report(simulate(engine.index, n_readers=60, rounds=6, policies=policies)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "trained_model.pkl")
