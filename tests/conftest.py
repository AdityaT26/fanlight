from __future__ import annotations

import copy

import pytest

from fanlight.config import Settings
from fanlight.engine import Engine
from fanlight.index import build_index
from fanlight.ingest import build_corpus
from fanlight.profiles import dataset_profiles
from fanlight.store import Store
from fanlight.synth import make_raw


@pytest.fixture(scope="session")
def raw():
    return make_raw(n_articles=500, n_users=40, seed=11)


@pytest.fixture(scope="session")
def index(raw):
    return build_index(build_corpus(copy.deepcopy(raw)))


@pytest.fixture(scope="session")
def profiles(raw, index):
    return dataset_profiles(raw, index.corpus)


@pytest.fixture()
def engine(index, profiles, tmp_path):
    """A fresh engine (own SQLite file) over the shared, read-only index."""
    store = Store(tmp_path / "test.db")
    yield Engine(index, profiles, store=store, settings=Settings(state_dir=tmp_path, cohorts=4))
    store.close()


def key_of(index, name: str) -> str:
    return next(k for k, c in index.corpus.categories.items() if c.name == name)
