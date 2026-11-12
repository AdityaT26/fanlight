"""Configuration: ranking knobs and runtime settings.

Everything that changes ranking behaviour lives in :class:`RankingConfig`, so a
feed is fully described by (profile, config). Runtime settings (paths, secrets)
come from the environment and never from source code.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

#: Signal names, in the order they are reported everywhere.
SIGNALS: tuple[str, ...] = (
    "category",
    "tags",
    "content",
    "freshness",
    "source",
    "popularity",
)


@dataclass(frozen=True)
class Weights:
    """Relative importance of each signal. Only ratios matter."""

    category: float = 0.32
    tags: float = 0.10
    content: float = 0.22
    freshness: float = 0.20
    source: float = 0.08
    popularity: float = 0.08

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in SIGNALS}

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Weights:
        unknown = set(values) - set(SIGNALS)
        if unknown:
            raise ValueError(f"unknown signal(s): {', '.join(sorted(unknown))}")
        clean: dict[str, float] = {}
        for name, raw in values.items():
            value = float(raw)
            if not (0.0 <= value <= 10.0):
                raise ValueError(f"weight '{name}' must be between 0 and 10")
            clean[name] = value
        return replace(cls(), **clean)


_RANGES: dict[str, tuple[float, float]] = {
    "half_life_hours": (1.0, 24.0 * 90),
    "diversity": (0.0, 1.0),
    "discovery": (0.0, 0.5),
    "pool_size": (20, 5000),
    "repeat_window": (1, 20),
    "event_half_life_days": (0.5, 3650.0),
    "seen_decay": (0.3, 1.0),
}


@dataclass(frozen=True)
class RankingConfig:
    """All ranking knobs. Immutable; use :meth:`with_overrides` to derive one."""

    weights: Weights = field(default_factory=Weights)
    #: Hours for the freshness signal to halve.
    half_life_hours: float = 48.0
    #: 0 = pure relevance order, 1 = strongest diversification.
    diversity: float = 0.25
    #: Share of feed slots reserved for stories outside the reader's known
    #: interests (0 = none, 0.5 = every other slot).
    discovery: float = 0.10
    #: Drop stories the reader already opened, liked or saved.
    hide_read: bool = True
    #: How many top-scoring candidates the diversity re-ranker may choose from.
    pool_size: int = 400
    #: Look-back window (in feed positions) for category/source repetition.
    repeat_window: int = 5
    #: Ignore stories older than this many days (None = no limit). If that
    #: leaves fewer stories than requested, the limit is lifted for that feed.
    max_age_days: float | None = 14.0
    #: Half-life, in days, of a reader's past interactions.
    event_half_life_days: float = 21.0
    #: Score multiplier per unanswered impression of the same story.
    seen_decay: float = 0.9

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> RankingConfig:
        """Return a copy with validated overrides applied."""
        if not overrides:
            return self
        allowed = {f.name for f in fields(self)}
        changes: dict[str, Any] = {}
        for key, value in overrides.items():
            if value is None and key != "max_age_days":
                continue
            if key not in allowed:
                raise ValueError(f"unknown ranking option '{key}'")
            if key == "weights":
                changes[key] = Weights.from_mapping({**self.weights.as_dict(), **dict(value)})
            elif key == "hide_read":
                changes[key] = bool(value)
            elif key == "max_age_days":
                if value is not None and float(value) <= 0:
                    raise ValueError("max_age_days must be positive")
                changes[key] = None if value is None else float(value)
            else:
                low, high = _RANGES[key]
                number = float(value)
                if not (low <= number <= high):
                    raise ValueError(f"{key} must be between {low:g} and {high:g}")
                changes[key] = int(number) if key in ("pool_size", "repeat_window") else number
        return replace(self, **changes)

    def public_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.as_dict(),
            "half_life_hours": self.half_life_hours,
            "diversity": self.diversity,
            "discovery": self.discovery,
            "hide_read": self.hide_read,
            "max_age_days": self.max_age_days,
        }


@dataclass(frozen=True)
class Settings:
    """Runtime settings, read from ``FANLIGHT_*`` environment variables."""

    data_dir: Path = Path("data")
    state_dir: Path = Path("var")
    #: ``latest`` (newest story in the corpus), ``now``, or an ISO timestamp.
    as_of: str = "latest"
    mongo_uri: str | None = None
    mongo_db: str = "unbiasly"
    cohorts: int = 8

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, dotenv: str | Path = ".env") -> Settings:
        """Read settings from ``env`` (default: a ``.env`` file overlaid by the process environment)."""
        if env is None:
            env = {**read_dotenv(dotenv), **os.environ}
        return cls(
            data_dir=Path(env.get("FANLIGHT_DATA_DIR", "data")),
            state_dir=Path(env.get("FANLIGHT_STATE_DIR", "var")),
            as_of=env.get("FANLIGHT_AS_OF", "latest"),
            mongo_uri=env.get("FANLIGHT_MONGO_URI") or None,
            mongo_db=env.get("FANLIGHT_MONGO_DB", "unbiasly"),
            cohorts=int(env.get("FANLIGHT_COHORTS", "8")),
        )

    @property
    def db_path(self) -> Path:
        return self.state_dir / "fanlight.db"


def read_dotenv(path: str | Path) -> dict[str, str]:
    """Minimal ``.env`` reader: ``KEY=value`` lines, ``#`` comments, optional quotes."""
    values: dict[str, str] = {}
    file = Path(path)
    if not file.is_file():
        return values
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values
