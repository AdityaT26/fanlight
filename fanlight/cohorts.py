"""Reader cohorts: k-means over stated category interests, with readable labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from .index import CorpusIndex
from .profiles import Profile

NEW_READER = "New readers"


@dataclass
class CohortModel:
    category_keys: list[str]
    centroids: np.ndarray
    labels: list[str]
    sizes: list[int]

    def assign(self, vector: np.ndarray) -> str:
        """Label of the nearest cohort for a category-affinity vector."""
        vector = np.clip(np.asarray(vector, dtype=np.float64), 0.0, None)
        norm = np.linalg.norm(vector)
        if norm == 0 or len(self.labels) == 0:
            return NEW_READER
        distances = np.linalg.norm(self.centroids - vector / norm, axis=1)
        return self.labels[int(distances.argmin())]


def _vector(profile: Profile, positions: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(positions), dtype=np.float64)
    for key, weight in profile.categories.items():
        if key in positions:
            vector[positions[key]] = max(float(weight), 0.0)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _labels(centroids: np.ndarray, baseline: np.ndarray, names: list[str]) -> list[str]:
    """Name each cohort after the categories it over-indexes on."""
    labels: list[str] = []
    for centroid in centroids:
        lift = centroid - baseline
        order = [j for j in np.argsort(-lift) if lift[j] > 0.02][:3]
        if not order:
            order = list(np.argsort(-centroid)[:2])
        label = " & ".join(names[j] for j in order[:2])
        if label in labels and len(order) > 2:
            label = " & ".join(names[j] for j in (order[0], order[2]))
        suffix = 2
        base = label
        while label in labels:
            label = f"{base} ({suffix})"
            suffix += 1
        labels.append(label)
    return labels


def fit_cohorts(
    index: CorpusIndex, profiles: list[Profile], k: int = 8, seed: int = 42
) -> tuple[CohortModel, dict[str, str]]:
    """Cluster profiles by stated interests. Returns the model and id -> label."""
    positions = {key: j for j, key in enumerate(index.category_keys)}
    names = [
        index.corpus.categories[key].name if key in index.corpus.categories else key
        for key in index.category_keys
    ]
    vectors = np.array([_vector(p, positions) for p in profiles]).reshape(len(profiles), len(positions))
    has_interests = np.linalg.norm(vectors, axis=1) > 0 if len(profiles) else np.zeros(0, dtype=bool)
    assignment = {p.id: NEW_READER for p in profiles}

    usable = vectors[has_interests]
    distinct = len(np.unique(usable.round(6), axis=0)) if len(usable) else 0
    k = min(k, distinct)
    if k < 1:
        empty = np.zeros((0, len(positions)))
        return CohortModel(index.category_keys, empty, [], []), assignment

    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(usable)
    labels = _labels(model.cluster_centers_, usable.mean(axis=0), names)
    members = [p for p, ok in zip(profiles, has_interests, strict=False) if ok]
    for profile, cluster in zip(members, model.labels_, strict=False):
        assignment[profile.id] = labels[int(cluster)]
    sizes = [int((model.labels_ == c).sum()) for c in range(k)]
    return CohortModel(index.category_keys, model.cluster_centers_, labels, sizes), assignment
