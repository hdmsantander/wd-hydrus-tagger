"""Unsupervised face clustering (radius neighbours + staged min_faces)."""

from __future__ import annotations

from collections import Counter
from typing import Callable, Literal

import numpy as np

DistanceMethod = Literal["cosine_similarity", "euclidean"]


def embedding_distance(a: np.ndarray, b: np.ndarray, method: DistanceMethod) -> float:
    if method == "cosine_similarity":
        return float(1.0 - np.dot(a, b))
    return float(np.linalg.norm(a - b))


def cluster_faces(
    faces: list[dict],
    *,
    max_distance: float,
    min_faces: int,
    allow_new: bool = True,
    distance_method: DistanceMethod = "cosine_similarity",
    create_person: Callable[[], str],
    assign_person: Callable[[int, str], None],
) -> int:
    """Assign ``person_id`` to face rows using radius neighbour clustering.

    Each face dict must include ``id`` (int row id) and ``emb`` (np.ndarray).
    ``create_person`` returns a new person id (e.g. ``p3``); ``assign_person(face_id, person_id)``
    persists the assignment.
    """
    if not faces:
        return 0

    from sklearn.neighbors import NearestNeighbors

    n = len(faces)
    x = np.vstack([f["emb"] for f in faces])
    metric = "cosine" if distance_method == "cosine_similarity" else "euclidean"
    nbrs = NearestNeighbors(radius=max_distance, metric=metric, n_jobs=-1)
    nbrs.fit(x)

    raw_neighbours = nbrs.radius_neighbors(x, return_distance=False)
    neighbour_indices = [np.setdiff1d(neigh, [i]) for i, neigh in enumerate(raw_neighbours)]
    degree = np.array([len(neigh) for neigh in neighbour_indices])
    order = np.argsort(-degree)

    person_ids: list[str | None] = [None] * n
    assigned = 0

    for idx in order:
        if person_ids[idx] is not None:
            continue

        neighbours = neighbour_indices[idx]
        existing = [person_ids[j] for j in neighbours if person_ids[j] is not None]
        if existing:
            best_pid = Counter(existing).most_common(1)[0][0]
        elif len(neighbours) >= min_faces and allow_new:
            best_pid = create_person()
        else:
            continue

        assign_person(int(faces[idx]["id"]), best_pid)
        person_ids[idx] = best_pid
        faces[idx]["person_id"] = best_pid
        assigned += 1

    return assigned


def staged_min_faces_list(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw if int(x) > 0]
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return [int(p) for p in parts if int(p) > 0]
