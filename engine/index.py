"""
PhotoVault — FAISS faces.index management (T7).

Maintains the local FAISS vector index (``faces.index``) for fast
cosine-similarity search over 512-dimensional ArcFace face embeddings.
Supports incremental additions by ID and persistence.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from engine.embeddings import FaceEmbeddingStore


class FacesIndex:
    """Wrapper around FAISS IndexIDMap2(IndexFlatIP) for face embeddings."""

    def __init__(self, index_path: str | Path, dimension: int = 128):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._ids: set[int] = set()

        if self.index_path.exists() and self.index_path.stat().st_size > 0:
            self._load()
        else:
            self._init_empty()

    def _init_empty(self) -> None:
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap2(base_index)
        self._ids = set()
        self._save()

    def _load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        self.dimension = self.index.d
        self._ids = set()
        if hasattr(self.index, "id_map"):
            id_array = faiss.vector_to_array(self.index.id_map)
            self._ids = {int(x) for x in id_array}

    def _save(self) -> None:
        tmp_path = self.index_path.with_name(self.index_path.name + ".tmp")
        faiss.write_index(self.index, str(tmp_path))
        if tmp_path.exists():
            os.replace(tmp_path, self.index_path)

    def add(
        self,
        ids: Sequence[int] | np.ndarray,
        vectors: np.ndarray,
    ) -> int:
        """Incrementally add vectors with explicit integer IDs.

        Any IDs that are already present in the index are skipped.
        Vectors are normalized float32 before addition.

        Returns the number of newly added vectors.
        """
        raw_ids = [int(i) for i in ids]
        raw_vecs = np.asarray(vectors, dtype=np.float32)

        if raw_vecs.ndim == 1:
            raw_vecs = np.expand_dims(raw_vecs, axis=0)

        if len(raw_ids) != raw_vecs.shape[0]:
            raise ValueError(f"IDs length ({len(raw_ids)}) must match vectors row count ({raw_vecs.shape[0]})")

        # Filter out existing IDs
        new_ids: list[int] = []
        new_vec_list: list[np.ndarray] = []
        for emb_id, vec in zip(raw_ids, raw_vecs):
            if emb_id not in self._ids:
                # Ensure L2 normalized
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                new_ids.append(emb_id)
                new_vec_list.append(vec)

        if not new_ids:
            return 0

        vec_matrix = np.stack(new_vec_list).astype(np.float32)
        id_arr = np.array(new_ids, dtype=np.int64)

        self.index.add_with_ids(vec_matrix, id_arr)
        self._ids.update(new_ids)
        self._save()

        return len(new_ids)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[tuple[int, float]]:
        """Search nearest neighbors by cosine similarity.

        Returns list of (embedding_id, similarity_score) tuples,
        sorted descending by similarity.
        """
        if len(self) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        q = np.expand_dims(q, axis=0)

        k = min(top_k, len(self))
        distances, indices = self.index.search(q, k)

        results: list[tuple[int, float]] = []
        for score, emb_id in zip(distances[0], indices[0]):
            if emb_id != -1 and (min_score is None or score >= min_score):
                results.append((int(emb_id), float(score)))

        return results

    def __contains__(self, item: int) -> bool:
        return int(item) in self._ids

    def __len__(self) -> int:
        return self.index.ntotal


def sync_faces_index(
    conn: sqlite3.Connection,
    store: FaceEmbeddingStore,
    index_or_path: FacesIndex | str | Path = "faces.index",
) -> int:
    """Synchronize all generated face embeddings from DB and store into faces.index.

    Returns the count of newly indexed face embeddings.
    """
    if isinstance(index_or_path, FacesIndex):
        index = index_or_path
    else:
        index = FacesIndex(index_or_path)

    # Fetch all face_embedding_ids recorded in database
    rows = conn.execute(
        "SELECT DISTINCT face_embedding_id FROM faces WHERE face_embedding_id IS NOT NULL ORDER BY face_embedding_id ASC"
    ).fetchall()

    missing_ids: list[int] = []
    missing_vecs: list[np.ndarray] = []

    for (emb_id,) in rows:
        emb_id = int(emb_id)
        if emb_id not in index:
            vec = store.get(emb_id)
            if vec is not None and len(vec) == index.dimension:
                missing_ids.append(emb_id)
                missing_vecs.append(vec)

    if not missing_ids:
        return 0

    return index.add(missing_ids, np.stack(missing_vecs))
