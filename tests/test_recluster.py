"""Tests for T9: HDBSCAN periodic re-cluster and merge job.

Verifies merging multiple person IDs, preserving display names,
and automated global HDBSCAN re-clustering of split face clusters.
"""

import sqlite3
from pathlib import Path
import numpy as np
import pytest

from engine.db.schema import init_db
from engine.embeddings import FaceEmbeddingStore
from engine.recluster import (
    merge_persons,
    recluster_faces_hdbscan,
)


# ── helpers ──────────────────────────────────────────────────────────

def _cluster_points(center_seed: int, count: int, noise: float = 0.02) -> list[np.ndarray]:
    """Generate tight cluster of normalized 512-dim vectors around a seed."""
    rng = np.random.RandomState(center_seed)
    center = rng.randn(512).astype(np.float32)
    center /= np.linalg.norm(center)

    points = []
    for _ in range(count):
        n = rng.randn(512).astype(np.float32) * noise
        p = center + n
        points.append(p / np.linalg.norm(p))
    return points


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    db_path = str(tmp_path / "vault.db")
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def store(tmp_path):
    return FaceEmbeddingStore(tmp_path / "embeddings")


# ── test cases ───────────────────────────────────────────────────────

class TestMergePersons:
    """Tests for explicit merge_persons functionality."""

    def test_merge_updates_faces_and_removes_secondary(self, db):
        # Create 2 persons
        db.execute("INSERT INTO persons (id, display_name) VALUES (1, 'Alice')")
        db.execute("INSERT INTO persons (id, display_name) VALUES (2, NULL)")
        db.execute("INSERT INTO photos (id, filepath) VALUES (1, '/p.jpg')")
        db.execute("INSERT INTO faces (id, photo_id, person_id) VALUES (10, 1, 1)")
        db.execute("INSERT INTO faces (id, photo_id, person_id) VALUES (20, 1, 2)")
        db.commit()

        count = merge_persons(primary_id=1, secondary_ids=[2], conn=db)
        assert count == 1

        # Both faces now belong to person 1
        rows = db.execute("SELECT id, person_id FROM faces ORDER BY id").fetchall()
        assert rows[0][1] == 1
        assert rows[1][1] == 1

        # Person 2 is deleted from persons table
        p_rows = db.execute("SELECT id FROM persons").fetchall()
        assert p_rows == [(1,)]

    def test_merge_adopts_secondary_name_if_primary_empty(self, db):
        db.execute("INSERT INTO persons (id, display_name) VALUES (1, NULL)")
        db.execute("INSERT INTO persons (id, display_name) VALUES (2, 'Bob')")
        db.execute("INSERT INTO photos (id, filepath) VALUES (1, '/p.jpg')")
        db.execute("INSERT INTO faces (id, photo_id, person_id) VALUES (10, 1, 1)")
        db.execute("INSERT INTO faces (id, photo_id, person_id) VALUES (20, 1, 2)")
        db.commit()

        merge_persons(primary_id=1, secondary_ids=[2], conn=db)
        name = db.execute("SELECT display_name FROM persons WHERE id = 1").fetchone()[0]
        assert name == "Bob"


class TestHdbscanRecluster:
    """Tests for global HDBSCAN re-clustering."""

    def test_hdbscan_merges_split_person_clusters(self, db, store):
        db.execute("INSERT INTO photos (id, filepath) VALUES (1, '/p1.jpg')")
        # Suppose Person A was split into two person records: 1 and 2
        db.execute("INSERT INTO persons (id, display_name) VALUES (1, 'Alice')")
        db.execute("INSERT INTO persons (id, display_name) VALUES (2, NULL)")

        # Suppose Person B is person 3
        db.execute("INSERT INTO persons (id, display_name) VALUES (3, 'Charlie')")

        # 4 vectors for Person A (very tight cluster)
        a_vecs = _cluster_points(center_seed=10, count=4, noise=0.01)
        # 4 vectors for Person B (separate tight cluster)
        b_vecs = _cluster_points(center_seed=99, count=4, noise=0.01)

        # Store embeddings
        for i, vec in enumerate(a_vecs[:2]):
            store.save(100 + i, vec)
            # Assigned to Person 1
            db.execute("INSERT INTO faces (photo_id, person_id, face_embedding_id) VALUES (1, 1, ?)", (100 + i,))

        for i, vec in enumerate(a_vecs[2:]):
            store.save(200 + i, vec)
            # Mistakenly assigned to Person 2
            db.execute("INSERT INTO faces (photo_id, person_id, face_embedding_id) VALUES (1, 2, ?)", (200 + i,))

        for i, vec in enumerate(b_vecs):
            store.save(300 + i, vec)
            # Assigned to Person 3
            db.execute("INSERT INTO faces (photo_id, person_id, face_embedding_id) VALUES (1, 3, ?)", (300 + i,))

        db.commit()

        # Run HDBSCAN re-clustering
        res = recluster_faces_hdbscan(db, store, min_cluster_size=2)
        assert res["merges_performed"] >= 1

        # Person 2 should now be merged into Person 1
        remaining_persons = db.execute("SELECT id, display_name FROM persons ORDER BY id").fetchall()
        p_ids = [r[0] for r in remaining_persons]
        assert 1 in p_ids
        assert 2 not in p_ids
        assert 3 in p_ids

        # All 4 faces of Person A now have person_id = 1
        a_faces = db.execute("SELECT DISTINCT person_id FROM faces WHERE face_embedding_id IN (100, 101, 200, 201)").fetchall()
        assert len(a_faces) == 1
        assert a_faces[0][0] == 1
