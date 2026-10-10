"""Tests for T8: Incremental face clustering by cosine threshold.

Verifies matching similar faces to existing person_id, creating new person_id
when similarity is below threshold, and DB batch clustering.
"""

import sqlite3
from pathlib import Path
import numpy as np
import pytest

from engine.db.schema import init_db
from engine.embeddings import FaceEmbeddingStore
from engine.index import FacesIndex
from engine.cluster import (
    match_or_create_person,
    cluster_unassigned_faces,
)


# ── helpers ──────────────────────────────────────────────────────────

def _make_unit_vector(seed: int, noise: float = 0.0, dim: int = 128) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = rng.randn(dim).astype(np.float32)
    base /= np.linalg.norm(base)

    if noise > 0:
        n = rng.randn(dim).astype(np.float32)
        n /= np.linalg.norm(n)
        v = base + noise * n
        return v / np.linalg.norm(v)
    return base


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


@pytest.fixture()
def index(tmp_path):
    return FacesIndex(tmp_path / "faces.index")


# ── test cases ───────────────────────────────────────────────────────

class TestMatchOrCreatePerson:
    """Tests for individual face match or new person assignment."""

    def test_first_face_creates_new_person(self, db, store, index):
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES ('/p1.jpg', 'embeddings_generated')")
        cur = db.execute("INSERT INTO faces (photo_id, bbox, face_embedding_id) VALUES (1, '0,0,10,10', 101)")
        face_id = cur.lastrowid
        db.commit()

        v1 = _make_unit_vector(seed=42)
        store.save(101, v1)
        index.add([101], np.expand_dims(v1, 0))

        person_id, is_new = match_or_create_person(face_id, db, store, index, threshold=0.65)
        assert is_new is True
        assert person_id is not None

        # Verify DB updated
        row = db.execute("SELECT person_id FROM faces WHERE id = ?", (face_id,)).fetchone()
        assert row[0] == person_id
        # Verify person exists in persons table
        p_row = db.execute("SELECT id, display_name FROM persons WHERE id = ?", (person_id,)).fetchone()
        assert p_row is not None
        assert p_row[0] == person_id

    def test_similar_face_matches_existing_person(self, db, store, index):
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES ('/p1.jpg', 'embeddings_generated')")
        cur = db.execute("INSERT INTO faces (photo_id, bbox, face_embedding_id) VALUES (1, '0,0,10,10', 101)")
        face1_id = cur.lastrowid
        cur = db.execute("INSERT INTO faces (photo_id, bbox, face_embedding_id) VALUES (1, '20,20,30,30', 102)")
        face2_id = cur.lastrowid
        db.commit()

        # Face 1 & 2 have very close embeddings (noise = 0.05 -> similarity > 0.95)
        v1 = _make_unit_vector(seed=42)
        v2 = _make_unit_vector(seed=42, noise=0.05)
        store.save(101, v1)
        store.save(102, v2)
        index.add([101], np.expand_dims(v1, 0))

        # First face gets person
        p1, _ = match_or_create_person(face1_id, db, store, index, threshold=0.65)

        # Index face 2
        index.add([102], np.expand_dims(v2, 0))

        # Second face should match person 1
        p2, is_new2 = match_or_create_person(face2_id, db, store, index, threshold=0.65)
        assert is_new2 is False
        assert p2 == p1

    def test_dissimilar_face_creates_different_person(self, db, store, index):
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES ('/p1.jpg', 'embeddings_generated')")
        cur = db.execute("INSERT INTO faces (photo_id, bbox, face_embedding_id) VALUES (1, '0,0,10,10', 101)")
        face1_id = cur.lastrowid
        cur = db.execute("INSERT INTO faces (photo_id, bbox, face_embedding_id) VALUES (1, '20,20,30,30', 102)")
        face2_id = cur.lastrowid
        db.commit()

        # Seed 42 and Seed 999 are orthogonal random vectors (similarity ~ 0)
        v1 = _make_unit_vector(seed=42)
        v2 = _make_unit_vector(seed=999)
        store.save(101, v1)
        store.save(102, v2)
        index.add([101], np.expand_dims(v1, 0))

        p1, _ = match_or_create_person(face1_id, db, store, index, threshold=0.65)

        index.add([102], np.expand_dims(v2, 0))
        p2, is_new2 = match_or_create_person(face2_id, db, store, index, threshold=0.65)

        assert is_new2 is True
        assert p2 != p1


class TestBatchCluster:
    """Tests for cluster_unassigned_faces batch function."""

    def test_batch_clustering(self, db, store, index):
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES ('/p1.jpg', 'embeddings_generated')")
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES ('/p2.jpg', 'embeddings_generated')")

        # Create 4 faces: two of Person A (seed 1), two of Person B (seed 2)
        va1 = _make_unit_vector(seed=1)
        va2 = _make_unit_vector(seed=1, noise=0.03)
        vb1 = _make_unit_vector(seed=2)
        vb2 = _make_unit_vector(seed=2, noise=0.03)

        store.save(1, va1)
        store.save(2, va2)
        store.save(3, vb1)
        store.save(4, vb2)

        db.execute("INSERT INTO faces (photo_id, bbox, face_embedding_id) VALUES (1, '0,0,1,1', 1)")
        db.execute("INSERT INTO faces (photo_id, bbox, face_embedding_id) VALUES (1, '2,2,3,3', 2)")
        db.execute("INSERT INTO faces (photo_id, bbox, face_embedding_id) VALUES (2, '0,0,1,1', 3)")
        db.execute("INSERT INTO faces (photo_id, bbox, face_embedding_id) VALUES (2, '2,2,3,3', 4)")
        db.commit()

        summary = cluster_unassigned_faces(db, store, index, threshold=0.65)
        assert summary["faces_processed"] == 4
        assert summary["new_persons"] == 2  # Exactly two distinct people created

        # Faces 1 and 2 should share the same person_id
        p_a1 = db.execute("SELECT person_id FROM faces WHERE id = 1").fetchone()[0]
        p_a2 = db.execute("SELECT person_id FROM faces WHERE id = 2").fetchone()[0]
        assert p_a1 == p_a2

        # Faces 3 and 4 should share the same person_id
        p_b1 = db.execute("SELECT person_id FROM faces WHERE id = 3").fetchone()[0]
        p_b2 = db.execute("SELECT person_id FROM faces WHERE id = 4").fetchone()[0]
        assert p_b1 == p_b2

        assert p_a1 != p_b1
