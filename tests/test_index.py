"""Tests for T7: FAISS faces.index build and incremental add.

Verifies index creation, on-disk persistence, incremental additions,
ID mapping, cosine similarity search, and DB synchronization.
"""

import sqlite3
from pathlib import Path
import numpy as np
import pytest

from engine.db.schema import init_db
from engine.embeddings import FaceEmbeddingStore
from engine.index import FacesIndex, sync_faces_index


# ── helpers ──────────────────────────────────────────────────────────

def _random_normalized_vector(dim: int = 128) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    db_path = str(tmp_path / "vault.db")
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def store(tmp_path):
    return FaceEmbeddingStore(tmp_path / "embeddings", dimension=128)


# ── test cases ───────────────────────────────────────────────────────

class TestFacesIndex:
    """Unit tests for FacesIndex."""

    def test_init_creates_new_empty_index(self, tmp_path):
        idx_path = tmp_path / "faces.index"
        index = FacesIndex(idx_path, dimension=128)
        assert len(index) == 0
        assert index.dimension == 128
        assert idx_path.exists()

    def test_add_and_search(self, tmp_path):
        idx_path = tmp_path / "faces.index"
        index = FacesIndex(idx_path)

        v1 = _random_normalized_vector()
        v2 = _random_normalized_vector()

        index.add([10, 20], np.stack([v1, v2]))
        assert len(index) == 2

        # Search with v1 should return ID 10 with similarity ~ 1.0
        results = index.search(v1, top_k=2)
        assert len(results) == 2
        assert results[0][0] == 10  # ID
        assert pytest.approx(results[0][1], rel=1e-3) == 1.0

    def test_incremental_add_and_persistence(self, tmp_path):
        idx_path = tmp_path / "faces.index"
        idx1 = FacesIndex(idx_path)

        v1 = _random_normalized_vector()
        idx1.add([1], np.expand_dims(v1, 0))
        assert len(idx1) == 1

        # Reload from disk in a second instance
        idx2 = FacesIndex(idx_path)
        assert len(idx2) == 1

        # Incrementally add second vector
        v2 = _random_normalized_vector()
        added = idx2.add([2], np.expand_dims(v2, 0))
        assert added == 1
        assert len(idx2) == 2

        # Attempting to re-add existing ID 1 should be skipped
        added_dup = idx2.add([1], np.expand_dims(v1, 0))
        assert added_dup == 0
        assert len(idx2) == 2

    def test_contains_id(self, tmp_path):
        idx_path = tmp_path / "faces.index"
        index = FacesIndex(idx_path)
        v = _random_normalized_vector()
        index.add([42], np.expand_dims(v, 0))

        assert 42 in index
        assert 99 not in index


class TestSyncFacesIndex:
    """Integration tests for synchronizing faces.index with DB and embedding store."""

    def test_sync_faces_index(self, db, store, tmp_path):
        idx_path = tmp_path / "faces.index"

        # Create photo and faces in DB
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES ('/tmp/p1.jpg', 'embeddings_generated')")
        cur = db.execute("INSERT INTO faces (photo_id, bbox, face_embedding_id) VALUES (1, '0,0,10,10', 101)")
        cur = db.execute("INSERT INTO faces (photo_id, bbox, face_embedding_id) VALUES (1, '20,20,30,30', 102)")
        db.commit()

        # Save embeddings in store
        v101 = _random_normalized_vector()
        v102 = _random_normalized_vector()
        store.save(101, v101)
        store.save(102, v102)

        # Sync into index
        added = sync_faces_index(db, store, idx_path)
        assert added == 2

        index = FacesIndex(idx_path)
        assert len(index) == 2
        assert 101 in index
        assert 102 in index

        # Second sync should add 0 new
        added_again = sync_faces_index(db, store, idx_path)
        assert added_again == 0
