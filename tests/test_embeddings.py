"""Tests for T6: ArcFace embedding generation for detected faces.

Verifies ArcFace embedding extraction against real local images in tmp_path
as required by AGENTS.md Definition of Done.
"""

import sqlite3
from pathlib import Path
import cv2
import numpy as np
import pytest
import insightface

from engine.db.schema import init_db
from engine.faces import process_photo_faces
from engine.embeddings import (
    FaceEmbeddingStore,
    get_face_embedder,
    generate_face_embeddings,
    batch_generate_face_embeddings,
)


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    """Return an initialised in-tmp-dir DB connection."""
    db_path = str(tmp_path / "vault.db")
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def store(tmp_path):
    """Return a fresh FaceEmbeddingStore in tmp_path."""
    return FaceEmbeddingStore(tmp_path / "embeddings")


@pytest.fixture()
def photo_with_faces(tmp_path):
    """Save a real photo with faces to tmp_path."""
    img = insightface.data.get_image("t1")
    p = tmp_path / "faces.jpg"
    cv2.imwrite(str(p), img)
    return p


# ── test cases ───────────────────────────────────────────────────────

class TestFaceEmbeddingStore:
    """Unit tests for the vector embedding storage layer."""

    def test_save_and_retrieve(self, tmp_path):
        store = FaceEmbeddingStore(tmp_path / "emb1")
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)

        store.save(101, vec)
        retrieved = store.get(101)
        assert retrieved is not None
        assert np.allclose(vec, retrieved)

    def test_persistence_on_reload(self, tmp_path):
        dir_path = tmp_path / "emb_persist"
        store1 = FaceEmbeddingStore(dir_path)
        vec1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
        vec2 = np.ones(512, dtype=np.float32) * 2 / np.sqrt(512 * 4)

        store1.save(1, vec1)
        store1.save(2, vec2)

        # Reopen with new instance pointing to same directory
        store2 = FaceEmbeddingStore(dir_path)
        assert len(store2) == 2
        assert np.allclose(store2.get(1), vec1)
        assert np.allclose(store2.get(2), vec2)

    def test_get_all(self, tmp_path):
        store = FaceEmbeddingStore(tmp_path / "emb_all")
        for i in range(5):
            vec = np.random.randn(512).astype(np.float32)
            store.save(i + 10, vec)

        ids, mat = store.get_all()
        assert len(ids) == 5
        assert mat.shape == (5, 512)
        assert set(ids) == {10, 11, 12, 13, 14}


class TestArcFaceEmbeddings:
    """Tests for ArcFace embedding generation on real images."""

    def test_embedder_initializes(self):
        embedder = get_face_embedder()
        assert embedder is not None

    def test_generate_face_embeddings(self, db, store, photo_with_faces):
        # 1. Ingest photo and detect faces (T5)
        fp = str(photo_with_faces)
        cur = db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (fp,))
        photo_id = cur.lastrowid
        db.commit()

        process_photo_faces(photo_id, db)
        face_rows_before = db.execute("SELECT id, face_embedding_id FROM faces WHERE photo_id = ?", (photo_id,)).fetchall()
        assert len(face_rows_before) > 0
        for r in face_rows_before:
            assert r[1] is None

        # 2. Generate embeddings (T6)
        generated = generate_face_embeddings(photo_id, db, store)
        assert len(generated) == len(face_rows_before)

        # Verify DB updated
        face_rows_after = db.execute("SELECT id, face_embedding_id FROM faces WHERE photo_id = ?", (photo_id,)).fetchall()
        for r in face_rows_after:
            emb_id = r[1]
            assert emb_id is not None
            # Verify stored vector is 128-dim (SFace) or 512-dim (ArcFace) and L2-normalized
            vec = store.get(emb_id)
            assert vec is not None
            assert vec.shape in ((128,), (512,))
            assert pytest.approx(float(np.linalg.norm(vec)), rel=1e-3) == 1.0

        # Verify photo processing_state updated
        p_row = db.execute("SELECT processing_state FROM photos WHERE id = ?", (photo_id,)).fetchone()
        assert p_row[0] == "embeddings_generated"

    def test_batch_generate_face_embeddings(self, db, store, photo_with_faces):
        fp = str(photo_with_faces)
        cur = db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (fp,))
        photo_id = cur.lastrowid
        db.commit()

        # Run face detection
        process_photo_faces(photo_id, db)

        # Run batch embedding generation
        summary = batch_generate_face_embeddings(db, store)
        assert summary["photos_processed"] == 1
        assert summary["embeddings_generated"] > 0
        assert summary["errors"] == 0

        # Calling it again processes 0 un-embedded photos
        summary2 = batch_generate_face_embeddings(db, store)
        assert summary2["photos_processed"] == 0
