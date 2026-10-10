"""Tests for T1: SQLite schema creation and idempotent migration.

Verifies that init_db creates the 4 approved tables (photos, persons,
faces, clip_embeddings) with the correct columns, and that calling it
twice is safe (idempotent).
"""

import os
import sqlite3
import tempfile

import pytest

from engine.db.schema import init_db


# ── helpers ──────────────────────────────────────────────────────────

def _table_columns(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    """Return [(col_name, col_type), ...] for *table*."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    # PRAGMA table_info → (cid, name, type, notnull, dflt_value, pk)
    return [(r[1], r[2]) for r in rows]


def _table_names(conn: sqlite3.Connection) -> set[str]:
    """Return the set of user-created table names in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def db_path(tmp_path):
    """Yield a temporary DB path, cleaned up automatically by pytest."""
    return str(tmp_path / "test_vault.db")


# ── test cases ───────────────────────────────────────────────────────

class TestInitDb:
    """Suite for init_db schema creation."""

    def test_creates_all_four_tables(self, db_path):
        """init_db must create photos, persons, faces, clip_embeddings."""
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        tables = _table_names(conn)
        conn.close()
        assert {"photos", "persons", "faces", "clip_embeddings"} <= tables

    def test_photos_columns(self, db_path):
        """photos table must have the approved columns."""
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = _table_columns(conn, "photos")
        conn.close()
        col_names = [c[0] for c in cols]
        assert "id" in col_names
        assert "filepath" in col_names
        assert "date_taken" in col_names
        assert "phash" in col_names
        assert "processing_state" in col_names
        assert "created_at" in col_names

    def test_persons_columns(self, db_path):
        """persons table must have the approved columns."""
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = _table_columns(conn, "persons")
        conn.close()
        col_names = [c[0] for c in cols]
        assert "id" in col_names
        assert "display_name" in col_names
        assert "created_at" in col_names

    def test_faces_columns(self, db_path):
        """faces table must have the approved columns."""
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = _table_columns(conn, "faces")
        conn.close()
        col_names = [c[0] for c in cols]
        assert "id" in col_names
        assert "photo_id" in col_names
        assert "person_id" in col_names
        assert "bbox" in col_names
        assert "face_embedding_id" in col_names

    def test_clip_embeddings_columns(self, db_path):
        """clip_embeddings table must have the approved columns."""
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = _table_columns(conn, "clip_embeddings")
        conn.close()
        col_names = [c[0] for c in cols]
        assert "id" in col_names
        assert "photo_id" in col_names
        assert "embedding_id" in col_names

    def test_idempotent(self, db_path):
        """Calling init_db twice must not raise or duplicate tables."""
        init_db(db_path)
        init_db(db_path)  # second call — should be safe
        conn = sqlite3.connect(db_path)
        tables = _table_names(conn)
        conn.close()
        assert {"photos", "persons", "faces", "clip_embeddings"} <= tables

    def test_photos_filepath_unique(self, db_path):
        """filepath in photos must have a UNIQUE constraint."""
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO photos (filepath, processing_state) VALUES (?, ?)",
            ("/a/b.jpg", "pending"),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO photos (filepath, processing_state) VALUES (?, ?)",
                ("/a/b.jpg", "pending"),
            )
        conn.close()

    def test_faces_fk_photo_id(self, db_path):
        """faces.photo_id must reference photos.id (FK enforced)."""
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        # Insert without a matching photo → should fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO faces (photo_id, bbox) VALUES (?, ?)",
                (9999, "0,0,100,100"),
            )
        conn.close()

    def test_clip_embeddings_fk_photo_id(self, db_path):
        """clip_embeddings.photo_id must reference photos.id (FK enforced)."""
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO clip_embeddings (photo_id, embedding_id) VALUES (?, ?)",
                (9999, 1),
            )
        conn.close()

    def test_db_file_created(self, db_path):
        """init_db must actually create the .db file on disk."""
        assert not os.path.exists(db_path)
        init_db(db_path)
        assert os.path.exists(db_path)
