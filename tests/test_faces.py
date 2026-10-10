"""Tests for T5: Face detection (SCRFD/RetinaFace ONNX).

Verifies face detection on real local image files in tmp_path (both
non-face images and bundled photo samples) as required by AGENTS.md
Definition of Done.
"""

import sqlite3
from pathlib import Path
import cv2
import numpy as np
import pytest
import insightface

from engine.db.schema import init_db
from engine.faces import (
    get_face_detector,
    detect_faces,
    process_photo_faces,
    batch_detect_faces,
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
def test_images(tmp_path):
    """Provide real image files on disk in tmp_path."""
    d = tmp_path / "face_photos"
    d.mkdir()

    # 1. Real photo with multiple faces (from bundled insightface data)
    sample_img = insightface.data.get_image("t1")
    p_with_faces = d / "people.jpg"
    cv2.imwrite(str(p_with_faces), sample_img)

    # 2. Real photo with no faces (plain landscape / gradient)
    no_faces_img = np.zeros((400, 400, 3), dtype=np.uint8)
    no_faces_img[:, :] = (120, 80, 40)
    p_no_faces = d / "landscape.jpg"
    cv2.imwrite(str(p_no_faces), no_faces_img)

    return {
        "with_faces": p_with_faces,
        "no_faces": p_no_faces,
    }


# ── test cases ───────────────────────────────────────────────────────

class TestDetectFaces:
    """Direct tests for detect_faces."""

    def test_detector_singleton_initializes(self):
        detector = get_face_detector()
        assert detector is not None

    def test_detects_faces_in_real_image(self, test_images):
        faces = detect_faces(test_images["with_faces"], min_score=0.5)
        assert len(faces) > 0
        for f in faces:
            assert "bbox" in f
            assert len(f["bbox"]) == 4
            assert f["score"] >= 0.5
            x1, y1, x2, y2 = f["bbox"]
            assert x2 > x1
            assert y2 > y1

    def test_no_faces_in_plain_image(self, test_images):
        faces = detect_faces(test_images["no_faces"])
        assert len(faces) == 0

    def test_missing_file_returns_empty(self, tmp_path):
        faces = detect_faces(tmp_path / "nonexistent.jpg")
        assert faces == []


class TestDbFacesIntegration:
    """Database integration tests for process_photo_faces and batch_detect_faces."""

    def test_process_photo_faces_inserts_to_db(self, db, test_images):
        fp = str(test_images["with_faces"])
        cur = db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (fp,))
        photo_id = cur.lastrowid
        db.commit()

        faces = process_photo_faces(photo_id, db)
        assert len(faces) > 0

        # Verify faces table rows
        rows = db.execute("SELECT id, photo_id, person_id, bbox, face_embedding_id FROM faces WHERE photo_id = ?", (photo_id,)).fetchall()
        assert len(rows) == len(faces)
        for r in rows:
            assert r[1] == photo_id
            assert r[2] is None  # person_id not yet assigned
            assert isinstance(r[3], str)  # bbox as comma-separated string
            bbox_parts = [float(v) for v in r[3].split(",")]
            assert len(bbox_parts) == 4
            assert r[4] is None  # embedding not yet computed

        # Verify photo processing_state updated
        p_row = db.execute("SELECT processing_state FROM photos WHERE id = ?", (photo_id,)).fetchone()
        assert p_row[0] == "faces_detected"

    def test_process_photo_no_faces(self, db, test_images):
        fp = str(test_images["no_faces"])
        cur = db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (fp,))
        photo_id = cur.lastrowid
        db.commit()

        faces = process_photo_faces(photo_id, db)
        assert len(faces) == 0

        # No rows in faces table
        rows = db.execute("SELECT * FROM faces WHERE photo_id = ?", (photo_id,)).fetchall()
        assert len(rows) == 0

        # Photo state still marked faces_detected
        p_row = db.execute("SELECT processing_state FROM photos WHERE id = ?", (photo_id,)).fetchone()
        assert p_row[0] == "faces_detected"

    def test_batch_detect_faces(self, db, test_images):
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (str(test_images["with_faces"]),))
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (str(test_images["no_faces"]),))
        # duplicate photos should be skipped
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'duplicate')", (str(test_images["with_faces"]) + ".dup",))
        db.commit()

        summary = batch_detect_faces(db)
        assert summary["photos_processed"] == 2
        assert summary["faces_detected"] > 0
        assert summary["errors"] == 0

        # All pending photos are now processed
        pending = db.execute("SELECT COUNT(*) FROM photos WHERE processing_state = 'pending'").fetchone()[0]
        assert pending == 0
