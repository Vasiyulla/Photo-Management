"""Tests for T10: OpenCLIP embedding generation and clip.index semantic search.

Verifies image and text encoding, clip.index FAISS index persistence,
database clip_embeddings records, and semantic search against real local images.
"""

import sqlite3
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import pytest

from engine.db.schema import init_db
from engine.clip import (
    get_clip_model,
    ClipEmbeddingStore,
    ClipIndex,
    process_photo_clip,
    batch_generate_clip_embeddings,
    search_photos_semantic,
)


# ── helpers ──────────────────────────────────────────────────────────

def _create_scene_image(path: Path, bg_color: tuple[int, int, int], shape: str):
    img = Image.new("RGB", (224, 224), color=bg_color)
    draw = ImageDraw.Draw(img)
    if shape == "sun":
        draw.ellipse([80, 80, 144, 144], fill=(255, 255, 0))
    elif shape == "blue_rect":
        draw.rectangle([50, 50, 174, 174], fill=(0, 0, 255))
    img.save(path)


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    db_path = str(tmp_path / "vault.db")
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def clip_store(tmp_path):
    return ClipEmbeddingStore(tmp_path / "clip_store")


@pytest.fixture()
def clip_index(tmp_path):
    return ClipIndex(tmp_path / "clip.index")


@pytest.fixture()
def scene_images(tmp_path):
    d = tmp_path / "scenes"
    d.mkdir()

    p_sun = d / "sun.jpg"
    _create_scene_image(p_sun, (255, 100, 0), "sun")  # Orange sky with bright yellow sun

    p_blue = d / "blue.jpg"
    _create_scene_image(p_blue, (0, 150, 255), "blue_rect")  # Ocean blue

    return {
        "sun": p_sun,
        "blue": p_blue,
    }


# ── test cases ───────────────────────────────────────────────────────

class TestClipModel:
    """Unit tests for ClipModel image and text encoders."""

    def test_model_initializes(self):
        clip = get_clip_model()
        assert clip is not None

    def test_encode_image(self, scene_images):
        clip = get_clip_model()
        emb = clip.encode_image(scene_images["sun"])
        assert emb.shape == (512,)
        assert emb.dtype == np.float32
        assert pytest.approx(float(np.linalg.norm(emb)), rel=1e-3) == 1.0

    def test_encode_text(self):
        clip = get_clip_model()
        emb = clip.encode_text("a yellow sun in the orange sky")
        assert emb.shape == (512,)
        assert emb.dtype == np.float32
        assert pytest.approx(float(np.linalg.norm(emb)), rel=1e-3) == 1.0


class TestClipIndexAndDb:
    """Integration tests for photo indexing and semantic search."""

    def test_process_photo_clip(self, db, clip_store, clip_index, scene_images):
        fp = str(scene_images["sun"])
        cur = db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (fp,))
        photo_id = cur.lastrowid
        db.commit()

        emb_id, vec = process_photo_clip(photo_id, db, clip_store, clip_index)
        assert emb_id is not None
        assert vec.shape == (512,)

        # Verify DB row in clip_embeddings
        row = db.execute("SELECT id, photo_id, embedding_id FROM clip_embeddings WHERE photo_id = ?", (photo_id,)).fetchone()
        assert row is not None
        assert row[1] == photo_id
        assert row[2] == emb_id

        # Verify present in index and store
        assert emb_id in clip_index
        assert clip_store.get(emb_id) is not None

    def test_semantic_search_ranks_relevant_image_higher(self, db, clip_store, clip_index, scene_images):
        cur1 = db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (str(scene_images["sun"]),))
        p1_id = cur1.lastrowid

        cur2 = db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (str(scene_images["blue"]),))
        p2_id = cur2.lastrowid
        db.commit()

        process_photo_clip(p1_id, db, clip_store, clip_index)
        process_photo_clip(p2_id, db, clip_store, clip_index)

        # Search for sun
        results_sun = search_photos_semantic("yellow sun", db, clip_index, top_k=2)
        assert len(results_sun) >= 1
        assert results_sun[0]["photo_id"] == p1_id  # Sun image ranked #1

        # Search for blue ocean
        results_blue = search_photos_semantic("blue ocean water", db, clip_index, top_k=2)
        assert len(results_blue) >= 1
        assert results_blue[0]["photo_id"] == p2_id  # Blue image ranked #1

    def test_batch_generate_clip_embeddings(self, db, clip_store, clip_index, scene_images):
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (str(scene_images["sun"]),))
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (str(scene_images["blue"]),))
        db.commit()

        summary = batch_generate_clip_embeddings(db, clip_store, clip_index)
        assert summary["processed"] == 2
        assert summary["errors"] == 0
        assert len(clip_index) == 2

        # Re-running batch does not duplicate
        summary2 = batch_generate_clip_embeddings(db, clip_store, clip_index)
        assert summary2["processed"] == 0
        assert len(clip_index) == 2
