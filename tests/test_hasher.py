"""Tests for T3: Perceptual hash (pHash) generation and deduplication.

Verifies pHash generation and deduplication against real local image files
(using PIL.Image to generate images in tmp_path) as required by AGENTS.md
Definition of Done.
"""

import sqlite3
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from engine.db.schema import init_db
from engine.hasher import (
    compute_phash,
    hamming_distance,
    find_duplicates,
    process_photo_phash,
    batch_process_phashes,
)


# ── helpers ──────────────────────────────────────────────────────────

def _create_test_image(path: Path, color: tuple[int, int, int], pattern: str = "plain"):
    """Create a real image file with specific content."""
    img = Image.new("RGB", (128, 128), color=color)
    draw = ImageDraw.Draw(img)
    if pattern == "circle":
        draw.ellipse([20, 20, 100, 100], fill=(255, 255, 255))
    elif pattern == "cross":
        draw.line([0, 0, 128, 128], fill=(0, 0, 0), width=5)
        draw.line([0, 128, 128, 0], fill=(0, 0, 0), width=5)
    img.save(path)


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    """Return an initialised in-tmp-dir DB connection."""
    db_path = str(tmp_path / "vault.db")
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def image_dir(tmp_path):
    """Generate real local images for testing."""
    d = tmp_path / "images"
    d.mkdir()
    
    # Original image 1
    _create_test_image(d / "img1.jpg", (200, 50, 50), "circle")
    
    # Exact duplicate of image 1 (same pixels, different filename)
    _create_test_image(d / "img1_copy.jpg", (200, 50, 50), "circle")
    
    # Resized version of image 1 (near-duplicate)
    img1 = Image.open(d / "img1.jpg")
    img1_resized = img1.resize((64, 64))
    img1_resized.save(d / "img1_resized.png")

    # Distinct image 2
    _create_test_image(d / "img2.jpg", (50, 200, 50), "cross")
    
    return d


# ── test cases ───────────────────────────────────────────────────────

class TestPHashCompute:
    """Tests for compute_phash and hamming_distance."""

    def test_compute_phash_returns_hex_string(self, image_dir):
        h = compute_phash(image_dir / "img1.jpg")
        assert isinstance(h, str)
        assert len(h) == 16  # standard 64-bit hex representation
        int(h, 16)  # should parse as valid hex

    def test_exact_duplicates_have_same_phash(self, image_dir):
        h1 = compute_phash(image_dir / "img1.jpg")
        h2 = compute_phash(image_dir / "img1_copy.jpg")
        assert h1 == h2
        assert hamming_distance(h1, h2) == 0

    def test_near_duplicates_have_low_distance(self, image_dir):
        h1 = compute_phash(image_dir / "img1.jpg")
        h_resized = compute_phash(image_dir / "img1_resized.png")
        dist = hamming_distance(h1, h_resized)
        assert dist <= 4  # Resized version has very low or zero distance

    def test_distinct_images_have_higher_distance(self, image_dir):
        h1 = compute_phash(image_dir / "img1.jpg")
        h2 = compute_phash(image_dir / "img2.jpg")
        dist = hamming_distance(h1, h2)
        assert dist > 5


class TestDedupInDb:
    """Tests for database deduplication check."""

    def test_find_duplicates_exact(self, db, image_dir):
        fp1 = str(image_dir / "img1.jpg")
        h1 = compute_phash(fp1)
        db.execute("INSERT INTO photos (filepath, phash, processing_state) VALUES (?, ?, 'pending')", (fp1, h1))
        db.commit()

        # Search with identical hash
        dups = find_duplicates(db, h1, threshold=0)
        assert len(dups) == 1
        assert dups[0]["filepath"] == fp1

        # Search excluding self
        cursor = db.execute("SELECT id FROM photos WHERE filepath = ?", (fp1,))
        photo_id = cursor.fetchone()[0]
        dups_ex = find_duplicates(db, h1, threshold=0, exclude_photo_id=photo_id)
        assert len(dups_ex) == 0

    def test_process_photo_phash_sets_hash_and_detects_duplicate(self, db, image_dir):
        fp1 = str(image_dir / "img1.jpg")
        fp2 = str(image_dir / "img1_copy.jpg")

        cursor = db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (fp1,))
        p1_id = cursor.lastrowid
        cursor = db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (fp2,))
        p2_id = cursor.lastrowid
        db.commit()

        # Process first photo
        h1, dups1 = process_photo_phash(p1_id, db)
        assert len(dups1) == 0
        row1 = db.execute("SELECT phash, processing_state FROM photos WHERE id = ?", (p1_id,)).fetchone()
        assert row1[0] == h1
        assert row1[1] == "pending"

        # Process duplicate photo
        h2, dups2 = process_photo_phash(p2_id, db)
        assert h1 == h2
        assert len(dups2) == 1
        assert dups2[0]["id"] == p1_id
        
        # When duplicate detected, processing_state marks duplicate
        row2 = db.execute("SELECT phash, processing_state FROM photos WHERE id = ?", (p2_id,)).fetchone()
        assert row2[0] == h2
        assert row2[1] == "duplicate"

    def test_batch_process_phashes(self, db, image_dir):
        fp1 = str(image_dir / "img1.jpg")
        fp2 = str(image_dir / "img1_copy.jpg")
        fp3 = str(image_dir / "img2.jpg")

        db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (fp1,))
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (fp2,))
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (fp3,))
        db.commit()

        summary = batch_process_phashes(db)
        assert summary["processed"] == 3
        assert summary["duplicates_found"] == 1

        unhashed = db.execute("SELECT COUNT(*) FROM photos WHERE phash IS NULL").fetchone()[0]
        assert unhashed == 0
