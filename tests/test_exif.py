"""Tests for T4: EXIF extraction (date_taken, GPS if present).

Verifies extraction against real local image files generated in tmp_path
as required by AGENTS.md Definition of Done.
"""

import sqlite3
from pathlib import Path
from PIL import Image
import pytest

from engine.db.schema import init_db
from engine.exif import (
    extract_exif,
    process_photo_exif,
    batch_process_exif,
)


# ── helpers ──────────────────────────────────────────────────────────

def _create_image_with_exif(path: Path, date_str: str | None = None, gps: tuple[float, float] | None = None):
    """Create a real JPEG image with EXIF metadata."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    exif = img.getexif()

    if date_str:
        # Tag 36867 = DateTimeOriginal, Tag 306 = DateTime
        exif[306] = date_str
        exif_ifd = exif.get_ifd(0x8769)  # ExifOffset IFD
        exif_ifd[36867] = date_str

    if gps:
        lat, lon = gps
        lat_ref = "N" if lat >= 0 else "S"
        lon_ref = "E" if lon >= 0 else "W"
        lat = abs(lat)
        lon = abs(lon)

        # Convert decimal degrees to (degrees, minutes, seconds)
        lat_d = int(lat)
        lat_m = int((lat - lat_d) * 60)
        lat_s = round(((lat - lat_d) * 60 - lat_m) * 60, 4)

        lon_d = int(lon)
        lon_m = int((lon - lon_d) * 60)
        lon_s = round(((lon - lon_d) * 60 - lon_m) * 60, 4)

        gps_ifd = exif.get_ifd(0x8825)  # GPSInfo IFD
        gps_ifd[1] = lat_ref
        gps_ifd[2] = (lat_d, lat_m, lat_s)
        gps_ifd[3] = lon_ref
        gps_ifd[4] = (lon_d, lon_m, lon_s)

    img.save(path, exif=exif)


def _create_image_without_exif(path: Path):
    """Create a plain PNG/JPEG without EXIF metadata."""
    img = Image.new("RGB", (64, 64), color=(50, 50, 50))
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
def sample_images(tmp_path):
    d = tmp_path / "exif_photos"
    d.mkdir()

    # Photo with both date and GPS (San Francisco approx: 37.7749 N, 122.4194 W)
    p_full = d / "full.jpg"
    _create_image_with_exif(p_full, "2023:08:15 17:45:30", (37.7749, -122.4194))

    # Photo with date only
    p_date_only = d / "date_only.jpg"
    _create_image_with_exif(p_date_only, "2022:12:25 09:00:00", None)

    # Photo with no EXIF
    p_none = d / "no_exif.jpg"
    _create_image_without_exif(p_none)

    return {
        "full": p_full,
        "date_only": p_date_only,
        "no_exif": p_none,
    }


# ── test cases ───────────────────────────────────────────────────────

class TestExtractExif:
    """Direct extraction tests for extract_exif."""

    def test_extract_full_exif(self, sample_images):
        data = extract_exif(sample_images["full"])
        assert data["date_taken"] == "2023-08-15T17:45:30"
        assert data["latitude"] is not None
        assert data["longitude"] is not None
        assert pytest.approx(data["latitude"], rel=1e-3) == 37.7749
        assert pytest.approx(data["longitude"], rel=1e-3) == -122.4194

    def test_extract_date_only(self, sample_images):
        data = extract_exif(sample_images["date_only"])
        assert data["date_taken"] == "2022-12-25T09:00:00"
        assert data["latitude"] is None
        assert data["longitude"] is None

    def test_extract_no_exif(self, sample_images):
        data = extract_exif(sample_images["no_exif"])
        assert data["date_taken"] is None
        assert data["latitude"] is None
        assert data["longitude"] is None

    def test_missing_file(self, tmp_path):
        data = extract_exif(tmp_path / "nonexistent.jpg")
        assert data["date_taken"] is None
        assert data["latitude"] is None
        assert data["longitude"] is None


class TestDbExifIntegration:
    """Database update tests for process_photo_exif and batch_process_exif."""

    def test_process_photo_exif_updates_db(self, db, sample_images):
        fp = str(sample_images["full"])
        cur = db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (fp,))
        photo_id = cur.lastrowid
        db.commit()

        exif_info = process_photo_exif(photo_id, db)
        assert exif_info["date_taken"] == "2023-08-15T17:45:30"
        assert pytest.approx(exif_info["latitude"], rel=1e-3) == 37.7749

        # Verify DB column date_taken is updated
        row = db.execute("SELECT date_taken FROM photos WHERE id = ?", (photo_id,)).fetchone()
        assert row[0] == "2023-08-15T17:45:30"

    def test_batch_process_exif(self, db, sample_images):
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (str(sample_images["full"]),))
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (str(sample_images["date_only"]),))
        db.execute("INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')", (str(sample_images["no_exif"]),))
        db.commit()

        summary = batch_process_exif(db)
        assert summary["processed"] == 3
        assert summary["with_date"] == 2
        assert summary["with_gps"] == 1

        rows = db.execute("SELECT filepath, date_taken FROM photos ORDER BY id").fetchall()
        assert rows[0][1] == "2023-08-15T17:45:30"
        assert rows[1][1] == "2022-12-25T09:00:00"
        assert rows[2][1] is None
