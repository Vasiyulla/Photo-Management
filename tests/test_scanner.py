"""Tests for T2: Filesystem scanner — detect new/removed image files.

Uses real files in tmp_path (not mocks) as required by AGENTS.md
Definition of Done for file I/O tasks.
"""

import sqlite3

import pytest

from engine.db.schema import init_db
from engine.scanner import IMAGE_EXTENSIONS, scan_folder, sync_folder


# ── helpers ──────────────────────────────────────────────────────────

def _create_file(path, content: bytes = b"\x00"):
    """Create a tiny dummy file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _db_filepaths(conn: sqlite3.Connection) -> set[str]:
    """Return the set of filepaths currently in the photos table."""
    rows = conn.execute("SELECT filepath FROM photos").fetchall()
    return {r[0] for r in rows}


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    """Return an initialised in-tmp-dir DB connection."""
    db_path = str(tmp_path / "vault.db")
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def photo_dir(tmp_path):
    """Return a tmp directory pre-populated with a few image files."""
    d = tmp_path / "photos"
    d.mkdir()
    _create_file(d / "a.jpg")
    _create_file(d / "b.png")
    _create_file(d / "sub" / "c.tiff")
    # non-image file — should be ignored
    _create_file(d / "readme.txt")
    return d


# ── scan_folder tests ────────────────────────────────────────────────

class TestScanFolder:
    """Unit tests for scan_folder (pure filesystem, no DB)."""

    def test_finds_images(self, photo_dir):
        found = scan_folder(str(photo_dir))
        basenames = {f.split("\\")[-1].split("/")[-1] for f in found}
        assert "a.jpg" in basenames
        assert "b.png" in basenames
        assert "c.tiff" in basenames

    def test_ignores_non_images(self, photo_dir):
        found = scan_folder(str(photo_dir))
        basenames = {f.split("\\")[-1].split("/")[-1] for f in found}
        assert "readme.txt" not in basenames

    def test_returns_absolute_paths(self, photo_dir):
        import os
        found = scan_folder(str(photo_dir))
        for f in found:
            assert os.path.isabs(f)

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert scan_folder(str(d)) == set()

    def test_case_insensitive_extension(self, tmp_path):
        """Uppercase extensions like .JPG should be detected."""
        d = tmp_path / "upper"
        d.mkdir()
        _create_file(d / "photo.JPG")
        _create_file(d / "image.Png")
        found = scan_folder(str(d))
        assert len(found) == 2


# ── sync_folder tests ────────────────────────────────────────────────

class TestSyncFolder:
    """Integration tests for sync_folder (filesystem + DB)."""

    def test_detects_new_files(self, photo_dir, db):
        new, removed = sync_folder(str(photo_dir), db)
        assert len(new) == 3  # a.jpg, b.png, sub/c.tiff
        assert len(removed) == 0

    def test_inserts_new_into_db(self, photo_dir, db):
        sync_folder(str(photo_dir), db)
        rows = _db_filepaths(db)
        assert len(rows) == 3

    def test_processing_state_is_pending(self, photo_dir, db):
        sync_folder(str(photo_dir), db)
        states = db.execute(
            "SELECT DISTINCT processing_state FROM photos"
        ).fetchall()
        assert states == [("pending",)]

    def test_detects_removed_files(self, photo_dir, db):
        # First sync — adds 3 files
        sync_folder(str(photo_dir), db)
        # Remove a file from disk
        (photo_dir / "a.jpg").unlink()
        # Second sync — should detect 1 removal
        new, removed = sync_folder(str(photo_dir), db)
        assert len(removed) == 1
        assert len(new) == 0

    def test_removes_from_db(self, photo_dir, db):
        sync_folder(str(photo_dir), db)
        (photo_dir / "a.jpg").unlink()
        sync_folder(str(photo_dir), db)
        rows = _db_filepaths(db)
        assert len(rows) == 2

    def test_idempotent(self, photo_dir, db):
        """Running sync twice with no changes yields empty new/removed."""
        sync_folder(str(photo_dir), db)
        new, removed = sync_folder(str(photo_dir), db)
        assert len(new) == 0
        assert len(removed) == 0

    def test_new_file_after_first_sync(self, photo_dir, db):
        """A file added after initial sync is detected on re-sync."""
        sync_folder(str(photo_dir), db)
        _create_file(photo_dir / "d.webp")
        new, removed = sync_folder(str(photo_dir), db)
        assert len(new) == 1
        assert len(removed) == 0
        assert len(_db_filepaths(db)) == 4
