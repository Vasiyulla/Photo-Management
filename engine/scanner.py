"""
PhotoVault — Filesystem scanner (T2).

Provides a polling-based scanner that detects new and removed image files
in a watched folder by comparing the filesystem state to the ``photos``
table in the vault database.

No external dependencies — uses only the Python standard library
(``pathlib``, ``os``).  This avoids adding ``watchdog`` or similar.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# ── configuration ────────────────────────────────────────────────────

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png",
    ".tiff", ".tif",
    ".bmp", ".webp",
    ".heic", ".heif",
    ".jfif", ".avif",
})


# ── public API ───────────────────────────────────────────────────────

def scan_folder(root_dir: str | Path) -> set[str]:
    """Recursively find all image files under *root_dir*.

    Returns a set of **absolute** file paths (as strings) whose
    extensions (case-insensitive) are in :data:`IMAGE_EXTENSIONS`.
    Non-image files and directories are silently skipped.
    """
    root = Path(root_dir).resolve()
    found: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                full = os.path.join(dirpath, fname)
                found.add(os.path.abspath(full))
    return found


def sync_folder(
    root_dir: str | Path,
    conn: sqlite3.Connection,
) -> tuple[set[str], set[str]]:
    """Compare on-disk images to the DB and reconcile.

    1. Any file on disk but **not** in ``photos`` → INSERT with
       ``processing_state = 'pending'``.
    2. Any row in ``photos`` whose file no longer exists on disk →
       DELETE the row.

    Returns ``(new_files, removed_files)`` — both are sets of absolute
    path strings.

    This is intentionally **incremental**: it never triggers a full
    library rescan for an incremental change (per AGENTS.md constraint).
    """
    root = Path(root_dir).resolve()
    root_str = str(root)
    on_disk = scan_folder(root)

    # Normalise DB paths to absolute for comparison
    rows = conn.execute("SELECT filepath FROM photos").fetchall()
    in_db: set[str] = {os.path.abspath(r[0]) for r in rows}

    new_files = on_disk - in_db

    # Only consider files under root_dir for removal
    def is_under_root(path_str: str) -> bool:
        try:
            return os.path.commonpath([root_str, path_str]) == root_str
        except (ValueError, OSError):
            return False

    files_under_root_in_db = {fp for fp in in_db if is_under_root(fp)}
    removed_files = files_under_root_in_db - on_disk

    # ── inserts ──────────────────────────────────────────────────────
    if new_files:
        conn.executemany(
            "INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')",
            [(fp,) for fp in sorted(new_files)],
        )
        conn.commit()

    # ── deletes ──────────────────────────────────────────────────────
    if removed_files:
        del_paths = [(fp,) for fp in sorted(removed_files)]
        conn.executemany(
            "DELETE FROM faces WHERE photo_id IN (SELECT id FROM photos WHERE filepath = ?)",
            del_paths,
        )
        conn.executemany(
            "DELETE FROM clip_embeddings WHERE photo_id IN (SELECT id FROM photos WHERE filepath = ?)",
            del_paths,
        )
        conn.executemany(
            "DELETE FROM photos WHERE filepath = ?",
            del_paths,
        )
        conn.commit()

    return new_files, removed_files
