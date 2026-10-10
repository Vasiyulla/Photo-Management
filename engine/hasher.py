"""
PhotoVault — Perceptual hash generation & deduplication (T3).

Computes perceptual hashes (pHash) for photos using Pillow and ImageHash,
stores the hash in the ``photos`` table, and detects duplicates.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from PIL import Image
import imagehash


def compute_phash(filepath: str | Path) -> str:
    """Compute the 64-bit perceptual hash (pHash) of an image file.

    Returns the 16-character hexadecimal string representation.
    """
    with Image.open(filepath) as img:
        # Convert to RGB/L if needed (ImageHash handles internally, but ensuring file is read)
        h = imagehash.phash(img)
        return str(h)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute the Hamming distance between two hex pHash strings.

    Returns the number of bit differences (0 = identical).
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return int(h1 - h2)


def find_duplicates(
    conn: sqlite3.Connection,
    target_phash: str,
    threshold: int = 0,
    exclude_photo_id: int | None = None,
) -> list[dict[str, Any]]:
    """Search for existing photos whose pHash is within *threshold* distance.

    *threshold* = 0 finds exact pHash duplicates.
    Higher values (e.g. 1-4) find near-duplicates.
    """
    conn.row_factory = sqlite3.Row
    query = "SELECT id, filepath, phash, processing_state FROM photos WHERE phash IS NOT NULL"
    params: list[Any] = []
    if exclude_photo_id is not None:
        query += " AND id != ?"
        params.append(exclude_photo_id)

    rows = conn.execute(query, params).fetchall()
    duplicates: list[dict[str, Any]] = []

    for row in rows:
        row_hash = row["phash"]
        if threshold == 0:
            if row_hash == target_phash:
                duplicates.append({
                    "id": row["id"],
                    "filepath": row["filepath"],
                    "phash": row_hash,
                    "distance": 0,
                    "processing_state": row["processing_state"],
                })
        else:
            dist = hamming_distance(target_phash, row_hash)
            if dist <= threshold:
                duplicates.append({
                    "id": row["id"],
                    "filepath": row["filepath"],
                    "phash": row_hash,
                    "distance": dist,
                    "processing_state": row["processing_state"],
                })

    return duplicates


def process_photo_phash(
    photo_id: int,
    conn: sqlite3.Connection,
    threshold: int = 0,
    mark_duplicate: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Compute pHash for a single photo, check for duplicates, and update DB.

    If duplicates are detected and *mark_duplicate* is True, sets
    ``processing_state = 'duplicate'``.
    """
    row = conn.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row:
        raise ValueError(f"Photo with id={photo_id} not found.")

    filepath = row[0]
    phash = compute_phash(filepath)
    duplicates = find_duplicates(conn, phash, threshold=threshold, exclude_photo_id=photo_id)

    if duplicates and mark_duplicate:
        conn.execute(
            "UPDATE photos SET phash = ?, processing_state = 'duplicate' WHERE id = ?",
            (phash, photo_id),
        )
    else:
        conn.execute(
            "UPDATE photos SET phash = ? WHERE id = ?",
            (phash, photo_id),
        )
    conn.commit()

    return phash, duplicates


def batch_process_phashes(
    conn: sqlite3.Connection,
    threshold: int = 0,
    mark_duplicate: bool = True,
) -> dict[str, int]:
    """Compute pHash for all photos where ``phash IS NULL``.

    Returns summary dictionary: ``{"processed": int, "duplicates_found": int, "errors": int}``.
    """
    rows = conn.execute("SELECT id FROM photos WHERE phash IS NULL ORDER BY id ASC").fetchall()
    summary = {
        "processed": 0,
        "duplicates_found": 0,
        "errors": 0,
    }

    for (photo_id,) in rows:
        try:
            _, dups = process_photo_phash(
                photo_id=photo_id,
                conn=conn,
                threshold=threshold,
                mark_duplicate=mark_duplicate,
            )
            summary["processed"] += 1
            if dups:
                summary["duplicates_found"] += 1
        except Exception:
            summary["errors"] += 1

    return summary
