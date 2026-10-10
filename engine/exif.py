"""
PhotoVault — EXIF metadata extraction (T4).

Extracts date_taken (formatted to ISO 8601) and GPS coordinates (decimal degrees)
from photos using Pillow, and updates the ``photos.date_taken`` database field.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from PIL import Image


def _parse_exif_date(date_raw: str | None) -> str | None:
    """Parse EXIF date string (typically 'YYYY:MM:DD HH:MM:SS') to ISO 8601.

    Returns ISO 8601 string 'YYYY-MM-DDTHH:MM:SS' or None if unparseable.
    """
    if not date_raw or not isinstance(date_raw, str):
        return None

    cleaned = date_raw.strip().replace("\x00", "")
    if not cleaned:
        return None

    # Matches "YYYY:MM:DD HH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
    m = re.match(r"^(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})", cleaned)
    if m:
        year, month, day, hour, minute, second = m.groups()
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}"

    return None


def _dms_to_decimal(dms: Any, ref: Any) -> float | None:
    """Convert (degrees, minutes, seconds) tuple/IFDRationals and ref ('N'/'S'/'E'/'W')

    to decimal degrees.
    """
    if not dms or len(dms) < 3:
        return None

    try:
        deg = float(dms[0])
        minute = float(dms[1])
        sec = float(dms[2])
        dec = deg + (minute / 60.0) + (sec / 3600.0)
        if ref and str(ref).strip().upper() in ("S", "W"):
            dec = -dec
        return round(dec, 6)
    except Exception:
        return None


def extract_exif(filepath: str | Path) -> dict[str, Any]:
    """Extract date_taken and GPS coordinates from an image file.

    Returns:
        dict with keys:
            - "date_taken": ISO 8601 string or None
            - "latitude": float (decimal degrees) or None
            - "longitude": float (decimal degrees) or None
    """
    result: dict[str, Any] = {
        "date_taken": None,
        "latitude": None,
        "longitude": None,
    }

    path = Path(filepath)
    if not path.is_file():
        return result

    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return result

            # 1. Date extraction: check DateTimeOriginal (36867) in ExifIFD first, then DateTime (306)
            date_raw = None
            try:
                exif_ifd = exif.get_ifd(0x8769)  # ExifOffset IFD
                date_raw = exif_ifd.get(36867) or exif_ifd.get(36868)  # DateTimeOriginal, DateTimeDigitized
            except Exception:
                pass

            if not date_raw:
                date_raw = exif.get(306)  # DateTime

            result["date_taken"] = _parse_exif_date(date_raw)

            # 2. GPS extraction: check GPSInfo IFD (0x8825 / 34853)
            try:
                gps_ifd = exif.get_ifd(0x8825)
                if gps_ifd:
                    lat_ref = gps_ifd.get(1)
                    lat_dms = gps_ifd.get(2)
                    lon_ref = gps_ifd.get(3)
                    lon_dms = gps_ifd.get(4)

                    result["latitude"] = _dms_to_decimal(lat_dms, lat_ref)
                    result["longitude"] = _dms_to_decimal(lon_dms, lon_ref)
            except Exception:
                pass

    except Exception:
        pass

    return result


def process_photo_exif(
    photo_id: int,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Extract EXIF data for a photo and update the date_taken column in DB.

    Returns extracted metadata dict including GPS coordinates.
    """
    row = conn.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row:
        raise ValueError(f"Photo with id={photo_id} not found.")

    filepath = row[0]
    exif_data = extract_exif(filepath)

    if exif_data["date_taken"]:
        conn.execute(
            "UPDATE photos SET date_taken = ? WHERE id = ?",
            (exif_data["date_taken"], photo_id),
        )
        conn.commit()

    return {
        "photo_id": photo_id,
        "date_taken": exif_data["date_taken"],
        "latitude": exif_data["latitude"],
        "longitude": exif_data["longitude"],
    }


def batch_process_exif(conn: sqlite3.Connection) -> dict[str, int]:
    """Extract and update EXIF data for all photos where ``date_taken IS NULL``

    and ``processing_state != 'duplicate'``.
    """
    rows = conn.execute(
        "SELECT id FROM photos WHERE date_taken IS NULL AND processing_state != 'duplicate' ORDER BY id ASC"
    ).fetchall()

    summary = {
        "processed": 0,
        "with_date": 0,
        "with_gps": 0,
        "errors": 0,
    }

    for (photo_id,) in rows:
        try:
            res = process_photo_exif(photo_id, conn)
            summary["processed"] += 1
            if res["date_taken"]:
                summary["with_date"] += 1
            if res["latitude"] is not None and res["longitude"] is not None:
                summary["with_gps"] += 1
        except Exception:
            summary["errors"] += 1

    return summary
