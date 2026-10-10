"""PhotoVault — Photo Folder Scanner & Indexer CLI.

Usage:
    python scan.py "C:\\path\\to\\your\\photos"
    python scan.py "C:\\path\\to\\your\\photos" --db vault.db
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.api import scan_library
from engine.db.schema import init_db


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan and index photos into PhotoVault")
    parser.add_argument("folder", help="Absolute or relative path to folder containing photos")
    parser.add_argument("--db", default="vault.db", help="Path to SQLite database (default: vault.db)")
    parser.add_argument("--storage", default=".", help="Directory for index files and vector stores (default: .)")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Folder '{folder}' does not exist or is not a directory.")
        return 1

    print(f"📸 Scanning folder: {folder}")
    conn = init_db(args.db)

    try:
        results = scan_library(str(folder), conn, storage_dir=args.storage)
        print("\n✅ Scan & Ingestion Complete!")
        print(f"   • New photos added: {results['new_photos']}")
        print(f"   • Removed photos:   {results['removed_photos']}")
        print(f"   • EXIF extracted:   {results['exif']['processed']} photos ({results['exif']['with_date']} with dates)")
        print(f"   • pHash computed:   {results['phashes']['processed']} photos ({results['phashes']['duplicates_found']} duplicates found)")
        print(f"   • Faces detected:   {results['faces']['faces_detected']} faces in {results['faces']['photos_processed']} photos")
        print(f"   • OpenCLIP encoded: {results['clip']['processed']} photos")
        print("\nRefresh your PhotoVault web browser (http://localhost:5173) to see your photos!")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
