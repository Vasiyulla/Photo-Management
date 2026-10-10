"""
PhotoVault — Library Re-indexing & Migration Utility.
Migrates photo library to OpenCV YuNet (FaceDetectorYN) + SFace (FaceRecognizerSF)
128-dimensional face embedding pipeline (digiKam architecture).
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

from engine.faces import batch_detect_faces
from engine.embeddings import FaceEmbeddingStore, batch_generate_face_embeddings
from engine.index import FacesIndex, sync_faces_index
from engine.cluster import cluster_unassigned_faces
from engine.recluster import recluster_faces_hdbscan


def backup_existing_face_data(
    db_path: str | Path = "vault.db",
    storage_dir: str | Path = ".",
) -> dict[str, str]:
    """Create safety backups of database, face_store, and faces.index."""
    storage = Path(storage_dir)
    db_file = Path(db_path)
    timestamp = int(time.time())
    backups: dict[str, str] = {}

    # 1. Database backup
    if db_file.exists():
        db_bak = db_file.with_name(f"{db_file.stem}_backup_{timestamp}.db")
        shutil.copy2(db_file, db_bak)
        backups["db"] = str(db_bak)

    # 2. face_store backup
    face_store_dir = storage / "face_store"
    if face_store_dir.exists():
        face_store_bak = storage / f"face_store_backup_{timestamp}"
        shutil.copytree(face_store_dir, face_store_bak)
        backups["face_store"] = str(face_store_bak)

    # 3. faces.index backup
    faces_idx = storage / "faces.index"
    if faces_idx.exists():
        faces_idx_bak = storage / f"faces_backup_{timestamp}.index"
        shutil.copy2(faces_idx, faces_idx_bak)
        backups["faces_index"] = str(faces_idx_bak)

    return backups


def reindex_library_faces(
    db_path: str | Path = "vault.db",
    storage_dir: str | Path = ".",
    create_backup: bool = True,
    preserve_named_persons: bool = True,
) -> dict[str, Any]:
    """
    Re-index all photos in vault.db using the OpenCV YuNet + SFace pipeline.

    Steps:
    1. Backs up existing data (if create_backup=True).
    2. Initializes clean 128D FaceEmbeddingStore and FacesIndex.
    3. Re-runs YuNet face detection across photos.
    4. Generates 128D SFace embeddings for all detected faces.
    5. Builds the 128D FAISS faces.index.
    6. Clusters faces using the optimal 0.36 cosine similarity threshold.
    """
    storage = Path(storage_dir)
    db_file = Path(db_path)

    if not db_file.exists():
        raise FileNotFoundError(f"Database not found at '{db_file}'")

    backups: dict[str, str] = {}
    if create_backup:
        print("[reindex] Creating safety backups...")
        backups = backup_existing_face_data(db_file, storage)
        for k, v in backups.items():
            print(f"  Backup created: {k} -> {v}")

    conn = sqlite3.connect(str(db_file))

    # Preserve named persons if requested
    named_map: dict[str, str] = {}
    if preserve_named_persons:
        rows = conn.execute(
            "SELECT id, display_name FROM persons WHERE display_name IS NOT NULL AND trim(display_name) != ''"
        ).fetchall()
        named_map = {r[0]: r[1] for r in rows}
        print(f"[reindex] Preserving {len(named_map)} named persons.")

    # Remove existing faces and unassigned/unnamed persons
    print("[reindex] Clearing existing face records...")
    conn.execute("DELETE FROM faces")
    if preserve_named_persons and named_map:
        placeholders = ",".join("?" for _ in named_map.keys())
        conn.execute(f"DELETE FROM persons WHERE id NOT IN ({placeholders})", list(named_map.keys()))
    else:
        conn.execute("DELETE FROM persons")

    # Reset non-duplicate photos to 'pending'
    conn.execute(
        "UPDATE photos SET processing_state = 'pending' "
        "WHERE processing_state != 'duplicate'"
    )
    conn.commit()

    # Reset face_store and faces.index to fresh 128-dim
    face_store_path = storage / "face_store"
    if face_store_path.exists():
        shutil.rmtree(face_store_path)
    face_store = FaceEmbeddingStore(face_store_path, dimension=128)

    faces_idx_path = storage / "faces.index"
    if faces_idx_path.exists():
        faces_idx_path.unlink()
    faces_idx = FacesIndex(faces_idx_path, dimension=128)

    # 1. Batch face detection with YuNet
    print("[reindex] Step 1/4: Running YuNet face detection...")
    t0 = time.time()
    det_stats = batch_detect_faces(conn, min_score=0.5)
    t_det = time.time() - t0
    print(f"  Detection completed in {t_det:.1f}s: {det_stats}")

    # 2. Batch embedding generation with SFace (128-dim)
    print("[reindex] Step 2/4: Generating 128D SFace embeddings...")
    t0 = time.time()
    emb_stats = batch_generate_face_embeddings(conn, store=face_store)
    t_emb = time.time() - t0
    print(f"  Embedding completed in {t_emb:.1f}s: {emb_stats}")

    # 3. Synchronize FAISS index
    print("[reindex] Step 3/4: Building FAISS 128D index...")
    t0 = time.time()
    added_to_idx = sync_faces_index(conn, face_store, faces_idx)
    t_idx = time.time() - t0
    print(f"  Indexed {added_to_idx} vectors in {t_idx:.1f}s (index size: {len(faces_idx)})")

    # 4. Incremental clustering & global agglomerative re-clustering (threshold 0.36)
    print("[reindex] Step 4/4: Clustering faces (cosine threshold 0.36)...")
    t0 = time.time()
    cluster_stats = cluster_unassigned_faces(conn, face_store, faces_idx, threshold=0.36)
    recluster_stats = recluster_faces_hdbscan(conn, face_store, distance_threshold=0.64)
    t_cls = time.time() - t0
    print(f"  Clustering completed in {t_cls:.1f}s: {cluster_stats}, merges: {recluster_stats['merges_performed']}")

    # Final summary query
    total_photos = conn.execute("SELECT count(*) FROM photos").fetchone()[0]
    total_faces = conn.execute("SELECT count(*) FROM faces").fetchone()[0]
    total_persons = conn.execute("SELECT count(*) FROM persons").fetchone()[0]
    conn.close()

    summary = {
        "status": "success",
        "total_photos": total_photos,
        "total_faces": total_faces,
        "total_persons": total_persons,
        "detection": det_stats,
        "embeddings": emb_stats,
        "vectors_indexed": added_to_idx,
        "cluster_summary": cluster_stats,
        "recluster_summary": recluster_stats,
        "backups": backups,
    }
    return summary


if __name__ == "__main__":
    import sys
    print("=" * 70)
    print("PhotoVault — Re-indexing Library with YuNet + SFace (digiKam Architecture)")
    print("=" * 70)
    res = reindex_library_faces()
    print("\nRe-indexing Complete!")
    print(f"Total Photos: {res['total_photos']}")
    print(f"Total Faces:  {res['total_faces']}")
    print(f"Total People: {res['total_persons']}")
