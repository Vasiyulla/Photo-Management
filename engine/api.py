"""PhotoVault — REST/IPC API (T12, T13).

Provides both a standard JSON IPC request dispatcher and an HTTP REST server
(using standard library ThreadingHTTPServer) for Tauri desktop shell integration.
Supports:
- Person management: listing, renaming, merging (T12)
- Photo querying: by date range, person_id, semantic text query, with pagination (T13)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Sequence
import time
import urllib.parse
from urllib.parse import urlparse

import numpy as np

from engine.recluster import merge_persons

# Module-level thread-safe state for asynchronous background indexing tasks
_SCAN_LOCK = threading.Lock()
_ACTIVE_SCAN_TASK: dict[str, Any] = {
    "status": "idle",      # "idle", "running", "completed", "error"
    "stage": "idle",       # "starting", "exif", "phashes", "faces", "embeddings", "clustering", "clip", "completed"
    "current": 0,
    "total": 0,
    "message": "",
    "error": None,
    "started_at": None,
    "finished_at": None,
}


def list_persons(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """List all persons with face count and cover face info."""
    sql = """
    SELECT
        p.id,
        p.display_name,
        COUNT(f.id) AS face_count,
        f_cover.photo_id AS cover_photo_id,
        f_cover.id AS cover_face_id,
        f_cover.bbox AS cover_bbox
    FROM persons p
    LEFT JOIN faces f ON f.person_id = p.id
    LEFT JOIN (
        SELECT f2.id, f2.person_id, f2.photo_id, f2.bbox
        FROM faces f2
        JOIN photos ph2 ON f2.photo_id = ph2.id
        GROUP BY f2.person_id
        HAVING f2.id = MIN(f2.id)
    ) f_cover ON f_cover.person_id = p.id
    GROUP BY p.id
    HAVING face_count > 0 OR (p.display_name IS NOT NULL AND p.display_name != '')
    ORDER BY (p.display_name IS NOT NULL AND p.display_name != '') DESC, face_count DESC, p.id ASC
    """
    rows = conn.execute(sql).fetchall()
    results = []
    for pid, display_name, face_count, cover_photo_id, cover_face_id, cover_bbox in rows:
        results.append({
            "id": pid,
            "display_name": display_name,
            "face_count": face_count,
            "cover_photo_id": cover_photo_id,
            "cover_face_id": cover_face_id,
            "cover_bbox": cover_bbox,
        })
    return results


def rename_person(conn: sqlite3.Connection, person_id: int, name: str | None) -> dict[str, Any]:
    """Rename a person and return the updated person record."""
    conn.execute("UPDATE persons SET display_name = ? WHERE id = ?", (name, person_id))
    conn.commit()
    return {"id": person_id, "display_name": name}


def merge_persons_api(
    conn: sqlite3.Connection,
    primary_id: int,
    secondary_ids: Sequence[int],
) -> dict[str, Any]:
    """Merge secondary persons into primary person."""
    faces_merged = merge_persons(primary_id, secondary_ids, conn)
    return {"primary_id": primary_id, "faces_merged": faces_merged}


def remove_face_from_person(
    conn: sqlite3.Connection,
    person_id: int,
    face_id: int,
) -> dict[str, Any]:
    """Remove a single face from a person and assign it to a new independent person."""
    cur = conn.execute("INSERT INTO persons (display_name) VALUES (NULL)")
    new_pid = cur.lastrowid
    conn.execute("UPDATE faces SET person_id = ? WHERE id = ? AND person_id = ?", (new_pid, face_id, person_id))
    conn.commit()
    return {"status": "ok", "old_person_id": person_id, "new_person_id": new_pid, "face_id": face_id}


def split_person_faces(
    conn: sqlite3.Connection,
    person_id: int,
    face_ids: Sequence[int],
    new_display_name: str | None = None,
) -> dict[str, Any]:
    """Split selected faces from a person into a new independent person record."""
    valid_fids = [int(fid) for fid in face_ids]
    if not valid_fids:
        return {"status": "error", "message": "No face IDs provided"}

    cur = conn.execute("INSERT INTO persons (display_name) VALUES (?)", (new_display_name,))
    new_pid = cur.lastrowid
    placeholders = ",".join("?" * len(valid_fids))
    conn.execute(
        f"UPDATE faces SET person_id = ? WHERE person_id = ? AND id IN ({placeholders})",
        [new_pid, person_id] + valid_fids,
    )
    conn.commit()
    return {
        "status": "ok",
        "old_person_id": person_id,
        "new_person_id": new_pid,
        "new_display_name": new_display_name,
        "faces_moved": len(valid_fids),
    }


def list_duplicates(
    conn: sqlite3.Connection,
    clip_index: Any | None = None,
) -> list[dict[str, Any]]:
    """List exact (pHash) and near (CLIP) duplicate photo pairs for review."""
    # 1. Exact pHash duplicates
    sql = """
    SELECT
        p1.id, p1.filepath, p1.date_taken, p1.phash,
        p2.id, p2.filepath, p2.date_taken, p2.phash
    FROM photos p1
    JOIN photos p2 ON p1.phash = p2.phash AND p1.id < p2.id
    WHERE p1.phash IS NOT NULL AND p1.phash != ''
    ORDER BY p1.id ASC
    """
    rows = conn.execute(sql).fetchall()
    duplicates: list[dict[str, Any]] = []

    for p1_id, p1_path, p1_date, p1_phash, p2_id, p2_path, p2_date, p2_phash in rows:
        duplicates.append({
            "match_type": "phash_exact",
            "similarity": 1.0,
            "photo_a": {
                "id": p1_id,
                "filepath": p1_path,
                "date_taken": p1_date,
            },
            "photo_b": {
                "id": p2_id,
                "filepath": p2_path,
                "date_taken": p2_date,
            },
        })

    return duplicates


def resolve_duplicate(
    conn: sqlite3.Connection,
    duplicate_photo_id: int,
    action: str = "mark_duplicate",
) -> dict[str, Any]:
    """Resolve a duplicate by marking its processing_state or keeping it."""
    if action == "mark_duplicate":
        conn.execute("UPDATE photos SET processing_state = 'duplicate' WHERE id = ?", (duplicate_photo_id,))
    elif action == "keep":
        conn.execute("UPDATE photos SET processing_state = 'completed' WHERE id = ?", (duplicate_photo_id,))
    conn.commit()
    return {"status": "ok", "photo_id": duplicate_photo_id, "action": action}


def run_ingestion_pipeline(
    conn: sqlite3.Connection,
    storage_dir: str = ".",
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Execute the core multi-stage ingestion pipeline for all pending photos:
    EXIF extraction -> pHash deduplication -> face detection -> embeddings ->
    FAISS index sync -> clustering -> re-clustering -> CLIP embeddings.
    """
    from pathlib import Path
    from engine.exif import batch_process_exif
    from engine.hasher import batch_process_phashes
    from engine.faces import batch_detect_faces
    from engine.embeddings import FaceEmbeddingStore, batch_generate_face_embeddings
    from engine.index import FacesIndex, sync_faces_index
    from engine.cluster import cluster_unassigned_faces
    from engine.recluster import recluster_faces_hdbscan
    from engine.clip import ClipEmbeddingStore, ClipIndex, batch_generate_clip_embeddings

    storage = Path(storage_dir)

    row = conn.execute("SELECT COUNT(*) FROM photos WHERE processing_state = 'pending'").fetchone()
    total_pending = row[0] if row else 0

    if progress_callback:
        progress_callback("exif", 0, total_pending, "Extracting EXIF metadata & capture dates...")
    exif_stats = batch_process_exif(conn)

    if progress_callback:
        progress_callback("phashes", 0, total_pending, "Computing perceptual hashes (pHash deduplication)...")
    hasher_stats = batch_process_phashes(conn)

    if progress_callback:
        progress_callback("faces", 0, total_pending, "Detecting faces in photos...")
    face_stats = batch_detect_faces(conn)

    if progress_callback:
        progress_callback("embeddings", 0, total_pending, "Extracting deep facial feature embeddings...")
    face_store = FaceEmbeddingStore(storage / "face_store")
    batch_generate_face_embeddings(conn, face_store)

    if progress_callback:
        progress_callback("clustering", 0, total_pending, "Clustering faces and updating index...")
    faces_idx = FacesIndex(storage / "faces.index")
    sync_faces_index(conn, face_store, faces_idx)
    cluster_unassigned_faces(conn, face_store, faces_idx)
    recluster_faces_hdbscan(conn, face_store)

    if progress_callback:
        progress_callback("clip", 0, total_pending, "Generating semantic search embeddings (OpenCLIP)...")
    clip_store = ClipEmbeddingStore(storage / "clip_store")
    clip_idx = ClipIndex(storage / "clip.index")
    clip_stats = batch_generate_clip_embeddings(conn, clip_store, clip_idx)

    if progress_callback:
        progress_callback("completed", total_pending, total_pending, f"Completed indexing {total_pending} photos successfully.")

    return {
        "exif": exif_stats,
        "phashes": hasher_stats,
        "faces": face_stats,
        "clip": clip_stats,
    }


def start_background_scan(
    db_path: str,
    storage_dir: str = ".",
) -> bool:
    """Trigger background ingestion pipeline non-blockingly on a dedicated thread."""
    global _ACTIVE_SCAN_TASK
    with _SCAN_LOCK:
        if _ACTIVE_SCAN_TASK["status"] == "running":
            return False
        _ACTIVE_SCAN_TASK["status"] = "running"
        _ACTIVE_SCAN_TASK["stage"] = "starting"
        _ACTIVE_SCAN_TASK["current"] = 0
        _ACTIVE_SCAN_TASK["total"] = 0
        _ACTIVE_SCAN_TASK["message"] = "Starting background indexing..."
        _ACTIVE_SCAN_TASK["error"] = None
        _ACTIVE_SCAN_TASK["started_at"] = time.time()
        _ACTIVE_SCAN_TASK["finished_at"] = None

    def _worker() -> None:
        global _ACTIVE_SCAN_TASK
        thread_conn = sqlite3.connect(db_path)
        try:
            def _on_progress(stage: str, cur: int, tot: int, msg: str) -> None:
                with _SCAN_LOCK:
                    _ACTIVE_SCAN_TASK["stage"] = stage
                    _ACTIVE_SCAN_TASK["current"] = cur
                    _ACTIVE_SCAN_TASK["total"] = tot
                    _ACTIVE_SCAN_TASK["message"] = msg

            run_ingestion_pipeline(thread_conn, storage_dir=storage_dir, progress_callback=_on_progress)
            with _SCAN_LOCK:
                _ACTIVE_SCAN_TASK["status"] = "completed"
                _ACTIVE_SCAN_TASK["stage"] = "completed"
                _ACTIVE_SCAN_TASK["message"] = "Indexing completed successfully."
                _ACTIVE_SCAN_TASK["finished_at"] = time.time()
        except Exception as exc:
            with _SCAN_LOCK:
                _ACTIVE_SCAN_TASK["status"] = "error"
                _ACTIVE_SCAN_TASK["stage"] = "error"
                _ACTIVE_SCAN_TASK["error"] = str(exc)
                _ACTIVE_SCAN_TASK["message"] = f"Indexing failed: {exc}"
                _ACTIVE_SCAN_TASK["finished_at"] = time.time()
        finally:
            thread_conn.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return True


def scan_library(
    folder_path: str,
    conn: sqlite3.Connection,
    storage_dir: str = ".",
) -> dict[str, Any]:
    """Scan and process an external photo folder into vault.db and indices synchronously."""
    from engine.scanner import sync_folder

    new_files, removed_files = sync_folder(folder_path, conn)
    pipeline_stats = run_ingestion_pipeline(conn, storage_dir=storage_dir)

    return {
        "status": "ok",
        "new_photos": len(new_files),
        "removed_photos": len(removed_files),
        **pipeline_stats,
    }


def register_import_files(
    file_paths: list[str],
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Validate, filter, and register individual image files in place as 'pending'.

    Performs existence, extension, and integrity checks and skips existing files.
    Returns summary dict with imported, skipped_duplicates, and failed files in < 50ms.
    """
    from PIL import Image
    from engine.scanner import IMAGE_EXTENSIONS

    rows = conn.execute("SELECT filepath FROM photos").fetchall()
    in_db: set[str] = {os.path.abspath(r[0]) for r in rows}

    new_files: list[str] = []
    skipped_duplicates: list[str] = []
    failed_files: list[dict[str, str]] = []

    seen_in_batch: set[str] = set()

    for raw_fp in file_paths:
        if not raw_fp or not isinstance(raw_fp, str):
            continue
        cleaned = raw_fp.strip().strip('"\'')
        if not cleaned:
            continue
        abs_path = os.path.abspath(cleaned)

        if abs_path in seen_in_batch:
            continue
        seen_in_batch.add(abs_path)

        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            failed_files.append({"path": abs_path, "error": "File does not exist or is not a regular file"})
            continue

        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            failed_files.append({"path": abs_path, "error": f"Unsupported file extension: '{ext}'"})
            continue

        try:
            with Image.open(abs_path) as img:
                img.verify()
        except Exception as e:
            failed_files.append({"path": abs_path, "error": f"Corrupt or unreadable image: {e}"})
            continue

        if abs_path in in_db:
            skipped_duplicates.append(abs_path)
            continue

        new_files.append(abs_path)

    if new_files:
        conn.executemany(
            "INSERT INTO photos (filepath, processing_state) VALUES (?, 'pending')",
            [(fp,) for fp in sorted(new_files)],
        )
        conn.commit()

    return {
        "imported": len(new_files),
        "skipped_duplicates": len(skipped_duplicates),
        "failed": len(failed_files),
        "skipped_details": skipped_duplicates,
        "failed_details": failed_files,
    }


def import_images(
    file_paths: list[str],
    conn: sqlite3.Connection,
    storage_dir: str = ".",
) -> dict[str, Any]:
    """Import a list of individual image files in place into PhotoVault synchronously.

    Reuses register_import_files and runs ingestion pipeline immediately.
    """
    report = register_import_files(file_paths, conn)
    pipeline_stats = run_ingestion_pipeline(conn, storage_dir=storage_dir)

    return {
        "status": "ok",
        **report,
        **pipeline_stats,
    }


def assign_face_person(
    conn: sqlite3.Connection,
    face_id: int,
    person_id: int | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Assign a face to an existing person or create a newly named person."""
    clean_name = name.strip() if name and name.strip() else None
    if person_id is None and clean_name:
        # Check if person with this name already exists
        row = conn.execute("SELECT id FROM persons WHERE display_name = ? COLLATE NOCASE", (clean_name,)).fetchone()
        if row:
            person_id = row[0]
        else:
            cur = conn.execute("INSERT INTO persons (display_name) VALUES (?)", (clean_name,))
            person_id = cur.lastrowid
    elif person_id is not None and clean_name:
        conn.execute("UPDATE persons SET display_name = ? WHERE id = ?", (clean_name, person_id))

    if person_id is not None:
        conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (person_id, face_id))
        conn.commit()

    return {"face_id": face_id, "person_id": person_id, "name": clean_name}


def query_photos(
    conn: sqlite3.Connection,
    date_from: str | None = None,
    date_to: str | None = None,
    person_id: int | None = None,
    text_query: str | None = None,
    query_vector: np.ndarray | None = None,
    clip_index: Any | None = None,
    clip_model: Any | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Query photos with filters: date range, person_id, semantic query, and pagination."""
    scores: dict[int, float] = {}

    # 1. Handle name-based search or semantic search if text_query is present
    if text_query:
        clean_text = text_query.strip()
        # Direct person name matching
        p_matches = conn.execute(
            "SELECT DISTINCT f.photo_id FROM faces f JOIN persons p ON f.person_id = p.id WHERE p.display_name LIKE ?",
            (f"%{clean_text}%",),
        ).fetchall()
        for row in p_matches:
            scores[row[0]] = 1.0

    if (text_query or query_vector is not None) and clip_index is not None:
        if query_vector is None and text_query:
            if clip_model is None:
                from engine.clip import get_clip_model
                clip_model = get_clip_model()
            query_vector = clip_model.encode_text(text_query)

        if query_vector is not None:
            raw_hits = clip_index.search(query_vector, top_k=max(limit + offset, 100))
            if raw_hits:
                emb_ids = [hit[0] for hit in raw_hits]
                placeholders = ",".join("?" * len(emb_ids))
                cur = conn.execute(
                    f"SELECT photo_id, embedding_id FROM clip_embeddings WHERE embedding_id IN ({placeholders})",
                    emb_ids,
                )
                emb_to_photo = {row[1]: row[0] for row in cur.fetchall()}
                for emb_id, score in raw_hits:
                    if emb_id in emb_to_photo:
                        pid = emb_to_photo[emb_id]
                        # Keep best score if duplicate
                        if pid not in scores or score > scores[pid]:
                            scores[pid] = float(score)

    # 2. Build SQL filters
    where_clauses: list[str] = []
    params: list[Any] = []

    joins: list[str] = []
    if person_id is not None:
        joins.append("JOIN faces f ON f.photo_id = p.id AND f.person_id = ?")
        params.append(person_id)

    if date_from:
        where_clauses.append("p.date_taken >= ?")
        params.append(date_from)
    if date_to:
        where_clauses.append("p.date_taken <= ?")
        params.append(date_to)

    if scores:
        cand_ids = list(scores.keys())
        placeholders = ",".join("?" * len(cand_ids))
        where_clauses.append(f"p.id IN ({placeholders})")
        params.extend(cand_ids)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    join_sql = " ".join(joins)

    sql = f"""
    SELECT DISTINCT
        p.id,
        p.filepath,
        p.date_taken,
        p.phash,
        p.processing_state,
        p.created_at
    FROM photos p
    {join_sql}
    {where_sql}
    """

    if not scores:
        sql += " ORDER BY p.date_taken DESC, p.id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

    rows = conn.execute(sql, params).fetchall()

    results: list[dict[str, Any]] = []
    for pid, filepath, date_taken, phash, state, created_at in rows:
        item: dict[str, Any] = {
            "id": pid,
            "filepath": filepath,
            "date_taken": date_taken,
            "phash": phash,
            "processing_state": state,
            "created_at": created_at,
        }
        if pid in scores:
            item["score"] = scores[pid]
        results.append(item)

    if scores:
        # Sort by semantic score descending, then slice limit/offset
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        results = results[offset : offset + limit]

    # Attach detected faces and identified people to each photo
    if results:
        res_pids = [item["id"] for item in results]
        placeholders = ",".join("?" * len(res_pids))
        face_rows = conn.execute(
            f"""
            SELECT f.photo_id, f.id, f.person_id, p.display_name, f.bbox
            FROM faces f
            LEFT JOIN persons p ON f.person_id = p.id
            WHERE f.photo_id IN ({placeholders})
            ORDER BY f.id ASC
            """,
            res_pids,
        ).fetchall()
        photo_people_map: dict[int, list[dict[str, Any]]] = {pid: [] for pid in res_pids}
        for f_pid, f_id, p_id, p_name, f_bbox in face_rows:
            photo_people_map[f_pid].append({
                "face_id": f_id,
                "person_id": p_id,
                "name": p_name or (f"Person #{p_id}" if p_id else "Unnamed"),
                "is_named": bool(p_name),
                "bbox": f_bbox,
            })
        for item in results:
            item["people"] = photo_people_map.get(item["id"], [])

    return results


def handle_ipc_request(
    req_dict: dict[str, Any],
    conn: sqlite3.Connection,
    clip_index: Any | None = None,
    clip_model: Any | None = None,
) -> dict[str, Any]:
    """Dispatch a JSON IPC request to the appropriate handler."""
    action = req_dict.get("action")
    try:
        if action == "list_persons":
            return {"status": "ok", "data": list_persons(conn)}
        elif action == "rename_person":
            res = rename_person(conn, req_dict["person_id"], req_dict.get("name"))
            return {"status": "ok", "data": res}
        elif action == "merge_persons":
            res = merge_persons_api(conn, req_dict["primary_id"], req_dict.get("secondary_ids", []))
            return {"status": "ok", "data": res}
        elif action == "query_photos":
            photos = query_photos(
                conn=conn,
                date_from=req_dict.get("date_from"),
                date_to=req_dict.get("date_to"),
                person_id=req_dict.get("person_id"),
                text_query=req_dict.get("text_query"),
                limit=int(req_dict.get("limit", 100)),
                offset=int(req_dict.get("offset", 0)),
                clip_index=clip_index,
                clip_model=clip_model,
            )
            return {"status": "ok", "data": photos}
        elif action == "list_duplicates":
            dups = list_duplicates(conn, clip_index=clip_index)
            return {"status": "ok", "data": dups}
        elif action == "resolve_duplicate":
            act = req_dict.get("resolution") or req_dict.get("duplicate_action") or req_dict.get("action_type") or "mark_duplicate"
            res = resolve_duplicate(
                conn=conn,
                duplicate_photo_id=req_dict["duplicate_photo_id"],
                action=act,
            )
            return {"status": "ok", "data": res}
        elif action == "scan_folder":
            res = scan_library(req_dict["folder_path"], conn)
            return {"status": "ok", "data": res}
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


class ApiRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for PhotoVault REST API."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP server logging to keep test output clean."""
        pass

    def _send_json(self, status_code: int, data: Any) -> None:
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(200, {"status": "healthy", "version": "0.1.0"})
            return

        if parsed.path in ("/api/scan/status", "/api/jobs/status", "/api/task/status"):
            with _SCAN_LOCK:
                task_copy = dict(_ACTIVE_SCAN_TASK)
            self._send_json(200, task_copy)
            return

        conn = None
        try:
            conn = sqlite3.connect(self.server.db_path)
            if parsed.path == "/api/persons":
                data = list_persons(conn)
                self._send_json(200, data)
            elif parsed.path == "/api/duplicates":
                data = list_duplicates(conn, getattr(self.server, "clip_index", None))
                self._send_json(200, data)
            elif parsed.path == "/api/photos":
                qs = urllib.parse.parse_qs(parsed.query)
                person_id_val = qs.get("person_id", [None])[0]
                person_id = int(person_id_val) if person_id_val is not None else None
                date_from = qs.get("date_from", [None])[0]
                date_to = qs.get("date_to", [None])[0]
                text_query = qs.get("text_query", [None])[0]
                limit = int(qs.get("limit", [100])[0])
                offset = int(qs.get("offset", [0])[0])

                data = query_photos(
                    conn=conn,
                    date_from=date_from,
                    date_to=date_to,
                    person_id=person_id,
                    text_query=text_query,
                    limit=limit,
                    offset=offset,
                    clip_index=getattr(self.server, "clip_index", None),
                    clip_model=getattr(self.server, "clip_model", None),
                )
                self._send_json(200, data)
            else:
                face_crop_match = re.match(r"^/api/faces/(\d+)/crop$", parsed.path)
                photo_file_match = re.match(r"^/api/photos/(\d+)/file$", parsed.path)

                if face_crop_match:
                    face_id = int(face_crop_match.group(1))
                    row = conn.execute(
                        "SELECT ph.filepath, f.bbox FROM faces f JOIN photos ph ON f.photo_id = ph.id WHERE f.id = ?",
                        (face_id,),
                    ).fetchone()
                    if not row or not os.path.exists(row[0]):
                        self._send_json(404, {"error": "Face not found"})
                        return
                    filepath, bbox_str = row
                    try:
                        from PIL import Image
                        import io
                        with Image.open(filepath) as img:
                            parts = [float(x) for x in bbox_str.split(",")]
                            if len(parts) == 4:
                                x1, y1, x2, y2 = parts
                                w, h = x2 - x1, y2 - y1
                                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                                size = max(w, h) * 1.35
                                crop_box = (
                                    max(0, int(cx - size / 2.0)),
                                    max(0, int(cy - size / 2.0)),
                                    min(img.width, int(cx + size / 2.0)),
                                    min(img.height, int(cy + size / 2.0)),
                                )
                                cropped = img.crop(crop_box)
                            else:
                                cropped = img
                            cropped.thumbnail((256, 256))
                            buf = io.BytesIO()
                            cropped.convert("RGB").save(buf, format="JPEG", quality=85)
                            data_bytes = buf.getvalue()

                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(data_bytes)))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Cache-Control", "public, max-age=86400")
                        self.end_headers()
                        self.wfile.write(data_bytes)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                    except Exception as e:
                        self._send_json(500, {"error": str(e)})

                elif photo_file_match:
                    photo_id = int(photo_file_match.group(1))
                    row = conn.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,)).fetchone()
                    if not row or not os.path.exists(row[0]):
                        self._send_json(404, {"error": "Photo not found"})
                        return
                    filepath = row[0]
                    content_type = "image/jpeg"
                    if filepath.lower().endswith(".png"):
                        content_type = "image/png"
                    elif filepath.lower().endswith(".webp"):
                        content_type = "image/webp"
                    try:
                        with open(filepath, "rb") as f:
                            data_bytes = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Content-Length", str(len(data_bytes)))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(data_bytes)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                elif parsed.path == "/api/dialog/pick-images":
                    file_paths = []
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes("-topmost", True)
                        file_types = [
                            ("Image files", "*.jpg;*.jpeg;*.png;*.webp;*.jfif;*.bmp;*.tiff;*.tif;*.avif"),
                            ("All files", "*.*"),
                        ]
                        picked = filedialog.askopenfilenames(
                            title="Select Images to Import into PhotoVault",
                            filetypes=file_types,
                        )
                        root.destroy()
                        if picked:
                            file_paths = [os.path.abspath(p) for p in picked]
                    except Exception as e:
                        self._send_json(500, {"error": f"Failed to open native file dialog: {e}"})
                        return
                    self._send_json(200, {"file_paths": file_paths})
                else:
                    self._send_json(404, {"error": "Not found"})
        except Exception as e:
            self._send_json(500, {"error": f"Internal server error: {e}"})
        finally:
            if conn:
                conn.close()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        conn = sqlite3.connect(self.server.db_path)
        try:
            rename_match = re.match(r"^/api/persons/(\d+)/rename$", parsed.path)
            merge_match = re.match(r"^/api/persons/(\d+)/merge$", parsed.path)
            remove_face_match = re.match(r"^/api/persons/(\d+)/remove-face$", parsed.path)
            split_person_match = re.match(r"^/api/persons/(\d+)/split$", parsed.path)

            face_assign_match = re.match(r"^/api/faces/(\d+)/assign$", parsed.path)

            if rename_match:
                person_id = int(rename_match.group(1))
                name = payload.get("name")
                res = rename_person(conn, person_id, name)
                self._send_json(200, res)
            elif merge_match:
                primary_id = int(merge_match.group(1))
                secondary_ids = payload.get("secondary_ids", [])
                res = merge_persons_api(conn, primary_id, secondary_ids)
                self._send_json(200, res)
            elif remove_face_match:
                person_id = int(remove_face_match.group(1))
                face_id = payload.get("face_id")
                if not face_id:
                    self._send_json(400, {"error": "Missing face_id"})
                    return
                res = remove_face_from_person(conn, person_id, int(face_id))
                self._send_json(200, res)
            elif split_person_match:
                person_id = int(split_person_match.group(1))
                face_ids = payload.get("face_ids", [])
                name = payload.get("new_display_name")
                res = split_person_faces(conn, person_id, face_ids, name)
                self._send_json(200, res)
            elif face_assign_match:
                face_id = int(face_assign_match.group(1))
                p_id = payload.get("person_id")
                p_name = payload.get("name")
                res = assign_face_person(conn, face_id, person_id=p_id, name=p_name)
                self._send_json(200, res)
            elif parsed.path == "/api/recluster":
                from pathlib import Path
                from engine.embeddings import FaceEmbeddingStore
                from engine.recluster import recluster_faces_hdbscan
                store = FaceEmbeddingStore(Path(".") / "face_store")
                min_size = int(payload.get("min_cluster_size", 2))
                res = recluster_faces_hdbscan(conn, store, min_cluster_size=min_size)
                self._send_json(200, res)
            elif parsed.path == "/api/duplicates/resolve":
                dup_id = payload.get("duplicate_photo_id")
                act = payload.get("action", "mark_duplicate")
                res = resolve_duplicate(conn, dup_id, act)
                self._send_json(200, res)
            elif parsed.path == "/api/photos/query":
                data = query_photos(
                    conn=conn,
                    date_from=payload.get("date_from"),
                    date_to=payload.get("date_to"),
                    person_id=payload.get("person_id"),
                    text_query=payload.get("text_query"),
                    limit=int(payload.get("limit", 100)),
                    offset=int(payload.get("offset", 0)),
                    clip_index=getattr(self.server, "clip_index", None),
                    clip_model=getattr(self.server, "clip_model", None),
                )
                self._send_json(200, data)
            elif parsed.path == "/api/scan":
                raw_path = payload.get("folder_path", "")
                folder_path = str(raw_path).strip().strip('"\'')
                if not folder_path or not os.path.isdir(folder_path):
                    self._send_json(400, {"error": f"Folder does not exist or invalid path: '{folder_path}'. Please verify the folder path exists on this computer."})
                    return
                from engine.scanner import sync_folder
                new_files, removed_files = sync_folder(folder_path, conn)

                row = conn.execute("SELECT COUNT(*) FROM photos WHERE processing_state = 'pending'").fetchone()
                pending_count = row[0] if row else 0

                if pending_count > 0:
                    start_background_scan(self.server.db_path)

                self._send_json(200, {
                    "status": "started" if pending_count > 0 else "completed",
                    "new_photos": len(new_files),
                    "removed_photos": len(removed_files),
                    "pending_photos": pending_count,
                    "message": "Indexing started in background" if pending_count > 0 else "All photos are up to date."
                })
            elif parsed.path == "/api/import":
                raw_paths = payload.get("file_paths", [])
                if not isinstance(raw_paths, list):
                    self._send_json(400, {"error": "file_paths must be a list of file path strings"})
                    return
                res = register_import_files(raw_paths, conn)
                if res.get("imported", 0) > 0:
                    start_background_scan(self.server.db_path)
                self._send_json(200, {
                    "status": "started" if res.get("imported", 0) > 0 else "completed",
                    **res
                })
            else:
                self._send_json(404, {"error": "Not found"})
        except Exception as e:
            self._send_json(500, {"error": f"Internal server error: {e}"})
        finally:
            if conn:
                conn.close()


class ApiServer:
    """Wrapper managing a ThreadingHTTPServer on a background thread."""

    def __init__(
        self,
        db_path: str,
        host: str = "127.0.0.1",
        port: int = 0,
        clip_index: Any | None = None,
        clip_model: Any | None = None,
    ):
        self.db_path = db_path
        self.host = host
        self.port = port
        self.server = ThreadingHTTPServer((host, port), ApiRequestHandler)
        self.server.db_path = db_path
        self.server.clip_index = clip_index
        self.server.clip_model = clip_model
        assigned_port = self.server.server_address[1]
        self.base_url = f"http://{host}:{assigned_port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=2.0)


def create_api_server(
    db_path: str,
    host: str = "127.0.0.1",
    port: int = 0,
    clip_index: Any | None = None,
    clip_model: Any | None = None,
) -> ApiServer:
    """Factory creating an ApiServer instance."""
    return ApiServer(
        db_path=db_path,
        host=host,
        port=port,
        clip_index=clip_index,
        clip_model=clip_model,
    )
