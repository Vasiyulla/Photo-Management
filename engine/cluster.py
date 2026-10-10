"""
PhotoVault — Incremental face clustering by cosine threshold (T8).

Matches face embeddings against existing indexed faces using cosine similarity
in FAISS. If a match is found above threshold, assigns the existing person_id;
otherwise creates a new person record in the ``persons`` table.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from engine.embeddings import FaceEmbeddingStore
from engine.index import FacesIndex, sync_faces_index


def match_or_create_person(
    face_id: int,
    conn: sqlite3.Connection,
    store: FaceEmbeddingStore,
    index: FacesIndex,
    threshold: float = 0.40,
) -> tuple[int, bool]:
    """Assign a face to an existing person if cosine similarity >= threshold,
    using multi-member consensus / centroid verification to prevent single-linkage chaining.

    Otherwise creates a new person in the ``persons`` table.

    Returns:
        (person_id, is_new_person)
    """
    row = conn.execute(
        "SELECT photo_id, face_embedding_id, person_id, bbox FROM faces WHERE id = ?",
        (face_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Face with id={face_id} not found.")

    photo_id, face_embedding_id, existing_person_id, bbox = row
    if existing_person_id is not None:
        return existing_person_id, False

    if face_embedding_id is None:
        raise ValueError(f"Face with id={face_id} has no face_embedding_id.")

    vec = store.get(face_embedding_id)
    if vec is None:
        raise ValueError(f"Embedding {face_embedding_id} not found in store.")

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    # Quality check: small face crops (< 70px width) require a stricter threshold
    effective_threshold = threshold
    if bbox:
        try:
            parts = [float(x) for x in str(bbox).split(",")]
            if len(parts) == 4:
                face_w = parts[2] - parts[0]
                if 0 < face_w < 70 and threshold < 0.45:
                    effective_threshold = max(threshold, 0.45)
        except Exception:
            pass

    # Search for nearest neighbors in the FAISS index
    matches = index.search(vec, top_k=25, min_score=effective_threshold)

    # Group matches by candidate person_id
    candidate_persons: dict[int, list[tuple[int, float]]] = {}
    for matched_emb_id, score in matches:
        if matched_emb_id == face_embedding_id:
            continue

        p_row = conn.execute(
            "SELECT person_id FROM faces WHERE face_embedding_id = ? AND person_id IS NOT NULL",
            (matched_emb_id,),
        ).fetchone()

        if p_row and p_row[0] is not None:
            pid = p_row[0]
            candidate_persons.setdefault(pid, []).append((matched_emb_id, float(score)))

    # Evaluate candidate persons using centroid and consensus matching
    best_candidate_id: int | None = None
    best_candidate_score: float = -1.0

    for pid, matched_list in candidate_persons.items():
        all_p_emb_rows = conn.execute(
            "SELECT face_embedding_id FROM faces WHERE person_id = ? AND face_embedding_id IS NOT NULL",
            (pid,),
        ).fetchall()

        all_emb_ids = [r[0] for r in all_p_emb_rows if r[0] is not None and r[0] != face_embedding_id]
        if not all_emb_ids:
            continue

        p_vecs = []
        for eid in all_emb_ids:
            v = store.get(eid)
            if v is not None:
                v_norm = np.linalg.norm(v)
                if v_norm > 0:
                    p_vecs.append(v / v_norm)

        if not p_vecs:
            continue

        # Case 1: Candidate person has only 1 face -> single match must meet effective_threshold
        if len(p_vecs) == 1:
            sim = float(np.dot(vec, p_vecs[0]))
            if sim >= effective_threshold and sim > best_candidate_score:
                best_candidate_id = pid
                best_candidate_score = sim
        else:
            # Case 2: Candidate person has >= 2 faces -> require centroid or consensus match
            centroid = np.mean(p_vecs, axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid /= c_norm
            centroid_sim = float(np.dot(vec, centroid))

            sims = [float(np.dot(vec, pv)) for pv in p_vecs]
            matching_count = sum(1 for s in sims if s >= effective_threshold)
            top_sims = sorted(sims, reverse=True)[:3]
            avg_top_sim = float(np.mean(top_sims))

            is_match = (centroid_sim >= effective_threshold) or (
                matching_count >= 2 and avg_top_sim >= effective_threshold
            )

            if is_match and centroid_sim > best_candidate_score:
                best_candidate_id = pid
                best_candidate_score = centroid_sim

    if best_candidate_id is not None:
        conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (best_candidate_id, face_id))
        conn.commit()
        return best_candidate_id, False

    # No match found: create a new person
    cur = conn.execute("INSERT INTO persons (display_name) VALUES (NULL)")
    new_person_id = cur.lastrowid
    conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (new_person_id, face_id))
    conn.commit()
    return new_person_id, True


def cluster_unassigned_faces(
    conn: sqlite3.Connection,
    store: FaceEmbeddingStore,
    index: FacesIndex,
    threshold: float = 0.40,
) -> dict[str, int]:
    """Cluster all faces that have an embedding but no person_id yet.

    Synchronizes the index first to ensure all embeddings are searchable.
    Returns summary dict: ``{"faces_processed": int, "new_persons": int, "errors": int}``.
    """
    # Ensure index has all existing embeddings
    sync_faces_index(conn, store, index)

    rows = conn.execute(
        "SELECT id FROM faces WHERE person_id IS NULL AND face_embedding_id IS NOT NULL ORDER BY id ASC"
    ).fetchall()

    summary = {
        "faces_processed": 0,
        "new_persons": 0,
        "errors": 0,
    }

    for (face_id,) in rows:
        try:
            person_id, is_new = match_or_create_person(
                face_id=face_id,
                conn=conn,
                store=store,
                index=index,
                threshold=threshold,
            )
            summary["faces_processed"] += 1
            if is_new:
                summary["new_persons"] += 1
        except Exception:
            summary["errors"] += 1

    return summary
