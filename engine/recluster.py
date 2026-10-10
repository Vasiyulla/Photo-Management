"""
PhotoVault — HDBSCAN periodic re-cluster + merge job (T9).

Runs global HDBSCAN clustering on all 512-dimensional face embeddings to
identify split person clusters and merges duplicate person entries in the
``persons`` table, maintaining data integrity and updating ``faces.person_id``.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Sequence

import numpy as np
from sklearn.cluster import HDBSCAN

from engine.embeddings import FaceEmbeddingStore


def merge_persons(
    primary_id: int,
    secondary_ids: Sequence[int],
    conn: sqlite3.Connection,
) -> int:
    """Merge secondary person records into a single primary person record.

    - Re-assigns all faces of secondary persons to *primary_id*.
    - Preserves display_name: if primary is unnamed but a secondary has a name,
      the primary adopts that name.
    - Removes secondary person records from ``persons``.

    Returns count of face records re-assigned.
    """
    valid_secondaries = [int(sid) for sid in secondary_ids if int(sid) != int(primary_id)]
    if not valid_secondaries:
        return 0

    placeholders = ",".join("?" * len(valid_secondaries))

    # Check if primary needs a display_name from a secondary
    primary_row = conn.execute("SELECT display_name FROM persons WHERE id = ?", (primary_id,)).fetchone()
    primary_name = primary_row[0] if primary_row else None

    if not primary_name:
        named_secondary = conn.execute(
            f"SELECT display_name FROM persons WHERE id IN ({placeholders}) AND display_name IS NOT NULL AND display_name != '' LIMIT 1",
            valid_secondaries,
        ).fetchone()
        if named_secondary and named_secondary[0]:
            conn.execute(
                "UPDATE persons SET display_name = ? WHERE id = ?",
                (named_secondary[0], primary_id),
            )

    # Re-assign faces
    cur = conn.execute(
        f"UPDATE faces SET person_id = ? WHERE person_id IN ({placeholders})",
        [primary_id] + valid_secondaries,
    )
    faces_moved = cur.rowcount

    # Delete secondary persons
    conn.execute(
        f"DELETE FROM persons WHERE id IN ({placeholders})",
        valid_secondaries,
    )
    conn.commit()

    return faces_moved


def split_fragmented_persons(
    conn: sqlite3.Connection,
    store: FaceEmbeddingStore,
    distance_threshold: float = 0.60,
) -> int:
    """Split existing persons whose faces have diverged into multiple distinct clusters.

    For each person with >= 2 faces, runs AgglomerativeClustering with average linkage.
    If multiple sub-clusters are detected, preserves the largest sub-cluster on the
    original person and assigns the diverging sub-clusters to new person records.

    Returns the count of split events performed.
    """
    from sklearn.cluster import AgglomerativeClustering

    # Find all persons with at least 2 faces having embeddings
    candidate_persons = conn.execute("""
        SELECT person_id, COUNT(id) as cnt
        FROM faces
        WHERE person_id IS NOT NULL AND face_embedding_id IS NOT NULL
        GROUP BY person_id
        HAVING cnt >= 2
    """).fetchall()

    splits_count = 0

    for pid, count in candidate_persons:
        rows = conn.execute("""
            SELECT id, face_embedding_id
            FROM faces
            WHERE person_id = ? AND face_embedding_id IS NOT NULL
            ORDER BY id ASC
        """, (pid,)).fetchall()

        valid_fids = []
        vecs = []
        for fid, eid in rows:
            v = store.get(int(eid))
            if v is not None:
                norm = np.linalg.norm(v)
                if norm > 0:
                    valid_fids.append(fid)
                    vecs.append(v / norm)

        if len(vecs) < 2:
            continue

        matrix = np.stack(vecs).astype(np.float32)
        cl = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = cl.fit_predict(matrix)

        unique_labels = set(labels)
        if len(unique_labels) <= 1:
            continue

        # Group face_ids by sub-cluster label
        sub_clusters: dict[int, list[int]] = {}
        for fid, lbl in zip(valid_fids, labels):
            sub_clusters.setdefault(int(lbl), []).append(fid)

        # Sort sub-clusters by size descending so largest stays with current person_id
        sorted_clusters = sorted(sub_clusters.values(), key=lambda c: len(c), reverse=True)

        # Keep sorted_clusters[0] with pid, move the rest to new persons
        for minority_fids in sorted_clusters[1:]:
            cur = conn.execute("INSERT INTO persons (display_name) VALUES (NULL)")
            new_pid = cur.lastrowid
            placeholders = ",".join("?" * len(minority_fids))
            conn.execute(
                f"UPDATE faces SET person_id = ? WHERE id IN ({placeholders})",
                [new_pid] + minority_fids,
            )
            splits_count += 1

        conn.commit()

    return splits_count


def recluster_faces_hdbscan(
    conn: sqlite3.Connection,
    store: FaceEmbeddingStore,
    min_cluster_size: int = 2,
    min_samples: int | None = None,
    algorithm: str = "auto",
    distance_threshold: float = 0.60,
) -> dict[str, int]:
    """Execute global clustering across all face embeddings to detect
    and merge fragmented person identities, after splitting any over-merged persons.

    Returns summary dict:
        {"total_faces": int, "clusters_found": int, "merges_performed": int, "splits_performed": int, "noise_faces": int}
    """
    # Phase 1: Intra-Person Split Pass to heal any over-merged clusters
    splits_performed = split_fragmented_persons(conn, store, distance_threshold=distance_threshold)

    rows = conn.execute(
        "SELECT id, person_id, face_embedding_id FROM faces WHERE face_embedding_id IS NOT NULL ORDER BY id ASC"
    ).fetchall()

    if len(rows) < min_cluster_size:
        return {
            "total_faces": len(rows),
            "clusters_found": 0,
            "merges_performed": 0,
            "splits_performed": splits_performed,
            "noise_faces": 0,
        }

    # Gather vectors
    valid_faces: list[dict[str, Any]] = []
    vector_list: list[np.ndarray] = []

    for face_id, person_id, emb_id in rows:
        vec = store.get(int(emb_id))
        if vec is not None:
            valid_faces.append({
                "face_id": face_id,
                "person_id": person_id,
                "emb_id": emb_id,
            })
            vector_list.append(vec)

    if len(vector_list) < min_cluster_size:
        return {
            "total_faces": len(rows),
            "clusters_found": 0,
            "merges_performed": 0,
            "noise_faces": 0,
        }

    matrix = np.stack(vector_list).astype(np.float32)

    # Use AgglomerativeClustering with cosine distance (or HDBSCAN if explicitly chosen)
    if algorithm == "agglomerative" or (algorithm == "auto" and len(vector_list) > 50):
        from sklearn.cluster import AgglomerativeClustering
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clusterer.fit_predict(matrix)
    else:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            copy=True,
        )
        labels = clusterer.fit_predict(matrix)

    clusters: dict[int, list[dict[str, Any]]] = {}
    noise_count = 0

    for label, face_info in zip(labels, valid_faces):
        if label >= 0:
            clusters.setdefault(int(label), []).append(face_info)
        else:
            noise_count += 1

    merges_performed = 0

    for cluster_id, cluster_faces in clusters.items():
        # Get all distinct person_ids currently in this cluster
        person_ids = {f["person_id"] for f in cluster_faces if f["person_id"] is not None}

        if len(person_ids) > 1:
            # Multiple person records detected in a single dense cluster -> Merge them
            # Prefer person with a display_name, or lowest ID
            candidates: list[tuple[int, bool]] = []
            for pid in sorted(person_ids):
                p_row = conn.execute("SELECT display_name FROM persons WHERE id = ?", (pid,)).fetchone()
                has_name = bool(p_row and p_row[0])
                candidates.append((pid, has_name))

            # Sort candidate with name first, then lowest id
            candidates.sort(key=lambda x: (not x[1], x[0]))
            primary_id = candidates[0][0]
            secondary_ids = [c[0] for c in candidates[1:]]

            merge_persons(primary_id, secondary_ids, conn)
            merges_performed += 1

            target_person_id = primary_id
        elif len(person_ids) == 1:
            target_person_id = next(iter(person_ids))
        else:
            # None of the faces in cluster have a person_id yet -> Create new person
            cur = conn.execute("INSERT INTO persons (display_name) VALUES (NULL)")
            target_person_id = cur.lastrowid
            conn.commit()

        # Update any unassigned faces in this cluster
        for f in cluster_faces:
            if f["person_id"] is None:
                conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (target_person_id, f["face_id"]))
        conn.commit()

    return {
        "total_faces": len(valid_faces),
        "clusters_found": len(clusters),
        "merges_performed": merges_performed,
        "splits_performed": splits_performed,
        "noise_faces": noise_count,
    }
