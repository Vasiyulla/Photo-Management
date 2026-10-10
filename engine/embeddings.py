"""
PhotoVault — SFace embedding generation.
Part of the digiKam-mimicking facial recognition pipeline.

Generates 128-dimensional normalized face embeddings using OpenCV SFace
(FaceRecognizerSF via native cv2.dnn ONNX Runtime / CPU).
Embeddings are persisted in a local vector store and linked to the
``faces.face_embedding_id`` database column.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any
import urllib.request

import cv2
import numpy as np

from engine.faces import get_face_detector

# Default SFace model path and download URL
DEFAULT_SFACE_PATH = Path("models/face_recognition_sface_2021dec.onnx")
SFACE_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# Cached global recognizer instance
_RECOGNIZER: cv2.FaceRecognizerSF | None = None


def ensure_sface_model(model_path: Path = DEFAULT_SFACE_PATH) -> Path:
    """Ensure SFace ONNX model is present on disk, downloading if necessary."""
    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[embeddings] Downloading SFace model to {model_path}...")
        urllib.request.urlretrieve(SFACE_DOWNLOAD_URL, str(model_path))
    return model_path


def get_face_recognizer(
    model_path: str | Path | None = None,
) -> cv2.FaceRecognizerSF:
    """Retrieve or initialize the cached OpenCV SFace face recognition model."""
    global _RECOGNIZER
    if _RECOGNIZER is None:
        p = Path(model_path) if model_path else DEFAULT_SFACE_PATH
        resolved_path = ensure_sface_model(p)
        _RECOGNIZER = cv2.FaceRecognizerSF.create(
            model=str(resolved_path),
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
    return _RECOGNIZER


# Backward compatibility alias
get_face_embedder = get_face_recognizer


class FaceEmbeddingStore:
    """Local vector store for 128-dim SFace embeddings backed by numpy arrays."""

    def __init__(self, dir_path: str | Path, dimension: int = 128):
        self.dir_path = Path(dir_path)
        self.dimension = dimension
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.ids_file = self.dir_path / "face_ids.npy"
        self.vectors_file = self.dir_path / "face_vectors.npy"
        self._data: dict[int, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if self.ids_file.exists() and self.vectors_file.exists():
            try:
                ids = np.load(str(self.ids_file))
                vectors = np.load(str(self.vectors_file))
                if len(vectors) > 0:
                    self.dimension = int(vectors.shape[1])
                for emb_id, vec in zip(ids, vectors):
                    self._data[int(emb_id)] = vec.astype(np.float32)
            except Exception:
                self._data = {}

    def _save_to_disk(self) -> None:
        if not self._data:
            return
        ids = np.array(sorted(self._data.keys()), dtype=np.int64)
        vectors = np.stack([self._data[i] for i in ids]).astype(np.float32)

        tmp_ids = self.dir_path / "face_ids.tmp.npy"
        tmp_vecs = self.dir_path / "face_vectors.tmp.npy"

        np.save(str(tmp_ids), ids)
        np.save(str(tmp_vecs), vectors)

        if tmp_ids.exists():
            os.replace(tmp_ids, self.ids_file)
        if tmp_vecs.exists():
            os.replace(tmp_vecs, self.vectors_file)

    def save(self, embedding_id: int, vector: np.ndarray) -> None:
        """Store an embedding vector normalized to unit length."""
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if vec.shape != (self.dimension,):
            # Auto-adapt if dimension changes
            if not self._data:
                self.dimension = vec.shape[0]
            else:
                raise ValueError(f"Expected {self.dimension}-dimensional vector, got {vec.shape}")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._data[int(embedding_id)] = vec
        self._save_to_disk()

    def get(self, embedding_id: int) -> np.ndarray | None:
        """Retrieve embedding vector by ID, or None if not present."""
        return self._data.get(int(embedding_id))

    def get_all(self) -> tuple[list[int], np.ndarray]:
        """Return (list_of_ids, 2d_matrix_of_vectors)."""
        if not self._data:
            return [], np.empty((0, self.dimension), dtype=np.float32)
        ids = sorted(self._data.keys())
        matrix = np.stack([self._data[i] for i in ids]).astype(np.float32)
        return ids, matrix

    def next_id(self) -> int:
        """Return the next unused integer ID."""
        if not self._data:
            return 1
        return max(self._data.keys()) + 1

    def __len__(self) -> int:
        return len(self._data)


def _bbox_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def generate_face_embeddings(
    photo_id: int,
    conn: sqlite3.Connection,
    store: FaceEmbeddingStore | None = None,
) -> list[int]:
    """Generate and store 128D SFace embeddings for all detected faces of a photo.

    Updates ``faces.face_embedding_id`` and ``photos.processing_state``.
    Returns list of assigned embedding IDs.
    """
    if store is None:
        store = FaceEmbeddingStore("face_store", dimension=128)

    row = conn.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row:
        raise ValueError(f"Photo with id={photo_id} not found.")

    filepath = Path(row[0])
    if not filepath.is_file():
        return []

    img = cv2.imread(str(filepath))
    if img is None:
        return []

    height, width = img.shape[:2]
    if width <= 0 or height <= 0:
        return []

    detector = get_face_detector()
    detector.setInputSize((width, height))
    _, raw_faces = detector.detect(img)

    recognizer = get_face_recognizer()

    # Fetch existing faces recorded for this photo
    face_rows = conn.execute(
        "SELECT id, bbox, face_embedding_id FROM faces WHERE photo_id = ? ORDER BY id ASC",
        (photo_id,),
    ).fetchall()

    assigned_ids: list[int] = []

    # If no faces were pre-recorded, record them now
    if not face_rows and raw_faces is not None and len(raw_faces) > 0:
        for rface in raw_faces:
            x, y, w, h = float(rface[0]), float(rface[1]), float(rface[2]), float(rface[3])
            bbox = [round(max(0.0, x), 2), round(max(0.0, y), 2), round(min(float(width), x + w), 2), round(min(float(height), y + h), 2)]
            bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            conn.execute(
                "INSERT INTO faces (photo_id, person_id, bbox, face_embedding_id) VALUES (?, NULL, ?, NULL)",
                (photo_id, bbox_str),
            )
        conn.commit()
        face_rows = conn.execute(
            "SELECT id, bbox, face_embedding_id FROM faces WHERE photo_id = ? ORDER BY id ASC",
            (photo_id,),
        ).fetchall()

    if not face_rows or raw_faces is None or len(raw_faces) == 0:
        conn.execute(
            "UPDATE photos SET processing_state = 'embeddings_generated' WHERE id = ?",
            (photo_id,),
        )
        conn.commit()
        return []

    # Match raw detected faces with database face rows
    used_raw_indices: set[int] = set()
    for face_id, bbox_str, existing_emb_id in face_rows:
        if existing_emb_id is not None:
            assigned_ids.append(existing_emb_id)
            continue

        try:
            target_bbox = [float(c) for c in bbox_str.split(",")]
        except Exception:
            continue

        # Find best matching face in raw_faces
        best_idx = -1
        best_iou = -1.0
        for idx, rface in enumerate(raw_faces):
            if idx in used_raw_indices:
                continue
            x, y, w, h = float(rface[0]), float(rface[1]), float(rface[2]), float(rface[3])
            r_box = [x, y, x + w, y + h]
            iou = _bbox_iou(target_bbox, r_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        # If matched or only one face
        if best_idx == -1 and len(raw_faces) == 1 and 0 not in used_raw_indices:
            best_idx = 0
            best_iou = 1.0

        if best_idx != -1 and (best_iou >= 0.25 or len(raw_faces) == 1):
            used_raw_indices.add(best_idx)
            primary_face = raw_faces[best_idx]

            # SFace alignment and crop
            aligned_face = recognizer.alignCrop(img, primary_face)
            feature = recognizer.feature(aligned_face)
            emb = feature.flatten().astype(np.float32)

            emb_id = store.next_id()
            store.save(emb_id, emb)
            conn.execute(
                "UPDATE faces SET face_embedding_id = ? WHERE id = ?",
                (emb_id, face_id),
            )
            assigned_ids.append(emb_id)

    conn.execute(
        "UPDATE photos SET processing_state = 'embeddings_generated' WHERE id = ?",
        (photo_id,),
    )
    conn.commit()

    return assigned_ids


def batch_generate_face_embeddings(
    conn: sqlite3.Connection,
    store: FaceEmbeddingStore | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Generate face embeddings for all photos that need them.

    Selects photos with ``processing_state = 'faces_detected'`` and
    processes each.
    """
    if store is None:
        store = FaceEmbeddingStore("face_store", dimension=128)

    query = (
        "SELECT id FROM photos WHERE processing_state = 'faces_detected' "
        "ORDER BY id ASC"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    rows = conn.execute(query).fetchall()

    summary = {
        "photos_processed": 0,
        "embeddings_generated": 0,
        "errors": 0,
    }

    for (photo_id,) in rows:
        try:
            assigned = generate_face_embeddings(photo_id, conn, store=store)
            summary["photos_processed"] += 1
            summary["embeddings_generated"] += len(assigned)
        except Exception:
            summary["errors"] += 1

    return summary
