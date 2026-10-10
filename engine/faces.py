"""
PhotoVault — Face detection using OpenCV YuNet (FaceDetectorYN).
Part of the digiKam-mimicking facial recognition pipeline.

Performs offline face detection on photos using OpenCV's native YuNet ONNX
model. Detected faces are stored in the ``faces`` table with their bounding
boxes (x1, y1, x2, y2), and ``photos.processing_state`` is updated.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
import urllib.request

import cv2
import numpy as np

# Default YuNet model path and download URL
DEFAULT_YUNET_PATH = Path("models/face_detection_yunet_2023mar.onnx")
YUNET_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# Cached global detector instance
_DETECTOR: cv2.FaceDetectorYN | None = None


def ensure_yunet_model(model_path: Path = DEFAULT_YUNET_PATH) -> Path:
    """Ensure YuNet ONNX model is present on disk, downloading if necessary."""
    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[faces] Downloading YuNet model to {model_path}...")
        urllib.request.urlretrieve(YUNET_DOWNLOAD_URL, str(model_path))
    return model_path


def get_face_detector(
    model_path: str | Path | None = None,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> cv2.FaceDetectorYN:
    """Retrieve or initialize the cached OpenCV YuNet face detection model."""
    global _DETECTOR
    if _DETECTOR is None:
        p = Path(model_path) if model_path else DEFAULT_YUNET_PATH
        resolved_path = ensure_yunet_model(p)
        _DETECTOR = cv2.FaceDetectorYN.create(
            model=str(resolved_path),
            config="",
            input_size=(320, 320),  # Placeholder, dynamically updated per image
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
    return _DETECTOR


def detect_faces(
    filepath_or_img: str | Path | np.ndarray,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Detect faces in an image file or numpy array using YuNet.

    Args:
        filepath_or_img: Absolute file path or BGR image numpy array.
        min_score: Minimum confidence detection threshold.

    Returns:
        List of dicts with keys:
            - "bbox": [x1, y1, x2, y2]
            - "bbox_str": "x1,y1,x2,y2"
            - "score": float confidence score
            - "kps": 5 facial landmarks [[x, y], ...] or None
            - "raw_face": 15-element raw detection array from YuNet
    """
    if isinstance(filepath_or_img, (str, Path)):
        p = Path(filepath_or_img)
        if not p.is_file():
            return []
        img = cv2.imread(str(p))
        if img is None:
            return []
    elif isinstance(filepath_or_img, np.ndarray):
        img = filepath_or_img
    else:
        return []

    height, width = img.shape[:2]
    if width <= 0 or height <= 0:
        return []

    detector = get_face_detector(score_threshold=min_score)
    # Dynamically set input size to match current image dimensions
    detector.setInputSize((width, height))

    _, raw_faces = detector.detect(img)
    if raw_faces is None or len(raw_faces) == 0:
        return []

    results: list[dict[str, Any]] = []
    for face in raw_faces:
        score = float(face[14])
        if score < min_score:
            continue

        x, y, w, h = float(face[0]), float(face[1]), float(face[2]), float(face[3])
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(width), x + w), min(float(height), y + h)

        bbox = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        kps = [
            [float(face[4 + 2 * i]), float(face[5 + 2 * i])]
            for i in range(5)
        ]

        results.append({
            "bbox": bbox,
            "bbox_str": bbox_str,
            "score": score,
            "kps": kps,
            "raw_face": face,
        })

    return results


def process_photo_faces(
    photo_id: int,
    conn: sqlite3.Connection,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Detect faces for a photo and record them in the ``faces`` table.

    Updates ``photos.processing_state = 'faces_detected'``.
    """
    row = conn.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row:
        raise ValueError(f"Photo with id={photo_id} not found.")

    filepath = row[0]
    detected = detect_faces(filepath, min_score=min_score)

    if detected:
        conn.executemany(
            "INSERT INTO faces (photo_id, person_id, bbox, face_embedding_id) VALUES (?, NULL, ?, NULL)",
            [(photo_id, f["bbox_str"]) for f in detected],
        )

    conn.execute(
        "UPDATE photos SET processing_state = 'faces_detected' WHERE id = ?",
        (photo_id,),
    )
    conn.commit()

    return detected


def batch_detect_faces(
    conn: sqlite3.Connection,
    min_score: float = 0.5,
    limit: int | None = None,
) -> dict[str, int]:
    """Run face detection on all unprocessed photos.

    Selects photos with ``processing_state = 'pending'`` and processes each.
    """
    query = "SELECT id FROM photos WHERE processing_state = 'pending' ORDER BY id ASC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    rows = conn.execute(query).fetchall()

    summary = {
        "photos_processed": 0,
        "faces_detected": 0,
        "errors": 0,
    }

    for (photo_id,) in rows:
        try:
            faces = process_photo_faces(photo_id, conn, min_score=min_score)
            summary["photos_processed"] += 1
            summary["faces_detected"] += len(faces)
        except Exception:
            summary["errors"] += 1

    return summary
