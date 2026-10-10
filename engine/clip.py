"""
PhotoVault — OpenCLIP embedding generation & semantic search (T10).

Generates 512-dimensional normalized OpenCLIP (ViT-B-32) embeddings for photos
and natural language text queries. Maintains the local FAISS index (``clip.index``)
and tracks photo associations in the ``clip_embeddings`` database table.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image

# Global cached CLIP model
_CLIP_MODEL: ClipModel | None = None


class ClipModel:
    """Wrapper around OpenCLIP ViT-B-32 for image and text embedding generation."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode_image(self, image_or_path: str | Path | Image.Image) -> np.ndarray:
        """Encode an image into a 512-dim L2-normalized float32 vector."""
        if isinstance(image_or_path, (str, Path)):
            with Image.open(image_or_path) as img:
                img_rgb = img.convert("RGB")
                tensor = self.preprocess(img_rgb).unsqueeze(0)
        elif isinstance(image_or_path, Image.Image):
            img_rgb = image_or_path.convert("RGB")
            tensor = self.preprocess(img_rgb).unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image type: {type(image_or_path)}")

        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a natural language text query into a 512-dim L2-normalized float32 vector."""
        tokens = self.tokenizer([text])
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten().astype(np.float32)


def get_clip_model() -> ClipModel:
    """Retrieve or initialize the global OpenCLIP model instance."""
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        _CLIP_MODEL = ClipModel()
    return _CLIP_MODEL


class ClipEmbeddingStore:
    """Local vector store for CLIP photo embeddings backed by numpy arrays."""

    def __init__(self, dir_path: str | Path):
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.ids_file = self.dir_path / "clip_ids.npy"
        self.vectors_file = self.dir_path / "clip_vectors.npy"
        self._data: dict[int, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if self.ids_file.exists() and self.vectors_file.exists():
            try:
                ids = np.load(str(self.ids_file))
                vectors = np.load(str(self.vectors_file))
                for emb_id, vec in zip(ids, vectors):
                    self._data[int(emb_id)] = vec.astype(np.float32)
            except Exception:
                self._data = {}

    def _save_to_disk(self) -> None:
        if not self._data:
            return
        ids = np.array(sorted(self._data.keys()), dtype=np.int64)
        vectors = np.stack([self._data[i] for i in ids]).astype(np.float32)

        tmp_ids = self.dir_path / "clip_ids.tmp.npy"
        tmp_vecs = self.dir_path / "clip_vectors.tmp.npy"

        np.save(str(tmp_ids), ids)
        np.save(str(tmp_vecs), vectors)

        if tmp_ids.exists():
            os.replace(tmp_ids, self.ids_file)
        if tmp_vecs.exists():
            os.replace(tmp_vecs, self.vectors_file)

    def save(self, embedding_id: int, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._data[int(embedding_id)] = vec
        self._save_to_disk()

    def get(self, embedding_id: int) -> np.ndarray | None:
        return self._data.get(int(embedding_id))

    def next_id(self) -> int:
        if not self._data:
            return 1
        return max(self._data.keys()) + 1

    def __len__(self) -> int:
        return len(self._data)


class ClipIndex:
    """Wrapper around FAISS IndexIDMap2(IndexFlatIP(512)) for CLIP image embeddings."""

    def __init__(self, index_path: str | Path, dimension: int = 512):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._ids: set[int] = set()

        if self.index_path.exists() and self.index_path.stat().st_size > 0:
            self._load()
        else:
            self._init_empty()

    def _init_empty(self) -> None:
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap2(base_index)
        self._ids = set()
        self._save()

    def _load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        self._ids = set()
        if hasattr(self.index, "id_map"):
            id_array = faiss.vector_to_array(self.index.id_map)
            self._ids = {int(x) for x in id_array}

    def _save(self) -> None:
        tmp_path = self.index_path.with_name(self.index_path.name + ".tmp")
        faiss.write_index(self.index, str(tmp_path))
        if tmp_path.exists():
            os.replace(tmp_path, self.index_path)

    def add(
        self,
        ids: Sequence[int] | np.ndarray,
        vectors: np.ndarray,
    ) -> int:
        raw_ids = [int(i) for i in ids]
        raw_vecs = np.asarray(vectors, dtype=np.float32)

        if raw_vecs.ndim == 1:
            raw_vecs = np.expand_dims(raw_vecs, axis=0)

        new_ids: list[int] = []
        new_vec_list: list[np.ndarray] = []
        for emb_id, vec in zip(raw_ids, raw_vecs):
            if emb_id not in self._ids:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                new_ids.append(emb_id)
                new_vec_list.append(vec)

        if not new_ids:
            return 0

        vec_matrix = np.stack(new_vec_list).astype(np.float32)
        id_arr = np.array(new_ids, dtype=np.int64)

        self.index.add_with_ids(vec_matrix, id_arr)
        self._ids.update(new_ids)
        self._save()

        return len(new_ids)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[tuple[int, float]]:
        if len(self) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        q = np.expand_dims(q, axis=0)

        k = min(top_k, len(self))
        distances, indices = self.index.search(q, k)

        results: list[tuple[int, float]] = []
        for score, emb_id in zip(distances[0], indices[0]):
            if emb_id != -1 and (min_score is None or score >= min_score):
                results.append((int(emb_id), float(score)))

        return results

    def __contains__(self, item: int) -> bool:
        return int(item) in self._ids

    def __len__(self) -> int:
        return self.index.ntotal


def process_photo_clip(
    photo_id: int,
    conn: sqlite3.Connection,
    store: ClipEmbeddingStore,
    index: ClipIndex,
) -> tuple[int, np.ndarray]:
    """Generate OpenCLIP image embedding, persist in store and index, and record in DB."""
    row = conn.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row:
        raise ValueError(f"Photo with id={photo_id} not found.")

    filepath = row[0]
    clip = get_clip_model()
    vec = clip.encode_image(filepath)

    emb_id = store.next_id()
    store.save(emb_id, vec)
    index.add([emb_id], np.expand_dims(vec, 0))

    conn.execute(
        "INSERT INTO clip_embeddings (photo_id, embedding_id) VALUES (?, ?)",
        (photo_id, emb_id),
    )
    conn.commit()

    return emb_id, vec


def batch_generate_clip_embeddings(
    conn: sqlite3.Connection,
    store: ClipEmbeddingStore,
    index: ClipIndex,
) -> dict[str, int]:
    """Generate CLIP embeddings for all photos without one in ``clip_embeddings``."""
    rows = conn.execute(
        """SELECT id FROM photos
           WHERE id NOT IN (SELECT photo_id FROM clip_embeddings)
             AND processing_state != 'duplicate'
           ORDER BY id ASC"""
    ).fetchall()

    summary = {
        "processed": 0,
        "errors": 0,
    }

    for (photo_id,) in rows:
        try:
            process_photo_clip(photo_id, conn, store, index)
            summary["processed"] += 1
        except Exception:
            summary["errors"] += 1

    return summary


def search_photos_semantic(
    query_text: str,
    conn: sqlite3.Connection,
    index: ClipIndex,
    top_k: int = 10,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Search photos matching natural language text using CLIP semantic similarity."""
    clip = get_clip_model()
    text_vec = clip.encode_text(query_text)

    matches = index.search(text_vec, top_k=top_k, min_score=min_score)
    if not matches:
        return []

    results: list[dict[str, Any]] = []
    for emb_id, score in matches:
        row = conn.execute(
            """SELECT p.id, p.filepath, p.date_taken
               FROM clip_embeddings c
               JOIN photos p ON p.id = c.photo_id
               WHERE c.embedding_id = ?""",
            (emb_id,),
        ).fetchone()

        if row:
            results.append({
                "photo_id": row[0],
                "filepath": row[1],
                "date_taken": row[2],
                "similarity": round(float(score), 4),
            })

    return results
