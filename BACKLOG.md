# BACKLOG.md — PhotoVault

Task IDs are stable — do not renumber. Check a box only when the
Definition of Done in AGENTS.md is fully met for that task.

- [x] T1 — SQLite schema + migration script for the 4 tables (photos, persons, faces, clip_embeddings)
- [x] T2 — Filesystem watcher: detect new/removed files in the granted folder
- [x] T3 — pHash dedup check on ingestion
- [x] T4 — EXIF extraction (date_taken, GPS if present) into `photos`
- [x] T5 — Face detection (RetinaFace/SCRFD, ONNX) on unprocessed photos
- [x] T6 — ArcFace embedding generation for each detected face
- [x] T7 — FAISS `faces.index` build + incremental add
- [x] T8 — Clustering: cosine-threshold match to existing person, else new person_id
- [x] T9 — HDBSCAN periodic re-cluster + merge job
- [x] T10 — CLIP embedding generation + `clip.index` for semantic search
- [x] T11 — Job queue table + worker loop (async, non-blocking)
- [x] T12 — REST/IPC API: list persons, rename person, merge persons
- [x] T13 — REST/IPC API: query photos by date range / person_id / text query
- [x] T14 — Tauri shell scaffold + sidecar process wiring to Python engine
- [x] T15 — React UI: People grid (cover photo per person, tap-to-rename)
- [x] T16 — React UI: Photo grid with date/person/text filter bar
- [x] T17 — Duplicate review queue UI (pHash exact + CLIP near-duplicate)
- [x] T18 — Packaging: PyInstaller/Nuitka build of the Python engine
- [x] T19 — End-to-end smoke test: fresh folder → scan → named people → filtered search

## Adding new tasks

Append new tasks below this line with the next free ID (T20, T21, ...).
Never reuse or renumber an existing ID, even if it's later removed —
PROGRESS.md may still reference it.

---
