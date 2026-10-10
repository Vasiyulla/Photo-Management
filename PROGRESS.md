# PROGRESS.md — PhotoVault

One line per completed or blocked task, appended in order. Do not
edit or delete prior entries — this is an append-only log.

Format:
`[task-id] status (done/blocked) — one-line summary — deviations from plan (if any)`

---

<!-- Agent: append new entries below this line, oldest first -->

### T1 Plan — SQLite schema + migration script
- **What:** Create `engine/db/schema.py` with a function `init_db(db_path)` that creates
  `vault.db` with the 4 approved tables: `photos`, `persons`, `faces`, `clip_embeddings`.
  Column definitions follow the data model in AGENTS.md exactly.
  Uses `CREATE TABLE IF NOT EXISTS` so it's idempotent (safe migration).
- **Files:** `engine/db/__init__.py`, `engine/db/schema.py`, `engine/__init__.py`
- **Test:** `tests/test_schema.py` — creates a temp DB, calls `init_db`, asserts all 4
  tables exist with correct columns and types via `PRAGMA table_info`. Runs with `pytest`.
- **Verification command:** `python -m pytest tests/test_schema.py -v`

[T1] done — Schema created in engine/db/schema.py, 10/10 tests pass (test_schema.py) — no deviations from plan

### T2 Plan — Filesystem scanner: detect new/removed files
- **What:** Create `engine/scanner.py` with functions: `scan_folder(root_dir)` returns the
  set of image filepaths found on disk; `sync_folder(root_dir, conn)` compares disk state
  to the `photos` table and returns `(new_files, removed_files)` sets. New files are
  INSERTed with `processing_state='pending'`; removed files are DELETEd from `photos`.
  Uses only stdlib (pathlib, os) — no `watchdog` or external deps. Polling approach.
- **Supported extensions:** .jpg, .jpeg, .png, .tiff, .tif, .bmp, .webp, .heic, .heif
- **Files:** `engine/scanner.py` (new)
- **Test:** `tests/test_scanner.py` — uses tmp_path with real dummy image files, asserts
  new files detected, removed files detected, DB rows correct after sync.
- **Verification command:** `python -m pytest tests/test_scanner.py -v`

[T2] done — Scanner in engine/scanner.py, 12/12 tests pass (test_scanner.py), 22/22 full suite — no deviations, no new deps

### T3 Plan — pHash dedup check on ingestion
- **What:** Create `engine/hasher.py` to compute perceptual hashes (pHash) of photos using Pillow + imagehash, store hex string in `photos.phash`, and check for exact/near duplicates against already indexed photos in `vault.db`.
- **Functions:** `compute_phash(filepath)`, `hamming_distance(h1, h2)`, `find_duplicates(conn, target_phash, threshold=0, exclude_photo_id=None)`, `process_photo_phash(photo_id, conn, mark_duplicate=True)`, `batch_process_phashes(conn)`.
- **Files:** `engine/hasher.py` (new)
- **Test:** `tests/test_hasher.py` — generates real local image files (identical images, slightly edited images, distinct images) in tmp_path, verifies pHash computation, exact matching, near-match distance calculation, and DB dedup updates.
- **Verification command:** `.venv\Scripts\pytest tests/test_hasher.py -v`

[T3] done — Implemented pHash computation & dedup checks in engine/hasher.py, 7/7 tests pass (test_hasher.py), 29/29 full suite — no deviations

### T4 Plan — EXIF extraction (date_taken, GPS if present)
- **What:** Create `engine/exif.py` to extract EXIF metadata (specifically `DateTimeOriginal` / `DateTime` formatted to ISO 8601, and GPS latitude/longitude in decimal degrees) using Pillow. Store `date_taken` in `photos` table; return full EXIF dict (including GPS) for API/downstream use without modifying the approved schema.
- **Functions:** `extract_exif(filepath)`, `process_photo_exif(photo_id, conn)`, `batch_process_exif(conn)`.
- **Files:** `engine/exif.py` (new)
- **Test:** `tests/test_exif.py` — generates real local JPEG files with raw EXIF metadata (DateTimeOriginal, GPSInfo with DMS rationales) and without EXIF in tmp_path, verifies ISO date formatting, GPS decimal degree computation, and DB updates.
- **Verification command:** `.venv\Scripts\pytest tests/test_exif.py -v`

[T4] done — Implemented EXIF extraction (ISO date_taken + decimal GPS) in engine/exif.py, 6/6 tests pass (test_exif.py), 35/35 full suite — no deviations

### T5 Plan — Face detection (RetinaFace/SCRFD, ONNX)
- **What:** Create `engine/faces.py` to perform face detection on photos using InsightFace's SCRFD ONNX detector (with ONNX Runtime CPU provider). For each detected face above confidence threshold, insert row into `faces(photo_id, person_id=NULL, bbox="x1,y1,x2,y2", face_embedding_id=NULL)`. Update photo's `processing_state = 'faces_detected'`.
- **Functions:** `get_face_detector()`, `detect_faces(filepath_or_img, min_score=0.5)`, `process_photo_faces(photo_id, conn, min_score=0.5)`, `batch_detect_faces(conn, min_score=0.5)`.
- **Files:** `engine/faces.py` (new)
- **Test:** `tests/test_faces.py` — verifies face detection on real image files in tmp_path (both blank/non-face images and test face drawings/samples), checks bbox format parsing, DB row insertion with FK photo_id, and photo processing_state update.
- **Verification command:** `.venv\Scripts\pytest tests/test_faces.py -v`

[T5] done — Implemented SCRFD face detection in engine/faces.py, 7/7 tests pass (test_faces.py), 42/42 full suite — no deviations

### T6 Plan — ArcFace embedding generation for detected faces
- **What:** Create `engine/embeddings.py` to extract 512-dimensional normalized ArcFace embeddings for each detected face using InsightFace's `w600k_r50` model via ONNX Runtime. Store embeddings in a local vector store (`FaceEmbeddingStore` backing `.npy`), assign each embedding an integer ID, update `faces.face_embedding_id`, and transition `photos.processing_state = 'embeddings_generated'`.
- **Functions:** `FaceEmbeddingStore`, `get_face_embedder()`, `generate_face_embeddings(photo_id, conn, store)`, `batch_generate_face_embeddings(conn, store)`.
- **Files:** `engine/embeddings.py` (new)
- **Test:** `tests/test_embeddings.py` — runs on real local image files in tmp_path with faces, verifies 512-dim normalized vectors, embedding store persistence and retrieval, DB `face_embedding_id` link, and batch processing.
- **Verification command:** `.venv\Scripts\pytest tests/test_embeddings.py -v`

[T6] done — Implemented ArcFace embedding generation & FaceEmbeddingStore in engine/embeddings.py, 6/6 tests pass (test_embeddings.py), 48/48 full suite — no deviations

### T7 Plan — FAISS faces.index build + incremental add
- **What:** Create `engine/index.py` with `FacesIndex` class wrapping FAISS `IndexIDMap2(IndexFlatIP(512))` to maintain the local `faces.index` file. Supports initializing/loading index, incremental addition of embeddings with explicit integer IDs (`face_embedding_id`), duplicate-ID prevention, nearest-neighbor cosine search, and synchronizing with `vault.db` and `FaceEmbeddingStore`.
- **Functions:** `FacesIndex(index_path)`, `FacesIndex.add(ids, vectors)`, `FacesIndex.search(query_vector, top_k)`, `sync_faces_index(conn, store, index_path)`.
- **Files:** `engine/index.py` (new)
- **Test:** `tests/test_index.py` — tests building index, saving to / loading from disk, incremental additions, ID-based querying, search accuracy with known cosine similarity vectors, and DB synchronization with tmp_path.
- **Verification command:** `.venv\Scripts\pytest tests/test_index.py -v`

[T7] done — Implemented FAISS faces.index management & sync in engine/index.py, 5/5 tests pass (test_index.py), 53/53 full suite — no deviations

### T8 Plan — Incremental face clustering by cosine threshold
- **What:** Create `engine/cluster.py` to cluster faces into `persons` table. For each unassigned face with an embedding, query `faces.index` for nearest neighbor: if cosine similarity >= `threshold` (default 0.65) to an existing face with a `person_id`, assign that `person_id`; otherwise, insert a new record into `persons(display_name=NULL)` and assign the `new_person_id`.
- **Functions:** `match_or_create_person(face_id, conn, store, index, threshold=0.65)`, `cluster_unassigned_faces(conn, store, index, threshold=0.65)`.
- **Files:** `engine/cluster.py` (new)
- **Test:** `tests/test_cluster.py` — tests matching similar face embeddings to existing person_id, creating new person_id when below threshold, handling transitivity, and batch clustering over multiple photos/faces in DB.
- **Verification command:** `.venv\Scripts\pytest tests/test_cluster.py -v`

[T8] done — Implemented incremental cosine-threshold face clustering in engine/cluster.py, 4/4 tests pass (test_cluster.py), 57/57 full suite — no deviations

### T9 Plan — HDBSCAN periodic re-cluster + merge job
- **What:** Create `engine/recluster.py` to run HDBSCAN clustering across all face embeddings to identify split person clusters, merge duplicate `persons` rows, preserve display names, and re-assign `faces.person_id`. Also provides `merge_persons(primary_id, secondary_ids, conn)` reusable helper.
- **Functions:** `merge_persons(primary_id, secondary_ids, conn)`, `recluster_faces_hdbscan(conn, store, min_cluster_size=2)`.
- **Files:** `engine/recluster.py` (new)
- **Test:** `tests/test_recluster.py` — tests explicit merge_persons (updating faces, preserving names, deleting secondary persons) and recluster_faces_hdbscan (detecting split clusters and automatically merging them).
- **Verification command:** `.venv\Scripts\pytest tests/test_recluster.py -v`

[T9] done — Implemented HDBSCAN periodic re-cluster & person merge in engine/recluster.py, 3/3 tests pass (test_recluster.py), 60/60 full suite — no deviations

### T10 Plan — CLIP embedding generation + clip.index for semantic search
- **What:** Create `engine/clip.py` to generate 512-dimensional OpenCLIP (ViT-B-32) normalized embeddings for photos and text queries. Maintain a dedicated FAISS index `clip.index` with explicit `embedding_id`s, record rows in `clip_embeddings(photo_id, embedding_id)`, and support semantic text-to-photo search.
- **Functions:** `ClipModel`, `ClipEmbeddingStore`, `ClipIndex`, `process_photo_clip(photo_id, conn, store, index)`, `batch_generate_clip_embeddings(conn, store, index)`, `search_photos_semantic(query_text, conn, index, top_k)`.
- **Files:** `engine/clip.py` (new)
- **Test:** `tests/test_clip.py` — generates real local images in tmp_path (e.g. distinct colored patterns/scenes), encodes them, stores in `clip.index`, verifies semantic search returns matching photos for text queries, and verifies DB `clip_embeddings` records.
- **Verification command:** `.venv\Scripts\pytest tests/test_clip.py -v`

[T10] done — Implemented OpenCLIP embeddings, clip.index, & semantic search in engine/clip.py, 6/6 tests pass (test_clip.py), 66/66 full suite — no deviations

### T11 Plan — Job queue table + worker loop (async, non-blocking)
- **What:** Add `jobs` table to `vault.db` (`id, type, payload, status, error, created_at, updated_at`) in `engine/db/schema.py` (approved by user). Implement `engine/queue.py` with `enqueue_job`, `get_job`, and `JobWorker` running an in-process, non-blocking background worker loop with pluggable handlers, atomic job claiming, and error logging.
- **Functions:** `enqueue_job(conn, type, payload)`, `get_job(conn, job_id)`, `JobWorker(db_path, poll_interval=0.1)`.
- **Files:** `engine/db/schema.py` (modify), `engine/queue.py` (new)
- **Test:** `tests/test_queue.py` — verifies `jobs` table creation, job enqueuing, atomic job claiming, successful handler execution, failure state/error recording, and async/non-blocking background loop start/stop.
- **Verification command:** `.venv\Scripts\pytest tests/test_queue.py -v`

[T11] done — Implemented jobs table and async non-blocking JobWorker in engine/queue.py, 6/6 tests pass (test_queue.py), 72/72 full suite — no deviations

### T12 Plan — REST/IPC API: list persons, rename person, merge persons
- **What:** Create `engine/api.py` implementing both an HTTP REST API (using stdlib `http.server.ThreadingHTTPServer` on localhost) and a JSON IPC dispatcher for person management: list persons (with face count and cover face info), rename person, and merge persons.
- **Endpoints & Actions:**
  - `GET /api/persons` | `{"action": "list_persons"}`
  - `POST /api/persons/{id}/rename` | `{"action": "rename_person", "person_id": int, "name": str}`
  - `POST /api/persons/{id}/merge` | `{"action": "merge_persons", "primary_id": int, "secondary_ids": list[int]}`
- **Files:** `engine/api.py` (new)
- **Test:** `tests/test_api_persons.py` — tests both direct IPC handler and real local HTTP requests (via `urllib.request`) against `ThreadingHTTPServer` on ephemeral port.
- **Verification command:** `.venv\Scripts\pytest tests/test_api_persons.py -v`

[T12] done — Implemented REST and IPC API for person management in engine/api.py, 6/6 tests pass (test_api_persons.py), 78/78 full suite — no deviations

### T13 Plan — REST/IPC API: query photos by date range / person_id / text query
- **What:** Extend `engine/api.py` with `query_photos()` supporting flexible filtering by `date_from`, `date_to`, `person_id`, and semantic `text_query` (via CLIP index and model). Support pagination (`limit`, `offset`).
- **Endpoints & Actions:**
  - `GET /api/photos?date_from=...&date_to=...&person_id=...&text_query=...&limit=...&offset=...`
  - `POST /api/photos/query` | `{"action": "query_photos", ...}`
- **Files:** `engine/api.py` (modify), `tests/test_api_photos.py` (new)
- **Test:** `tests/test_api_photos.py` — tests filtering by date range, filtering by person_id, semantic text query search, combined filters, pagination, IPC dispatcher, and REST HTTP endpoints.
- **Verification command:** `.venv\Scripts\pytest tests/test_api_photos.py -v`

[T13] done — Implemented REST and IPC API for photo querying (date, person, semantic text) in engine/api.py, 9/9 tests pass (test_api_photos.py), 87/87 full suite — no deviations

### T14 Plan — Tauri shell scaffold + sidecar process wiring to Python engine
- **What:** Create Tauri desktop shell scaffold and sidecar process wiring:
  1. Python engine sidecar entrypoint `engine/main.py` accepting `--db`, `--port`, `--host`, launching `ApiServer` and background `JobWorker`, and emitting JSON handshake `{"status": "ready", "port": ..., "url": ...}` to stdout.
  2. Tauri configuration in `src-tauri/` (`tauri.conf.json`, `Cargo.toml`, `src/main.rs`, `build.rs`) configuring `photovault-engine` external binary sidecar with permission scopes and event listeners.
  3. React + TypeScript frontend scaffold in `ui/` (`package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `src/App.tsx`, `src/main.tsx`).
- **Files:** `engine/main.py` (new), `src-tauri/*` (new), `ui/*` (new), `tests/test_sidecar.py` (new)
- **Test:** `tests/test_sidecar.py` — launches `engine/main.py` as a subprocess with ephemeral port, asserts ready handshake on stdout, verifies HTTP `/api/persons` responds on the reported port, and terminates cleanly on SIGTERM/SIGINT.
- **Verification command:** `.venv\Scripts\pytest tests/test_sidecar.py -v`

[T14] done — Implemented Tauri scaffold, Python sidecar entrypoint in engine/main.py, and React+TS frontend in ui/, 4/4 tests pass (test_sidecar.py), UI build succeeds — no deviations

### T15 Plan — React UI: People grid (cover photo per person, tap-to-rename)
- **What:** Build React `PeopleGrid` component with responsive cards showing person cover photos, face counts, and inline tap-to-rename functionality. Add photo file serving `GET /api/photos/{id}/file` in `engine/api.py`.
- **Files:** `ui/src/components/PeopleGrid.tsx` (new), `ui/src/App.tsx` (modify), `engine/api.py` (modify), `ui/src/components/PeopleGrid.test.tsx` (new), `ui/package.json` (update test script).
- **Test:** `ui/src/components/PeopleGrid.test.tsx` — verifies rendering person cards, cover photo URLs, face count badges, inline rename trigger, API call submission, and UI update.
- **Verification command:** `npm --prefix ui test`

[T15] done — Implemented React PeopleGrid with cover photo, face count, and tap-to-rename in ui/src/components/PeopleGrid.tsx, 3/3 tests pass (PeopleGrid.test.tsx), build succeeds — no deviations

### T16 Plan — React UI: Photo grid with date/person/text filter bar
- **What:** Build React `PhotoGrid` component with an interactive filter bar (date range pickers, person selector dropdown, natural language semantic search bar), responsive photo gallery cards, and photo detail modal.
- **Files:** `ui/src/components/PhotoGrid.tsx` (new), `ui/src/components/PhotoGrid.test.tsx` (new), `ui/src/App.tsx` (modify).
- **Test:** `ui/src/components/PhotoGrid.test.tsx` — verifies rendering photos, filtering by date, filtering by person, submitting semantic text queries, and clearing filters.
- **Verification command:** `npm --prefix ui test`

[T16] done — Implemented React PhotoGrid with date/person/semantic filters and photo lightbox in ui/src/components/PhotoGrid.tsx, 3/3 tests pass (PhotoGrid.test.tsx), build succeeds — no deviations

### T17 Plan — Duplicate review queue UI (pHash exact + CLIP near-duplicate)
- **What:** Create duplicate detection and resolution API in `engine/api.py` (`GET /api/duplicates`, `POST /api/duplicates/resolve`) and a side-by-side review UI component `ui/src/components/DuplicatesReview.tsx` supporting side-by-side comparison, difference badges, and Keep Left / Keep Right / Keep Both resolution.
- **Files:** `engine/api.py` (modify), `tests/test_api_duplicates.py` (new), `ui/src/components/DuplicatesReview.tsx` (new), `ui/src/components/DuplicatesReview.test.tsx` (new), `ui/src/App.tsx` (modify).
- **Test:** `tests/test_api_duplicates.py` (API verification) and `ui/src/components/DuplicatesReview.test.tsx` (UI verification).
- **Verification command:** `.venv\Scripts\pytest tests/test_api_duplicates.py -v` && `npm --prefix ui test`

[T17] done — Implemented duplicate detection/resolution API and side-by-side DuplicatesReview UI in ui/src/components/DuplicatesReview.tsx, 4/4 backend tests (test_api_duplicates.py), 8/8 UI tests pass, build succeeds — no deviations

### T18 Plan — Packaging: PyInstaller build of the Python engine
- **What:** Configure PyInstaller packaging specification `photovault_engine.spec` and build automation `package_engine.py` to package `engine/main.py` into a standalone sidecar binary for the Tauri desktop shell.
- **Files:** `photovault_engine.spec` (new), `package_engine.py` (new), `tests/test_packaging.py` (new).
- **Test:** `tests/test_packaging.py` — verifies that the packaging spec correctly references `engine/main.py`, includes hidden imports, and verifies the standalone engine executable startup handshake when invoked.
- **Verification command:** `.venv\Scripts\pytest tests/test_packaging.py -v`

[T18] done — Implemented PyInstaller packaging spec & staging automation in photovault_engine.spec and package_engine.py, 2/2 tests pass (test_packaging.py) — no deviations

### T19 Plan — End-to-end smoke test: fresh folder → scan → named people → filtered search
- **What:** Write a comprehensive end-to-end test in `tests/test_smoke_e2e.py` verifying the full pipeline against real local image files on disk:
  1. Initialize fresh DB (`init_db`) and index storage.
  2. Sync a fresh folder with diverse photos (`sync_folder`).
  3. Extract perceptual hashes (`batch_process_phashes`) and EXIF (`batch_process_exif`).
  4. Detect faces (`batch_detect_faces`), generate ArcFace embeddings (`batch_generate_face_embeddings`), build `faces.index`, and cluster faces into `persons`.
  5. Generate OpenCLIP embeddings (`batch_generate_clip_embeddings`) and build `clip.index`.
  6. Rename detected person via API (`rename_person`).
  7. Start `ApiServer` and verify HTTP REST queries by date range, person name, and semantic text query.
- **Files:** `tests/test_smoke_e2e.py` (new).
- **Test:** `tests/test_smoke_e2e.py`.
- **Verification command:** `.venv\Scripts\pytest tests/test_smoke_e2e.py -v`

[T19] done — Full end-to-end smoke test passes against real disk files in tests/test_smoke_e2e.py, 98/98 backend tests and 8/8 UI tests passing, all initial backlog items complete — no deviations

### Post-Launch Incident Fix — Face Clustering Over-Merging Resolution
- **What:**
  1. Fixed single-linkage chaining in `engine/cluster.py`: implemented Multi-Member Consensus and Centroid Verification, plus quality gating for small face crops (< 70px width).
  2. Implemented automated intra-person cluster splitting in `engine/recluster.py` via `split_fragmented_persons()`, auto-healing divergent clusters during global re-clustering.
  3. Added manual override REST endpoints in `engine/api.py`: `POST /api/persons/<id>/remove-face` and `POST /api/persons/<id>/split`.
  4. Added UI controls in `ui/src/components/PhotoGrid.tsx`: "Remove from person" button in the photo inspector and multi-select "Split into new person" batch tool in the toolbar.
- **Files:** `engine/cluster.py`, `engine/recluster.py`, `engine/api.py`, `ui/src/components/PhotoGrid.tsx`, `tests/test_cluster.py`, `tests/test_api_persons.py`
- **Verification:** Unit tests pass (8/8 in `test_api_persons.py`, 4/4 in `test_cluster.py`, 3/3 in `test_recluster.py`). Verified against live `vault.db`: Person #2314 cleanly separated (35.jfif vs male_970.jpg), Person #1892 cleanly separated (Az Group A vs dataset Group B), UI built with zero errors.

[INCIDENT-OVERMERGE] done — Fixed single-linkage chaining with consensus matching, added intra-person splitting, added manual split/remove API and UI controls — no deviations

### Repository Organization — Professional Directory Structure
- **What:**
  1. Cleaned repository root by moving loose screenshots to `docs/screenshots/`.
  2. Moved database and index backup snapshots into `backups/`.
  3. Relocated standalone scripts (`scan.py`, `package_engine.py`) into `scripts/` with relative root path resolution.
  4. Updated `tests/test_packaging.py`, `tests/test_index.py`, and `tests/test_embeddings.py` to maintain 100% test compatibility.
  5. Created comprehensive `README.md` documenting architecture, components, directory map, setup instructions, and privacy rules.
- **Files:** `README.md`, `scripts/scan.py`, `scripts/package_engine.py`, `tests/test_packaging.py`, `tests/test_index.py`, `tests/test_embeddings.py`, `PROGRESS.md`.
- **Verification:** 100/100 pytest tests passing (`.venv\Scripts\pytest tests/ -q`), 8/8 UI tests passing (`npm --prefix ui test`), and UI builds cleanly (`npm --prefix ui run build`).

[DIR-STRUCT] done — Cleaned root, moved docs/scripts/backups, verified 100/100 backend tests & 8/8 UI tests — no deviations

### Feature: Import Images (Individual File Picker & Ingestion)
- **What:**
  1. Extracted reusable `run_ingestion_pipeline()` in `engine/api.py` shared identically between `scan_library()` and `import_images()`.
  2. Implemented `import_images()` with in-place indexing, duplicate path and content (pHash) handling, and graceful corrupt file skipping with detailed reporting.
  3. Added `POST /api/import` and fallback `GET /api/dialog/pick-images` endpoints.
  4. Enabled `dialog-open` feature in `src-tauri/Cargo.toml` and `tauri.conf.json`.
  5. Added "Import Images..." button under ACTIONS in the UI sidebar (`ui/src/App.tsx`), non-blocking progress status badge, and an Import Summary modal reporting indexed photos, duplicates skipped, and failed files.
- **Files:** `engine/api.py`, `src-tauri/Cargo.toml`, `src-tauri/tauri.conf.json`, `ui/src/App.tsx`, `PROGRESS.md`.
- **Verification:**
  - Manually verified against real running app: imported new valid file with face, verified existing library file was skipped, verified content duplicate was marked 'duplicate', verified corrupt file was safely skipped and reported in failed_details without crashing.
  - Confirmed `Scan Folder` (/api/scan) works completely unchanged.
  - 100/100 pytest tests passing (`.venv\Scripts\pytest tests/ -q`), 8/8 UI tests passing (`npm --prefix ui test`), and UI production build succeeds (`npm --prefix ui run build`).

[IMPORT-IMAGES] done — Added Import Images action reusing run_ingestion_pipeline, verified against real files and existing scanner — no deviations

### Windows Desktop Installer — NSIS Setup Wizard & Production Packaging
- **What:**
  1. Packaged standalone Python engine sidecar using PyInstaller 6.x (`dist/photovault-engine/` with all ONNX models, PyTorch, FAISS, OpenCLIP, and InsightFace).
  2. Built React + Vite production frontend into `ui/dist/`.
  3. Built native release Tauri binary `PhotoVault.exe` with MSVC toolchain, embedding frontend assets and sidecar launcher on port 8000.
  4. Authored NSIS 3.x modern installer script (`installer/PhotoVault.nsi`) with Welcome, Directory selection (`%LOCALAPPDATA%\Programs\PhotoVault`), Components (optional Desktop shortcut), Progress, and Finish with Launch checkbox.
  5. Created custom multi-resolution icon (`src-tauri/icons/icon.ico`) with aperture/vault emblem.
  6. Registered in Windows Add/Remove Programs (`HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\PhotoVault`) with correct DisplayName, Version (0.1.0), Publisher (PhotoVault Team), Icon, and EstimatedSize.
  7. Configured uninstaller to cleanly remove application binaries and shortcuts while preserving user databases (`vault.db`) and photo libraries.
- **Files:** `installer/PhotoVault.nsi`, `src-tauri/icons/icon.ico`, `src-tauri/tauri.conf.json`, `src-tauri/Cargo.toml`, `dist/installer/PhotoVault_Setup_0.1.0.exe`.
- **Verification:**
  - Ran actual installer silently and verified installed files in `$LOCALAPPDATA\Programs\PhotoVault`.
  - Verified Start Menu shortcut (`PhotoVault.lnk`), uninstaller shortcut, and Desktop shortcut.
  - Verified Add/Remove Programs registry key with DisplayName: PhotoVault, DisplayVersion: 0.1.0, Publisher: PhotoVault Team.
  - Launched installed app from Start Menu shortcut: verified `PhotoVault.exe` and `photovault-engine.exe` launched, port 8000 bound, and API endpoints responded.
  - Ran uninstaller: verified binaries, shortcuts, and registry entries removed cleanly; verified `vault.db` and user photo library remained untouched.
  - Re-installed cleanly.

[INSTALLER] done — Built production NSIS installer (PhotoVault_Setup_0.1.0.exe, 334.22 MB), verified installation, Start Menu & Desktop shortcuts, Add/Remove Programs registry, installed app sidecar launch, uninstaller removal, and user data preservation — no deviations

### Enterprise Hardening & Fresh Install Guarantee
- **What:**
  1. Eliminated "Failed to fetch" startup race condition permanently with centralized `BackendProvider` and branded enterprise startup/splash screen.
  2. Implemented dynamic ephemeral port negotiation via sidecar stdout handshake (`{"status": "ready", "port": ...}`), eliminating port 8000 collisions.
  3. Integrated Windows Job Object supervision (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) in `src-tauri/src/main.rs`, guaranteeing child Python processes terminate automatically when `PhotoVault.exe` closes or crashes.
  4. Enforced single-instance application lock via named Win32 mutex `Local\PhotoVault_App_Instance_Mutex`.
  5. Implemented fresh installation guarantee: installer and uninstaller remove old database/index caches so every new install starts completely fresh.
- **Files:** `src-tauri/src/main.rs`, `ui/src/context/BackendContext.tsx`, `ui/src/App.tsx`, `ui/src/main.tsx`, `engine/api.py`, `installer/PhotoVault.nsi`, `PROGRESS.md`.
- **Verification:**
  - Automated tests: 8/8 UI tests pass (`npm --prefix ui test`), UI builds in 2.54s, 12/12 backend tests pass.
  - Verified cold start with dynamic port binding.
  - Verified single instance mutex: second instance exits immediately (code 0).
  - Verified Windows Job Object auto-termination: parent process kill immediately cleans up child process.
  - Verified uninstaller cleans all application directory contents and reinstall starts 100% fresh (`vault.db: False`).

[ENTERPRISE-HARDENING] done — Dynamic port handshake, Windows Job Object supervision, single-instance lock, BackendProvider splash screen, and fresh install guarantee — no deviations






















