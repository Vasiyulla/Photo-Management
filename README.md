# PhotoVault 📸

**PhotoVault** is a fully offline, privacy-first desktop photo manager featuring automatic face detection, facial clustering, and semantic text-to-image search. All indexing and processing happen locally on-device without cloud services or external network calls.

---

## 🏗️ Architecture & Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **Core Engine** | Python 3.11+ / FastAPI | Face detection, embeddings, FAISS indexes, REST API |
| **Face Detection & Recognition** | OpenCV YuNet + SFace ONNX | Fast, lightweight, on-device face bounding boxes & 128D embeddings |
| **Semantic Search** | OpenCLIP (ViT-B-32) ONNX | 512D multi-modal text and image embeddings |
| **Vector Search** | FAISS | Instant similarity search for faces (`faces.index`) and CLIP (`clip.index`) |
| **Metadata & State** | SQLite (`vault.db`) | ACID-compliant relational storage for photos, people, faces, and job queue |
| **Desktop Shell** | Tauri (Rust) | Native, lightweight desktop window wrapping the frontend & sidecar |
| **Frontend UI** | React 18 + TypeScript + Vite | Virtualized photo grid, person management, duplicate review queue |

---

## 📂 Project Directory Structure

```text
Photo/
├── backups/                    # Timestamped backups of databases & vector indexes
│   ├── face_store_backup_*/    # Raw face embedding numpy archives
│   ├── faces_backup_*.index    # FAISS face index snapshots
│   └── vault_backup_*.db       # SQLite database snapshots
│
├── docs/                       # Project documentation & visual assets
│   └── screenshots/            # UI screenshots (people grid, photo grid, duplicate review)
│
├── engine/                     # Core Python Backend Engine
│   ├── api/                    # REST / IPC endpoints (photos, people, duplicates, system)
│   ├── db/                     # SQLite schema definitions, migrations & connection helpers
│   ├── clustering.py           # Single-pass threshold clustering & HDBSCAN merge logic
│   ├── config.py               # Centralized configuration & environment settings
│   ├── duplicates.py           # Exact pHash and near-duplicate vector similarity review
│   ├── embeddings.py           # SFace facial embedding generator & vector storage
│   ├── faces.py                # YuNet face detector & bounding-box extractor
│   ├── index.py                # FAISS index wrapper (IndexFlatIP with IDMap2)
│   ├── main.py                 # FastAPI application entrypoint & CLI runner
│   ├── queue.py                # SQLite-backed background job queue & worker loop
│   └── watcher.py              # Inotify/ReadDirectoryChangesW filesystem watcher
│
├── models/                     # Local ONNX Model Weights
│   ├── face_detection_yunet_*.onnx
│   ├── face_recognition_sface_*.onnx
│   └── open_clip_*.onnx
│
├── scripts/                    # Standalone CLI Utilities
│   ├── package_engine.py       # PyInstaller standalone sidecar packager
│   └── scan.py                 # CLI tool to scan & index local photo directories
│
├── src-tauri/                  # Desktop Shell (Tauri / Rust)
│   ├── src/                    # Rust backend commands & sidecar process supervisor
│   ├── binaries/               # Bundled Python sidecar binaries
│   ├── Cargo.toml              # Rust crate dependencies
│   └── tauri.conf.json         # Tauri application configuration
│
├── tests/                      # Pytest Test Suite
│   ├── fixtures/               # Sample images, corrupted files, test fixtures
│   ├── test_api.py             # REST API endpoint tests
│   ├── test_clustering.py      # Face clustering & re-clustering tests
│   ├── test_duplicates.py      # Duplicate detection verification
│   ├── test_embeddings.py      # SFace & ArcFace embedding tests
│   ├── test_faces.py           # Face detection pipeline tests
│   ├── test_index.py           # FAISS index persistence & search tests
│   ├── test_queue.py           # Background job queue processing tests
│   └── test_watcher.py         # Directory watcher tests
│
├── ui/                         # Frontend Application (React + TypeScript)
│   ├── src/
│   │   ├── components/         # PhotoGrid, PeopleGrid, DuplicateReview, FilterBar
│   │   ├── services/           # Backend API clients & Tauri invoke bridges
│   │   ├── App.tsx             # Root application view
│   │   └── index.css           # Modern theme & utility styling
│   ├── package.json            # Node.js dependencies
│   └── vite.config.ts          # Vite build configuration
│
├── clip_store/                 # Local numpy store for CLIP embeddings
├── face_store/                 # Local numpy store for Face embeddings
├── clip.index                  # FAISS vector index for text-to-photo semantic search
├── faces.index                 # FAISS vector index for face recognition
├── vault.db                    # Primary SQLite metadata database
├── photovault_engine.spec      # PyInstaller spec for sidecar distribution
├── AGENTS.md                   # Agent protocol, architecture contracts & rules
├── BACKLOG.md                  # Task roadmap & status
├── GUARDRAILS.md               # Safety guidelines & known edge cases
└── PROGRESS.md                 # Iteration audit log & changelog
```

---

## 🚀 Getting Started

### 1. Backend Engine Setup
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run backend server directly
python -m engine.main --db vault.db --port 8000
```

### 2. Frontend Development Server
```bash
cd ui
npm install
npm run dev
```

### 3. Run Scanning Script
To index and process photos from any folder:
```bash
python scripts/scan.py "C:\path\to\your\photos"
```

### 4. Running the Full Test Suite
```bash
.venv\Scripts\pytest tests/ -v
```

---

## 🔒 Privacy & Safety Principles
- **Zero Cloud Leakage**: No telemetry, no external API calls, no third-party cloud processing.
- **In-Place Indexing**: Source photo files are never moved, renamed, or modified.
- **Incremental State Machine**: All processing is driven by SQLite `processing_state` tags (`pending` → `faces_detected` → `embeddings_generated` → `indexed`).
