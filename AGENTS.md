# AGENTS.md — PhotoVault (Local Face-Aware Photo Manager)

## Project Context (do not re-derive, treat as fixed)

You are building **PhotoVault**, a fully offline, disk-only desktop photo
manager with automatic face clustering and semantic search. No cloud
services, no external servers, no network calls except local model
downloads at setup time.

**Approved architecture — do not deviate without explicit user approval:**

| Layer | Technology |
|---|---|
| Core engine | Python 3.11 (InsightFace/ArcFace, OpenCLIP, HDBSCAN, ONNX Runtime) |
| Metadata DB | SQLite (single file, `vault.db`) |
| Vector search | FAISS — two local index files: `faces.index`, `clip.index` |
| Job queue | SQLite-backed table, in-process workers (no Redis/Celery) |
| Desktop shell | Tauri (Rust) |
| UI | React + TypeScript, virtualized photo grid |
| Packaging | PyInstaller/Nuitka → Python engine shipped as a Tauri sidecar |

**Data model (do not change without approval):**
```
photos(id, filepath, date_taken, phash, processing_state, created_at)
persons(id, display_name, created_at)
faces(id, photo_id, person_id, bbox, face_embedding_id)
clip_embeddings(id, photo_id, embedding_id)
```

**Non-negotiable constraints:**
- No photo files are ever copied or moved — only indexed in place.
- No data leaves the local machine.
- `processing_state` drives all reprocessing — never trigger a full
  library rescan for an incremental change.

---

## The Loop Protocol

Work in fixed iterations. Each iteration follows this exact cycle —
do not skip a phase, do not merge phases:

1. **Explore** — Read `BACKLOG.md`. Identify the single highest-priority
   unchecked task. Read only the files relevant to that task (use
   `@workspace_scope` to stay inside the module you're touching).
2. **Plan** — Write a short implementation plan (5–10 lines) to
   `PROGRESS.md` before touching code: what will change, which files,
   what test proves it works.
3. **Implement** — Make the change. Stay inside the file/module
   boundary named in the plan. Do not refactor unrelated code.
4. **Verify** — Run the local test command for that module
   (`pytest` for the Python engine, `npm test` for the UI). If no
   test exists for this task yet, write one first, then implement
   against it.
5. **Checkpoint** — If and only if tests pass: mark the task checked
   off in `BACKLOG.md`. Do NOT run `git commit`, `git push`, or any
   git command — this project is not using git for now. Local file
   changes are the only record of progress. If tests fail after 3
   attempts, STOP — do not keep retrying. Write the failure to
   `GUARDRAILS.md` and move to the next task instead.
6. **Log** — Append one line to `PROGRESS.md`: task id, status
   (done/blocked), and any deviation from plan.
7. **Loop** — Check the iteration budget below. If under budget and
   `BACKLOG.md` has unchecked tasks, return to step 1.

**Iteration budget:** stop after 20 iterations or when `BACKLOG.md`
is fully checked, whichever comes first. Do not exceed this without
explicit user confirmation — report status and wait.

---

## Guardrails (read before every iteration)

- Do NOT use git in any form — no `git init`, `git commit`, `git push`,
  or GitHub API calls. Progress is tracked only via `BACKLOG.md` and
  `PROGRESS.md`. If the agent proposes initializing a git repo, refuse
  and remind it of this rule.
- If you find yourself repeating the same fix for the same test
  failure more than 3 times, STOP. Log it in `GUARDRAILS.md` under
  a `## Known stuck points` heading and skip to the next backlog item.
- Never modify `vault.db` schema, the FAISS index format, or the
  approved architecture table above without pausing and asking the
  user directly.
- Never add a new external dependency (especially anything requiring
  network access, a server process, or a cloud SDK) without asking first.
- Never touch files outside the module named in the current task's plan.
- If a task's acceptance criteria in `BACKLOG.md` is ambiguous, pause
  and ask — do not guess and proceed.

---

## Definition of Done (per task)

A backlog task is only checked off when:
- A test exists and passes for it.
- It was verified against real local files (not just mocked), where
  the task involves file I/O, face detection, or embedding generation.
- `PROGRESS.md` has a corresponding entry.
- No approved-architecture constraint above was violated.

---

## BACKLOG.md (seed — expand as needed, keep task IDs stable)

- [ ] T1 — SQLite schema + migration script for the 4 tables above
- [ ] T2 — Filesystem watcher: detect new/removed files in the granted folder
- [ ] T3 — pHash dedup check on ingestion
- [ ] T4 — EXIF extraction (date_taken, GPS if present) into `photos`
- [ ] T5 — Face detection (RetinaFace/SCRFD, ONNX) on unprocessed photos
- [ ] T6 — ArcFace embedding generation for each detected face
- [ ] T7 — FAISS `faces.index` build + incremental add
- [ ] T8 — Clustering: cosine-threshold match to existing person, else new person_id
- [ ] T9 — HDBSCAN periodic re-cluster + merge job
- [ ] T10 — CLIP embedding generation + `clip.index` for semantic search
- [ ] T11 — Job queue table + worker loop (async, non-blocking)
- [ ] T12 — REST/IPC API: list persons, rename person, merge persons
- [ ] T13 — REST/IPC API: query photos by date range / person_id / text query
- [ ] T14 — Tauri shell scaffold + sidecar process wiring to Python engine
- [ ] T15 — React UI: People grid (cover photo per person, tap-to-rename)
- [ ] T16 — React UI: Photo grid with date/person/text filter bar
- [ ] T17 — Duplicate review queue UI (pHash exact + CLIP near-duplicate)
- [ ] T18 — Packaging: PyInstaller/Nuitka build of the Python engine
- [ ] T19 — End-to-end smoke test: fresh folder → scan → named people → filtered search

---

## How to run this loop in Antigravity

In the Antigravity Agent Manager, point a new agent session at this
repo with this file as its root instruction set, then run:

```
Follow AGENTS.md exactly. Start at the top of BACKLOG.md.
Do not exceed the iteration budget without checking in.
```

For CLI-driven looping instead of the Manager UI:
```bash
limit=20
for ((i=1; i<=limit; i++)); do
  agy -p - < AGENTS.md || break
  ./verify.sh && continue || break
done
```
