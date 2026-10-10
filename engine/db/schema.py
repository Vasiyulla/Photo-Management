"""
PhotoVault — SQLite schema initialisation (T1).

Provides :func:`init_db` which creates (or migrates) the four approved
tables in ``vault.db``.  Uses ``CREATE TABLE IF NOT EXISTS`` so the
function is safe to call on every startup (idempotent migration).

Approved data model (from AGENTS.md — do NOT change without user approval):

    photos(id, filepath, date_taken, phash, processing_state, created_at)
    persons(id, display_name, created_at)
    faces(id, photo_id, person_id, bbox, face_embedding_id)
    clip_embeddings(id, photo_id, embedding_id)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# ── SQL DDL ──────────────────────────────────────────────────────────

_SCHEMA_SQL = """\
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS photos (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath         TEXT    NOT NULL UNIQUE,
    date_taken       TEXT,
    phash            TEXT,
    processing_state TEXT    NOT NULL DEFAULT 'pending',
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS persons (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS faces (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id          INTEGER NOT NULL REFERENCES photos(id),
    person_id         INTEGER REFERENCES persons(id),
    bbox              TEXT,
    face_embedding_id INTEGER,
    FOREIGN KEY (photo_id)  REFERENCES photos(id),
    FOREIGN KEY (person_id) REFERENCES persons(id)
);

CREATE TABLE IF NOT EXISTS clip_embeddings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id     INTEGER NOT NULL REFERENCES photos(id),
    embedding_id INTEGER,
    FOREIGN KEY (photo_id) REFERENCES photos(id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    type       TEXT    NOT NULL,
    payload    TEXT,
    status     TEXT    NOT NULL DEFAULT 'pending',
    error      TEXT,
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


# ── public API ───────────────────────────────────────────────────────

def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Create / migrate the vault database at *db_path*.

    Returns the open :class:`sqlite3.Connection` for further use by
    the caller.  The connection has ``foreign_keys`` enabled and uses
    WAL journalling for safe concurrent reads.
    """
    db_path = str(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA_SQL)
    return conn
