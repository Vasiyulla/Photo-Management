"""
PhotoVault — In-process SQLite-backed Job Queue & Worker Loop (T11).

Provides an asynchronous, non-blocking job queue backed by the ``jobs``
table in ``vault.db``. Workers run in-process with pluggable handlers,
atomic job claiming, and automatic error tracking. No Redis or Celery.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable


def enqueue_job(
    conn: sqlite3.Connection,
    job_type: str,
    payload: Any = None,
) -> int:
    """Enqueue a new job with 'pending' status into the jobs table.

    Returns the new job ID.
    """
    if payload is not None and not isinstance(payload, str):
        payload_str = json.dumps(payload)
    else:
        payload_str = payload

    cur = conn.execute(
        "INSERT INTO jobs (type, payload, status) VALUES (?, ?, 'pending')",
        (job_type, payload_str),
    )
    conn.commit()
    return cur.lastrowid


def get_job(conn: sqlite3.Connection, job_id: int) -> dict[str, Any] | None:
    """Retrieve details of a job by ID."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, type, payload, status, error, created_at, updated_at FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()

    if not row:
        return None

    raw_payload = row["payload"]
    parsed_payload = raw_payload
    if raw_payload:
        try:
            parsed_payload = json.loads(raw_payload)
        except Exception:
            pass

    return {
        "id": row["id"],
        "type": row["type"],
        "payload": parsed_payload,
        "status": row["status"],
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


class JobWorker:
    """In-process background worker executing queued jobs from vault.db."""

    def __init__(self, db_path: str | Path, poll_interval: float = 0.1):
        self.db_path = str(db_path)
        self.poll_interval = poll_interval
        self._handlers: dict[str, Callable[[Any, sqlite3.Connection], Any]] = {}
        self._running = False
        self._thread: threading.Thread | None = None

    def register_handler(
        self,
        job_type: str,
        handler: Callable[[Any, sqlite3.Connection], Any],
    ) -> None:
        """Register a handler function for a specific job_type."""
        self._handlers[job_type] = handler

    def claim_next_job(self, conn: sqlite3.Connection) -> dict[str, Any] | None:
        """Atomically claim the next pending job and mark it as running."""
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id, type, payload FROM jobs WHERE status = 'pending' ORDER BY id ASC LIMIT 1"
            ).fetchone()

            if not row:
                conn.commit()
                return None

            job_id, jtype, raw_payload = row
            conn.execute(
                "UPDATE jobs SET status = 'running', updated_at = datetime('now') WHERE id = ?",
                (job_id,),
            )
            conn.commit()

            payload = raw_payload
            if raw_payload:
                try:
                    payload = json.loads(raw_payload)
                except Exception:
                    pass

            return {
                "id": job_id,
                "type": jtype,
                "payload": payload,
            }
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return None

    def process_one_job(self) -> bool:
        """Attempt to claim and execute one job.

        Returns True if a job was processed, False if queue was empty.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            job = self.claim_next_job(conn)
            if not job:
                return False

            job_id = job["id"]
            job_type = job["type"]
            payload = job["payload"]

            handler = self._handlers.get(job_type)
            if not handler:
                conn.execute(
                    "UPDATE jobs SET status = 'failed', error = ?, updated_at = datetime('now') WHERE id = ?",
                    (f"No handler registered for job type '{job_type}'", job_id),
                )
                conn.commit()
                return True

            try:
                handler(payload, conn)
                conn.execute(
                    "UPDATE jobs SET status = 'completed', updated_at = datetime('now') WHERE id = ?",
                    (job_id,),
                )
                conn.commit()
            except Exception as e:
                conn.execute(
                    "UPDATE jobs SET status = 'failed', error = ?, updated_at = datetime('now') WHERE id = ?",
                    (str(e), job_id),
                )
                conn.commit()

            return True
        finally:
            conn.close()

    def _loop(self) -> None:
        """Background thread worker loop."""
        while self._running:
            processed = self.process_one_job()
            if not processed:
                time.sleep(self.poll_interval)

    def start(self) -> None:
        """Start the background worker thread non-blockingly."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker to stop and wait for completion."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()
