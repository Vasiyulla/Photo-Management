"""Tests for T11: SQLite-backed Job Queue and in-process non-blocking worker.

Verifies jobs table schema, enqueuing, atomic job claiming,
handler execution, error logging, and non-blocking background loop.
"""

import sqlite3
import time
from pathlib import Path
import pytest

from engine.db.schema import init_db
from engine.queue import (
    enqueue_job,
    get_job,
    JobWorker,
)


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def db_path(tmp_path):
    p = str(tmp_path / "vault.db")
    conn = init_db(p)
    conn.close()
    return p


@pytest.fixture()
def db(db_path):
    conn = sqlite3.connect(db_path)
    yield conn
    conn.close()


# ── test cases ───────────────────────────────────────────────────────

class TestJobsTableAndEnqueue:
    """Tests for jobs table creation and enqueuing."""

    def test_jobs_table_exists(self, db):
        rows = db.execute("PRAGMA table_info(jobs)").fetchall()
        cols = {r[1] for r in rows}
        assert {"id", "type", "payload", "status", "error", "created_at", "updated_at"} <= cols

    def test_enqueue_and_get_job(self, db):
        payload = {"folder": "C:/Photos", "count": 10}
        job_id = enqueue_job(db, "scan_folder", payload)
        assert isinstance(job_id, int)
        assert job_id > 0

        job = get_job(db, job_id)
        assert job is not None
        assert job["id"] == job_id
        assert job["type"] == "scan_folder"
        assert job["payload"] == payload
        assert job["status"] == "pending"
        assert job["error"] is None


class TestJobWorkerExecution:
    """Tests for job processing, handler execution, and error handling."""

    def test_process_one_job_success(self, db_path, db):
        worker = JobWorker(db_path)
        executed_payloads = []

        def handle_test(payload, conn):
            executed_payloads.append(payload)

        worker.register_handler("test_job", handle_test)

        job_id = enqueue_job(db, "test_job", {"value": 42})
        processed = worker.process_one_job()
        assert processed is True
        assert executed_payloads == [{"value": 42}]

        job = get_job(db, job_id)
        assert job["status"] == "completed"
        assert job["error"] is None

    def test_process_one_job_failure_records_error(self, db_path, db):
        worker = JobWorker(db_path)

        def handle_fail(payload, conn):
            raise RuntimeError("Something went wrong in processing!")

        worker.register_handler("fail_job", handle_fail)

        job_id = enqueue_job(db, "fail_job", {"file": "bad.jpg"})
        processed = worker.process_one_job()
        assert processed is True

        job = get_job(db, job_id)
        assert job["status"] == "failed"
        assert "Something went wrong" in job["error"]

    def test_process_one_job_when_empty_returns_false(self, db_path):
        worker = JobWorker(db_path)
        assert worker.process_one_job() is False

    def test_background_worker_loop_non_blocking(self, db_path, db):
        worker = JobWorker(db_path, poll_interval=0.05)
        results = []

        def handle_async(payload, conn):
            results.append(payload["item"])

        worker.register_handler("async_task", handle_async)

        worker.start()
        assert worker.is_running is True

        try:
            # Enqueue jobs while worker is running in background
            job1 = enqueue_job(db, "async_task", {"item": "A"})
            job2 = enqueue_job(db, "async_task", {"item": "B"})

            # Wait briefly for background loop to pick up and finish
            deadline = time.time() + 3.0
            while len(results) < 2 and time.time() < deadline:
                time.sleep(0.05)

            assert results == ["A", "B"]

            j1 = get_job(db, job1)
            j2 = get_job(db, job2)
            assert j1["status"] == "completed"
            assert j2["status"] == "completed"

        finally:
            worker.stop()
            assert worker.is_running is False
