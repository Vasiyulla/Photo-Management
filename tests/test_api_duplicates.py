"""Tests for T17: Duplicate detection and resolution API.

Verifies:
- Detection of exact pHash duplicates.
- Resolving duplicates (marking duplicate or keeping).
- HTTP REST endpoints for duplicates.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.request
import pytest

from engine.db.schema import init_db
from engine.api import (
    create_api_server,
    list_duplicates,
    resolve_duplicate,
    handle_ipc_request,
)


@pytest.fixture()
def db_path(tmp_path):
    p = str(tmp_path / "vault.db")
    conn = init_db(p)
    conn.close()
    return p


@pytest.fixture()
def db_with_duplicates(db_path):
    conn = sqlite3.connect(db_path)
    # 2 photos with identical pHash
    conn.execute(
        "INSERT INTO photos (id, filepath, date_taken, phash, processing_state) VALUES (1, '/photos/orig.jpg', '2023-01-01', 'hash123', 'completed')"
    )
    conn.execute(
        "INSERT INTO photos (id, filepath, date_taken, phash, processing_state) VALUES (2, '/photos/copy.jpg', '2023-01-02', 'hash123', 'pending')"
    )
    # Distinct photo
    conn.execute(
        "INSERT INTO photos (id, filepath, date_taken, phash, processing_state) VALUES (3, '/photos/other.jpg', '2023-01-03', 'diff999', 'completed')"
    )
    conn.commit()
    yield conn
    conn.close()


class TestDuplicatesApi:
    """Test duplicate detection and resolution functions."""

    def test_list_duplicates_finds_exact_phash_pair(self, db_with_duplicates):
        dups = list_duplicates(db_with_duplicates)
        assert len(dups) == 1
        pair = dups[0]
        assert pair["match_type"] == "phash_exact"
        assert pair["photo_a"]["id"] == 1
        assert pair["photo_b"]["id"] == 2

    def test_resolve_duplicate_marks_duplicate_state(self, db_with_duplicates):
        res = resolve_duplicate(db_with_duplicates, duplicate_photo_id=2, action="mark_duplicate")
        assert res["status"] == "ok"
        state = db_with_duplicates.execute("SELECT processing_state FROM photos WHERE id = 2").fetchone()[0]
        assert state == "duplicate"

    def test_ipc_duplicate_workflow(self, db_with_duplicates):
        res_list = handle_ipc_request({"action": "list_duplicates"}, db_with_duplicates)
        assert res_list["status"] == "ok"
        assert len(res_list["data"]) == 1

        res_resolve = handle_ipc_request(
            {"action": "resolve_duplicate", "duplicate_photo_id": 2, "resolution": "keep"},
            db_with_duplicates,
        )
        assert res_resolve["status"] == "ok"


class TestDuplicatesHttp:
    """Test HTTP REST endpoints for duplicates."""

    @pytest.fixture()
    def http_server(self, db_path, db_with_duplicates):
        server = create_api_server(db_path, host="127.0.0.1", port=0)
        server.start()
        yield server
        server.stop()

    def test_http_get_and_resolve_duplicates(self, http_server, db_with_duplicates):
        # GET /api/duplicates
        url = f"{http_server.base_url}/api/duplicates"
        with urllib.request.urlopen(url) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert len(data) == 1

        # POST /api/duplicates/resolve
        resolve_url = f"{http_server.base_url}/api/duplicates/resolve"
        payload = json.dumps({"duplicate_photo_id": 2, "action": "mark_duplicate"}).encode("utf-8")
        req = urllib.request.Request(
            resolve_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["status"] == "ok"
