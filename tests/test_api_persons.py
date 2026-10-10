"""Tests for T12: REST/IPC API for Person management (list, rename, merge).

Verifies both direct JSON IPC dispatching and actual HTTP REST requests
using standard library urllib against ThreadingHTTPServer.
"""

import json
import sqlite3
import urllib.request
import urllib.error
from pathlib import Path
import pytest

from engine.db.schema import init_db
from engine.api import (
    handle_ipc_request,
    create_api_server,
    list_persons,
    rename_person,
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
    # Seed initial photos, persons, and faces
    conn.execute("INSERT INTO photos (id, filepath) VALUES (1, '/photos/img1.jpg')")
    conn.execute("INSERT INTO photos (id, filepath) VALUES (2, '/photos/img2.jpg')")

    conn.execute("INSERT INTO persons (id, display_name) VALUES (1, 'Alice')")
    conn.execute("INSERT INTO persons (id, display_name) VALUES (2, NULL)")

    conn.execute("INSERT INTO faces (id, photo_id, person_id, bbox) VALUES (10, 1, 1, '10,10,50,50')")
    conn.execute("INSERT INTO faces (id, photo_id, person_id, bbox) VALUES (11, 2, 1, '15,15,55,55')")
    conn.execute("INSERT INTO faces (id, photo_id, person_id, bbox) VALUES (20, 2, 2, '30,30,80,80')")
    conn.commit()

    yield conn
    conn.close()


@pytest.fixture()
def http_server(db_path, db):
    server = create_api_server(db_path, host="127.0.0.1", port=0)
    server.start()
    yield server
    server.stop()


# ── test cases ───────────────────────────────────────────────────────

class TestPersonsIpc:
    """Tests for direct IPC request handler."""

    def test_list_persons(self, db):
        res = handle_ipc_request({"action": "list_persons"}, db)
        assert res["status"] == "ok"
        persons = res["data"]
        assert len(persons) == 2

        # Alice has 2 faces, cover face from photo 1
        alice = next(p for p in persons if p["id"] == 1)
        assert alice["display_name"] == "Alice"
        assert alice["face_count"] == 2
        assert alice["cover_photo_id"] == 1
        assert alice["cover_bbox"] == "10,10,50,50"

        # Unnamed person 2 has 1 face
        p2 = next(p for p in persons if p["id"] == 2)
        assert p2["display_name"] is None
        assert p2["face_count"] == 1

    def test_rename_person(self, db):
        res = handle_ipc_request({"action": "rename_person", "person_id": 2, "name": "Bob"}, db)
        assert res["status"] == "ok"
        assert res["data"]["id"] == 2
        assert res["data"]["display_name"] == "Bob"

        # Verify DB updated
        name = db.execute("SELECT display_name FROM persons WHERE id = 2").fetchone()[0]
        assert name == "Bob"

    def test_merge_persons(self, db):
        res = handle_ipc_request({"action": "merge_persons", "primary_id": 1, "secondary_ids": [2]}, db)
        assert res["status"] == "ok"
        assert res["data"]["primary_id"] == 1
        assert res["data"]["faces_merged"] == 1

        # Person 2 deleted, Alice now has 3 faces
        p_rows = db.execute("SELECT id FROM persons").fetchall()
        assert p_rows == [(1,)]
        face_count = db.execute("SELECT COUNT(*) FROM faces WHERE person_id = 1").fetchone()[0]
        assert face_count == 3


class TestPersonsRestHttp:
    """Tests for HTTP REST API."""

    def test_get_persons(self, http_server):
        url = f"{http_server.base_url}/api/persons"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert len(data) == 2
            assert data[0]["id"] == 1
            assert data[0]["display_name"] == "Alice"

    def test_post_rename_person(self, http_server):
        url = f"{http_server.base_url}/api/persons/2/rename"
        payload = json.dumps({"name": "Charlie"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["id"] == 2
            assert data["display_name"] == "Charlie"

    def test_post_merge_persons(self, http_server):
        url = f"{http_server.base_url}/api/persons/1/merge"
        payload = json.dumps({"secondary_ids": [2]}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["primary_id"] == 1
            assert data["faces_merged"] == 1

    def test_post_remove_face(self, http_server):
        url = f"{http_server.base_url}/api/persons/1/remove-face"
        payload = json.dumps({"face_id": 10}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert data["old_person_id"] == 1
            assert data["face_id"] == 10
            assert data["new_person_id"] != 1

    def test_post_split_person(self, http_server):
        url = f"{http_server.base_url}/api/persons/1/split"
        payload = json.dumps({"face_ids": [11], "new_display_name": "Split Person"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert data["old_person_id"] == 1
            assert data["faces_moved"] == 1
            assert data["new_display_name"] == "Split Person"

