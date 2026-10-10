"""Tests for T13: REST/IPC API for photo querying (date range, person_id, text query).

Tests filtering, pagination, semantic CLIP integration, IPC dispatcher,
and HTTP REST requests.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path
import numpy as np
import pytest

from engine.clip import ClipIndex
from engine.db.schema import init_db
from engine.api import (
    create_api_server,
    handle_ipc_request,
    query_photos,
)


@pytest.fixture()
def db_path(tmp_path):
    p = str(tmp_path / "vault.db")
    conn = init_db(p)
    conn.close()
    return p


@pytest.fixture()
def seeded_db(db_path):
    conn = sqlite3.connect(db_path)
    # 4 photos with different dates and processing states
    conn.execute(
        "INSERT INTO photos (id, filepath, date_taken, phash, processing_state) VALUES (1, '/photos/cat.jpg', '2023-01-15T12:00:00', 'h1', 'completed')"
    )
    conn.execute(
        "INSERT INTO photos (id, filepath, date_taken, phash, processing_state) VALUES (2, '/photos/dog.jpg', '2023-06-20T14:30:00', 'h2', 'completed')"
    )
    conn.execute(
        "INSERT INTO photos (id, filepath, date_taken, phash, processing_state) VALUES (3, '/photos/sunset.jpg', '2024-02-10T18:00:00', 'h3', 'completed')"
    )
    conn.execute(
        "INSERT INTO photos (id, filepath, date_taken, phash, processing_state) VALUES (4, '/photos/beach.jpg', '2024-07-04T10:00:00', 'h4', 'completed')"
    )

    # 2 persons: 1 = Alice, 2 = Bob
    conn.execute("INSERT INTO persons (id, display_name) VALUES (1, 'Alice')")
    conn.execute("INSERT INTO persons (id, display_name) VALUES (2, 'Bob')")

    # Photo 1 has Alice
    conn.execute("INSERT INTO faces (id, photo_id, person_id, bbox) VALUES (101, 1, 1, '10,10,50,50')")
    # Photo 2 has Bob
    conn.execute("INSERT INTO faces (id, photo_id, person_id, bbox) VALUES (102, 2, 2, '20,20,60,60')")
    # Photo 4 has both Alice and Bob
    conn.execute("INSERT INTO faces (id, photo_id, person_id, bbox) VALUES (103, 4, 1, '15,15,55,55')")
    conn.execute("INSERT INTO faces (id, photo_id, person_id, bbox) VALUES (104, 4, 2, '70,70,90,90')")
    # Photo 3 has no faces

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture()
def clip_index_seeded(tmp_path, seeded_db):
    idx_path = str(tmp_path / "clip.index")
    index = ClipIndex(index_path=idx_path, dimension=512)

    # Add synthetic normalized vectors for photo 1 and photo 2
    rng = np.random.default_rng(42)
    vec1 = rng.standard_normal(512).astype(np.float32)
    vec1 /= np.linalg.norm(vec1)

    vec2 = rng.standard_normal(512).astype(np.float32)
    vec2 /= np.linalg.norm(vec2)

    index.add([1, 2], np.stack([vec1, vec2]))

    # Link in clip_embeddings table
    seeded_db.execute("INSERT INTO clip_embeddings (id, photo_id, embedding_id) VALUES (1, 1, 1)")
    seeded_db.execute("INSERT INTO clip_embeddings (id, photo_id, embedding_id) VALUES (2, 2, 2)")
    seeded_db.commit()

    return index, vec1


class TestPhotoQueryDirect:
    """Direct functional tests for query_photos()."""

    def test_query_all_photos(self, seeded_db):
        photos = query_photos(seeded_db)
        assert len(photos) == 4
        # Default order should be by date_taken DESC or id DESC
        ids = [p["id"] for p in photos]
        assert 4 in ids and 1 in ids

    def test_query_by_date_range(self, seeded_db):
        # 2023 only
        photos = query_photos(seeded_db, date_from="2023-01-01", date_to="2023-12-31")
        assert len(photos) == 2
        assert {p["id"] for p in photos} == {1, 2}

        # From 2024 onwards
        photos_2024 = query_photos(seeded_db, date_from="2024-01-01")
        assert len(photos_2024) == 2
        assert {p["id"] for p in photos_2024} == {3, 4}

    def test_query_by_person(self, seeded_db):
        # Alice (person_id 1) is in photo 1 and 4
        alice_photos = query_photos(seeded_db, person_id=1)
        assert len(alice_photos) == 2
        assert {p["id"] for p in alice_photos} == {1, 4}

        # Bob (person_id 2) is in photo 2 and 4
        bob_photos = query_photos(seeded_db, person_id=2)
        assert len(bob_photos) == 2
        assert {p["id"] for p in bob_photos} == {2, 4}

    def test_query_combined_filters(self, seeded_db):
        # Alice in 2024 (only photo 4)
        res = query_photos(seeded_db, person_id=1, date_from="2024-01-01")
        assert len(res) == 1
        assert res[0]["id"] == 4
        assert res[0]["filepath"] == "/photos/beach.jpg"

    def test_pagination(self, seeded_db):
        page1 = query_photos(seeded_db, limit=2, offset=0)
        assert len(page1) == 2
        page2 = query_photos(seeded_db, limit=2, offset=2)
        assert len(page2) == 2
        assert {p["id"] for p in page1}.isdisjoint({p["id"] for p in page2})

    def test_semantic_search_integration(self, seeded_db, clip_index_seeded):
        index, vec1 = clip_index_seeded
        # Using a mock query embedding equal to vec1 to test semantic ranking
        photos = query_photos(seeded_db, query_vector=vec1, clip_index=index)
        assert len(photos) >= 2
        assert photos[0]["id"] == 1  # exact match to vec1 ranks first
        assert "score" in photos[0]
        assert photos[0]["score"] > 0.99


class TestPhotoQueryIpc:
    """Test IPC JSON dispatcher for query_photos."""

    def test_ipc_query_photos(self, seeded_db):
        res = handle_ipc_request(
            {"action": "query_photos", "person_id": 1, "date_from": "2024-01-01"},
            seeded_db,
        )
        assert res["status"] == "ok"
        photos = res["data"]
        assert len(photos) == 1
        assert photos[0]["id"] == 4


class TestPhotoQueryHttp:
    """Test HTTP REST endpoints for photo queries."""

    @pytest.fixture()
    def http_server(self, db_path, seeded_db):
        server = create_api_server(db_path, host="127.0.0.1", port=0)
        server.start()
        yield server
        server.stop()

    def test_get_photos_filtered(self, http_server):
        url = f"{http_server.base_url}/api/photos?person_id=1&date_from=2024-01-01"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert len(data) == 1
            assert data[0]["id"] == 4

    def test_post_query_photos(self, http_server):
        url = f"{http_server.base_url}/api/photos/query"
        payload = json.dumps({"person_id": 2}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert len(data) == 2
            assert {p["id"] for p in data} == {2, 4}
