"""Tests for asynchronous non-blocking scan and import endpoints."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import urllib.request
from pathlib import Path
from PIL import Image

from engine.api import create_api_server
from engine.db.schema import init_db


def test_async_scan_and_status(tmp_path: Path):
    db_path = str(tmp_path / "vault.db")
    init_db(db_path)

    # Create dummy image folder
    img_dir = tmp_path / "photos"
    img_dir.mkdir()
    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 40, 100, 150))
        img.save(img_dir / f"test_{i}.jpg")

    server = create_api_server(db_path=db_path, host="127.0.0.1", port=0)
    server.start()

    try:
        base_url = server.base_url

        # 1. Test GET /api/scan/status initially idle
        req = urllib.request.Request(f"{base_url}/api/scan/status")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            task = json.loads(resp.read().decode("utf-8"))
            assert "status" in task

        # 2. Test POST /api/scan responds quickly (< 1s) and starts task
        scan_payload = json.dumps({"folder_path": str(img_dir)}).encode("utf-8")
        scan_req = urllib.request.Request(
            f"{base_url}/api/scan",
            data=scan_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        with urllib.request.urlopen(scan_req) as resp:
            elapsed = time.time() - t0
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert elapsed < 1.0, f"Scan took {elapsed}s, expected < 1.0s"
            assert data["new_photos"] == 3
            assert data["status"] in ("started", "completed")

        # 3. Test GET /api/scan/status while/after scanning
        req = urllib.request.Request(f"{base_url}/api/scan/status")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            task = json.loads(resp.read().decode("utf-8"))
            assert task["status"] in ("running", "completed")

    finally:
        server.stop()
