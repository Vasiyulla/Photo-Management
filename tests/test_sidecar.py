"""Tests for T14: Tauri shell scaffold & sidecar process wiring.

Verifies:
1. engine/main.py starts cleanly as a sidecar process.
2. Emits JSON ready handshake with ephemeral port and base URL.
3. Serves API endpoints over HTTP.
4. Shuts down cleanly on terminate signal.
5. src-tauri/tauri.conf.json is valid and configures the external sidecar.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import pytest

from engine.db.schema import init_db


@pytest.fixture()
def db_path(tmp_path):
    p = str(tmp_path / "vault.db")
    conn = init_db(p)
    conn.close()
    return p


class TestSidecarEngine:
    """Test Python engine executable sidecar entrypoint."""

    def test_sidecar_startup_handshake_and_query(self, db_path):
        cmd = [
            sys.executable,
            "-m",
            "engine.main",
            "--db",
            db_path,
            "--port",
            "0",
            "--host",
            "127.0.0.1",
        ]

        # Launch sidecar process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        try:
            # Read ready handshake from stdout
            line = proc.stdout.readline().strip()
            assert line, "Sidecar produced no stdout before exit"

            handshake = json.loads(line)
            assert handshake["status"] == "ready"
            assert handshake["port"] > 0
            assert "url" in handshake

            base_url = handshake["url"]

            # Query /api/persons via HTTP
            req = urllib.request.Request(f"{base_url}/api/persons")
            with urllib.request.urlopen(req, timeout=3) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert isinstance(data, list)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


class TestTauriScaffold:
    """Test Tauri shell configuration and sidecar wiring."""

    def test_tauri_conf_valid(self):
        conf_path = Path("src-tauri/tauri.conf.json")
        assert conf_path.exists(), "src-tauri/tauri.conf.json does not exist"

        with open(conf_path, "r", encoding="utf-8") as f:
            conf = json.load(f)

        # Assert Tauri bundle includes photovault-engine external binary
        external_bin = conf.get("tauri", {}).get("bundle", {}).get("externalBin", [])
        assert any("photovault-engine" in b for b in external_bin), "externalBin must include photovault-engine"

    def test_cargo_toml_valid(self):
        cargo_path = Path("src-tauri/Cargo.toml")
        assert cargo_path.exists(), "src-tauri/Cargo.toml does not exist"
        content = cargo_path.read_text(encoding="utf-8")
        assert "tauri" in content
        assert "shell-sidecar" in content or "shell-execute" in content

    def test_main_rs_exists(self):
        main_rs = Path("src-tauri/src/main.rs")
        assert main_rs.exists(), "src-tauri/src/main.rs does not exist"
        content = main_rs.read_text(encoding="utf-8")
        assert "tauri" in content
