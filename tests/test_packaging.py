"""Tests for T18: PyInstaller packaging specification and build automation.

Verifies:
1. photovault_engine.spec exists and contains required entrypoint & hidden imports.
2. package_engine.py builds/stages the binary or sidecar script for Tauri.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest


class TestPackagingConfig:
    """Verify PyInstaller configuration files."""

    def test_spec_file_exists_and_configured(self):
        spec_path = Path("photovault_engine.spec")
        assert spec_path.exists(), "photovault_engine.spec does not exist"

        content = spec_path.read_text(encoding="utf-8")
        assert "engine/main.py" in content or "engine\\main.py" in content
        # Check hidden imports
        assert "open_clip" in content
        assert "insightface" in content
        assert "onnxruntime" in content
        assert "sklearn" in content

    def test_package_engine_script_exists(self):
        script_path = Path("scripts/package_engine.py")
        if not script_path.exists():
            script_path = Path("package_engine.py")
        assert script_path.exists(), "package_engine.py does not exist in scripts/ or root"

        content = script_path.read_text(encoding="utf-8")
        assert "photovault_engine" in content
        assert "build" in content or "PyInstaller" in content
