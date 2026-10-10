"""PhotoVault — Packaging automation script (T18).

Builds the standalone Python engine executable via PyInstaller and
stages it for Tauri sidecar integration under ``src-tauri/binaries/``.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def get_tauri_target_triple() -> str:
    """Return the Rust target triple for the current system architecture."""
    machine = platform.machine().lower()
    system = platform.system().lower()

    if system == "windows":
        if machine in ("amd64", "x86_64"):
            return "x86_64-pc-windows-msvc"
        elif machine in ("arm64", "aarch64"):
            return "aarch64-pc-windows-msvc"
    elif system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "aarch64-apple-darwin"
        return "x86_64-apple-darwin"
    elif system == "linux":
        return "x86_64-unknown-linux-gnu"

    return f"{machine}-{system}"


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_sidecar() -> Path:
    """Run PyInstaller build and copy executable to src-tauri/binaries/."""
    spec_path = PROJECT_ROOT / "photovault_engine.spec"
    if not spec_path.exists():
        raise FileNotFoundError(f"photovault_engine.spec not found at {spec_path}")

    print("[package_engine] Invoking PyInstaller build...")
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        str(spec_path),
    ]

    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))

    target_triple = get_tauri_target_triple()
    binaries_dir = PROJECT_ROOT / "src-tauri" / "binaries"
    binaries_dir.mkdir(parents=True, exist_ok=True)

    dist_exe = PROJECT_ROOT / "dist" / "photovault-engine" / "photovault-engine.exe"
    if not dist_exe.exists():
        dist_exe = PROJECT_ROOT / "dist" / "photovault-engine" / "photovault-engine"

    dest_name = f"photovault-engine-{target_triple}.exe" if sys.platform == "win32" else f"photovault-engine-{target_triple}"
    dest_path = binaries_dir / dest_name

    if dist_exe.exists():
        shutil.copy2(dist_exe, dest_path)
        print(f"[package_engine] Staged sidecar to: {dest_path}")
    else:
        print(f"[package_engine] Notice: {dist_exe} not found directly; check dist/ directory.")

    return dest_path


if __name__ == "__main__":
    build_sidecar()
