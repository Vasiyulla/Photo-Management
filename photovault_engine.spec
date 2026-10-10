# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None

datas = [
    ('models', 'models'),
]
hiddenimports = [
    'engine',
    'engine.api',
    'engine.clip',
    'engine.cluster',
    'engine.embeddings',
    'engine.exif',
    'engine.faces',
    'engine.hasher',
    'engine.index',
    'engine.queue',
    'engine.recluster',
    'engine.scanner',
    'engine.db.schema',
    'open_clip',
    'insightface',
    'onnxruntime',
    'sklearn',
    'faiss',
    'PIL',
    'imagehash',
]

a = Analysis(
    ['engine/main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='photovault-engine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='photovault-engine',
)
