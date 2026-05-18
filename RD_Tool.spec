# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

project_root = Path(SPECPATH)
hiddenimports = ['cv2', 'modules.files_renamer', 'modules.bili_videos', 'modules.gen_subtitles', 'modules.mac_poop_scooper', 'modules.merge_colors', 'modules.split_colors', 'modules.twist_shape', 'modules.ECG_handler', 'modules.dicom_to_imgs']
hiddenimports += collect_submodules('pydicom')


a = Analysis(
    [str(project_root / 'main.py')],
    pathex=[],
    binaries=[],
    datas=[
        (str(project_root / 'ui' / 'qss'), 'ui/qss'),
        (str(project_root / 'configs'), 'configs'),
        (str(project_root / 'i18n'), 'i18n'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['faster_whisper', 'torch', 'tensorboard', 'ctranslate2', 'onnxruntime', 'av', 'tokenizers', 'transformers'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RD_Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RD_Tool',
)
app = BUNDLE(
    coll,
    name='RD_Tool.app',
    icon=None,
    bundle_identifier=None,
)
