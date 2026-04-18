# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for VideoSynth.

Build with:
    pyinstaller video_synth.spec --clean --noconfirm

Output: dist/VideoSynth/VideoSynth.exe
"""

import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# ---------------------------------------------------------------------------
# Helper: resolve a path relative to the spec file's directory
# ---------------------------------------------------------------------------
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))   # .../build/
ROOT_DIR = os.path.dirname(SPEC_DIR)                # .../video_synth/ (project root)


def src(rel):
    """Return an absolute path relative to the project root."""
    return os.path.join(ROOT_DIR, rel)


# ---------------------------------------------------------------------------
# Data files bundled into the package
# ---------------------------------------------------------------------------
datas = []

# Web UI built by React/Vite (run 'cd web && npm run build' first)
web_dist = src('web/dist')
if os.path.isdir(web_dist):
    datas.append((web_dist, 'web/dist'))
else:
    print(
        "WARNING: web/dist not found. The web UI will not be available in the bundle.\n"
        "         Run 'cd web && npm install && npm run build' before packaging."
    )

# GLSL shader files (may be empty if shaders are embedded in Python source)
shaders_dir = src('shaders')
if os.path.isdir(shaders_dir):
    datas.append((shaders_dir, 'shaders'))

# Default patches and MIDI mappings
save_dir = src('save')
if os.path.isdir(save_dir):
    datas.append((save_dir, 'save'))

# ---------------------------------------------------------------------------
# Collect binaries / data / hiddenimports from packages that PyInstaller
# often misses because they use lazy loading or C extensions.
# ---------------------------------------------------------------------------
binaries = []
hiddenimports = []

for pkg in ('moderngl', 'glcontext', 'PyQt6'):
    _d, _b, _h = collect_all(pkg)
    datas    += _d
    binaries += _b
    hiddenimports += _h

# ---------------------------------------------------------------------------
# Explicit hidden imports
# ---------------------------------------------------------------------------
hiddenimports += [
    # OpenGL context backends
    'moderngl',
    'moderngl.mgl',
    'glcontext',

    # Qt
    'PyQt6.QtWidgets',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtMultimedia',
    'PyQt6.QtMultimediaWidgets',
    'PyQt6.QtOpenGL',
    'PyQt6.QtOpenGLWidgets',
    'PyQt6.sip',

    # Computer vision / numerics
    'cv2',
    'numpy',
    'numpy.core._multiarray_umath',
    'numpy.core._multiarray_extras',

    # FastAPI / ASGI stack
    'fastapi',
    'fastapi.staticfiles',
    'fastapi.middleware.cors',
    'starlette',
    'starlette.middleware',
    'starlette.middleware.cors',
    'starlette.staticfiles',
    'starlette.responses',
    'starlette.routing',
    'uvicorn',
    'uvicorn.config',
    'uvicorn.main',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.loops.asyncio',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.protocols.websockets.websockets_impl',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'anyio',
    'anyio._backends._asyncio',
    'aiofiles',
    'websockets',
    'h11',

    # Async / HTTP
    'httpx',
    'httpcore',

    # Pydantic (FastAPI models)
    'pydantic',
    'pydantic.v1',

    # MIDI
    'mido',
    'mido.backends',
    'mido.backends.rtmidi',
    'rtmidi',

    # Audio
    'sounddevice',

    # OSC
    'pythonosc',
    'pythonosc.dispatcher',
    'pythonosc.osc_server',

    # Noise / generative
    'noise',

    # Virtual camera
    'pyvirtualcam',

    # OBS WebSocket
    'obsws_python',
    'obsws_python.requests',
    'obsws_python.events',

    # Image processing
    'PIL',
    'PIL.Image',
    'PIL.ImageFilter',

    # Windows-specific helpers
    'win32api',
    'win32con',
    'win32com',
    'win32com.client',
    'pywintypes',
    'wmi',

    # USB monitoring
    'usb_monitor',

    # Keyboard
    'keyboard',

    # YAML
    'yaml',

    # Ollama (optional AI integration)
    'ollama',
]

# ---------------------------------------------------------------------------
# Animation submodules (all discovered files in video_synth/animations/)
# ---------------------------------------------------------------------------
_animation_modules = [
    'video_synth.animations.base',
    'video_synth.animations.chladni',
    'video_synth.animations.dla',
    'video_synth.animations.drift_field',
    'video_synth.animations.enums',
    'video_synth.animations.fractal_zoom',
    'video_synth.animations.harmonic_interference',
    'video_synth.animations.lenia',
    'video_synth.animations.metaballs',
    'video_synth.animations.moire',
    'video_synth.animations.oscillator_grid',
    'video_synth.animations.perlin_noise',
    'video_synth.animations.physarum',
    'video_synth.animations.plasma',
    'video_synth.animations.reaction_diffusion',
    'video_synth.animations.shaders',
    'video_synth.animations.shaders2',
    'video_synth.animations.shaders3',
    'video_synth.animations.strange_attractor',
    'video_synth.animations.voronoi',
]
hiddenimports += _animation_modules

# Also collect as plain module names (video_synth package uses relative imports)
_animation_plain = [m.split('.')[-1] for m in _animation_modules if m != 'video_synth.animations.enums']
hiddenimports += [f'animations.{m}' for m in _animation_plain]

# ---------------------------------------------------------------------------
# Effect submodules
# ---------------------------------------------------------------------------
_effect_modules = [
    'video_synth.effects.base',
    'video_synth.effects.color',
    'video_synth.effects.enums',
    'video_synth.effects.erosion',
    'video_synth.effects.feedback',
    'video_synth.effects.glitch',
    'video_synth.effects.image_noiser',
    'video_synth.effects.lissajous',
    'video_synth.effects.pixels',
    'video_synth.effects.ptz',
    'video_synth.effects.reflector',
    'video_synth.effects.shapes',
    'video_synth.effects.sync',
    'video_synth.effects.warp',
]
hiddenimports += _effect_modules

_effect_plain = [m.split('.')[-1] for m in _effect_modules if m != 'video_synth.effects.enums']
hiddenimports += [f'effects.{m}' for m in _effect_plain]

# Also collect all submodules dynamically for robustness
hiddenimports += collect_submodules('video_synth')

# Remove duplicates
hiddenimports = sorted(set(hiddenimports))

# ---------------------------------------------------------------------------
# Runtime hooks directory
# ---------------------------------------------------------------------------
runtime_hooks = [os.path.join(SPEC_DIR, 'hooks/runtime_opengl.py')]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [src('src/video_synth/__main__.py')],
    pathex=[
        src('src/video_synth'),   # bare imports like 'from api import ...' resolve
        src('src'),               # package-level imports like 'import video_synth'
        src('.'),
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(SPEC_DIR, 'hooks')],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[
        # Exclude large packages not needed at runtime
        'tkinter',
        'matplotlib',
        'scipy',
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'pytest',
        'black',
        'mypy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ---------------------------------------------------------------------------
# EXE
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir mode: binaries live in COLLECT below
    name='VideoSynth',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # windowed app — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon omitted: no .ico found in project root; add 'images/icon.ico' if created
    # icon='images/icon.ico',
)

# ---------------------------------------------------------------------------
# COLLECT — one-directory bundle (faster startup, easier debugging than onefile)
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VideoSynth',
)
