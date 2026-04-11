"""
PyInstaller runtime hook — OpenGL environment setup.

Executed before the main application code, inside the frozen bundle.
Sets up:
  - sys.path so the bundle root is on the path
  - LIBGL fallback for systems without a dedicated GPU (Linux/Mesa)
  - PYOPENGL_PLATFORM for headless / software rendering scenarios
  - Bundle root resolution via sys._MEIPASS
"""

import os
import sys

# -------------------------------------------------------------------------
# Bundle root: sys._MEIPASS points to the temp extraction dir (onefile) or
# the dist/VideoSynth directory (onedir).  Make it importable.
# -------------------------------------------------------------------------
if hasattr(sys, '_MEIPASS'):
    bundle_root = sys._MEIPASS
else:
    bundle_root = os.path.dirname(os.path.abspath(__file__))

if bundle_root not in sys.path:
    sys.path.insert(0, bundle_root)

# -------------------------------------------------------------------------
# Linux/macOS: fall back to Mesa software renderer if no GPU is detected.
# This keeps the app functional on CI machines and headless servers.
# Remove or guard with a real GPU detection if software rendering is too slow.
# -------------------------------------------------------------------------
if sys.platform.startswith('linux'):
    # Only force software render if the env var is not already set by the user
    if 'LIBGL_ALWAYS_SOFTWARE' not in os.environ:
        # Try to detect a real GPU by checking for DRI devices
        has_gpu = any(
            f.startswith('card') or f.startswith('renderD')
            for f in os.listdir('/dev/dri') if os.path.exists('/dev/dri')
        )
        if not has_gpu:
            os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

# -------------------------------------------------------------------------
# Windows: ensure the Qt6 plugin directory is on PATH so platform DLLs load.
# PyInstaller copies them to PyQt6/Qt6/plugins/platforms/ inside the bundle.
# -------------------------------------------------------------------------
if sys.platform == 'win32':
    qt_plugins = os.path.join(bundle_root, 'PyQt6', 'Qt6', 'plugins')
    if os.path.isdir(qt_plugins):
        os.environ['QT_PLUGIN_PATH'] = qt_plugins

    # Ensure the bundle root itself is on PATH so OpenGL / VC runtime DLLs resolve
    os.environ['PATH'] = bundle_root + os.pathsep + os.environ.get('PATH', '')
