# PyInstaller hook for glcontext
# Ensures the OpenGL context backend (WGL on Windows, EGL on Linux) is collected.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('glcontext')
