# PyInstaller hook for ModernGL
# Ensures all backend DLLs, data files, and submodules are collected.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('moderngl')
