# Building VideoSynth as a Standalone Executable

This document explains how to package VideoSynth into a self-contained executable
using [PyInstaller](https://pyinstaller.org/).  The result is a directory
(`dist/VideoSynth/`) that can be zipped and distributed — no Python installation
required on the target machine.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 | Runtime + packaging |
| Node.js | 18+ | Build the React web UI |
| npm | 9+ | Bundled with Node.js |
| PyInstaller | 6.x | Creates the executable |
| Git | any | Source checkout |

> **Windows only (for now):** The CI workflow and `build.bat` target Windows.
> Linux/macOS builds work via `build.sh` but are not CI-tested yet.

### Optional runtime dependencies (not bundled)

- **OBS Virtual Camera driver** — required for the `--virtualcam` output.
  Install OBS Studio (https://obsproject.com/) which ships the virtual camera driver.
  The driver must be installed separately; it cannot be bundled in the executable.

- **FFmpeg** — required for `--ffmpeg` recording/streaming output.
  Download a static build from https://ffmpeg.org/download.html and place `ffmpeg.exe`
  on your PATH or in the same directory as `VideoSynth.exe`.

---

## Step-by-step: Windows

### 1. Clone the repository and create a virtual environment

```cmd
git clone https://github.com/your-org/video_synth.git
cd video_synth
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Python dependencies

```cmd
pip install -r requirements.txt
```

### 3. Build the React web UI

The web UI is served by the FastAPI server at `/ui`.  If you skip this step the
API server will still work but the browser-based control panel will be unavailable.

```cmd
cd web
npm install
npm run build
cd ..
```

This produces `web/dist/` which PyInstaller bundles into the executable.

### 4. Run the build script

```cmd
build.bat
```

Or manually:

```cmd
pip install pyinstaller
pyinstaller video_synth.spec --clean --noconfirm
```

### 5. Test the executable

```cmd
dist\VideoSynth\VideoSynth.exe --help
dist\VideoSynth\VideoSynth.exe
```

---

## Step-by-step: Linux / macOS

```bash
# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build web UI
cd web && npm install && npm run build && cd ..

# Package
bash build.sh
```

The executable is at `dist/VideoSynth/VideoSynth`.

---

## Automated CI Build (GitHub Actions)

The workflow at `.github/workflows/build.yml` runs on:

- **Tag push** matching `v*.*.*` — builds and creates a GitHub Release with the ZIP.
- **Manual trigger** (`workflow_dispatch`) — builds and uploads an artifact.

To trigger a release build:

```bash
git tag v1.0.0
git push origin v1.0.0
```

---

## Known Issues and Workarounds

### ModernGL / OpenGL context

**Problem:** ModernGL requires a working OpenGL context.  On systems without a GPU
(headless servers, some VMs) the app will fail with `Error: Unable to create GL context`.

**Workaround (Linux):** Install Mesa software renderer and set:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
dist/VideoSynth/VideoSynth --headless --api
```

The runtime hook (`hooks/runtime_opengl.py`) automatically sets `LIBGL_ALWAYS_SOFTWARE=1`
on Linux when no DRI device is detected.

**Workaround (Windows):** Install [ANGLE](https://chromium.googlesource.com/angle/angle)
or enable the "Basic display adapter" (WARP) in Device Manager.

---

### PyQt6 platform plugins

**Problem:** The app launches but immediately crashes with
`qt.qpa.plugin: Could not load the Qt platform plugin "windows"`.

**Cause:** PyQt6's `qwindows.dll` platform plugin is missing or not found.

**Fix:** The bundle must contain `PyQt6/Qt6/plugins/platforms/qwindows.dll`.
PyInstaller's `collect_all('PyQt6')` handles this, but if it fails, manually
copy the plugins directory from your `.venv`:

```
.venv\Lib\site-packages\PyQt6\Qt6\plugins\  →  dist\VideoSynth\PyQt6\Qt6\plugins\
```

The runtime hook also sets `QT_PLUGIN_PATH` automatically inside the bundle.

---

### Web UI not available

**Problem:** The app starts but `http://localhost:8000/ui` returns 404.

**Cause:** `web/dist/` was not built before running PyInstaller.

**Fix:** Run the web build step before packaging:

```cmd
cd web && npm install && npm run build
```

`video_synth.spec` prints a warning (not an error) if `web/dist/` is missing,
so the build will still succeed — the web UI simply won't be bundled.

---

### Virtual camera not working on Windows

**Problem:** `VirtualCamOutput` raises an error or the virtual camera device does
not appear in other applications.

**Cause:** `pyvirtualcam` on Windows relies on the **OBS Virtual Camera** driver
which must be installed independently.

**Fix:**
1. Download and install [OBS Studio](https://obsproject.com/) (free).
2. Launch OBS at least once so it registers the virtual camera driver.
3. You do not need to keep OBS running for VideoSynth's virtual camera to work.

To disable the virtual camera entirely: `VideoSynth.exe --no-virtualcam`

---

### MIDI not detected

**Problem:** MIDI controller is not recognised.

**Cause:** `python-rtmidi` ships a compiled `.pyd` that must be bundled correctly.
The hidden import `mido.backends.rtmidi` pulls in the backend.

**Fix:** Ensure `python-rtmidi` is installed in the **same environment** used for
the PyInstaller build, not a different venv.

---

### Large bundle size

The default bundle is large (~500 MB) because it includes full PyQt6 and OpenCV
distributions.  Strategies to reduce size:

1. **Strip unused Qt modules** — after building, delete Qt6 DLLs you do not need:
   ```
   dist\VideoSynth\PyQt6\Qt6\bin\Qt6WebEngineCore.dll  (large, not needed)
   dist\VideoSynth\PyQt6\Qt6\bin\Qt6Quick.dll           (not needed)
   ```
   Be careful: test the exe after each deletion.

2. **UPX compression** — already enabled in `video_synth.spec` (`upx=True`).
   Install UPX (https://upx.github.io/) and put it on PATH before building.

3. **Exclude optional packages** — if you never use `ollama`, `obsws_python`, etc.,
   add them to the `excludes` list in `video_synth.spec`.

---

## Testing the Built Executable

```cmd
:: Basic GUI launch
dist\VideoSynth\VideoSynth.exe

:: Headless mode with API server
dist\VideoSynth\VideoSynth.exe --headless --api --api-host 0.0.0.0

:: Specific patch and layout
dist\VideoSynth\VideoSynth.exe --patch 2 --control-layout QUAD_PREVIEW

:: With FFmpeg output
dist\VideoSynth\VideoSynth.exe --ffmpeg --ffmpeg-output output.mp4

:: Debug logging
dist\VideoSynth\VideoSynth.exe --log-level DEBUG
```

Check that:
- The GUI window opens (if not headless)
- The API docs are reachable at `http://127.0.0.1:8000/docs` (if `--api` is set)
- The web UI is reachable at `http://127.0.0.1:8000/ui` (if web was built)
- Video output renders without errors in the log
