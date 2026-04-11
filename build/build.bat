@echo off
:: ============================================================================
:: VideoSynth -- Windows build script
:: ============================================================================
:: Prerequisites:
::   - Python 3.11 installed and on PATH
::   - Node.js 18+ installed (for web UI build)
::   - Activated virtual environment (recommended)
::
:: Usage: build.bat
:: Output: dist\VideoSynth\VideoSynth.exe
:: ============================================================================

setlocal ENABLEDELAYEDEXPANSION

echo.
echo ============================================================
echo  VideoSynth -- Windows Build
echo ============================================================
echo.

:: ---- 1. Install / upgrade PyInstaller ----
echo [1/4] Installing PyInstaller...
pip install pyinstaller --upgrade --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install PyInstaller.
    exit /b 1
)

:: ---- 2. Install Python dependencies ----
echo [2/4] Installing Python dependencies...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install requirements.
    exit /b 1
)

:: ---- 3. Build the React web UI (skip if web/dist already exists) ----
echo [3/4] Building web UI...
if not exist "web\dist\index.html" (
    if exist "web\package.json" (
        pushd web
        echo      Running npm install...
        call npm install --silent
        if %ERRORLEVEL% NEQ 0 (
            echo WARNING: npm install failed. Web UI will not be available.
        ) else (
            echo      Running npm run build...
            call npm run build
            if %ERRORLEVEL% NEQ 0 (
                echo WARNING: npm build failed. Web UI will not be available.
            )
        )
        popd
    ) else (
        echo      web\package.json not found -- skipping web build.
    )
) else (
    echo      web\dist already exists -- skipping web build.
)

:: ---- 4. Run PyInstaller ----
echo [4/4] Running PyInstaller...
pyinstaller build\video_synth.spec --clean --noconfirm
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyInstaller build failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Build complete!
echo  Output : dist\VideoSynth\
echo  Run    : dist\VideoSynth\VideoSynth.exe
echo ============================================================
echo.

endlocal
