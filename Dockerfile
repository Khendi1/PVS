FROM python:3.11-slim

# Install system dependencies for headless OpenGL, Qt6, audio/MIDI, and video
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenGL / EGL for headless rendering (ModernGL + PyOpenGL)
    libegl1 \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libgles2-mesa \
    libgbm1 \
    # Virtual framebuffer so Qt6 can initialize without a physical display
    xvfb \
    x11-utils \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libdbus-1-3 \
    # Audio (sounddevice / portaudio)
    portaudio19-dev \
    libasound2-dev \
    # MIDI (python-rtmidi)
    libasound2 \
    # Video encoding (opencv / ffmpeg)
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    ffmpeg \
    # Build tools
    gcc \
    g++ \
    python3-dev \
    # Utilities
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy application source
COPY video_synth/ ./video_synth/
COPY web/ ./web/
COPY save/ ./save/
COPY shaders/ ./shaders/

# Headless OpenGL via software renderer + Xvfb virtual display
ENV DISPLAY=:99
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV QT_QPA_PLATFORM=offscreen

EXPOSE 8000

COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
