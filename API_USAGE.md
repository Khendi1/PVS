# Video Synthesizer API & FFmpeg Usage

## Overview

The video synthesizer can now be controlled remotely via REST API and output to FFmpeg for recording or streaming.

## Features

- **REST API**: Control all parameters remotely via HTTP
- **FFmpeg Output**: Record to file or stream to RTMP
- **Headless Mode**: Run without GUI for server deployments
- **Agent Control**: Perfect for AI agents or automation scripts

## Installation

Install the additional dependencies:

```bash
pip install fastapi uvicorn
```

Make sure FFmpeg is installed and in your PATH:
- Windows: Download from https://ffmpeg.org/download.html
- macOS: `brew install ffmpeg`
- Linux: `apt-get install ffmpeg` or `yum install ffmpeg`

## Usage

### Enable API Server

Start the synthesizer with API enabled:

```bash
python -m video_synth --api
```

The API server will start at `http://127.0.0.1:8000`

Custom host and port:

```bash
python -m video_synth --api --api-host 0.0.0.0 --api-port 8080
```

### Enable FFmpeg Output

Record to file:

```bash
python -m video_synth --ffmpeg --ffmpeg-output output.mp4
```

Stream to RTMP server:

```bash
python -m video_synth --ffmpeg --ffmpeg-output rtmp://localhost/live/stream
```

Encoding options:

```bash
python -m video_synth --ffmpeg --ffmpeg-output output.mp4 \
  --ffmpeg-preset fast --ffmpeg-crf 20
```

Presets: `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`, `slower`, `veryslow`
CRF: 0-51 (lower = better quality, 23 is default)

### Headless Mode

Run without GUI (requires `--api` or `--ffmpeg`):

```bash
python -m video_synth --headless --api --ffmpeg --ffmpeg-output output.mp4
```

This is useful for server deployments or when running on systems without a display.

### Combined Usage

API + FFmpeg + Headless for remote-controlled recording:

```bash
python -m video_synth --headless --api --ffmpeg --ffmpeg-output recording.mp4
```

## API Endpoints

The REST API provides the following endpoints:

### Get All Parameters

```http
GET http://127.0.0.1:8000/params
```

Returns a list of all parameters with their current values, min/max bounds, and metadata.

Example response:

```json
[
  {
    "name": "glitch_intensity_max",
    "value": 50,
    "min": 0,
    "max": 100,
    "default": 50,
    "group": "Groups.SRC_1_EFFECTS",
    "subgroup": "Glitch_General",
    "type": "Widget.SLIDER"
  },
  ...
]
```

### Get Specific Parameter

```http
GET http://127.0.0.1:8000/params/{param_name}
```

Example:

```http
GET http://127.0.0.1:8000/params/glitch_intensity_max
```

### Set Parameter Value

```http
PUT http://127.0.0.1:8000/params/{param_name}
Content-Type: application/json

{
  "value": 75
}
```

Example with curl:

```bash
curl -X PUT http://127.0.0.1:8000/params/glitch_intensity_max \
  -H "Content-Type: application/json" \
  -d '{"value": 75}'
```

Example with Python requests:

```python
import requests

response = requests.put(
    'http://127.0.0.1:8000/params/glitch_intensity_max',
    json={'value': 75}
)
print(response.json())
```

### Reset Parameter

```http
POST http://127.0.0.1:8000/params/reset/{param_name}
```

Resets the parameter to its default value.

### Get Snapshot

```http
GET http://127.0.0.1:8000/snapshot
```

Returns the current frame as a JPEG image. Useful for monitoring or analysis.

Example with Python:

```python
import requests
from PIL import Image
import io

response = requests.get('http://127.0.0.1:8000/snapshot')
image = Image.open(io.BytesIO(response.content))
image.show()
```

### API Documentation

Interactive API documentation (Swagger UI) is available at:

```
http://127.0.0.1:8000/docs
```

## Agent Control Examples

### Python Agent Example

```python
import requests
import time

API_BASE = 'http://127.0.0.1:8000'

class VideoSynthAgent:
    def __init__(self, base_url=API_BASE):
        self.base_url = base_url

    def get_params(self):
        """Get all parameters."""
        response = requests.get(f'{self.base_url}/params')
        return response.json()

    def set_param(self, name, value):
        """Set a parameter value."""
        response = requests.put(
            f'{self.base_url}/params/{name}',
            json={'value': value}
        )
        return response.json()

    def get_snapshot(self):
        """Get current frame as PIL Image."""
        from PIL import Image
        import io
        response = requests.get(f'{self.base_url}/snapshot')
        return Image.open(io.BytesIO(response.content))

    def animate_parameter(self, param_name, start, end, duration=5.0, steps=100):
        """Smoothly animate a parameter from start to end value."""
        for i in range(steps):
            progress = i / (steps - 1)
            value = start + (end - start) * progress
            self.set_param(param_name, value)
            time.sleep(duration / steps)

# Usage
agent = VideoSynthAgent()

# Get all parameters
params = agent.get_params()
print(f"Found {len(params)} parameters")

# Animate glitch intensity
agent.animate_parameter('glitch_intensity_max', 0, 100, duration=10.0)

# Set multiple parameters
agent.set_param('pattern_alpha', 0.8)
agent.set_param('pattern_speed', 2.5)

# Capture snapshot
image = agent.get_snapshot()
image.save('snapshot.jpg')
```

### LLM Agent Integration

The API is designed to be easily used by LLM agents. Example prompt:

```
You are controlling a video synthesizer via REST API.

Available endpoints:
- GET /params - list all parameters
- PUT /params/{name} - set parameter value (JSON: {"value": number})
- GET /snapshot - get current frame as image

Task: Create a psychedelic visual effect by:
1. Enabling pattern feedback
2. Setting high pattern warp
3. Animating the pattern speed

API base URL: http://127.0.0.1:8000
```

### Automation Example

```python
import requests
import time
import random

API_BASE = 'http://127.0.0.1:8000'

def random_glitch_sequence():
    """Create a random glitch art sequence."""
    glitch_params = [
        'enable_pixel_shift',
        'enable_color_split',
        'enable_block_corruption',
        'enable_slitscan'
    ]

    # Randomly enable/disable glitch effects
    for param in glitch_params:
        value = random.choice([0, 1])
        requests.put(f'{API_BASE}/params/{param}', json={'value': value})

    # Randomize intensity
    intensity = random.randint(30, 100)
    requests.put(f'{API_BASE}/params/glitch_intensity_max', json={'value': intensity})

# Run random glitch sequence every 5 seconds
while True:
    random_glitch_sequence()
    time.sleep(5)
```

## OBS Integration

The video synthesizer can be integrated with OBS Studio in several ways.

### Method 1: RTMP Stream to OBS (Recommended)

This is the most reliable method for live integration.

#### Step 1: Setup RTMP Server

**Using Docker (Easiest):**
```bash
docker run -d -p 1935:1935 --name rtmp-server tiangolo/nginx-rtmp
```

**Using nginx-rtmp:**

Windows: Download from https://github.com/illuspas/nginx-rtmp-windows-module

macOS:
```bash
brew tap denji/nginx
brew install nginx-full --with-rtmp-module
```

Linux:
```bash
sudo apt-get install nginx libnginx-mod-rtmp
```

Create/edit `nginx.conf`:
```nginx
rtmp {
    server {
        listen 1935;
        chunk_size 4096;

        application live {
            live on;
            record off;
            # Optional: allow only localhost
            # allow publish 127.0.0.1;
            # deny publish all;
        }
    }
}
```

Start nginx:
```bash
# Windows
nginx.exe

# macOS/Linux
sudo nginx
# or
sudo systemctl start nginx
```

#### Step 2: Stream from Video Synth
```bash
python -m video_synth --ffmpeg \
  --ffmpeg-output rtmp://localhost/live/stream \
  --ffmpeg-preset veryfast
```

#### Step 3: Add to OBS
1. In OBS, add a **Media Source**
2. Uncheck "Local File"
3. Input: `rtmp://localhost/live/stream`
4. Check "Restart playback when source becomes active"
5. Set to "Close file when inactive" = Off

**Tips:**
- Use `veryfast` or `ultrafast` preset for low latency
- For best quality, use `medium` preset but expect ~1-2 second delay
- Adjust buffer size in OBS Media Source settings if you experience lag

### Method 2: OBS WebSocket Control

Control OBS programmatically while using virtual camera or RTMP.

#### Install Dependencies
```bash
pip install obs-websocket-py
```

#### Enable OBS WebSocket
1. In OBS, go to **Tools > WebSocket Server Settings**
2. Enable WebSocket server
3. Set a password (optional but recommended)
4. Note the port (default: 4455 for OBS 28+, 4444 for older versions)

#### Use OBS Controller

```python
from obs_controller import OBSController

# Connect to OBS
obs = OBSController(password="your_password")
obs.connect()

# Start recording
obs.start_recording()

# Switch scene
obs.set_scene("Scene 2")

# Stop recording after some time
import time
time.sleep(60)
obs.stop_recording()

# Disconnect
obs.disconnect()
```

#### Combined API + OBS Control Example

```python
import requests
import time
from obs_controller import OBSController

API_BASE = 'http://127.0.0.1:8000'

# Connect to OBS
obs = OBSController(password="your_password")
obs.connect()

# Start OBS recording
obs.start_recording()

# Animate video synth parameters via API
for i in range(100):
    intensity = int(i)
    requests.put(f'{API_BASE}/params/glitch_intensity_max',
                 json={'value': intensity})
    time.sleep(0.1)

# Stop recording
obs.stop_recording()
obs.disconnect()
```

### Method 3: Virtual Camera

Use OBS Virtual Camera as an intermediate device.

#### Step 1: Enable OBS Virtual Camera
1. In OBS, click **Start Virtual Camera**
2. This creates a virtual webcam device

#### Step 2: Use Virtual Camera in Video Synth
The video synth can capture from the virtual camera as a video device:
```bash
# List available devices first
python -m video_synth

# In the GUI, select "DEVICE_X" that corresponds to OBS Virtual Camera
```

#### Step 3: Create Feedback Loop
This creates interesting feedback effects:
1. Video Synth → OBS (via RTMP)
2. OBS → Virtual Camera
3. Virtual Camera → Video Synth (as input)

**Warning:** This can create intense visual feedback! Start with low effect intensities.

### Method 4: NDI (Network Device Interface)

NDI allows low-latency video over network.

#### Install NDI Tools
Download from: https://ndi.tv/tools/

#### Install OBS NDI Plugin
Download from: https://github.com/obs-ndi/obs-ndi/releases

#### Install NDI Python Library
```bash
pip install ndi-python
```

#### Stream via NDI
```python
# This would require implementing NDI output in the video synth
# Currently not implemented, but could be added similar to FFmpeg output
```

### Method 5: SRT Protocol (Low Latency Alternative to RTMP)

SRT provides lower latency than RTMP.

#### Install SRT
```bash
# Windows: Download from https://github.com/Haivision/srt/releases
# macOS:
brew install srt

# Linux:
sudo apt-get install srt-tools
```

#### Stream with SRT
```bash
python -m video_synth --ffmpeg \
  --ffmpeg-output "srt://localhost:9999?mode=listener" \
  --ffmpeg-preset ultrafast
```

#### Add to OBS
1. Add **Media Source**
2. Input: `srt://localhost:9999`
3. Enable hardware decoding

### Automated Recording Workflow

Complete example: Automated video generation with OBS recording.

```python
import requests
import time
from obs_controller import OBSController

API_BASE = 'http://127.0.0.1:8000'

def automated_recording_session():
    """Automated 5-minute recording with parameter automation."""

    # Connect to OBS
    obs = OBSController(password="your_password")
    obs.connect()

    # Setup initial parameters
    requests.put(f'{API_BASE}/params/pattern_alpha', json={'value': 0.5})
    requests.put(f'{API_BASE}/params/pattern_fb_enable', json={'value': 1})

    # Start recording
    obs.start_recording()
    print("Recording started...")

    # Animate parameters over 5 minutes
    duration = 300  # 5 minutes
    steps = 300

    for i in range(steps):
        progress = i / steps

        # Animate multiple parameters
        params = {
            'glitch_intensity_max': int(progress * 100),
            'pattern_speed': progress * 3.0,
            'pattern_fb_warp': progress * 15.0,
            'warp_angle_amt': int(progress * 180)
        }

        for param, value in params.items():
            requests.put(f'{API_BASE}/params/{param}', json={'value': value})

        time.sleep(duration / steps)

        # Print progress
        if i % 30 == 0:
            print(f"Progress: {int(progress * 100)}%")

    # Stop recording
    obs.stop_recording()
    print("Recording stopped")

    # Wait for OBS to finalize the file
    time.sleep(2)

    # Get recording info
    status = obs.get_recording_status()
    print(f"Recording status: {status}")

    obs.disconnect()

if __name__ == "__main__":
    # Start the video synth with RTMP output
    print("Start video_synth with:")
    print("python -m video_synth --api --ffmpeg --ffmpeg-output rtmp://localhost/live/stream --ffmpeg-preset veryfast")
    print("\nThen run this script to automate the recording")

    input("Press Enter when video_synth and OBS are ready...")
    automated_recording_session()
```

### Recommended Setup for Best Results

**For Live Streaming:**
```bash
# Terminal 1: Start video synth with RTMP output
python -m video_synth --api --ffmpeg \
  --ffmpeg-output rtmp://localhost/live/stream \
  --ffmpeg-preset ultrafast

# Terminal 2: Run automation script
python automation_script.py
```

**For High-Quality Recording:**
```bash
# Terminal 1: Video synth with medium quality
python -m video_synth --api --ffmpeg \
  --ffmpeg-output rtmp://localhost/live/stream \
  --ffmpeg-preset medium

# OBS: Record at high quality settings
# OBS Settings > Output > Recording Quality: "High Quality, Medium File Size"
```

**For Headless Server:**
```bash
# No GUI, just API and RTMP output
python -m video_synth --headless --api --ffmpeg \
  --ffmpeg-output rtmp://your-server:1935/live/stream \
  --ffmpeg-preset fast
```

## Performance Notes

- API calls are thread-safe and non-blocking
- Parameter changes take effect immediately (next frame)
- Snapshot endpoint may have slight delay depending on frame rate
- FFmpeg encoding adds minimal overhead with appropriate presets
- For real-time streaming, use `ultrafast` or `veryfast` preset

## Troubleshooting

### FFmpeg not found

Make sure FFmpeg is installed and in your system PATH:

```bash
ffmpeg -version
```

### API server won't start

Check if port is already in use:

```bash
# Windows
netstat -ano | findstr :8000

# macOS/Linux
lsof -i :8000
```

Use a different port with `--api-port`.

### Headless mode validation error

Headless mode requires either `--api` or `--ffmpeg` to be enabled.

### Permission errors

On Linux/macOS, you may need to allow the port in your firewall:

```bash
# Allow port 8000
sudo ufw allow 8000
```
