# Video Synthesizer Examples

This directory contains example scripts demonstrating how to use the video synthesizer API and OBS integration.

## Files

- `obs_automation_example.py` - Complete automation example with OBS recording

## Setup

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Setup RTMP Server

**Using Docker (Recommended):**
```bash
docker run -d -p 1935:1935 --name rtmp-server tiangolo/nginx-rtmp
```

**Or install nginx-rtmp manually** - see [../API_USAGE.md](../API_USAGE.md) for details.

### 3. Setup OBS

1. Install OBS Studio
2. Enable WebSocket server:
   - Tools > WebSocket Server Settings
   - Enable WebSocket server
   - Set password (optional)
   - Note the port (default: 4455)
3. Add Media Source:
   - Add new Media Source
   - Uncheck "Local File"
   - Input: `rtmp://localhost/live/stream`
   - Check "Restart playback when source becomes active"

### 4. Start Video Synthesizer

```bash
python -m video_synth --api --ffmpeg \
  --ffmpeg-output rtmp://localhost/live/stream \
  --ffmpeg-preset veryfast
```

## Running Examples

### Full Automated Recording

This creates a 3-minute automated video with different visual sequences.

```bash
python obs_automation_example.py
```

Select option `f` for full automation.

The script will:
1. Connect to video synth API
2. Connect to OBS WebSocket
3. Start OBS recording
4. Run three 60-second sequences:
   - Glitch Art (animated intensity)
   - Pattern Feedback (evolving patterns)
   - Warp Chaos (feedback warp)
5. Stop recording
6. Disconnect

### Manual Control Demo

Interactive demo for experimenting with parameters.

```bash
python obs_automation_example.py
```

Select option `m` for manual control.

Available commands:
- Toggle glitch effects on/off
- Adjust pattern speed
- Enable feedback warp
- Capture snapshots
- Reset all parameters

## Customizing Sequences

### Create Your Own Sequence

```python
def my_custom_sequence(duration=30):
    """Create a custom visual sequence."""
    import requests
    import time

    API_BASE = 'http://127.0.0.1:8000'
    steps = duration * 10  # Updates per second

    for i in range(steps):
        progress = i / steps

        # Your parameter automation here
        params = {
            'param_name': min_value + (max_value - min_value) * progress,
            # Add more parameters...
        }

        for param, value in params.items():
            requests.put(f'{API_BASE}/params/{param}', json={'value': value})

        time.sleep(duration / steps)
```

### Add to Main Session

```python
def full_automated_session():
    # ... existing code ...

    # Add your sequence
    print("[03:00 - 03:30] Sequence 4: My Custom Sequence")
    my_custom_sequence(duration=30)

    # ... rest of code ...
```

## Useful Parameter Combinations

### Glitch Art
```python
params = {
    'enable_pixel_shift': 1,
    'enable_color_split': 1,
    'enable_slitscan': 1,
    'glitch_intensity_max': 75,
    'slitscan_speed': 2.5
}
```

### Pattern Feedback Chaos
```python
params = {
    'pattern_fb_enable': 1,
    'pattern_fb_decay': 0.95,
    'pattern_fb_strength': 0.7,
    'pattern_fb_warp': 12.0,
    'pattern_speed': 2.5,
    'pattern_alpha': 0.8
}
```

### Warp Feedback
```python
params = {
    'warp_type': 6,  # FEEDBACK
    'fb_warp_decay': 0.95,
    'fb_warp_strength': 30.0,
    'fb_warp_freq': 8.0,
    'alpha': 0.8  # Overall feedback
}
```

### Psychedelic Mix
```python
params = {
    'pattern_type': 4,  # RADIAL
    'pattern_alpha': 0.6,
    'pattern_fb_enable': 1,
    'pattern_speed': 3.0,
    'warp_type': 6,
    'alpha': 0.5,
    'enable_color_split': 1
}
```

## Tips

1. **Start Simple**: Begin with single parameter changes before complex sequences
2. **Use Snapshots**: Capture snapshots to see what parameters look good
3. **Monitor Performance**: Use `--diagnose 100` flag to monitor frame timing
4. **Preset Quality**: Use `veryfast` for live streaming, `medium` for recordings
5. **Reset Between Sequences**: Reset parameters between sequences for clean transitions

## Troubleshooting

### Cannot connect to API
- Make sure video synth is running with `--api` flag
- Check that port 8000 is not blocked
- Try accessing http://127.0.0.1:8000 in browser

### Cannot connect to OBS
- Verify OBS is running
- Check WebSocket is enabled in OBS settings
- Verify password matches
- Check port (default 4455 for OBS 28+, 4444 for older)

### No video in OBS
- Check RTMP server is running (`docker ps` or `nginx status`)
- Verify Media Source URL is correct: `rtmp://localhost/live/stream`
- Check video synth FFmpeg output is working (look for frame count logs)

### Low FPS / Lag
- Use faster encoding preset (`ultrafast` or `veryfast`)
- Reduce resolution in video synth settings
- Close other applications
- Check CPU/GPU usage

## Advanced Examples

### AI-Driven Parameter Control

```python
import requests
import random

API_BASE = 'http://127.0.0.1:8000'

def ai_generative_sequence(duration=60):
    """
    Simulated AI-driven parameter evolution.
    Could be replaced with actual ML model.
    """
    # Get all parameters
    all_params = requests.get(f'{API_BASE}/params').json()

    # Filter numeric slider params
    slider_params = [p for p in all_params if p['type'] == 'Widget.SLIDER']

    steps = duration * 5

    for i in range(steps):
        # Randomly select 3 parameters to modify
        params_to_change = random.sample(slider_params, 3)

        for param in params_to_change:
            # Random walk within bounds
            current = param['value']
            min_val = param['min']
            max_val = param['max']

            # Small random change
            delta = random.uniform(-5, 5)
            new_value = max(min_val, min(max_val, current + delta))

            requests.put(f'{API_BASE}/params/{param["name"]}',
                        json={'value': new_value})

        time.sleep(duration / steps)
```

### Reactive to Audio (External)

```python
import requests
import numpy as np
import sounddevice as sd

API_BASE = 'http://127.0.0.1:8000'

def audio_reactive_control():
    """
    Control parameters based on external audio input.
    """
    def audio_callback(indata, frames, time, status):
        # Calculate RMS (loudness)
        rms = np.sqrt(np.mean(indata**2)) * 1000

        # Map to glitch intensity
        intensity = int(min(100, rms))
        requests.put(f'{API_BASE}/params/glitch_intensity_max',
                    json={'value': intensity})

    with sd.InputStream(callback=audio_callback):
        input("Press Enter to stop...\n")
```

## Further Reading

- [API Usage Guide](../API_USAGE.md) - Complete API documentation
- [OBS WebSocket Protocol](https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
