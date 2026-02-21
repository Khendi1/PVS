"""
Example automation script showing how to control both the video synthesizer
and OBS Studio programmatically for automated video production.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'video_synth'))

import requests
import time
from obs_controller import OBSController

API_BASE = 'http://127.0.0.1:8000'


def check_connection():
    """Check if video synth API is accessible."""
    try:
        response = requests.get(f'{API_BASE}/')
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def animated_glitch_sequence(duration=30):
    """
    Create an animated glitch art sequence.

    Args:
        duration: Duration in seconds
    """
    print(f"Running {duration}s glitch sequence...")

    steps = duration * 10  # 10 updates per second

    for i in range(steps):
        progress = i / steps

        # Gradually increase glitch intensity
        intensity = int(progress * 100)
        requests.put(f'{API_BASE}/params/glitch_intensity_max',
                     json={'value': intensity})

        # Randomly toggle glitch types
        if i % 50 == 0:  # Every 5 seconds
            import random
            glitch_types = [
                'enable_pixel_shift',
                'enable_color_split',
                'enable_slitscan'
            ]
            for glitch_type in glitch_types:
                value = random.choice([0, 1])
                requests.put(f'{API_BASE}/params/{glitch_type}',
                             json={'value': value})

        time.sleep(duration / steps)


def pattern_feedback_sequence(duration=30):
    """
    Create evolving pattern feedback sequence.

    Args:
        duration: Duration in seconds
    """
    print(f"Running {duration}s pattern feedback sequence...")

    # Enable pattern feedback
    requests.put(f'{API_BASE}/params/pattern_fb_enable', json={'value': 1})

    steps = duration * 10

    for i in range(steps):
        progress = i / steps

        # Animate pattern parameters
        params = {
            'pattern_speed': progress * 3.0,
            'pattern_fb_warp': progress * 15.0,
            'pattern_fb_decay': 0.85 + (progress * 0.1),
            'pattern_alpha': 0.3 + (progress * 0.5)
        }

        for param, value in params.items():
            requests.put(f'{API_BASE}/params/{param}', json={'value': value})

        time.sleep(duration / steps)


def warp_chaos_sequence(duration=30):
    """
    Create chaotic warp feedback sequence.

    Args:
        duration: Duration in seconds
    """
    print(f"Running {duration}s warp chaos sequence...")

    # Set warp type to FEEDBACK (enum value 6)
    requests.put(f'{API_BASE}/params/warp_type', json={'value': 6})

    # Enable feedback
    requests.put(f'{API_BASE}/params/alpha', json={'value': 0.7})

    steps = duration * 10

    for i in range(steps):
        progress = i / steps

        # Animate warp parameters
        params = {
            'fb_warp_decay': 0.9 + (progress * 0.08),
            'fb_warp_strength': progress * 40.0,
            'fb_warp_freq': 3.0 + (progress * 10.0)
        }

        for param, value in params.items():
            requests.put(f'{API_BASE}/params/{param}', json={'value': value})

        time.sleep(duration / steps)


def full_automated_session():
    """
    Complete automated recording session with OBS integration.
    Creates a 3-minute video with different visual sequences.
    """

    print("=" * 60)
    print("OBS Automation Example - Automated Recording Session")
    print("=" * 60)

    # Check API connection
    if not check_connection():
        print("ERROR: Cannot connect to video synth API")
        print("Please start the video synth with:")
        print("  python -m video_synth --api --ffmpeg \\")
        print("    --ffmpeg-output udp://127.0.0.1:1234 \\")
        print("    --ffmpeg-preset veryfast")
        return

    # Connect to OBS
    print("\nConnecting to OBS...")
    obs = OBSController(password="")  # Set your OBS password here
    obs.connect()

    if not obs.connected:
        print("ERROR: Failed to connect to OBS")
        print("Please check:")
        print("  1. OBS is running")
        print("  2. WebSocket server is enabled (Tools > WebSocket Server Settings)")
        print("  3. Password is correct")
        return

    # Get available scenes
    scenes = obs.get_scenes()
    print(f"Available OBS scenes: {scenes}")

    # Reset all parameters to defaults
    print("\nResetting parameters to defaults...")
    params_to_reset = [
        'glitch_intensity_max', 'pattern_alpha', 'pattern_fb_enable',
        'warp_type', 'alpha'
    ]
    for param in params_to_reset:
        try:
            requests.post(f'{API_BASE}/params/reset/{param}')
        except:
            pass

    time.sleep(2)

    # Start recording
    print("\n" + "=" * 60)
    print("Starting OBS recording...")
    print("=" * 60)
    obs.start_recording()

    # Sequence 1: Glitch Art (60 seconds)
    print("\n[00:00 - 01:00] Sequence 1: Glitch Art")
    animated_glitch_sequence(duration=60)

    # Sequence 2: Pattern Feedback (60 seconds)
    print("\n[01:00 - 02:00] Sequence 2: Pattern Feedback")
    pattern_feedback_sequence(duration=60)

    # Sequence 3: Warp Chaos (60 seconds)
    print("\n[02:00 - 03:00] Sequence 3: Warp Chaos")
    warp_chaos_sequence(duration=60)

    # Stop recording
    print("\n" + "=" * 60)
    print("Stopping OBS recording...")
    print("=" * 60)
    obs.stop_recording()

    # Wait for finalization
    time.sleep(3)

    # Get recording status
    status = obs.get_recording_status()
    print(f"\nRecording status: {status}")

    # Disconnect
    obs.disconnect()

    print("\n" + "=" * 60)
    print("Recording session complete!")
    print("=" * 60)
    print("\nCheck your OBS recordings folder for the output video.")


def manual_control_demo():
    """
    Simple interactive demo for manual parameter control.
    """
    print("=" * 60)
    print("Manual Control Demo")
    print("=" * 60)

    if not check_connection():
        print("ERROR: Cannot connect to video synth API")
        return

    while True:
        print("\nCommands:")
        print("  1 - Toggle glitch effects")
        print("  2 - Increase pattern speed")
        print("  3 - Enable feedback warp")
        print("  4 - Get current snapshot")
        print("  5 - Reset all parameters")
        print("  q - Quit")

        choice = input("\nEnter choice: ").strip()

        if choice == '1':
            value = int(input("Enable (1) or disable (0)? "))
            for effect in ['enable_pixel_shift', 'enable_color_split', 'enable_slitscan']:
                requests.put(f'{API_BASE}/params/{effect}', json={'value': value})
            print("Glitch effects toggled")

        elif choice == '2':
            speed = float(input("Enter speed (0.0 - 5.0): "))
            requests.put(f'{API_BASE}/params/pattern_speed', json={'value': speed})
            print(f"Pattern speed set to {speed}")

        elif choice == '3':
            requests.put(f'{API_BASE}/params/warp_type', json={'value': 6})
            requests.put(f'{API_BASE}/params/alpha', json={'value': 0.7})
            print("Feedback warp enabled")

        elif choice == '4':
            response = requests.get(f'{API_BASE}/snapshot')
            filename = f'snapshot_{int(time.time())}.jpg'
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Snapshot saved to {filename}")

        elif choice == '5':
            # Get all params and reset them
            params = requests.get(f'{API_BASE}/params').json()
            for param in params:
                requests.post(f'{API_BASE}/params/reset/{param["name"]}')
            print("All parameters reset to defaults")

        elif choice == 'q':
            break


if __name__ == "__main__":
    print("\nOBS Automation Example")
    print("=" * 60)
    print("\nBefore running this script, make sure:")
    print("1. Video synth is running with API and FFmpeg:")
    print("   python -m video_synth --api --ffmpeg \\")
    print("     --ffmpeg-output udp://127.0.0.1:1234 \\")
    print("     --ffmpeg-preset veryfast")
    print("\n2. OBS is running with:")
    print("   - WebSocket server enabled")
    print("   - Media source added: udp://127.0.0.1:1234")
    print("\n" + "=" * 60)

    choice = input("\nRun (f)ull automation or (m)anual control? [f/m]: ").strip().lower()

    if choice == 'f':
        input("\nPress Enter when everything is ready...")
        full_automated_session()
    elif choice == 'm':
        manual_control_demo()
    else:
        print("Invalid choice")
