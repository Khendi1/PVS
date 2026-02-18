"""
OBS WebSocket controller for programmatic control of OBS Studio.
Allows the video synthesizer to control OBS recording, streaming, and scenes.
"""

import logging
from obswebsocket import obsws, requests as obs_requests
from typing import Optional

log = logging.getLogger(__name__)


class OBSController:
    """Controller for OBS Studio via WebSocket."""

    def __init__(self, host: str = "localhost", port: int = 4455, password: str = ""):
        """
        Initialize OBS WebSocket connection.

        Args:
            host: OBS WebSocket host (default: localhost)
            port: OBS WebSocket port (default: 4455 for OBS 28+, 4444 for older)
            password: WebSocket password (set in OBS Tools > WebSocket Server Settings)
        """
        self.host = host
        self.port = port
        self.password = password
        self.ws: Optional[obsws] = None
        self.connected = False

    def connect(self):
        """Connect to OBS WebSocket server."""
        try:
            self.ws = obsws(self.host, self.port, self.password)
            self.ws.connect()
            self.connected = True
            log.info(f"Connected to OBS at {self.host}:{self.port}")

            # Get OBS version info
            version = self.ws.call(obs_requests.GetVersion())
            log.info(f"OBS Version: {version.getObsVersion()}")

        except Exception as e:
            log.error(f"Failed to connect to OBS: {e}")
            self.connected = False

    def disconnect(self):
        """Disconnect from OBS."""
        if self.ws and self.connected:
            self.ws.disconnect()
            self.connected = False
            log.info("Disconnected from OBS")

    def start_recording(self):
        """Start OBS recording."""
        if not self.connected:
            log.error("Not connected to OBS")
            return False

        try:
            self.ws.call(obs_requests.StartRecord())
            log.info("Started OBS recording")
            return True
        except Exception as e:
            log.error(f"Failed to start recording: {e}")
            return False

    def stop_recording(self):
        """Stop OBS recording."""
        if not self.connected:
            log.error("Not connected to OBS")
            return False

        try:
            self.ws.call(obs_requests.StopRecord())
            log.info("Stopped OBS recording")
            return True
        except Exception as e:
            log.error(f"Failed to stop recording: {e}")
            return False

    def start_streaming(self):
        """Start OBS streaming."""
        if not self.connected:
            log.error("Not connected to OBS")
            return False

        try:
            self.ws.call(obs_requests.StartStream())
            log.info("Started OBS streaming")
            return True
        except Exception as e:
            log.error(f"Failed to start streaming: {e}")
            return False

    def stop_streaming(self):
        """Stop OBS streaming."""
        if not self.connected:
            log.error("Not connected to OBS")
            return False

        try:
            self.ws.call(obs_requests.StopStream())
            log.info("Stopped OBS streaming")
            return True
        except Exception as e:
            log.error(f"Failed to stop streaming: {e}")
            return False

    def set_scene(self, scene_name: str):
        """Switch to a specific scene."""
        if not self.connected:
            log.error("Not connected to OBS")
            return False

        try:
            self.ws.call(obs_requests.SetCurrentProgramScene(sceneName=scene_name))
            log.info(f"Switched to scene: {scene_name}")
            return True
        except Exception as e:
            log.error(f"Failed to switch scene: {e}")
            return False

    def get_scenes(self):
        """Get list of available scenes."""
        if not self.connected:
            log.error("Not connected to OBS")
            return []

        try:
            response = self.ws.call(obs_requests.GetSceneList())
            scenes = [scene['sceneName'] for scene in response.getScenes()]
            log.info(f"Available scenes: {scenes}")
            return scenes
        except Exception as e:
            log.error(f"Failed to get scenes: {e}")
            return []

    def set_source_visibility(self, scene_name: str, source_name: str, visible: bool):
        """Show or hide a source in a scene."""
        if not self.connected:
            log.error("Not connected to OBS")
            return False

        try:
            self.ws.call(obs_requests.SetSceneItemEnabled(
                sceneName=scene_name,
                sceneItemId=source_name,
                sceneItemEnabled=visible
            ))
            log.info(f"Set {source_name} visibility to {visible}")
            return True
        except Exception as e:
            log.error(f"Failed to set source visibility: {e}")
            return False

    def get_recording_status(self):
        """Get current recording status."""
        if not self.connected:
            return None

        try:
            response = self.ws.call(obs_requests.GetRecordStatus())
            return {
                'recording': response.getOutputActive(),
                'paused': response.getOutputPaused(),
                'duration': response.getOutputDuration() if hasattr(response, 'getOutputDuration') else None
            }
        except Exception as e:
            log.error(f"Failed to get recording status: {e}")
            return None

    def set_source_settings(self, source_name: str, settings: dict):
        """Update settings for a specific source."""
        if not self.connected:
            log.error("Not connected to OBS")
            return False

        try:
            self.ws.call(obs_requests.SetInputSettings(
                inputName=source_name,
                inputSettings=settings
            ))
            log.info(f"Updated settings for {source_name}")
            return True
        except Exception as e:
            log.error(f"Failed to set source settings: {e}")
            return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create controller
    obs = OBSController(password="your_password_here")

    # Connect to OBS
    obs.connect()

    # Get available scenes
    scenes = obs.get_scenes()
    print(f"Scenes: {scenes}")

    # Start recording
    obs.start_recording()

    # Wait for user input
    input("Press Enter to stop recording...")

    # Stop recording
    obs.stop_recording()

    # Disconnect
    obs.disconnect()
