"""
OBS WebSocket controller for programmatic control of OBS Studio.
Uses obsws-python (v5 protocol) for OBS 28+.
Allows the video synthesizer to control OBS recording, streaming, scenes, and source filters.
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)


class OBSController:
    """Controller for OBS Studio via WebSocket v5."""

    def __init__(self, host: str = "localhost", port: int = 4455, password: str = ""):
        self.host = host
        self.port = port
        self.password = password
        self.cl = None
        self.connected = False

    def connect(self):
        """Connect to OBS WebSocket server."""
        try:
            import obsws_python as obs
            self.cl = obs.ReqClient(host=self.host, port=self.port, password=self.password, timeout=3)
            self.connected = True

            version = self.cl.get_version()
            log.info(f"Connected to OBS at {self.host}:{self.port} (OBS {version.obs_version})")

        except Exception as e:
            log.error(f"Failed to connect to OBS: {e}")
            self.connected = False

    def disconnect(self):
        """Disconnect from OBS."""
        if self.cl and self.connected:
            try:
                self.cl.base_client.ws.close()
            except Exception:
                pass
            self.connected = False
            log.info("Disconnected from OBS")

    def start_recording(self):
        if not self.connected:
            log.error("Not connected to OBS")
            return False
        try:
            self.cl.start_record()
            log.info("Started OBS recording")
            return True
        except Exception as e:
            log.error(f"Failed to start recording: {e}")
            return False

    def stop_recording(self):
        if not self.connected:
            log.error("Not connected to OBS")
            return False
        try:
            self.cl.stop_record()
            log.info("Stopped OBS recording")
            return True
        except Exception as e:
            log.error(f"Failed to stop recording: {e}")
            return False

    def start_streaming(self):
        if not self.connected:
            log.error("Not connected to OBS")
            return False
        try:
            self.cl.start_stream()
            log.info("Started OBS streaming")
            return True
        except Exception as e:
            log.error(f"Failed to start streaming: {e}")
            return False

    def stop_streaming(self):
        if not self.connected:
            log.error("Not connected to OBS")
            return False
        try:
            self.cl.stop_stream()
            log.info("Stopped OBS streaming")
            return True
        except Exception as e:
            log.error(f"Failed to stop streaming: {e}")
            return False

    def set_scene(self, scene_name: str):
        if not self.connected:
            log.error("Not connected to OBS")
            return False
        try:
            self.cl.set_current_program_scene(scene_name)
            log.info(f"Switched to scene: {scene_name}")
            return True
        except Exception as e:
            log.error(f"Failed to switch scene: {e}")
            return False

    def get_scenes(self):
        if not self.connected:
            log.error("Not connected to OBS")
            return []
        try:
            response = self.cl.get_scene_list()
            scenes = [scene['sceneName'] for scene in response.scenes]
            return scenes
        except Exception as e:
            log.error(f"Failed to get scenes: {e}")
            return []

    def get_input_list(self):
        """Get list of all inputs (sources) in OBS."""
        if not self.connected:
            return []
        try:
            response = self.cl.get_input_list()
            return [inp['inputName'] for inp in response.inputs]
        except Exception as e:
            log.error(f"Failed to get input list: {e}")
            return []

    def get_source_filters(self, source_name: str):
        """Get list of filters on a source."""
        if not self.connected:
            return []
        try:
            response = self.cl.get_source_filter_list(source_name)
            return response.filters
        except Exception as e:
            log.error(f"Failed to get filters for '{source_name}': {e}")
            return []

    def create_filter(self, source_name: str, filter_name: str, filter_kind: str, settings: dict = None):
        """Create a new filter on a source."""
        if not self.connected:
            return False
        try:
            self.cl.create_source_filter(source_name, filter_name, filter_kind, settings or {})
            log.info(f"Created filter '{filter_name}' ({filter_kind}) on '{source_name}'")
            return True
        except Exception as e:
            log.error(f"Failed to create filter '{filter_name}' on '{source_name}': {e}")
            return False

    def set_filter_settings(self, source_name: str, filter_name: str, settings: dict, overlay: bool = True):
        """Update settings for a filter on a source."""
        if not self.connected:
            return False
        try:
            self.cl.set_source_filter_settings(source_name, filter_name, settings, overlay)
            return True
        except Exception as e:
            log.error(f"Failed to set filter settings for '{filter_name}' on '{source_name}': {e}")
            return False

    def set_filter_enabled(self, source_name: str, filter_name: str, enabled: bool):
        """Enable or disable a filter on a source."""
        if not self.connected:
            return False
        try:
            self.cl.set_source_filter_enabled(source_name, filter_name, enabled)
            return True
        except Exception as e:
            log.error(f"Failed to set filter enabled for '{filter_name}': {e}")
            return False

    def set_source_visibility(self, scene_name: str, scene_item_id: int, visible: bool):
        if not self.connected:
            return False
        try:
            self.cl.set_scene_item_enabled(scene_name, scene_item_id, visible)
            log.info(f"Set item {scene_item_id} visibility to {visible}")
            return True
        except Exception as e:
            log.error(f"Failed to set source visibility: {e}")
            return False

    def set_source_transform(self, scene_name: str, scene_item_id: int, transform: dict):
        """Set transform properties (position, rotation, scale, crop) for a scene item."""
        if not self.connected:
            return False
        try:
            self.cl.set_scene_item_transform(scene_name, scene_item_id, transform)
            return True
        except Exception as e:
            log.error(f"Failed to set source transform: {e}")
            return False

    def get_scene_item_id(self, scene_name: str, source_name: str):
        """Get the scene item ID for a source in a scene."""
        if not self.connected:
            return None
        try:
            response = self.cl.get_scene_item_id(scene_name, source_name)
            return response.scene_item_id
        except Exception as e:
            log.error(f"Failed to get scene item ID for '{source_name}': {e}")
            return None

    def get_recording_status(self):
        if not self.connected:
            return None
        try:
            response = self.cl.get_record_status()
            return {
                'recording': response.output_active,
                'paused': response.output_paused,
            }
        except Exception as e:
            log.error(f"Failed to get recording status: {e}")
            return None

    def set_source_settings(self, source_name: str, settings: dict):
        if not self.connected:
            return False
        try:
            self.cl.set_input_settings(source_name, settings, True)
            return True
        except Exception as e:
            log.error(f"Failed to set source settings: {e}")
            return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    obs = OBSController(password="your_password_here")
    obs.connect()

    scenes = obs.get_scenes()
    print(f"Scenes: {scenes}")

    inputs = obs.get_input_list()
    print(f"Inputs: {inputs}")

    obs.disconnect()
