from param import Param, ParamTable
from common import *


class UserSettings():
    def __init__(self, control_layout: str, output_mode: str, devices: int, patch: int, log_level: str, file: str, diagnose: int,
                 api: bool = False, api_host: str = "127.0.0.1", api_port: int = 8000,
                 ffmpeg: bool = False, ffmpeg_output: str = "output.mp4", ffmpeg_preset: str = "medium", ffmpeg_crf: int = 23,
                 no_virtualcam: bool = False,
                 headless: bool = False,
                 obs: bool = False, obs_host: str = "localhost", obs_port: int = 4455, obs_password: str = "",
                 **kwargs):
        """Initialize user settings with given parameters."""
        group = Groups.USER_SETTINGS
        subgroup = group.name
        self.params = ParamTable(group=group)

        # API settings (not exposed in GUI)
        self.api = api
        self.api_host = api_host
        self.api_port = api_port

        # FFmpeg settings (not exposed in GUI)
        self.ffmpeg = ffmpeg
        self.ffmpeg_output = ffmpeg_output
        self.ffmpeg_preset = ffmpeg_preset
        self.ffmpeg_crf = ffmpeg_crf

        # Virtual camera output (enabled by default, --no-virtualcam to disable)
        self.no_virtualcam = no_virtualcam

        # Headless mode (not exposed in GUI)
        self.headless = headless

        # OBS WebSocket settings (not exposed in GUI)
        self.obs = obs
        self.obs_host = obs_host
        self.obs_port = obs_port
        self.obs_password = obs_password

        self.layout = self.params.add("layout",
                                       min=0, max=len(Layout), default=Layout[control_layout].value,
                                       group=group, subgroup=subgroup,
                                       type=Widget.DROPDOWN, options=Layout)
        self.output_mode = self.params.add("output_mode",
                                           min=0, max=len(OutputMode), default=OutputMode[output_mode].value,
                                           group=group, subgroup=subgroup,
                                           type=Widget.DROPDOWN, options=OutputMode)
        self.num_devices = self.params.add("num_devices",
                                           min=1, max=MAX_DEVICES, default=devices,
                                           group=group, subgroup=subgroup,
                                           type=Widget.DROPDOWN, options={str(i): str(i) for i in range(1, MAX_DEVICES + 1)})
        self.patch_index = self.params.add("patch_index",
                                           min=0, max=100, default=patch,
                                           group=group, subgroup=subgroup,
                                           type=Widget.SLIDER)
        self.log_level = self.params.add("log_level",
                                         min=0, max=len(logging._nameToLevel), default=log_level,
                                         group=group, subgroup=subgroup,
                                         type=Widget.DROPDOWN, options=logging._nameToLevel)
        self.save_file = self.params.add("save_file",
                                         min=0, max=0, default=0,
                                         group=group, subgroup=subgroup,
                                         type=Widget.DROPDOWN, options={"saved_values.yaml": "saved_values.yaml", "alternate_values.yaml": "alternate_values.yaml"})
        self.diagnose_frames = self.params.add("diagnose_frames",
                                               min=0, max=1000, default=diagnose,
                                               group=group, subgroup=subgroup)
        self.api_enabled = self.params.add("api_enabled",
                                           min=0, max=1, default=int(api),
                                           group=group, subgroup=subgroup,
                                           type=Widget.TOGGLE)

        