from param import Param, ParamTable
from common import *


class UserSettings():
    def __init__(self, control_layout: str, output_mode: str, devices: int, patch: int, log_level: str, file: str, diagnose: int):
        """Initialize user settings with given parameters."""
        group = Groups.USER_SETTINGS
        subgroup = group.name
        self.params = ParamTable(group=group)

        self.layout = self.params.add("layout",
                                       min=0, max=len(Layout), default=Layout[control_layout].value,
                                       group=group, subgroup=subgroup,
                                       type=Widget.SLIDER)
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
        
        