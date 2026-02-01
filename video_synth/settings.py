from param import Param, ParamTable
from common import ParentClass, WidgetType, LayoutType, OutputMode

class UserSettings():
    def __init__(self, **kwargs):
        self.params = ParamTable(parent=ParentClass.SETTINGS)
        print(kwargs)
        self.layout = self.params.add("layout", 0, 10, LayoutType[kwargs['control_layout']].value, subclass="Settings", parent=ParentClass.SETTINGS)
        self.output_mode = self.params.add(
            "output_mode", 0, 2, OutputMode[kwargs['output_mode']].value, subclass="Settings", parent=ParentClass.SETTINGS,
            type=WidgetType.DROPDOWN,
            options={
                0: "No Output",
                1: "Windowed Output",
                2: "Fullscreen Output"
            }
        )
        self.num_devices = self.params.add(
            "num_devices", 1, 10, kwargs['devices'], subclass="Settings", parent=ParentClass.SETTINGS,
            type=WidgetType.DROPDOWN,
            options={i: str(i) for i in range(1, 11)}
        )
        self.patch_index = self.params.add(
            "patch_index", 0, 100, kwargs['patch'], subclass="Settings", parent=ParentClass.SETTINGS
        )

        # self.

