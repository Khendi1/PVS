from param import Param, ParamTable
from config import ParentClass, WidgetType

class UserSettings():
    def __init__(self, params):
        self.view = params.add("view", 0, 10, 0, subclass="Settings", parent=ParentClass.SETTINGS)
