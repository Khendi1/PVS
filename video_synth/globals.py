from effects import EffectManager

"""
This module stores shared objects used across the application.

Now classes with modifyable params are expected to define a create_gui_panel function and call it gui.py 
gui panels within
"""

effects = EffectManager() # initialized with params in main.py

