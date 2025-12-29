import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox, QRadioButton, QScrollArea, QToolButton, QSizePolicy, QLineEdit, QTabWidget, QComboBox, QDialog
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
import logging
from param import ParamTable, Param
from config import ParentClass, SourceIndex, WidgetType

log = logging.getLogger(__name__)


class CollapsibleGroupBox(QWidget):
    def __init__(self, title="", header_color="#607D8B", parent=None): # Default grey color
        super().__init__(parent)
        self.toggle_button = QToolButton(self)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False) # Start collapsed
        self.toggle_button.setStyleSheet(f"QToolButton {{ border: none; background-color: {header_color}; color: white; padding: 5px; text-align: left; }} QToolButton::hover {{ background-color: {header_color}; }}")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.clicked.connect(self.toggle_content)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed) # Make button stretch horizontally

        self.content_area = QWidget(self)
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setSpacing(-20) # Reduced spacing between widgets
        self.content_area.setVisible(False) # Start hidden

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setContentsMargins(0, 0, 5, 5)

    def toggle_content(self):
        checked = self.toggle_button.isChecked()
        self.content_area.setVisible(checked)
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


class LFOManagerDialog(QDialog):
    def __init__(self, param, osc_bank, gui_instance, mod_button, parent=None):
        super().__init__(parent)
        self.param = param
        self.osc_bank = osc_bank
        self.gui_instance = gui_instance
        self.mod_button = mod_button # Store reference to the LFO button
        self.setWindowTitle(f"LFO for {param.name}")
        self.setWindowFlags(Qt.WindowType.Popup)

        self.layout = QVBoxLayout(self)
        self.rebuild_ui()

    def rebuild_ui(self):
        # Clear existing layout
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if self.param.linked_oscillator is None:
            link_button = QPushButton("Link New LFO")
            link_button.clicked.connect(self.link_new_lfo)
            self.layout.addWidget(link_button)
        else:
            unlink_button = QPushButton("Unlink LFO")
            unlink_button.clicked.connect(self.unlink_lfo)
            self.layout.addWidget(unlink_button)

            # Container for oscillator controls
            self.controls_container = QWidget()
            self.controls_layout = QVBoxLayout(self.controls_container)
            self.layout.addWidget(self.controls_container)
            
            osc = self.param.linked_oscillator
            # Create widgets for oscillator parameters
            for param_name in ['frequency', 'amplitude', 'phase', 'shape', 'noise_octaves', 'noise_persistence', 'noise_lacunarity', 'noise_repeat', 'noise_base']:
                if hasattr(osc, param_name):
                    param = getattr(osc, param_name)
                    widget = self.gui_instance.create_param_widget(param)
                    self.controls_layout.addWidget(widget)

    def link_new_lfo(self):
        # Create a unique name for the new oscillator
        osc_name = f"{self.param.name}"
        new_osc = self.osc_bank.add_oscillator(name=osc_name)
        new_osc.link_param(self.param)
        self.param.linked_oscillator = new_osc
        self.mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_LINKED_STYLE) # Update button style
        self.rebuild_ui()

    def unlink_lfo(self):
        if self.param.linked_oscillator:
            self.osc_bank.remove_oscillator(self.param.linked_oscillator)
            self.param.linked_oscillator.unlink_param()
            self.param.linked_oscillator = None
            self.mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_UNLINKED_STYLE) # Update button style
            self.rebuild_ui()

class PyQTGUI(QMainWindow):
    LFO_BUTTON_UNLINKED_STYLE = "QPushButton { background-color: #607D8B; color: white; }" # Default grey
    LFO_BUTTON_LINKED_STYLE = "QPushButton { background-color: #4CAF50; color: white; }" # Green

    def __init__(self, effects, layout='split', mixer=None):
        super().__init__()
        self.effects = effects
        self.mixer = mixer
        self.src_1_effects = effects[SourceIndex.SRC_1]
        self.src_2_effects = effects[SourceIndex.SRC_2]
        self.post_effects = effects[SourceIndex.POST]

        self.src_1_params = self.src_1_effects.params
        self.src_2_params = self.src_2_effects.params
        self.post_params = self.post_effects.params

        self.src_1_toggles = self.src_1_effects.toggles
        self.src_2_toggles = self.src_2_effects.toggles
        self.post_toggles = self.post_effects.toggles

        self.setWindowTitle("PyQt Control Panel")

        # Get the primary screen
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.geometry()

        # Calculate new dimensions
        control_panel_width = int(screen_geometry.width() * 0.50)
        control_panel_height = int(screen_geometry.height() * 0.80)

        # Set the geometry (x, y, width, height)
        self.setGeometry(0, 0, control_panel_width, control_panel_height)

        # Set up the main layout (root vertical layout)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.root_layout = QVBoxLayout(self.central_widget)

        if layout == 'tabbed':
            # Create a tab widget for the top section
            self.top_tab_widget = QTabWidget()
            self.root_layout.addWidget(self.top_tab_widget)

            # Left Pane (SRC_1) as a tab
            self.left_pane_widget = QWidget()
            self.layout = QVBoxLayout(self.left_pane_widget)
            self.layout.addStretch(1)
            left_scroll_area = QScrollArea()
            left_scroll_area.setWidgetResizable(True)
            left_scroll_area.setWidget(self.left_pane_widget)
            left_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.top_tab_widget.addTab(left_scroll_area, "Source 1")

            # Right Pane (SRC_2) as a tab
            self.right_pane_widget = QWidget()
            self.second_pane_layout = QVBoxLayout(self.right_pane_widget)
            self.second_pane_layout.addStretch(1)
            right_scroll_area = QScrollArea()
            right_scroll_area.setWidgetResizable(True)
            right_scroll_area.setWidget(self.right_pane_widget)
            right_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.top_tab_widget.addTab(right_scroll_area, "Source 2")

        else: # 'split' layout (default)
            self.top_horizontal_layout = QHBoxLayout()
            self.root_layout.addLayout(self.top_horizontal_layout)

            # Left Pane (SRC_1 Animations and Effects)
            self.left_pane_widget = QWidget()
            self.layout = QVBoxLayout(self.left_pane_widget) 
            self.layout.addStretch(1)

            self.left_scroll_area = QScrollArea()
            self.left_scroll_area.setWidgetResizable(True)
            self.left_scroll_area.setWidget(self.left_pane_widget)
            self.left_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.top_horizontal_layout.addWidget(self.left_scroll_area)

            # Right Pane (SRC_2 Animations and Effects)
            self.right_pane_widget = QWidget()
            self.second_pane_layout = QVBoxLayout(self.right_pane_widget)
            self.second_pane_layout.addStretch(1)

            self.right_scroll_area = QScrollArea()
            self.right_scroll_area.setWidgetResizable(True)
            self.right_scroll_area.setWidget(self.right_pane_widget)
            self.right_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.top_horizontal_layout.addWidget(self.right_scroll_area)

        # Bottom Section: Full width for Mixer, Post Effects, Uncategorized
        self.bottom_pane_widget = QWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_pane_widget)
        self.bottom_layout.addStretch(1)

        self.bottom_scroll_area = QScrollArea()
        self.bottom_scroll_area.setWidgetResizable(True)
        self.bottom_scroll_area.setWidget(self.bottom_pane_widget)
        self.bottom_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.root_layout.addWidget(self.bottom_scroll_area)

        self.create_ui() # Call create_ui after layouts are initialized


    def open_lfo_dialog(self, param, button):
        if param.parent in [ParentClass.SRC_1_ANIMATIONS, ParentClass.SRC_1_EFFECTS]:
            osc_bank = self.src_1_effects.oscs
        elif param.parent in [ParentClass.SRC_2_ANIMATIONS, ParentClass.SRC_2_EFFECTS]:
            osc_bank = self.src_2_effects.oscs
        else:
            osc_bank = self.post_effects.oscs
        
        dialog = LFOManagerDialog(param, osc_bank, self, button, self)
        
        button_pos = button.mapToGlobal(button.rect().bottomLeft())
        dialog.move(button_pos)
        
        dialog.exec()
        
    def create_ui(self):
        parent_groups = {}
        for params_collection in [self.src_1_params, self.src_2_params, self.post_params]:
            for param in params_collection.values():
                if param.parent is None:
                    parent_key = "Uncategorized"
                else:
                    parent_key = param.parent

                if not isinstance(parent_key, (str, int, float, bool, tuple, type(None), ParentClass)):
                    log.warning(f"Unhashable parent_key for param '{param.name}': type={type(parent_key)}, value={parent_key}. Assigning to 'Uncategorized'.")
                    parent_key = "Uncategorized"

                if parent_key not in parent_groups:
                    parent_groups[parent_key] = []
                parent_groups[parent_key].append(param)

        for parent_enum_or_str, params_in_group in parent_groups.items():
            if parent_enum_or_str == "Uncategorized":
                title = "Uncategorized"
                header_color = "#AAAAAA"
            else:
                title = parent_enum_or_str.name.replace("_", " ")
                header_color = parent_enum_or_str.value

            parent_collapsible_group_box = CollapsibleGroupBox(title, header_color)
            
            family_groups = {}
            for param in params_in_group:
                if param.family not in family_groups:
                    family_groups[param.family] = []
                family_groups[param.family].append(param)
            
            for family_name, params_in_family in family_groups.items():
                if family_name == "None":
                    family_title = "General"
                    family_header_color = "#888888"
                else:
                    family_title = family_name.replace("_", " ").title()
                    family_header_color = "#546E7A"

                family_collapsible_group_box = CollapsibleGroupBox(family_title, family_header_color)
                for param in params_in_family:
                    widget = self.create_param_widget(param)
                    family_collapsible_group_box.add_widget(widget)
                
                parent_collapsible_group_box.add_widget(family_collapsible_group_box)
            
            if parent_enum_or_str in [ParentClass.SRC_1_ANIMATIONS, ParentClass.SRC_1_EFFECTS]:
                self.layout.addWidget(parent_collapsible_group_box)
            elif parent_enum_or_str in [ParentClass.SRC_2_ANIMATIONS, ParentClass.SRC_2_EFFECTS]:
                self.second_pane_layout.addWidget(parent_collapsible_group_box)
            elif parent_enum_or_str in [ParentClass.MIXER, ParentClass.POST_EFFECTS] or parent_enum_or_str == "Uncategorized":
                self.bottom_layout.addWidget(parent_collapsible_group_box)

    def create_param_widget(self, param: Param):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(param.name)
        label.setFixedWidth(100)
        layout.addWidget(label)

        if param.type == WidgetType.RADIO:
            radio_button_group = QWidget()
            radio_layout = QHBoxLayout(radio_button_group)
            for option in param.options:
                radio_button = QRadioButton(str(param.options(option).name))
                radio_button.toggled.connect(lambda checked, p=param, v=option: self.on_radio_change(p, v, checked))
                if option == param.value:
                    radio_button.setChecked(True)
                radio_layout.addWidget(radio_button)
            layout.addWidget(radio_button_group)

        elif param.type == WidgetType.DROPDOWN:
            combo_box = QComboBox()
            combo_box.addItems([str(o) for o in param.options])
            combo_box.setCurrentText(str(param.value))
            combo_box.currentTextChanged.connect(lambda text, p=param: self.on_dropdown_change(p, text))
            layout.addWidget(combo_box)

        else:  # Default to SLIDER
            mod_button = QPushButton("LFO")
            mod_button.setFixedWidth(35)
            mod_button.clicked.connect(lambda: self.open_lfo_dialog(param, mod_button))
            
            # Set initial style based on linked_oscillator status
            if param.linked_oscillator:
                mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_LINKED_STYLE)
            else:
                mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_UNLINKED_STYLE)
            
            layout.addWidget(mod_button)

            slider = QSlider(Qt.Orientation.Horizontal)
            value_input = QLineEdit()
            value_input.setFixedWidth(50)

            if isinstance(param.default_val, float):
                slider.setRange(int(param.min * 1000), int(param.max * 1000))
                slider.setValue(int(param.value * 1000))
                value_input.setText(str(round(param.value, 3)))
            else:
                slider.setRange(int(param.min), int(param.max))
                slider.setValue(param.value)
                value_input.setText(str(param.value))

            slider.valueChanged.connect(lambda value, p=param: self.on_slider_change(p, value))
            layout.addWidget(slider)
            
            slider.setProperty("value_input", value_input)
            value_input.editingFinished.connect(lambda p=param, vi=value_input, s=slider: self.on_text_input_change(p, vi, s))
            layout.addWidget(value_input)

        reset_button = QPushButton("↩️")
        reset_button.setFixedWidth(25)
        reset_button.clicked.connect(lambda: self.on_reset_click(param, widget))
        layout.addWidget(reset_button)

        return widget


    def on_slider_change(self, param: Param, value):
        if isinstance(param.default_val, float):
            param.value = value / 1000.0
        else:
            param.value = value
        
        slider = self.sender()
        value_input = slider.property("value_input")
        if value_input:
            value_input.setText(str(round(param.value, 3) if isinstance(param.default_val, float) else param.value))


    def on_text_input_change(self, param: Param, value_input: QLineEdit, slider: QSlider):
        try:
            new_value_str = value_input.text()
            if isinstance(param.default_val, float):
                new_value = float(new_value_str)
            else:
                new_value = int(new_value_str)

            new_value = max(param.min, min(param.max, new_value))
            param.value = new_value

            if isinstance(param.default_val, float):
                slider.setValue(int(param.value * 1000))
            else:
                slider.setValue(param.value)
            value_input.setText(str(round(param.value, 3) if isinstance(param.default_val, float) else param.value))
        except ValueError:
            value_input.setText(str(round(param.value, 3) if isinstance(param.default_val, float) else param.value))


    def on_radio_change(self, param: Param, value, checked):
        if checked:
            param.value = int(param.options(value).value)

    
    def open_lfo_dialog(self, param, button):
        if param.parent in [ParentClass.SRC_1_ANIMATIONS, ParentClass.SRC_1_EFFECTS]:
            osc_bank = self.src_1_effects.oscs
        elif param.parent in [ParentClass.SRC_2_ANIMATIONS, ParentClass.SRC_2_EFFECTS]:
            osc_bank = self.src_2_effects.oscs
        else:
            osc_bank = self.post_effects.oscs
        
        dialog = LFOManagerDialog(param, osc_bank, self, button, self)
        
        button_pos = button.mapToGlobal(button.rect().bottomLeft())
        dialog.move(button_pos)
        
        dialog.exec()
        
    def create_ui(self):
        parent_groups = {}
        for params_collection in [self.src_1_params, self.src_2_params, self.post_params]:
            for param in params_collection.values():
                if param.parent is None:
                    parent_key = "Uncategorized"
                else:
                    parent_key = param.parent

                if not isinstance(parent_key, (str, int, float, bool, tuple, type(None), ParentClass)):
                    log.warning(f"Unhashable parent_key for param '{param.name}': type={type(parent_key)}, value={parent_key}. Assigning to 'Uncategorized'.")
                    parent_key = "Uncategorized"

                if parent_key not in parent_groups:
                    parent_groups[parent_key] = []
                parent_groups[parent_key].append(param)

        for parent_enum_or_str, params_in_group in parent_groups.items():
            if parent_enum_or_str == "Uncategorized":
                title = "Uncategorized"
                header_color = "#AAAAAA"
            else:
                title = parent_enum_or_str.name.replace("_", " ")
                header_color = parent_enum_or_str.value

            parent_collapsible_group_box = CollapsibleGroupBox(title, header_color)
            
            family_groups = {}
            for param in params_in_group:
                if param.family not in family_groups:
                    family_groups[param.family] = []
                family_groups[param.family].append(param)
            
            for family_name, params_in_family in family_groups.items():
                if family_name == "None":
                    family_title = "General"
                    family_header_color = "#888888"
                else:
                    family_title = family_name.replace("_", " ").title()
                    family_header_color = "#546E7A"

                family_collapsible_group_box = CollapsibleGroupBox(family_title, family_header_color)
                for param in params_in_family:
                    widget = self.create_param_widget(param)
                    family_collapsible_group_box.add_widget(widget)
                
                parent_collapsible_group_box.add_widget(family_collapsible_group_box)
            
            if parent_enum_or_str in [ParentClass.SRC_1_ANIMATIONS, ParentClass.SRC_1_EFFECTS]:
                self.layout.addWidget(parent_collapsible_group_box)
            elif parent_enum_or_str in [ParentClass.SRC_2_ANIMATIONS, ParentClass.SRC_2_EFFECTS]:
                self.second_pane_layout.addWidget(parent_collapsible_group_box)
            elif parent_enum_or_str in [ParentClass.MIXER, ParentClass.POST_EFFECTS] or parent_enum_or_str == "Uncategorized":
                self.bottom_layout.addWidget(parent_collapsible_group_box)

    def create_param_widget(self, param: Param):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(param.name)
        label.setFixedWidth(100)
        layout.addWidget(label)

        if param.type == WidgetType.RADIO:
            radio_button_group = QWidget()
            radio_layout = QHBoxLayout(radio_button_group)
            for option in param.options:
                radio_button = QRadioButton(str(param.options(option).name))
                radio_button.toggled.connect(lambda checked, p=param, v=option: self.on_radio_change(p, v, checked))
                if option == param.value:
                    radio_button.setChecked(True)
                radio_layout.addWidget(radio_button)
            layout.addWidget(radio_button_group)

        elif param.type == WidgetType.DROPDOWN:
            combo_box = QComboBox()
            combo_box.addItems([str(o) for o in param.options])
            combo_box.setCurrentText(str(param.value))
            combo_box.currentTextChanged.connect(lambda text, p=param: self.on_dropdown_change(p, text))
            layout.addWidget(combo_box)

        else:  # Default to SLIDER
            mod_button = QPushButton("LFO")
            mod_button.setFixedWidth(35)
            mod_button.clicked.connect(lambda: self.open_lfo_dialog(param, mod_button))
            
            # Set initial style based on linked_oscillator status
            if param.linked_oscillator:
                mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_LINKED_STYLE)
            else:
                mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_UNLINKED_STYLE)
            
            layout.addWidget(mod_button)

            slider = QSlider(Qt.Orientation.Horizontal)
            value_input = QLineEdit()
            value_input.setFixedWidth(50)

            if isinstance(param.default_val, float):
                slider.setRange(int(param.min * 1000), int(param.max * 1000))
                slider.setValue(int(param.value * 1000))
                value_input.setText(str(round(param.value, 3)))
            else:
                slider.setRange(int(param.min), int(param.max))
                slider.setValue(param.value)
                value_input.setText(str(param.value))

            slider.valueChanged.connect(lambda value, p=param: self.on_slider_change(p, value))
            layout.addWidget(slider)
            
            slider.setProperty("value_input", value_input)
            value_input.editingFinished.connect(lambda p=param, vi=value_input, s=slider: self.on_text_input_change(p, vi, s))
            layout.addWidget(value_input)

        reset_button = QPushButton("↩️")
        reset_button.setFixedWidth(25)
        reset_button.clicked.connect(lambda: self.on_reset_click(param, widget))
        layout.addWidget(reset_button)

        return widget

    def on_slider_change(self, param: Param, value):
        if isinstance(param.default_val, float):
            param.value = value / 1000.0
        else:
            param.value = value
        
        slider = self.sender()
        value_input = slider.property("value_input")
        if value_input:
            value_input.setText(str(round(param.value, 3) if isinstance(param.default_val, float) else param.value))

    def on_text_input_change(self, param: Param, value_input: QLineEdit, slider: QSlider):
        try:
            new_value_str = value_input.text()
            if isinstance(param.default_val, float):
                new_value = float(new_value_str)
            else:
                new_value = int(new_value_str)

            new_value = max(param.min, min(param.max, new_value))
            param.value = new_value

            if isinstance(param.default_val, float):
                slider.setValue(int(param.value * 1000))
            else:
                slider.setValue(param.value)
            value_input.setText(str(round(param.value, 3) if isinstance(param.default_val, float) else param.value))
        except ValueError:
            value_input.setText(str(round(param.value, 3) if isinstance(param.default_val, float) else param.value))

    def on_radio_change(self, param: Param, value, checked):
        if checked:
            param.value = int(param.options(value).value)

    def on_dropdown_change(self, param: Param, text_value):
        try:
            if isinstance(param.default_val, int) and not isinstance(text_value, str):
                value = int(text_value)
            elif isinstance(param.default_val, float):
                value = float(text_value)
            else:
                value = text_value
            param.value = value

            if self.mixer:
                if param.name == "source_1":
                    self.mixer.start_video(value, SourceIndex.SRC_1)
                elif param.name == "source_2":
                    self.mixer.start_video(value, SourceIndex.SRC_2)
        except (ValueError, TypeError):
            pass # Ignore if conversion fails

    def on_reset_click(self, param: Param, widget: QWidget):
        param.reset()
        slider = widget.findChild(QSlider)
        if slider:
            if isinstance(param.default_val, float):
                slider.setValue(int(param.value * 1000))
            else:
                slider.setValue(param.value)
        
        value_input = widget.findChild(QLineEdit)
        if value_input:
            value_input.setText(str(round(param.value, 3) if isinstance(param.default_val, float) else param.value))

        radio_buttons = widget.findChildren(QRadioButton)
        if radio_buttons:
            for rb in radio_buttons:
                if rb.text() == str(param.value):
                    rb.setChecked(True)
                    break
        
        combo_box = widget.findChild(QComboBox)
        if combo_box:
            combo_box.setCurrentText(str(param.value))


    def closeEvent(self, event):
        """Handle the window closing event."""
        # print("Closing PyQt window.")
        QApplication.instance().quit()
        event.accept()

def create_pyqt_gui(src_1_effects, src_2_effects, post_effects):
    main_window = PyQTGUI(src_1_effects, src_2_effects, post_effects)
    return main_window
