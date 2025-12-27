import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox, QRadioButton, QScrollArea, QToolButton, QSizePolicy
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
import logging
from param import ParamTable, Param
from config import ParentClass, SourceIndex # Import the title from config

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
        self.content_layout.setContentsMargins(5, 0, 5, 5)

    def toggle_content(self):
        checked = self.toggle_button.isChecked()
        self.content_area.setVisible(checked)
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


class PyQTGUI(QMainWindow):
    def __init__(self, effects):
        super().__init__()

        self.src_1_params = effects[SourceIndex.SRC_1].params
        self.src_2_params = effects[SourceIndex.SRC_2].params
        self.post_params = effects[SourceIndex.POST].params

        self.src_1_toggles = effects[SourceIndex.SRC_1].toggles
        self.src_2_toggles = effects[SourceIndex.SRC_2].toggles
        self.post_toggles = effects[SourceIndex.POST].toggles

        self.setWindowTitle("PyQt Control Panel")

        # Top Section: Two horizontal panes
        # Set up the main layout (root vertical layout)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.root_layout = QVBoxLayout(self.central_widget)

        self.top_horizontal_layout = QHBoxLayout()
        self.root_layout.addLayout(self.top_horizontal_layout)

        # Left Pane (SRC_1 Animations and Effects)
        self.left_pane_widget = QWidget()
        self.layout = QVBoxLayout(self.left_pane_widget) # This will be the layout for the left scroll area
        self.layout.addStretch(1) # Add stretch to push content to the top

        self.left_scroll_area = QScrollArea()
        self.left_scroll_area.setWidgetResizable(True)
        self.left_scroll_area.setWidget(self.left_pane_widget)
        self.left_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.top_horizontal_layout.addWidget(self.left_scroll_area)

        # Right Pane (SRC_2 Animations and Effects)
        self.right_pane_widget = QWidget()
        self.second_pane_layout = QVBoxLayout(self.right_pane_widget) # This will be the layout for the right scroll area
        self.second_pane_layout.addStretch(1) # Add stretch to push content to the top

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

    def create_ui(self):
        parent_groups = {}
        print("--- DEBUG: create_ui ---")
        for params in [self.src_1_params, self.src_2_params, self.post_params]:
            print(f"DEBUG: Params group: {[param.name for param in params.values()]}")
        for params in [self.src_1_params.values(), self.src_2_params.values(), self.post_params.values()]:
            for param in params:
                # print(f"DEBUG: Processing param: {param}, param.parent: {param.parent}")
                if param.parent is None:
                    parent_key = "Uncategorized"
                else:
                    parent_key = param.parent # param.parent is a ParentClass enum member

                if parent_key not in parent_groups:
                    parent_groups[parent_key] = []
                parent_groups[parent_key].append(param)
        print(f"DEBUG: parent_groups: {parent_groups.keys()}")

        for parent_enum_or_str, params_in_group in parent_groups.items():
            if parent_enum_or_str == "Uncategorized":
                title = "Uncategorized"
                header_color = "#AAAAAA" # Grey for uncategorized
            else:
                title = parent_enum_or_str.name.replace("_", " ") # Convert enum name to readable title
                header_color = parent_enum_or_str.value # Get hex color from enum value

            parent_collapsible_group_box = CollapsibleGroupBox(title, header_color)
            
            # Group parameters by family within this parent group
            family_groups = {}
            for param in params_in_group:
                if param.family not in family_groups:
                    family_groups[param.family] = []
                family_groups[param.family].append(param)
            
            for family_name, params_in_family in family_groups.items():
                if family_name == "None": # Handle parameters without an explicit family
                    family_title = "General"
                    family_header_color = "#888888" # Slightly darker grey for general family
                else:
                    family_title = family_name.replace("_", " ").title() # Convert family name to readable title
                    family_header_color = "#546E7A" # A slightly different grey-blue for nested groups

                family_collapsible_group_box = CollapsibleGroupBox(family_title, family_header_color)
                for param in params_in_family:
                    widget = self.create_param_widget(param)
                    family_collapsible_group_box.add_widget(widget)
                
                parent_collapsible_group_box.add_widget(family_collapsible_group_box) # Add family box to parent box
            
            # Pane placement logic
            if parent_enum_or_str in [ParentClass.SRC_1_ANIMATIONS, ParentClass.SRC_1_EFFECTS]:
                self.layout.addWidget(parent_collapsible_group_box) # Top-Left pane
                print(f"DEBUG: Adding '{title}' to LEFT pane.")
            elif parent_enum_or_str in [ParentClass.SRC_2_ANIMATIONS, ParentClass.SRC_2_EFFECTS]:
                self.second_pane_layout.addWidget(parent_collapsible_group_box) # Top-Right pane
                print(f"DEBUG: Adding '{title}' to RIGHT pane.")
            elif parent_enum_or_str in [ParentClass.MIXER, ParentClass.POST_EFFECTS] or parent_enum_or_str == "Uncategorized":
                self.bottom_layout.addWidget(parent_collapsible_group_box) # Bottom full-width pane
                print(f"DEBUG: Adding '{title}' to BOTTOM pane.")


    def create_param_widget(self, param: Param):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0) # Remove margins for individual parameter layout

        # Label
        label = QLabel(param.name)
        label.setFixedWidth(150)
        layout.addWidget(label)

        # Heuristic for widget type
        if isinstance(param.default_val, int) and param.max - param.min <= 0:
            # Radio buttons for small integer ranges
            radio_button_group = QWidget()
            radio_layout = QHBoxLayout(radio_button_group)
            for i in range(int(param.min), int(param.max) + 1):
                radio_button = QRadioButton(str(i))
                radio_button.toggled.connect(lambda checked, p=param, v=i: self.on_radio_change(p, v, checked))
                if i == param.value:
                    radio_button.setChecked(True)
                radio_layout.addWidget(radio_button)
            layout.addWidget(radio_button_group)
        else:
            # Slider for other types
            slider = QSlider(Qt.Orientation.Horizontal)
            value_label = QLabel(str(param.value))
            if isinstance(param.default_val, float):
                # Scale float to int for slider
                slider.setRange(int(param.min * 1000), int(param.max * 1000))
                slider.setValue(int(param.value * 1000))
            else:
                slider.setRange(int(param.min), int(param.max))
                slider.setValue(param.value)
            
            slider.valueChanged.connect(lambda value, p=param: self.on_slider_change(p, value))
            layout.addWidget(slider)
            
            # Value Label
            slider.setProperty("value_label", value_label) # Store reference
            layout.addWidget(value_label)


        # Reset Button
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(lambda: self.on_reset_click(param, widget)) # pass widget to find control
        layout.addWidget(reset_button)

        return widget

    def on_slider_change(self, param: Param, value):
        if isinstance(param.default_val, float):
            param.value = value / 1000.0
        else:
            param.value = value
        
        # Find the associated value label and update it
        slider = self.sender()
        value_label = slider.property("value_label")
        if value_label:
            value_label.setText(str(round(param.value, 3)))

    def on_radio_change(self, param: Param, value, checked):
        if checked:
            param.value = value

    def on_reset_click(self, param: Param, widget: QWidget):
        param.reset()
        # Find the control and reset it
        slider = widget.findChild(QSlider)
        if slider:
            if isinstance(param.default_val, float):
                slider.setValue(int(param.value * 1000))
            else:
                slider.setValue(param.value)
        
        radio_buttons = widget.findChildren(QRadioButton)
        if radio_buttons:
            for rb in radio_buttons:
                if int(rb.text()) == param.value:
                    rb.setChecked(True)
                    break

    def closeEvent(self, event):
        """Handle the window closing event."""
        # print("Closing PyQt window.")
        QApplication.instance().quit()
        event.accept()

def create_pyqt_gui(src_1_effects, src_2_effects, post_effects):
    main_window = PyQTGUI(src_1_effects, src_2_effects, post_effects)
    return main_window
