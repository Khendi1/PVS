import enum
import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox, QRadioButton, QScrollArea, QToolButton, QSizePolicy, QLineEdit, QTabWidget, QComboBox, QDialog, QGridLayout, QColorDialog
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, pyqtSlot, QTimer
import logging
from param import ParamTable, Param
from common import ParentClass, SourceIndex, WidgetType, LayoutType
from mix import MixModes # Import MixModes
from save import SaveController


log = logging.getLogger(__name__)


"""
Custom color picker widget for selecting HSV colors.
"""
class ColorPickerWidget(QWidget):
    def __init__(self, name, h_param, s_param, v_param, parent=None):
        super().__init__(parent)
        self.h_param = h_param
        self.s_param = s_param
        self.v_param = v_param
        self.name = name

        self.layout = QHBoxLayout(self)
        self.label = QLabel(self.name)
        self.label.setFixedWidth(100)
        self.color_button = QPushButton()
        self.color_button.setFixedWidth(100)
        self.color_button.clicked.connect(self.open_color_dialog)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.color_button)
        self.update_button_color()

    def open_color_dialog(self):
        color = QColorDialog.getColor(self.get_current_color(), self, "Select Color")
        if color.isValid():
            self.h_param.value = color.hue() / 2 
            self.s_param.value = color.saturation()
            self.v_param.value = color.value()
            self.update_button_color()

    def get_current_color(self):
        return QColor.fromHsv(int(self.h_param.value * 2), self.s_param.value, self.v_param.value)

    def update_button_color(self):
        color = self.get_current_color()
        self.color_button.setStyleSheet(f"background-color: {color.name()}")

"""
Widget to display video frames with aspect ratio preservation.
"""
class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._pixmap = None

    @pyqtSlot(QImage)
    def set_image(self, image):
        self._pixmap = QPixmap.fromImage(image)
        self.setPixmap(self._pixmap)
        self.updateGeometry()

    def hasHeightForWidth(self):
        return self._pixmap is not None

    def heightForWidth(self, width: int) -> int:
        if self._pixmap:
            if self._pixmap.width() == 0:
                return self.height()
            return int(width * (self._pixmap.height() / self._pixmap.width()))
        return self.height()

    def paintEvent(self, event):
        if self._pixmap:
            pm = self._pixmap
            widget_size = self.size()
            
            scaled_pixmap = pm.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            x = (widget_size.width() - scaled_pixmap.width()) / 2
            y = (widget_size.height() - scaled_pixmap.height()) / 2

            painter = QPainter(self)
            painter.drawPixmap(int(x), int(y), scaled_pixmap)
        else:
            super().paintEvent(event)


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
            for param_name in ['shape', 'frequency', 'amplitude', 'phase', 'noise_octaves', 'noise_persistence', 'noise_lacunarity', 'noise_repeat', 'noise_base']:
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
    video_frame_ready = pyqtSignal(QImage)

    def __init__(self, effects, layout, mixer=None):
        super().__init__()
        self.layout_style = layout
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

        # TODO: test if this works
        self.all_params = ParamTable()
        self.all_params.params.update(self.src_1_params)
        self.all_params.params.update(self.src_2_params)
        self.all_params.params.update(self.post_params)
        self.save_controller = SaveController(self.all_params)
        self.save_controller.patch_loaded_callback = self.refresh_all_widgets

        self.mixer_widgets = {}
        self.param_widgets = {}

        self.osc_banks = [(effect_manager.oscs, effect_manager.parent.name) for effect_manager in effects]

        self.setWindowTitle("PyQt Control Panel")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.showMaximized()

        self._create_layout()
        self.create_ui()

        self.lfo_refresh_timer = QTimer(self)
        self.lfo_refresh_timer.timeout.connect(self.refresh_lfo_buttons)
        self.lfo_refresh_timer.start(250)
        
    def refresh_lfo_buttons(self):
        self.all_params = ParamTable()
        self.all_params.params.update(self.src_1_effects.params)
        self.all_params.params.update(self.src_2_effects.params)
        self.all_params.params.update(self.post_effects.params)
        for param_name, widget in self.param_widgets.items():
            param = self.all_params.get(param_name)
            if param:
                mod_button = widget.findChild(QPushButton)
                if mod_button and mod_button.text() == "LFO":
                    if param.linked_oscillator:
                        mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_LINKED_STYLE)
                    else:
                        mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_UNLINKED_STYLE)

    def _create_layout(self):
        print(f"Creating layout: {self.layout_style}")
        if self.layout_style == LayoutType.QUAD_PREVIEW.value:
            self._create_quad_layout()
        elif self.layout_style == LayoutType.SPLIT.value:
            self._create_tabbed_layout()

    def _create_quad_layout(self):
        self.root_layout = QGridLayout(self.central_widget)
        self.root_layout.setContentsMargins(0, 0, 0, 0) # No margins around the grid
        self.root_layout.setSpacing(0) # No space between cells in the grid

        # --- Left Column (as a single widget spanning two rows) ---
        left_column_widget = QWidget()
        left_column_layout = QVBoxLayout(left_column_widget)
        
        # Top-Left Tabs
        top_left_tabs = QTabWidget()
        left_column_layout.addWidget(top_left_tabs, 1) # 50% height

        # Src 1 Effects Tab
        src1_effects_container = QWidget()
        self.src1_effects_layout = QVBoxLayout(src1_effects_container)
        src1_effects_scroll = QScrollArea()
        src1_effects_scroll.setWidgetResizable(True)
        src1_effects_scroll.setWidget(src1_effects_container)
        top_left_tabs.addTab(src1_effects_scroll, "Src 1 Effects")

        # Src 1 Animations Tab
        src1_animations_container = QWidget()
        self.src1_animations_layout = QVBoxLayout(src1_animations_container)
        src1_animations_scroll = QScrollArea()
        src1_animations_scroll.setWidgetResizable(True)
        src1_animations_scroll.setWidget(src1_animations_container)
        top_left_tabs.addTab(src1_animations_scroll, "Src 1 Animations")

        # Src 2 Effects Tab
        src2_effects_container = QWidget()
        self.src2_effects_layout = QVBoxLayout(src2_effects_container)
        src2_effects_scroll = QScrollArea()
        src2_effects_scroll.setWidgetResizable(True)
        src2_effects_scroll.setWidget(src2_effects_container)
        top_left_tabs.addTab(src2_effects_scroll, "Src 2 Effects")

        # Src 2 Animations Tab
        src2_animations_container = QWidget()
        self.src2_animations_layout = QVBoxLayout(src2_animations_container)
        src2_animations_scroll = QScrollArea()
        src2_animations_scroll.setWidgetResizable(True)
        src2_animations_scroll.setWidget(src2_animations_container)
        top_left_tabs.addTab(src2_animations_scroll, "Src 2 Animations")

        # Bottom-Left Tabs
        bottom_left_tabs = QTabWidget()
        left_column_layout.addWidget(bottom_left_tabs, 1) # 50% height
        self.post_effects_container = QWidget() # Make it an attribute
        self.post_effects_layout = QVBoxLayout(self.post_effects_container)
        bottom_left_tabs.addTab(self.post_effects_container, "Post Effects") # Directly add the container

        uncategorized_container = QWidget()
        self.uncategorized_layout = QVBoxLayout(uncategorized_container)
        uncategorized_scroll = QScrollArea()
        uncategorized_scroll.setWidgetResizable(True)
        uncategorized_scroll.setWidget(uncategorized_container)
        bottom_left_tabs.addTab(uncategorized_scroll, "Uncategorized")

        self.root_layout.addWidget(left_column_widget, 0, 0, 2, 1)

        # --- Right Column ---
        # Top-Right: Video with Buttons
        top_right_container = QWidget()
        top_right_layout = QHBoxLayout(top_right_container)
        top_right_layout.setContentsMargins(0, 0, 0, 0)
        top_right_layout.setSpacing(0)

        button_column_layout = QVBoxLayout()
        
        button_r1 = QPushButton("R1")
        button_r1.setFixedWidth(30)
        button_r1.setToolTip("Reset Source 1 Params")
        button_r1.clicked.connect(self.reset_src1_params)
        button_column_layout.addWidget(button_r1)

        button_r2 = QPushButton("R2")
        button_r2.setFixedWidth(30)
        button_r2.setToolTip("Reset Source 2 Params")
        button_r2.clicked.connect(self.reset_src2_params)
        button_column_layout.addWidget(button_r2)

        button_rp = QPushButton("RP")
        button_rp.setFixedWidth(30)
        button_rp.setToolTip("Reset Post-Processing Params")
        button_rp.clicked.connect(self.reset_post_params)
        button_column_layout.addWidget(button_rp)

        button_ra = QPushButton("RA")
        button_ra.setFixedWidth(30)
        button_ra.setToolTip("Reset All Params")
        button_ra.clicked.connect(self.reset_all_params)
        button_column_layout.addWidget(button_ra)

        button_column_layout.addStretch(1) # Pushes buttons to the top
        top_right_layout.addLayout(button_column_layout)

        self.video_widget = VideoWidget()
        top_right_layout.addWidget(self.video_widget, 1) # Add stretch to video widget
        self.video_frame_ready.connect(self.video_widget.set_image)
        
        self.root_layout.addWidget(top_right_container, 0, 1)

        # Bottom-Right: Tabs for Mixer and patch recall buttons
        bottom_right_tabs = QTabWidget()
        
        # Mixer Tab
        mixer_container = QWidget()
        self.mixer_layout = QVBoxLayout(mixer_container)
        mixer_scroll = QScrollArea()
        mixer_scroll.setWidgetResizable(True)
        mixer_scroll.setWidget(mixer_container)
        bottom_right_tabs.addTab(mixer_scroll, "Mixer")

        self.root_layout.addWidget(bottom_right_tabs, 1, 1)
        
        # --- Set Stretch Factors ---
        self.root_layout.setColumnStretch(0, 1)
        self.root_layout.setColumnStretch(1, 1)


    def _create_tabbed_layout(self):
        self.root_layout = QVBoxLayout(self.central_widget)

        # Top Pane
        top_pane_tabs = QTabWidget()
        self.root_layout.addWidget(top_pane_tabs)

        # Src 1 Effects Tab
        src1_effects_container = QWidget()
        self.src1_effects_layout = QVBoxLayout(src1_effects_container)
        src1_effects_scroll = QScrollArea()
        src1_effects_scroll.setWidgetResizable(True)
        src1_effects_scroll.setWidget(src1_effects_container)
        top_pane_tabs.addTab(src1_effects_scroll, "Src 1 Effects")

        # Src 1 Animations Tab
        src1_animations_container = QWidget()
        self.src1_animations_layout = QVBoxLayout(src1_animations_container)
        src1_animations_scroll = QScrollArea()
        src1_animations_scroll.setWidgetResizable(True)
        src1_animations_scroll.setWidget(src1_animations_container)
        top_pane_tabs.addTab(src1_animations_scroll, "Src 1 Animations")

        # Src 2 Effects Tab
        src2_effects_container = QWidget()
        self.src2_effects_layout = QVBoxLayout(src2_effects_container)
        src2_effects_scroll = QScrollArea()
        src2_effects_scroll.setWidgetResizable(True)
        src2_effects_scroll.setWidget(src2_effects_container)
        top_pane_tabs.addTab(src2_effects_scroll, "Src 2 Effects")

        # Src 2 Animations Tab
        src2_animations_container = QWidget()
        self.src2_animations_layout = QVBoxLayout(src2_animations_container)
        src2_animations_scroll = QScrollArea()
        src2_animations_scroll.setWidgetResizable(True)
        src2_animations_scroll.setWidget(src2_animations_container)
        top_pane_tabs.addTab(src2_animations_scroll, "Src 2 Animations")

        # Bottom Pane
        bottom_pane_tabs = QTabWidget()
        self.root_layout.addWidget(bottom_pane_tabs)

        # Mixer Tab
        mixer_container = QWidget()
        self.mixer_layout = QVBoxLayout(mixer_container)
        mixer_scroll = QScrollArea()
        mixer_scroll.setWidgetResizable(True)
        mixer_scroll.setWidget(mixer_container)
        bottom_pane_tabs.addTab(mixer_scroll, "Mixer")

        # Post Effects Tab
        self.post_effects_container = QWidget()
        self.post_effects_layout = QVBoxLayout(self.post_effects_container)
        bottom_pane_tabs.addTab(self.post_effects_container, "Post Effects")

        # Uncategorized Tab
        uncategorized_container = QWidget()
        self.uncategorized_layout = QVBoxLayout(uncategorized_container)
        uncategorized_scroll = QScrollArea()
        uncategorized_scroll.setWidgetResizable(True)
        uncategorized_scroll.setWidget(uncategorized_container)
        bottom_pane_tabs.addTab(uncategorized_scroll, "Uncategorized")


    def _mixer_tab(self, parent_groups, layout_map):
        params_in_group = parent_groups.pop(ParentClass.MIXER)
        target_layout = layout_map[ParentClass.MIXER]

        save_button = QPushButton("Save Patch")
        save_button.clicked.connect(self.save_controller.save_patch)
        target_layout.addWidget(save_button)

        patch_recall_layout = QHBoxLayout()
        load_prev_button = QPushButton("Load Previous Patch")
        load_prev_button.clicked.connect(self.save_controller.load_prev_patch)
        patch_recall_layout.addWidget(load_prev_button)

        load_random_button = QPushButton("Load Random Patch")
        load_random_button.clicked.connect(self.save_controller.load_random_patch)
        patch_recall_layout.addWidget(load_random_button)

        load_next_button = QPushButton("Load Next Patch")
        load_next_button.clicked.connect(self.save_controller.load_next_patch)
        patch_recall_layout.addWidget(load_next_button)
        
        target_layout.addLayout(patch_recall_layout)

        family_groups = {}
        for param in params_in_group:
            if param.family not in family_groups:
                family_groups[param.family] = []
            family_groups[param.family].append(param)
        
        for family_name, params_in_family in family_groups.items():

            blend_mode_param, chroma_key_hue_sat_val_params, source_params = None, [], {}
            
            all_mixer_family_params = list(params_in_family)
            params_in_family = []
            
            for param in all_mixer_family_params:
                if param.name == 'blend_mode':
                    blend_mode_param = param
                elif param.name in ['upper_hue', 'upper_sat', 'upper_val', 'lower_hue', 'lower_sat', 'lower_val']:
                    chroma_key_hue_sat_val_params.append(param)
                elif param.name in ['source_1', 'source_2']:
                    source_params[param.name] = param
                else:
                    params_in_family.append(param)
            
            if len(source_params) == 2:
                source_layout = QHBoxLayout()
                
                s1_param = source_params['source_1']
                s1_label = QLabel(s1_param.name)
                s1_combo = QComboBox()
                s1_combo.addItems([str(o) for o in s1_param.options])
                s1_combo.setCurrentText(str(s1_param.value))
                s1_combo.currentTextChanged.connect(lambda text, p=s1_param: self.on_dropdown_change(p, text))
                source_layout.addWidget(s1_label)
                source_layout.addWidget(s1_combo)

                s2_param = source_params['source_2']
                s2_label = QLabel(s2_param.name)
                s2_combo = QComboBox()
                s2_combo.addItems([str(o) for o in s2_param.options])
                s2_combo.setCurrentText(str(s2_param.value))
                s2_combo.currentTextChanged.connect(lambda text, p=s2_param: self.on_dropdown_change(p, text))
                source_layout.addWidget(s2_label)
                source_layout.addWidget(s2_combo)

                swap_button = QPushButton("Swap")
                swap_button.clicked.connect(self.mixer.swap.toggle)
                source_layout.addWidget(swap_button)
                
                target_layout.addLayout(source_layout)
            
            if blend_mode_param:
                widget = self.create_param_widget(blend_mode_param)
                target_layout.addWidget(widget)

            if len(chroma_key_hue_sat_val_params) == 6:
                chroma_key_dict = {p.name: p for p in chroma_key_hue_sat_val_params}
                color_pickers_layout = QHBoxLayout()
                
                upper_picker = ColorPickerWidget('Upper Color', chroma_key_dict['upper_hue'], chroma_key_dict['upper_sat'], chroma_key_dict['upper_val'])
                lower_picker = ColorPickerWidget('Lower Color', chroma_key_dict['lower_hue'], chroma_key_dict['lower_sat'], chroma_key_dict['lower_val'])
                
                color_pickers_layout.addWidget(upper_picker)
                color_pickers_layout.addWidget(lower_picker)
                
                target_layout.addLayout(color_pickers_layout)
                
                self.mixer_widgets['upper_color_picker'] = upper_picker
                self.mixer_widgets['lower_color_picker'] = lower_picker

            for param in params_in_family:
                widget = self.create_param_widget(param)
                target_layout.addWidget(widget)
        
        target_layout.addStretch(1)


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
        dialog.setMinimumWidth(400)
        
        dialog.exec() 


    def create_ui(self):
        """
        Dynamically creates and arranges the user interface elements based on the application's parameters
        and the chosen layout.
        """
        all_params_list = list(self.src_1_params.values()) + \
                          list(self.src_2_params.values()) + \
                          list(self.post_params.values())

        parent_groups = {}
        for param in all_params_list:
            parent_key = param.parent if param.parent is not None else "Uncategorized"
            if not isinstance(parent_key, (str, int, float, bool, tuple, type(None), ParentClass)):
                log.warning(f"Unhashable parent_key for param '{param.name}': type={{type(parent_key)}}, value={{parent_key}}. Assigning to 'Uncategorized'.")
                parent_key = "Uncategorized"
            if parent_key not in parent_groups:
                parent_groups[parent_key] = []
            parent_groups[parent_key].append(param)
        
        layout_map = {
            ParentClass.SRC_1_EFFECTS: self.src1_effects_layout,
            ParentClass.SRC_1_ANIMATIONS: self.src1_animations_layout,
            ParentClass.SRC_2_EFFECTS: self.src2_effects_layout,
            ParentClass.SRC_2_ANIMATIONS: self.src2_animations_layout,
            ParentClass.POST_EFFECTS: self.post_effects_layout,
            ParentClass.MIXER: self.mixer_layout,
            "Uncategorized": self.uncategorized_layout,
        }

        if ParentClass.MIXER in parent_groups:
            self._mixer_tab(parent_groups, layout_map)

        for parent_enum_or_str, params_in_group in parent_groups.items():
            target_layout = layout_map.get(parent_enum_or_str)

            if target_layout is None:
                log.warning(f"No target layout found for parameter group '{parent_enum_or_str}'. Skipping.")
                continue

            family_tab_widget = QTabWidget()
            target_layout.addWidget(family_tab_widget)

            family_groups = {}
            for param in params_in_group:
                if param.family not in family_groups:
                    family_groups[param.family] = []
                family_groups[param.family].append(param)
            
            for family_name, params_in_family in family_groups.items():
                family_widget = QWidget()
                family_layout = QVBoxLayout(family_widget)

                for param in params_in_family:
                    widget = self.create_param_widget(param)
                    family_layout.addWidget(widget)

                family_scroll = QScrollArea()
                family_scroll.setWidgetResizable(True)
                family_scroll.setWidget(family_widget)
                
                tab_title = family_name.replace("_", " ").title()
                family_tab_widget.addTab(family_scroll, tab_title)

        self.update_mixer_visibility()


    def create_param_widget(self, param: Param):
        widget = QWidget()
        widget.setProperty("param_name", param.name)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(param.name.replace("_", " ").title())
        label.setFixedWidth(100)
        layout.addWidget(label)

        if param.type == WidgetType.RADIO:
            radio_button_group = QWidget()
            radio_layout = QHBoxLayout(radio_button_group)
            for option in param.options:
                radio_button = QRadioButton(option.name.replace('_', ' ').title())
                radio_button.toggled.connect(lambda checked, p=param, v=option.value: self.on_radio_change(p, v, checked))
                if option.value == param.value:
                    radio_button.setChecked(True)
                radio_layout.addWidget(radio_button)
            layout.addWidget(radio_button_group)

        elif param.type == WidgetType.DROPDOWN:
            combo_box = QComboBox()
            if isinstance(param.options, dict):
                for key, value in param.options.items():
                    combo_box.addItem(key.replace('_', ' ').title(), value)
            else: # Enum or list of strings
                for option in param.options:
                    if isinstance(option, str):
                        combo_box.addItem(option.replace('_', ' ').title(), option)
                    else: # Enum member
                        combo_box.addItem(option.name.replace('_', ' ').title(), option)

            # Find the index of the current value and set it
            if isinstance(param.options, dict):
                # Find the index corresponding to the param's value in the dict's values
                try:
                    idx = list(param.options.values()).index(param.value)
                    combo_box.setCurrentIndex(idx)
                except ValueError:
                    pass # Value not in dict
            else: # Enum or list of strings
                try:
                    # Check if the options are enum members
                    if param.options and hasattr(list(param.options)[0], 'value'):
                        idx = [option.value for option in param.options].index(param.value)
                    else: # It's a list of strings (or other values)
                        idx = list(param.options).index(param.value)
                    combo_box.setCurrentIndex(idx)
                except (ValueError, IndexError):
                    log.warning(f"Could not set dropdown index for {param.name} with value {param.value}")
                    pass


            combo_box.currentIndexChanged.connect(lambda index, p=param, c=combo_box: self.on_dropdown_change(p, c.itemData(index)))
            layout.addWidget(combo_box)

        else:  # Default to SLIDER
            mod_button = QPushButton("LFO")
            mod_button.setProperty("param_name", param.name)
            mod_button.setFixedWidth(35)
            mod_button.clicked.connect(lambda: self.open_lfo_dialog(param, mod_button))
            
            if param.linked_oscillator:
                mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_LINKED_STYLE)
            else:
                mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_UNLINKED_STYLE)
            
            layout.addWidget(mod_button)

            slider = QSlider(Qt.Orientation.Horizontal)
            value_input = QLineEdit()
            value_input.setFixedWidth(50)

            if isinstance(param.default, float):
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

        reset_button = QPushButton("R")
        reset_button.setFixedWidth(25)
        reset_button.clicked.connect(lambda: self.on_reset_click(param, widget))
        layout.addWidget(reset_button)

        if param.parent == ParentClass.MIXER:
            self.mixer_widgets[param.name] = widget
        else:
            self.param_widgets[param.name] = widget

        return widget


    def on_slider_change(self, param: Param, value):
        if isinstance(param.default, float):
            param.value = value / 1000.0
        else:
            param.value = value
        
        log.info(f"Slider changed: {param.name} = {param.value}")
        
        slider = self.sender()
        value_input = slider.property("value_input")
        if value_input:
            value_input.setText(str(round(param.value, 3) if isinstance(param.default, float) else param.value))


    def on_text_input_change(self, param: Param, value_input: QLineEdit, slider: QSlider):
        try:
            new_value_str = value_input.text()
            if isinstance(param.default, float):
                new_value = float(new_value_str)
            else:
                new_value = int(new_value_str)

            new_value = max(param.min, min(param.max, new_value))
            param.value = new_value

            if isinstance(param.default, float):
                slider.setValue(int(param.value * 1000))
            else:
                slider.setValue(param.value)
            value_input.setText(str(round(param.value, 3) if isinstance(param.default, float) else param.value))
        except ValueError:
            value_input.setText(str(round(param.value, 3) if isinstance(param.default, float) else param.value))


    def on_radio_change(self, param: Param, value, checked):
        if checked:
            param.value = value
            if param.name == 'blend_mode':
                self.update_mixer_visibility()


    def on_dropdown_change(self, param: Param, data):
        try:
            if isinstance(data, enum.Enum):
                param.value = data.value
            else:
                param.value = data
            log.info(f"Dropdown changed: {param.name} = {param.value}")

            if self.mixer:
                if param.name == "source_1":
                    self.mixer.start_video(data, SourceIndex.SRC_1)
                elif param.name == "source_2":
                    self.mixer.start_video(data, SourceIndex.SRC_2)
        except (ValueError, TypeError):
            pass

    def on_reset_click(self, param: Param, widget: QWidget):
        param.reset()
        slider = widget.findChild(QSlider)
        if slider:
            if isinstance(param.default, float):
                slider.setValue(int(param.value * 1000))
            else:
                slider.setValue(param.value)
        
        value_input = widget.findChild(QLineEdit)
        if value_input:
            value_input.setText(str(round(param.value, 3) if isinstance(param.default, float) else param.value))

        radio_buttons = widget.findChildren(QRadioButton)
        if radio_buttons:
            for rb in radio_buttons:
                if rb.text() == str(param.value):
                    rb.setChecked(True)
                    break
        
        combo_box = widget.findChild(QComboBox)
        if combo_box:
            combo_box.setCurrentText(str(param.value))

    def reset_src1_params(self):
        for param in self.src_1_params.values():
            param.reset()
        self.refresh_all_widgets()

    def reset_src2_params(self):
        for param in self.src_2_params.values():
            param.reset()
        self.refresh_all_widgets()

    def reset_post_params(self):
        for param in self.post_params.values():
            param.reset()
        self.refresh_all_widgets()

    def reset_all_params(self):
        for param in self.all_params.values():
            param.reset()
        self.refresh_all_widgets()


    def update_mixer_visibility(self):
        blend_mode = self.mixer.blend_mode.value

        widgets_visibility = {
            'alpha_blend': blend_mode == MixModes.ALPHA_BLEND.value,
            'luma_threshold': blend_mode == MixModes.LUMA_KEY.value,
            'luma_selection': blend_mode == MixModes.LUMA_KEY.value,
            'upper_color_picker': blend_mode == MixModes.CHROMA_KEY.value,
            'lower_color_picker': blend_mode == MixModes.CHROMA_KEY.value,
        }

        for name, widget in self.mixer_widgets.items():
            if name in widgets_visibility:
                widget.setVisible(widgets_visibility[name])
            elif name == 'blend_mode':
                widget.setVisible(True) # Always show the blend mode selector


    def refresh_all_widgets(self):
        for widget in self.findChildren(QWidget):
            param_name = widget.property("param_name")
            if param_name and param_name in self.all_params:
                param = self.all_params[param_name]
                
                # Update slider and line edit
                slider = widget.findChild(QSlider)
                if slider:
                    if isinstance(param.default, float):
                        slider.setValue(int(param.value * 1000))
                    else:
                        slider.setValue(param.value)
                
                value_input = widget.findChild(QLineEdit)
                if value_input:
                    if isinstance(param.default, float):
                        value_input.setText(str(round(param.value, 3)))
                    else:
                        value_input.setText(str(param.value))

                # Update radio buttons
                radio_buttons = widget.findChildren(QRadioButton)
                if radio_buttons:
                    for rb in radio_buttons:
                        if rb.text().replace(' ', '_').lower() == param.value.name.lower():
                            rb.setChecked(True)
                            break
                
                # Update combo box
                combo_box = widget.findChild(QComboBox)
                if combo_box:
                    if isinstance(param.options, dict):
                        try:
                            idx = list(param.options.values()).index(param.value)
                            combo_box.setCurrentIndex(idx)
                        except ValueError:
                            pass
                    else: # Enum
                        try:
                            idx = [option for option in param.options].index(param.value)
                            combo_box.setCurrentIndex(idx)
                        except ValueError:
                            pass


    def closeEvent(self, event):
        QApplication.instance().quit()
        event.accept()