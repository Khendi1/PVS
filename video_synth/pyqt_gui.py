import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox, QRadioButton, QScrollArea, QToolButton, QSizePolicy, QLineEdit, QTabWidget, QComboBox, QDialog, QGridLayout, QColorDialog
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, pyqtSlot
import logging
from param import ParamTable, Param
from config import ParentClass, SourceIndex, WidgetType
from mix import MixModes # Import MixModes

log = logging.getLogger(__name__)


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

class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.setPixmap(QPixmap.fromImage(image))

    def paintEvent(self, event):
        p = self.pixmap()
        if p:
            pm = self.pixmap()
            widget_size = self.size()
            pixmap_size = pm.size()
            
            scaled_pixmap = pm.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            x = (widget_size.width() - scaled_pixmap.width()) / 2
            y = (widget_size.height() - scaled_pixmap.height()) / 2

            painter = QPainter(self)
            painter.drawPixmap(int(x), int(y), scaled_pixmap)
        else:
            super().paintEvent(event)

    def sizeHint(self):
        return QSize(1, 1)

    def minimumSizeHint(self):
        return QSize(1, 1)


# class CollapsibleGroupBox(QWidget): # REMOVED
#     def __init__(self, title="", header_color="#607D8B", parent=None): # Default grey color
#         super().__init__(parent)
#         self.toggle_button = QToolButton(self)
#         self.toggle_button.setText(title)
#         self.toggle_button.setCheckable(True)
#         self.toggle_button.setChecked(False) # Start collapsed
#         self.toggle_button.setStyleSheet(f"QToolButton {{ border: none; background-color: {header_color}; color: white; padding: 5px; text-align: left; }} QToolButton::hover {{ background-color: {header_color}; }}")
#         self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
#         self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
#         self.toggle_button.clicked.connect(self.toggle_content)
#         self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed) # Make button stretch horizontally

#         self.content_area = QWidget(self)
#         self.content_layout = QVBoxLayout(self.content_area)
#         self.content_layout.setSpacing(-20) # Reduced spacing between widgets
#         self.content_area.setVisible(False) # Start hidden

#         main_layout = QVBoxLayout(self)
#         main_layout.addWidget(self.toggle_button)
#         main_layout.addWidget(self.content_area)
#         main_layout.setContentsMargins(0, 0, 0, 0)
#         self.content_layout.setContentsMargins(0, 0, 5, 5)

#     def toggle_content(self):
#         checked = self.toggle_button.isChecked()
#         self.content_area.setVisible(checked)
#         self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

#     def add_widget(self, widget):
#         self.content_layout.addWidget(widget)


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
    video_frame_ready = pyqtSignal(QImage)

    def __init__(self, effects, layout='quad', mixer=None):
        super().__init__()
        self.is_quad_layout = (layout == 'quad')
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

        self.mixer_widgets = {}

        self.osc_banks = [(effect_manager.oscs, effect_manager.parent.name) for effect_manager in effects]

        self.setWindowTitle("PyQt Control Panel")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        if self.is_quad_layout:
            self.showMaximized()
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
            # self.post_effects_layout will not be created here.
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
            button_vid_b1 = QPushButton("B1")
            button_vid_b1.setFixedWidth(30)
            button_column_layout.addWidget(button_vid_b1)

            button_vid_b2 = QPushButton("B2")
            button_vid_b2.setFixedWidth(30)
            button_column_layout.addWidget(button_vid_b2)

            button_vid_b3 = QPushButton("B3")
            button_vid_b3.setFixedWidth(30)
            button_column_layout.addWidget(button_vid_b3)
            button_column_layout.addStretch(1) # Pushes buttons to the top
            top_right_layout.addLayout(button_column_layout)

            self.video_widget = VideoWidget()
            top_right_layout.addWidget(self.video_widget, 1) # Add stretch to video widget
            self.video_frame_ready.connect(self.video_widget.set_image)
            
            self.root_layout.addWidget(top_right_container, 0, 1)

            # Bottom-Right: Tabs for Mixer and General LFOs
            bottom_right_tabs = QTabWidget()
            
            # Mixer Tab
            mixer_container = QWidget()
            self.mixer_layout = QVBoxLayout(mixer_container)
            self.mixer_layout.addStretch(1)
            mixer_scroll = QScrollArea()
            mixer_scroll.setWidgetResizable(True)
            mixer_scroll.setWidget(mixer_container)
            bottom_right_tabs.addTab(mixer_scroll, "Mixer")

            # General LFOs / Oscillators Tab
            general_lfos_container = QWidget()
            self.general_lfos_layout = QVBoxLayout(general_lfos_container)
            self.general_lfos_layout.addStretch(1)
            general_lfos_scroll = QScrollArea()
            general_lfos_scroll.setWidgetResizable(True)
            general_lfos_scroll.setWidget(general_lfos_container)
            bottom_right_tabs.addTab(general_lfos_scroll, "General LFOs") # Changed tab title here

            self.root_layout.addWidget(bottom_right_tabs, 1, 1)
            
            # --- Set Stretch Factors ---
            self.root_layout.setColumnStretch(0, 1)
            self.root_layout.setColumnStretch(1, 1)
        else:
            # Fallback to original layouts
            self.root_layout = QVBoxLayout(self.central_widget)
            self.layout = QVBoxLayout() # Placeholder
            self.second_pane_layout = QVBoxLayout() # Placeholder
            self.bottom_layout = QVBoxLayout() # Placeholder
            self.root_layout.addLayout(self.layout)
            self.root_layout.addLayout(self.second_pane_layout)
            self.root_layout.addLayout(self.bottom_layout)

        self.create_ui()

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
        and the chosen layout (quad in this case).

        This function performs the following steps:
        1. Gathers all tunable parameters from Source 1 Effects, Source 2 Effects, and Post Effects.
        2. Categorizes these parameters into groups based on their 'parent' attribute.
        3. Initializes a layout_map to direct parameter groups to their respective QVBoxLayouts
           within the quad layout (Src 1 Effects, Src 1 Animations, Src 2 Effects, Src 2 Animations, Mixer, Uncategorized).
        4. Explicitly handles the 'POST_EFFECTS' group: it creates a QTabWidget where each family of
           post-effects is presented as a separate tab. The parameters within each family are added
           directly to their respective family_widget.
        5. Explicitly handles the 'MIXER' group: it presents parameters grouped by family, with each
           family having a non-collapsible QLabel header, followed by its parameters.
        6. Processes remaining parameter groups (like Source 1/2 Effects/Animations, and Uncategorized)
           for the left column tabs. For Effects/Animations, it creates a QTabWidget for families,
           with each family getting its own tab containing parameters. For Uncategorized, it uses
           CollapsibleGroupBoxes.
        7. Explicitly populates the 'General LFOs' tab in the bottom-right quad. It iterates through
           each oscillator in the osc_banks, creating a tab for each, containing its specific parameters
           (frequency, amplitude, phase, etc.).

        No parameters are directly passed to this function, as it operates on the instance's attributes.
        It modifies the central_widget's layout by adding various QWidgets, QScrollAreas, QTabWidgets,
        and parameter-specific controls (sliders, radio buttons, dropdowns).
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
            ParentClass.MIXER: self.mixer_layout,
            "Uncategorized": self.uncategorized_layout,
        }

        # The first duplicate POST_EFFECTS handling block has been removed.
        
        # Explicitly handle POST_EFFECTS and MIXER first to prevent duplication
        log.info(f"--- create_ui started ---")
        log.info(f"Initial parent_groups keys: {list(parent_groups.keys())}")
        log.info(f"layout_map: {layout_map}")

        # Explicitly handle POST_EFFECTS and MIXER first to prevent duplication
        if self.is_quad_layout:
            if ParentClass.POST_EFFECTS in parent_groups:
                log.info(f"Processing ParentClass.POST_EFFECTS explicitly.")
                params_in_group = parent_groups[ParentClass.POST_EFFECTS]
                
                # Create a QVBoxLayout for self.post_effects_container
                post_effects_layout_for_tab = QVBoxLayout(self.post_effects_container)
                post_effects_layout_for_tab.setContentsMargins(0,0,0,0)
                # post_effects_layout_for_tab.addStretch(1) # REMOVED

                family_tab_widget = QTabWidget()
                post_effects_layout_for_tab.addWidget(family_tab_widget) # Add the family tab widget to this layout

                family_groups = {}
                for param in params_in_group:
                    if param.family not in family_groups:
                        family_groups[param.family] = []
                    family_groups[param.family].append(param)
                
                for family_name, params_in_family in family_groups.items():
                    family_widget = QWidget()
                    family_layout = QVBoxLayout(family_widget)
                    # family_layout.addStretch(1) # REMOVED

                    for param in params_in_family:
                        widget = self.create_param_widget(param)
                        family_layout.addWidget(widget)

                    family_scroll = QScrollArea()
                    family_scroll.setWidgetResizable(True)
                    family_scroll.setWidget(family_widget)
                    
                    tab_title = family_name.replace("_", " ").title()
                    family_tab_widget.addTab(family_scroll, tab_title)
                
                del parent_groups[ParentClass.POST_EFFECTS] # Remove after processing
                log.info(f"ParentClass.POST_EFFECTS deleted. parent_groups keys now: {list(parent_groups.keys())}")


            if ParentClass.MIXER in parent_groups:
                params_in_group = parent_groups[ParentClass.MIXER]
                target_layout = layout_map[ParentClass.MIXER]

                family_groups = {}
                for param in params_in_group:
                    if param.family not in family_groups:
                        family_groups[param.family] = []
                    family_groups[param.family].append(param)
                
                for family_name, params_in_family in family_groups.items():
                    # Add a non-collapsible header for the family
                    family_header = QLabel(family_name.replace("_", " ").title())
                    font = family_header.font()
                    font.setBold(True)
                    family_header.setFont(font)
                    family_header.setStyleSheet("QLabel { padding: 5px; background-color: #546E7A; color: white; }")
                    target_layout.addWidget(family_header)

                    # Add parameters directly to the layout
                    source_params = {p.name: p for p in params_in_family if p.name in ['source_1', 'source_2']}
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
                        
                        target_layout.addLayout(source_layout)
                        params_in_family = [p for p in params_in_family if p.name not in source_params]


                    chroma_key_params = {p.name: p for p in params_in_family if p.name in ['upper_hue', 'upper_sat', 'upper_val', 'lower_hue', 'lower_sat', 'lower_val']}
                    if len(chroma_key_params) == 6:
                        upper_picker = ColorPickerWidget('Upper Color', chroma_key_params['upper_hue'], chroma_key_params['upper_sat'], chroma_key_params['upper_val'])
                        lower_picker = ColorPickerWidget('Lower Color', chroma_key_params['lower_hue'], chroma_key_params['lower_sat'], chroma_key_params['lower_val'])
                        target_layout.addWidget(upper_picker)
                        target_layout.addWidget(lower_picker)
                        self.mixer_widgets['upper_color_picker'] = upper_picker
                        self.mixer_widgets['lower_color_picker'] = lower_picker
                        # Remove the chroma key params from the list to avoid creating individual sliders
                        params_in_family = [p for p in params_in_family if p.name not in chroma_key_params]

                    for param in params_in_family:
                        widget = self.create_param_widget(param)
                        target_layout.addWidget(widget)
                del parent_groups[ParentClass.MIXER] # Remove after processing
        
        # Process remaining parent groups
        for parent_enum_or_str, params_in_group in parent_groups.items():
            target_layout = None
            if self.is_quad_layout:
                if parent_enum_or_str in layout_map:
                    target_layout = layout_map[parent_enum_or_str]
            else: 
                if parent_enum_or_str in [ParentClass.SRC_1_ANIMATIONS, ParentClass.SRC_1_EFFECTS]:
                    target_layout = self.layout
                elif parent_enum_or_str in [ParentClass.SRC_2_ANIMATIONS, ParentClass.SRC_2_EFFECTS]:
                    target_layout = self.second_pane_layout
                else:
                    target_layout = self.bottom_layout

            if target_layout is None:
                log.warning(f"No target layout found for parameter group '{parent_enum_or_str}'. Skipping.")
                continue

            else: # Standard handling for all other groups
                # For SRC_1_EFFECTS, SRC_1_ANIMATIONS, SRC_2_EFFECTS, SRC_2_ANIMATIONS
                # Replace collapsible groups with tab groups for families
                if self.is_quad_layout and parent_enum_or_str in [
                    ParentClass.SRC_1_EFFECTS, ParentClass.SRC_1_ANIMATIONS,
                    ParentClass.SRC_2_EFFECTS, ParentClass.SRC_2_ANIMATIONS
                ]:
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
                        # family_layout.addStretch(1) # REMOVED

                        for param in params_in_family:
                            widget = self.create_param_widget(param)
                            family_layout.addWidget(widget)

                        family_scroll = QScrollArea()
                        family_scroll.setWidgetResizable(True)
                        family_scroll.setWidget(family_widget)
                        
                        tab_title = family_name.replace("_", " ").title()
                        family_tab_widget.addTab(family_scroll, tab_title)
                else: # Fallback to collapsible groups for other uncategorized groups
                    # if parent_enum_or_str == "Uncategorized": # This is handled directly in __init__ now
                    #     title = "Uncategorized"
                    #     header_color = "#AAAAAA"
                    # else:
                    #     title = parent_enum_or_str.name.replace("_", " ")
                    #     header_color = parent_enum_or_str.value

                    parent_tab_widget = QTabWidget() # Renamed to be clearer
                    
                    family_groups = {}
                    for param in params_in_group:
                        if param.family not in family_groups:
                            family_groups[param.family] = []
                        family_groups[param.family].append(param)
                    
                    for family_name, params_in_family in family_groups.items():
                        # Each family becomes a tab within the parent_tab_widget
                        family_widget = QWidget()
                        family_layout = QVBoxLayout(family_widget)
                        for param in params_in_family:
                            widget = self.create_param_widget(param)
                            family_layout.addWidget(widget)

                        family_scroll = QScrollArea()
                        family_scroll.setWidgetResizable(True)
                        family_scroll.setWidget(family_widget)
                        
                        tab_title = family_name.replace("_", " ").title()
                        parent_tab_widget.addTab(family_scroll, tab_title)
                    
                    target_layout.addWidget(parent_tab_widget) # Add the QTabWidget to the target_layout
        
        # Explicitly populate the General LFOs tab outside the parent_groups loop
        general_lfos_family_tab_widget = QTabWidget()
        self.general_lfos_layout.addWidget(general_lfos_family_tab_widget) # Add the family tab widget to the general lfos layout

        for osc_bank, osc_bank_name in self.osc_banks:
            for i, osc in enumerate(osc_bank.oscillators):
                osc_title = f"{osc_bank_name.replace('_EFFECTS', '').replace('_', ' ').title()} Oscillator {i+1}"
                
                osc_widget = QWidget()
                osc_layout = QVBoxLayout(osc_widget)
                # osc_layout.addStretch(1) # REMOVED

                osc_params = [
                    osc.frequency, osc.amplitude, osc.phase, osc.shape,
                    osc.noise_octaves, osc.noise_persistence, osc.noise_lacunarity,
                    osc.noise_repeat, osc.noise_base
                ]
                for param in osc_params:
                    widget = self.create_param_widget(param)
                    osc_layout.addWidget(widget)

                osc_scroll = QScrollArea()
                osc_scroll.setWidgetResizable(True)
                osc_scroll.setWidget(osc_widget)
                
                general_lfos_family_tab_widget.addTab(osc_scroll, osc_title)

        self.update_mixer_visibility()

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

        reset_button = QPushButton("R")
        reset_button.setFixedWidth(25)
        reset_button.clicked.connect(lambda: self.on_reset_click(param, widget))
        layout.addWidget(reset_button)

        if param.parent == ParentClass.MIXER:
            self.mixer_widgets[param.name] = widget

        return widget


    def on_slider_change(self, param: Param, value):
        if isinstance(param.default_val, float):
            param.value = value / 1000.0
        else:
            param.value = value
        
        log.info(f"Slider changed: {param.name} = {param.value}")
        
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
            if param.name == 'blend_mode':
                self.update_mixer_visibility()


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
            pass

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

    def closeEvent(self, event):
        QApplication.instance().quit()
        event.accept()
