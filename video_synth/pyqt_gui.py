import enum
import logging
from param import ParamTable, Param
from common import Groups, MixerSource, Widget, Layout
from audio_reactive import BAND_NAMES
from mixer import MixModes, FileSource
from save import SaveController
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox, QRadioButton, QScrollArea, QToolButton, QSizePolicy, QLineEdit, QTabWidget, QComboBox, QDialog, QGridLayout, QListWidget, QColorDialog, QTextEdit, QCheckBox
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap, QPainter, QColor, QTextCursor
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, pyqtSlot, QTimer


log = logging.getLogger(__name__)


"""
Custom Qt logging handler that emits log messages to a QTextEdit widget.
"""
class QTextEditLogger(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)
        # Auto-scroll to bottom
        self.widget.moveCursor(QTextCursor.MoveOperation.End)


"""
Custom color picker widget for selecting HSV colors.
"""
class ColorPickerWidget(QWidget):
    def __init__(self, name, h_param, s_param, v_param, group=None):
        super().__init__(group)
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
    def __init__(self, group=None):
        super().__init__(group)
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

"""
Dialog for managing LFO linkage to a parameter and editing LFO settings.
"""
class LFOManagerDialog(QDialog):
    def __init__(self, param, osc_bank, gui_instance, mod_button, group=None):
        super().__init__(group)
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
            for param_name in ['shape', 'frequency', 'amplitude', 'phase', 'seed', 'cutoff_min', 'cutoff_max', 'noise_octaves', 'noise_persistence', 'noise_lacunarity', 'noise_repeat', 'noise_base']:
                if hasattr(osc, param_name):
                    param = getattr(osc, param_name)
                    # Strip the oscillator name prefix to create a clean display name
                    display_name = param_name.replace("_", " ").title()
                    widget = self.gui_instance._create_param_widget(param, register=False, display_name=display_name)
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


"""Dialog for managing audio band linkage to a parameter."""
class AudioLinkDialog(QDialog):
    def __init__(self, param, audio_module, gui_instance, aud_button, parent=None):
        super().__init__(parent)
        self.param = param
        self.audio_module = audio_module
        self.gui_instance = gui_instance
        self.aud_button = aud_button
        self.setWindowTitle(f"Audio for {param.name}")
        self.setWindowFlags(Qt.WindowType.Popup)

        self.layout = QVBoxLayout(self)
        self.rebuild_ui()

    def rebuild_ui(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not self.audio_module.available:
            label = QLabel("No audio device available.")
            self.layout.addWidget(label)
            return

        if self.param.linked_audio_band is None:
            link_button = QPushButton("Link Audio Band")
            link_button.clicked.connect(self.link_new_band)
            self.layout.addWidget(link_button)
        else:
            unlink_button = QPushButton("Unlink Audio Band")
            unlink_button.clicked.connect(self.unlink_band)
            self.layout.addWidget(unlink_button)

            controls_container = QWidget()
            controls_layout = QVBoxLayout(controls_container)
            self.layout.addWidget(controls_container)

            band = self.param.linked_audio_band
            for param_name in ['band', 'sensitivity', 'attack', 'decay', 'cutoff_min', 'cutoff_max']:
                if hasattr(band, param_name):
                    p = getattr(band, param_name)
                elif hasattr(band, f'{param_name}_select'):
                    p = getattr(band, f'{param_name}_select')
                else:
                    continue
                display_name = param_name.replace("_", " ").title()
                widget = self.gui_instance._create_param_widget(p, register=False, display_name=display_name)
                controls_layout.addWidget(widget)

    def link_new_band(self):
        band_name = f"{self.param.name}_audio"
        new_band = self.audio_module.add_band(band_name, band_index=0)
        new_band.link_param(self.param)
        self.param.linked_audio_band = new_band
        self.aud_button.setStyleSheet(PyQTGUI.AUD_BUTTON_LINKED_STYLE)
        self.rebuild_ui()

    def unlink_band(self):
        if self.param.linked_audio_band:
            self.audio_module.remove_band(self.param.linked_audio_band)
            self.param.linked_audio_band.unlink_param()
            self.param.linked_audio_band = None
            self.aud_button.setStyleSheet(PyQTGUI.AUD_BUTTON_UNLINKED_STYLE)
            self.rebuild_ui()


"""Widget for reordering the effects processing chain via drag-and-drop."""
class SequencerWidget(QWidget):
    def __init__(self, effect_manager, parent=None):
        super().__init__(parent)
        self.effect_manager = effect_manager
        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.DropAction.MoveAction)
        layout.addWidget(self.list_widget)

        self._populate_list()
        self.list_widget.model().rowsMoved.connect(self._on_reorder)

    def _populate_list(self):
        self.list_widget.clear()
        for method in self.effect_manager.all_methods:
            class_name = method.__self__.__class__.__name__
            self.list_widget.addItem(f"{class_name}.{method.__name__}")

    def _on_reorder(self):
        method_lookup = {}
        for method in self.effect_manager.all_methods:
            class_name = method.__self__.__class__.__name__
            key = f"{class_name}.{method.__name__}"
            method_lookup[key] = method

        new_order = []
        for i in range(self.list_widget.count()):
            item_text = self.list_widget.item(i).text()
            if item_text in method_lookup:
                new_order.append(method_lookup[item_text])

        if len(new_order) == len(self.effect_manager.all_methods):
            self.effect_manager.all_methods = new_order
        else:
            log.error("Sequencer reorder failed: method count mismatch. Resetting.")
            self._populate_list()


"""Main PyQt GUI class for the video synthesizer application."""
class PyQTGUI(QMainWindow):
    LFO_BUTTON_UNLINKED_STYLE = "QPushButton { background-color: #607D8B; color: white; }" # Default grey
    LFO_BUTTON_LINKED_STYLE = "QPushButton { background-color: #4CAF50; color: white; }" # Green
    AUD_BUTTON_UNLINKED_STYLE = "QPushButton { background-color: #607D8B; color: white; }" # Default grey
    AUD_BUTTON_LINKED_STYLE = "QPushButton { background-color: #FF9800; color: white; }" # Orange
    video_frame_ready = pyqtSignal(QImage)

    def __init__(self, effects, settings, mixer=None, audio_module=None):
        super().__init__()
        self.layout_style = settings.layout.value
        self.effects = effects
        self.mixer = mixer
        self.audio_module = audio_module
        self.src_1_effects = effects[MixerSource.SRC_1]
        self.src_2_effects = effects[MixerSource.SRC_2]
        self.post_effects = effects[MixerSource.POST]

        self.src_1_params = self.src_1_effects.params
        self.src_2_params = self.src_2_effects.params
        self.post_params = self.post_effects.params
        self.mixer_params = mixer.params if mixer else ParamTable(group="Mixer")
        self.settings_params = settings.params

        self.all_params = ParamTable()
        self.all_params.params.update(self.src_1_params)
        self.all_params.params.update(self.src_2_params)
        self.all_params.params.update(self.post_params)
        self.all_params.params.update(self.mixer_params)
        self.all_params.params.update(self.settings_params)
        self.save_controller = SaveController({
            "src_1": self.src_1_params,
            "src_2": self.src_2_params,
            "post": self.post_params,
            "mixer": self.mixer_params,
            "settings": self.settings_params,
        })
        self.save_controller.patch_loaded_callback = self._refresh_all_widgets

        self.mixer_widgets = {}
        self.param_widgets = {}

        self.osc_banks = [(effect_manager.oscs, effect_manager.group.name) for effect_manager in effects]

        self.setWindowTitle("PyQt Control Panel")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.showMaximized()

        self._create_layout()
        self.create_ui()

        # Set up logging to GUI
        self._setup_logging()

        self.mod_refresh_timer = QTimer(self)
        self.mod_refresh_timer.timeout.connect(self._refresh_mod_buttons)
        self.mod_refresh_timer.start(250)
  
        
    def _refresh_mod_buttons(self):
        self.all_params = ParamTable()
        self.all_params.params.update(self.src_1_effects.params)
        self.all_params.params.update(self.src_2_effects.params)
        self.all_params.params.update(self.post_effects.params)
        self.all_params.params.update(self.mixer_params)
        self.all_params.params.update(self.settings_params)
        for param_name, widget in self.param_widgets.items():
            param = self.all_params.get(param_name)
            if param:
                for btn in widget.findChildren(QPushButton):
                    if btn.text() == "LFO":
                        if param.linked_oscillator:
                            btn.setStyleSheet(PyQTGUI.LFO_BUTTON_LINKED_STYLE)
                        else:
                            btn.setStyleSheet(PyQTGUI.LFO_BUTTON_UNLINKED_STYLE)
                    elif btn.text() == "AUD":
                        if param.linked_audio_band:
                            btn.setStyleSheet(PyQTGUI.AUD_BUTTON_LINKED_STYLE)
                        else:
                            btn.setStyleSheet(PyQTGUI.AUD_BUTTON_UNLINKED_STYLE)

    def _setup_logging(self):
        """Connect the logging system to the GUI log viewer."""
        log_handler = QTextEditLogger(self.log_viewer)
        log_handler.setFormatter(logging.Formatter('%(levelname).1s | %(module)s | %(message)s'))
        logging.getLogger().addHandler(log_handler)
        log.info("GUI log viewer initialized")


    def _create_layout(self):
        log.debug(f"Creating layout: {self.layout_style}/{Layout(self.layout_style).name}")
        match Layout(self.layout_style):
            case Layout.QUAD_FULL:
                log.error("QUAD_FULL layout is not yet implemented. Defaulting to QUAD_PREVIEW.")
                self._create_quad_layout()
            case Layout.QUAD_PREVIEW:
                self._create_quad_layout()
            case Layout.SPLIT:
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
        button_r1.clicked.connect(self._reset_src1_params)
        button_column_layout.addWidget(button_r1)

        button_r2 = QPushButton("R2")
        button_r2.setFixedWidth(30)
        button_r2.setToolTip("Reset Source 2 Params")
        button_r2.clicked.connect(self._reset_src2_params)
        button_column_layout.addWidget(button_r2)

        button_rp = QPushButton("RP")
        button_rp.setFixedWidth(30)
        button_rp.setToolTip("Reset Post-Processing Params")
        button_rp.clicked.connect(self._reset_post_params)
        button_column_layout.addWidget(button_rp)

        button_ra = QPushButton("RA")
        button_ra.setFixedWidth(30)
        button_ra.setToolTip("Reset All Params")
        button_ra.clicked.connect(self._reset_all_params)
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

        # User Settings Tab
        user_settings_container = QWidget()
        self.user_settings_layout = QVBoxLayout(user_settings_container)
        user_settings_scroll = QScrollArea()
        user_settings_scroll.setWidgetResizable(True)
        user_settings_scroll.setWidget(user_settings_container)
        bottom_right_tabs.addTab(user_settings_scroll, "User Settings")

        # Logs Tab
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        bottom_right_tabs.addTab(self.log_viewer, "Logs")

        self.root_layout.addWidget(bottom_right_tabs, 1, 1)

        # --- Set Stretch Factors ---
        self.root_layout.setColumnStretch(0, 1)
        self.root_layout.setColumnStretch(1, 1)
        self.root_layout.setRowStretch(0, 1)
        self.root_layout.setRowStretch(1, 1)


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

        # User Settings Tab
        user_settings_container = QWidget()
        self.user_settings_layout = QVBoxLayout(user_settings_container)
        user_settings_scroll = QScrollArea()
        user_settings_scroll.setWidgetResizable(True)
        user_settings_scroll.setWidget(user_settings_container)
        bottom_pane_tabs.addTab(user_settings_scroll, "User Settings")

        # Logs Tab
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        bottom_pane_tabs.addTab(self.log_viewer, "Logs")


    def _open_lfo_dialog(self, param, button):
        if param.group in [Groups.SRC_1_ANIMATIONS, Groups.SRC_1_EFFECTS]:
            osc_bank = self.src_1_effects.oscs
        elif param.group in [Groups.SRC_2_ANIMATIONS, Groups.SRC_2_EFFECTS]:
            osc_bank = self.src_2_effects.oscs
        else:
            osc_bank = self.post_effects.oscs

        dialog = LFOManagerDialog(param, osc_bank, self, button, self)
        dialog.setMinimumWidth(600)

        # Position dialog below button, with screen boundary detection
        button_pos = button.mapToGlobal(button.rect().bottomLeft())
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()

        # Adjust size to ensure it fits on screen
        dialog.adjustSize()
        dialog_size = dialog.size()

        # Check if dialog would go off right edge
        if button_pos.x() + dialog_size.width() > screen_geometry.right():
            button_pos.setX(screen_geometry.right() - dialog_size.width())

        # Check if dialog would go off left edge
        if button_pos.x() < screen_geometry.left():
            button_pos.setX(screen_geometry.left())

        # Check if dialog would go off bottom edge
        if button_pos.y() + dialog_size.height() > screen_geometry.bottom():
            # Position above button instead
            button_pos = button.mapToGlobal(button.rect().topLeft())
            button_pos.setY(button_pos.y() - dialog_size.height())

        # Check if dialog would go off top edge
        if button_pos.y() < screen_geometry.top():
            button_pos.setY(screen_geometry.top())

        dialog.move(button_pos)
        dialog.exec()


    def _open_audio_dialog(self, param, button):
        if self.audio_module is None:
            return

        dialog = AudioLinkDialog(param, self.audio_module, self, button, self)
        dialog.setMinimumWidth(600)

        # Position dialog below button, with screen boundary detection
        button_pos = button.mapToGlobal(button.rect().bottomLeft())
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()

        dialog.adjustSize()
        dialog_size = dialog.size()

        if button_pos.x() + dialog_size.width() > screen_geometry.right():
            button_pos.setX(screen_geometry.right() - dialog_size.width())
        if button_pos.x() < screen_geometry.left():
            button_pos.setX(screen_geometry.left())
        if button_pos.y() + dialog_size.height() > screen_geometry.bottom():
            button_pos = button.mapToGlobal(button.rect().topLeft())
            button_pos.setY(button_pos.y() - dialog_size.height())
        if button_pos.y() < screen_geometry.top():
            button_pos.setY(screen_geometry.top())

        dialog.move(button_pos)
        dialog.exec()


    def create_ui(self):
        """
        Dynamically creates and arranges the user interface elements based on the application's parameters
        and the chosen layout.
        """
        all_params_list = list(self.src_1_params.values()) + \
                          list(self.src_2_params.values()) + \
                          list(self.post_params.values()) + \
                          list(self.mixer_params.values()) + \
                          list(self.settings_params.values())

        groups = {}
        for param in all_params_list:
            group_key = param.group if param.group is not None else "Uncategorized"
            if not isinstance(group_key, (str, int, float, bool, tuple, type(None), Groups)):
                log.warning(f"Unhashable group_key for param '{param.name}': type={{type(group_key)}}, value={{group_key}}. Assigning to 'Uncategorized'.")
                group_key = "Uncategorized"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(param)
        
        layout_map = {
            Groups.SRC_1_EFFECTS: self.src1_effects_layout,
            Groups.SRC_1_ANIMATIONS: self.src1_animations_layout,
            Groups.SRC_2_EFFECTS: self.src2_effects_layout,
            Groups.SRC_2_ANIMATIONS: self.src2_animations_layout,
            Groups.POST_EFFECTS: self.post_effects_layout,
            Groups.MIXER: self.mixer_layout,
            Groups.USER_SETTINGS: self.user_settings_layout,
            "Uncategorized": self.uncategorized_layout,
        }

        for group_enum_or_str, params_in_subgroup in groups.items():
            target_layout = layout_map.get(group_enum_or_str)
            if target_layout is None:
                log.warning(f"No target layout found for parameter subgroup '{group_enum_or_str}'. Skipping.")
                continue

            subgroups = {}
            for param in params_in_subgroup:
                if param.subgroup not in subgroups:
                    subgroups[param.subgroup] = []
                subgroups[param.subgroup].append(param)
            
            use_sub_tabs = True
            if len(subgroups) == 1:
                subgroup_name = list(subgroups.keys())[0]
                
                group_name_str = ""
                if isinstance(group_enum_or_str, enum.Enum):
                    group_name_str = group_enum_or_str.name
                else:
                    group_name_str = group_enum_or_str

                subgroup_name_str = ""
                if isinstance(subgroup_name, enum.Enum):
                    subgroup_name_str = subgroup_name.name
                else:
                    subgroup_name_str = subgroup_name

                if subgroup_name_str.replace('_', '').replace(' ', '').lower() == group_name_str.replace('_', '').replace(' ', '').lower():
                    use_sub_tabs = False

            if not use_sub_tabs:
                # No sub-tabs, add widgets directly
                if group_enum_or_str == Groups.MIXER:
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

                    for params_in_subgroup in subgroups.values():
                        blend_mode_param, chroma_key_hue_sat_val_params, source_params, remaining_params = None, [], {}, []
                        
                        all_mixer_subgroup_params = list(params_in_subgroup)
                        
                        for param in all_mixer_subgroup_params:
                            if param.name == 'blend_mode':
                                blend_mode_param = param
                            elif param.name in ['upper_hue', 'upper_sat', 'upper_val', 'lower_hue', 'lower_sat', 'lower_val']:
                                chroma_key_hue_sat_val_params.append(param)
                            elif param.name in ['source_1', 'source_2']:
                                source_params[param.name] = param
                            else:
                                remaining_params.append(param)

                        if len(source_params) == 2:
                            source_layout = QHBoxLayout()
                            
                            s1_param = source_params['source_1']
                            s1_label = QLabel(s1_param.name)
                            s1_combo = QComboBox()
                            s1_combo.addItems([str(o) for o in s1_param.options])
                            s1_combo.setCurrentText(str(s1_param.value))
                            s1_combo.currentTextChanged.connect(lambda text, p=s1_param: self._on_dropdown_change(p, text))
                            source_layout.addWidget(s1_label)
                            source_layout.addWidget(s1_combo)

                            s2_param = source_params['source_2']
                            s2_label = QLabel(s2_param.name)
                            s2_combo = QComboBox()
                            s2_combo.addItems([str(o) for o in s2_param.options])
                            s2_combo.setCurrentText(str(s2_param.value))
                            s2_combo.currentTextChanged.connect(lambda text, p=s2_param: self._on_dropdown_change(p, text))
                            source_layout.addWidget(s2_label)
                            source_layout.addWidget(s2_combo)

                            swap_button = QPushButton("Swap")
                            swap_button.clicked.connect(lambda: self._swap_sources(s1_param, s2_param, s1_combo, s2_combo))
                            source_layout.addWidget(swap_button)
                            
                            target_layout.addLayout(source_layout)

                        if blend_mode_param:
                            widget = self._create_param_widget(blend_mode_param)
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

                        for param in remaining_params:
                            widget = self._create_param_widget(param)
                            target_layout.addWidget(widget)

                else:
                    for params_in_subgroup in subgroups.values():
                        for param in params_in_subgroup:
                            widget = self._create_param_widget(param)
                            target_layout.addWidget(widget)
                target_layout.addStretch(1)
            else:
                # Create sub-tabs for each subgroup
                subgroup_tab_widget = QTabWidget()
                target_layout.addWidget(subgroup_tab_widget)

                for subgroup_name, params_in_subgroup in subgroups.items():
                    subgroup_widget = QWidget()
                    subgroup_layout = QVBoxLayout(subgroup_widget)

                    for param in params_in_subgroup:
                        widget = self._create_param_widget(param)
                        subgroup_layout.addWidget(widget)

                    subgroup_scroll = QScrollArea()
                    subgroup_scroll.setWidgetResizable(True)
                    subgroup_scroll.setWidget(subgroup_widget)

                    tab_title = subgroup_name.replace("_", " ").title() if isinstance(subgroup_name, str) else subgroup_name.name.replace("_", " ").title()
                    subgroup_tab_widget.addTab(subgroup_scroll, tab_title)

                # Add sequencer subtab within effects sub-tabs
                effect_manager = {
                    Groups.SRC_1_EFFECTS: self.src_1_effects,
                    Groups.SRC_2_EFFECTS: self.src_2_effects,
                    Groups.POST_EFFECTS: self.post_effects,
                }.get(group_enum_or_str)
                if effect_manager:
                    seq_widget = SequencerWidget(effect_manager)
                    subgroup_tab_widget.addTab(seq_widget, "Sequence")

        self._update_mixer_visibility()


    def _create_param_widget(self, param: Param, register=True, display_name=None):
        widget = QWidget()
        widget.setProperty("param_name", param.name)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label_text = display_name if display_name else param.name.replace("_", " ").title()
        label = QLabel(label_text)
        label.setFixedWidth(100)
        layout.addWidget(label)

        if param.type == Widget.RADIO:
            radio_button_subgroup = QWidget()
            radio_layout = QHBoxLayout(radio_button_subgroup)
            for option in param.options:
                radio_button = QRadioButton(option.name.replace('_', ' ').title())
                radio_button.toggled.connect(lambda checked, p=param, v=option.value: self._on_radio_change(p, v, checked))
                if option.value == param.value:
                    radio_button.setChecked(True)
                radio_layout.addWidget(radio_button)
            layout.addWidget(radio_button_subgroup)

        elif param.type == Widget.DROPDOWN:
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


            combo_box.currentIndexChanged.connect(lambda index, p=param, c=combo_box: self._on_dropdown_change(p, c.itemData(index)))
            layout.addWidget(combo_box)

        elif param.type == Widget.TOGGLE:
            checkbox = QCheckBox()
            checkbox.setChecked(bool(param.value))
            checkbox.stateChanged.connect(lambda state, p=param: self._on_toggle_change(p, state))
            layout.addWidget(checkbox)

        else:  # Default to SLIDER
            mod_button = QPushButton("LFO")
            mod_button.setProperty("param_name", param.name)
            mod_button.setFixedWidth(35)
            mod_button.clicked.connect(lambda: self._open_lfo_dialog(param, mod_button))
            
            if param.linked_oscillator:
                mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_LINKED_STYLE)
            else:
                mod_button.setStyleSheet(PyQTGUI.LFO_BUTTON_UNLINKED_STYLE)
            
            layout.addWidget(mod_button)

            aud_button = QPushButton("AUD")
            aud_button.setProperty("param_name", param.name)
            aud_button.setFixedWidth(35)
            aud_button.clicked.connect(lambda _, p=param, b=aud_button: self._open_audio_dialog(p, b))

            if param.linked_audio_band:
                aud_button.setStyleSheet(PyQTGUI.AUD_BUTTON_LINKED_STYLE)
            else:
                aud_button.setStyleSheet(PyQTGUI.AUD_BUTTON_UNLINKED_STYLE)

            layout.addWidget(aud_button)

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

            slider.valueChanged.connect(lambda value, p=param: self._on_slider_change(p, value))
            layout.addWidget(slider)
            
            slider.setProperty("value_input", value_input)
            value_input.editingFinished.connect(lambda p=param, vi=value_input, s=slider: self._on_text_input_change(p, vi, s))
            layout.addWidget(value_input)

        reset_button = QPushButton("R")
        reset_button.setFixedWidth(25)
        reset_button.clicked.connect(lambda: self._on_reset_click(param, widget))
        layout.addWidget(reset_button)

        if register:
            if param.group == Groups.MIXER:
                self.mixer_widgets[param.name] = widget
            else:
                self.param_widgets[param.name] = widget

        return widget


    def _on_slider_change(self, param: Param, value):
        if isinstance(param.default, float):
            param.value = value / 1000.0
        else:
            param.value = value
        
        # log.info(f"Slider changed: {param.name} = {param.value}")
        
        slider = self.sender()
        value_input = slider.property("value_input")
        if value_input:
            value_input.setText(str(round(param.value, 3) if isinstance(param.default, float) else param.value))


    def _on_text_input_change(self, param: Param, value_input: QLineEdit, slider: QSlider):
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


    def _on_radio_change(self, param: Param, value, checked):
        if checked:
            param.value = value
            if param.name == 'blend_mode':
                self._update_mixer_visibility()


    def _on_dropdown_change(self, param: Param, data):
        try:
            if isinstance(data, enum.Enum):
                param.value = data.value
            else:
                param.value = data
            log.info(f"Dropdown changed: {param.name} = {param.value}")

            if self.mixer:
                if param.name == "source_1":
                    self.mixer.start_video(data, MixerSource.SRC_1)
                elif param.name == "source_2":
                    self.mixer.start_video(data, MixerSource.SRC_2)
                elif param.name == "video_file_src1" and self.mixer.selected_source1.value == FileSource.VIDEO.name:
                    self.mixer.start_video(FileSource.VIDEO.name, MixerSource.SRC_1)
                elif param.name == "video_file_src2" and self.mixer.selected_source2.value == FileSource.VIDEO.name:
                    self.mixer.start_video(FileSource.VIDEO.name, MixerSource.SRC_2)
                elif param.name == "image_file_src1" and self.mixer.selected_source1.value == FileSource.IMAGE.name:
                    self.mixer.start_video(FileSource.IMAGE.name, MixerSource.SRC_1)
                elif param.name == "image_file_src2" and self.mixer.selected_source2.value == FileSource.IMAGE.name:
                    self.mixer.start_video(FileSource.IMAGE.name, MixerSource.SRC_2)
        except (ValueError, TypeError):
            pass


    def _on_toggle_change(self, param: Param, state):
        """Handle toggle checkbox state changes. Qt.CheckState: Unchecked=0, Checked=2"""
        param.value = 1 if state == Qt.CheckState.Checked.value else 0
        log.info(f"Toggle changed: {param.name} = {param.value}")


    def _on_reset_click(self, param: Param, widget: QWidget):
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

        checkbox = widget.findChild(QCheckBox)
        if checkbox:
            checkbox.setChecked(bool(param.value))


    def _reset_src1_params(self):
        for param in self.src_1_params.values():
            param.reset()
        self._refresh_all_widgets()


    def _reset_src2_params(self):
        for param in self.src_2_params.values():
            param.reset()
        self._refresh_all_widgets()


    def _reset_post_params(self):
        for param in self.post_params.values():
            param.reset()
        self._refresh_all_widgets()


    def _reset_all_params(self):
        for param in self.all_params.values():
            param.reset()
        self._refresh_all_widgets()


    def _update_mixer_visibility(self):
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


    def _refresh_all_widgets(self):
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
