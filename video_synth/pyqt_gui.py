import enum
import logging
from param import ParamTable, Param
from common import Groups, MixerSource, Widget, Layout
from audio_reactive import BAND_NAMES
from mixer import MixModes, FileSource
from save import SaveController
from pyqt_widgets import (QTextEditLogger, ColorPickerWidget, VideoWidget, LFOManagerDialog,
                           AudioLinkDialog, SequencerWidget, MidiMapperWidget, OSCMapperWidget,
                           LFO_BUTTON_LINKED_STYLE, LFO_BUTTON_UNLINKED_STYLE,
                           AUD_BUTTON_LINKED_STYLE, AUD_BUTTON_UNLINKED_STYLE)
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox, QRadioButton, QScrollArea, QToolButton, QSizePolicy, QLineEdit, QTabWidget, QComboBox, QDialog, QGridLayout, QListWidget, QColorDialog, QTextEdit, QCheckBox
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap, QPainter, QColor, QTextCursor
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, pyqtSlot, QTimer


log = logging.getLogger(__name__)


"""Main PyQt GUI class for the video synthesizer application."""
class PyQTGUI(QMainWindow):
    LFO_BUTTON_UNLINKED_STYLE = LFO_BUTTON_UNLINKED_STYLE
    LFO_BUTTON_LINKED_STYLE = LFO_BUTTON_LINKED_STYLE
    AUD_BUTTON_UNLINKED_STYLE = AUD_BUTTON_UNLINKED_STYLE
    AUD_BUTTON_LINKED_STYLE = AUD_BUTTON_LINKED_STYLE
    video_frame_ready = pyqtSignal(QImage)

    def __init__(self, effects, settings, mixer=None, audio_module=None, obs_filters=None, api_server=None, midi_mapper=None, osc_controller=None):
        super().__init__()
        self.layout_style = settings.layout.value
        self.effects = effects
        self.settings = settings
        self.mixer = mixer
        self.audio_module = audio_module
        self.obs_filters = obs_filters
        self.api_server = api_server
        self.midi_mapper = midi_mapper
        self.osc_controller = osc_controller
        self.src_1_effects = effects[MixerSource.SRC_1]
        self.src_2_effects = effects[MixerSource.SRC_2]
        self.post_effects = effects[MixerSource.POST]

        self.src_1_params = self.src_1_effects.params
        self.src_2_params = self.src_2_effects.params
        self.post_params = self.post_effects.params
        self.mixer_params = mixer.params if mixer else ParamTable(group="Mixer")
        self.settings_params = settings.params
        self.obs_params = obs_filters.params if obs_filters else ParamTable(group="OBS")

        self.all_params = ParamTable()
        self.all_params.params.update(self.src_1_params)
        self.all_params.params.update(self.src_2_params)
        self.all_params.params.update(self.post_params)
        self.all_params.params.update(self.mixer_params)
        self.all_params.params.update(self.settings_params)
        self.all_params.params.update(self.obs_params)
        save_tables = {
            "src_1": self.src_1_params,
            "src_2": self.src_2_params,
            "post": self.post_params,
            "mixer": self.mixer_params,
            "settings": self.settings_params,
        }
        if obs_filters:
            save_tables["obs"] = self.obs_params
        self.save_controller = SaveController(save_tables)
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
        self.all_params.params.update(self.obs_params)
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


    def _rebuild_layout(self):
        """Tear down and rebuild the entire GUI layout when layout param changes."""
        new_style = self.settings.layout.value
        if isinstance(new_style, enum.Enum):
            new_style = new_style.value
        if new_style == self.layout_style:
            return
        log.info(f"Rebuilding layout: {Layout(self.layout_style).name} -> {Layout(new_style).name}")
        self.layout_style = new_style

        # Remove old log handler before destroying widgets
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, QTextEditLogger):
                root_logger.removeHandler(handler)

        # Clear widget tracking
        self.mixer_widgets.clear()
        self.param_widgets.clear()

        # Replace central widget (destroys all child widgets)
        old_widget = self.central_widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        old_widget.deleteLater()

        self._create_layout()
        self.create_ui()
        self._setup_logging()

    def _create_layout(self):
        log.debug(f"Creating layout: {self.layout_style}/{Layout(self.layout_style).name}")
        match Layout(self.layout_style):
            case Layout.QUAD_FULL:
                self._create_quad_full_layout()
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

        # OBS Tab
        obs_container = QWidget()
        self.obs_layout = QVBoxLayout(obs_container)
        obs_scroll = QScrollArea()
        obs_scroll.setWidgetResizable(True)
        obs_scroll.setWidget(obs_container)
        bottom_right_tabs.addTab(obs_scroll, "OBS")

        # MIDI Mapper Tab
        self.midi_mapper_widget = MidiMapperWidget(self.midi_mapper) if self.midi_mapper else QWidget()
        bottom_right_tabs.addTab(self.midi_mapper_widget, "MIDI")

        # OSC Mapper Tab
        self.osc_mapper_widget = OSCMapperWidget(self.osc_controller) if self.osc_controller else QWidget()
        bottom_right_tabs.addTab(self.osc_mapper_widget, "OSC")

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


    def _create_quad_full_layout(self):
        """
        4-quadrant layout without video preview.
        Src 2 Effects/Animations tabs replace the video player in the top-right.
        All 4 quadrants are equal size.
        """
        self.root_layout = QGridLayout(self.central_widget)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        # --- Top-Left: Src 1 Effects + Src 1 Animations ---
        top_left_tabs = QTabWidget()

        src1_effects_container = QWidget()
        self.src1_effects_layout = QVBoxLayout(src1_effects_container)
        src1_effects_scroll = QScrollArea()
        src1_effects_scroll.setWidgetResizable(True)
        src1_effects_scroll.setWidget(src1_effects_container)
        top_left_tabs.addTab(src1_effects_scroll, "Src 1 Effects")

        src1_animations_container = QWidget()
        self.src1_animations_layout = QVBoxLayout(src1_animations_container)
        src1_animations_scroll = QScrollArea()
        src1_animations_scroll.setWidgetResizable(True)
        src1_animations_scroll.setWidget(src1_animations_container)
        top_left_tabs.addTab(src1_animations_scroll, "Src 1 Animations")

        self.root_layout.addWidget(top_left_tabs, 0, 0)

        # --- Top-Right: Src 2 Effects + Src 2 Animations + Reset Buttons ---
        top_right_widget = QWidget()
        top_right_outer = QHBoxLayout(top_right_widget)
        top_right_outer.setContentsMargins(0, 0, 0, 0)
        top_right_outer.setSpacing(0)

        # Reset button column
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

        button_column_layout.addStretch(1)
        top_right_outer.addLayout(button_column_layout)

        # Src 2 tabs
        top_right_tabs = QTabWidget()

        src2_effects_container = QWidget()
        self.src2_effects_layout = QVBoxLayout(src2_effects_container)
        src2_effects_scroll = QScrollArea()
        src2_effects_scroll.setWidgetResizable(True)
        src2_effects_scroll.setWidget(src2_effects_container)
        top_right_tabs.addTab(src2_effects_scroll, "Src 2 Effects")

        src2_animations_container = QWidget()
        self.src2_animations_layout = QVBoxLayout(src2_animations_container)
        src2_animations_scroll = QScrollArea()
        src2_animations_scroll.setWidgetResizable(True)
        src2_animations_scroll.setWidget(src2_animations_container)
        top_right_tabs.addTab(src2_animations_scroll, "Src 2 Animations")

        top_right_outer.addWidget(top_right_tabs, 1)

        self.root_layout.addWidget(top_right_widget, 0, 1)

        # --- Bottom-Left: Post Effects + Uncategorized ---
        bottom_left_tabs = QTabWidget()

        self.post_effects_container = QWidget()
        self.post_effects_layout = QVBoxLayout(self.post_effects_container)
        bottom_left_tabs.addTab(self.post_effects_container, "Post Effects")

        uncategorized_container = QWidget()
        self.uncategorized_layout = QVBoxLayout(uncategorized_container)
        uncategorized_scroll = QScrollArea()
        uncategorized_scroll.setWidgetResizable(True)
        uncategorized_scroll.setWidget(uncategorized_container)
        bottom_left_tabs.addTab(uncategorized_scroll, "Uncategorized")

        self.root_layout.addWidget(bottom_left_tabs, 1, 0)

        # --- Bottom-Right: Mixer + User Settings + Logs ---
        bottom_right_tabs = QTabWidget()

        mixer_container = QWidget()
        self.mixer_layout = QVBoxLayout(mixer_container)
        mixer_scroll = QScrollArea()
        mixer_scroll.setWidgetResizable(True)
        mixer_scroll.setWidget(mixer_container)
        bottom_right_tabs.addTab(mixer_scroll, "Mixer")

        user_settings_container = QWidget()
        self.user_settings_layout = QVBoxLayout(user_settings_container)
        user_settings_scroll = QScrollArea()
        user_settings_scroll.setWidgetResizable(True)
        user_settings_scroll.setWidget(user_settings_container)
        bottom_right_tabs.addTab(user_settings_scroll, "User Settings")

        # OBS Tab
        obs_container = QWidget()
        self.obs_layout = QVBoxLayout(obs_container)
        obs_scroll = QScrollArea()
        obs_scroll.setWidgetResizable(True)
        obs_scroll.setWidget(obs_container)
        bottom_right_tabs.addTab(obs_scroll, "OBS")

        # MIDI Mapper Tab
        self.midi_mapper_widget = MidiMapperWidget(self.midi_mapper) if self.midi_mapper else QWidget()
        bottom_right_tabs.addTab(self.midi_mapper_widget, "MIDI")

        # OSC Mapper Tab
        self.osc_mapper_widget = OSCMapperWidget(self.osc_controller) if self.osc_controller else QWidget()
        bottom_right_tabs.addTab(self.osc_mapper_widget, "OSC")

        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        bottom_right_tabs.addTab(self.log_viewer, "Logs")

        self.root_layout.addWidget(bottom_right_tabs, 1, 1)

        # --- Equal quadrant sizing ---
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

        # OBS Tab
        obs_container = QWidget()
        self.obs_layout = QVBoxLayout(obs_container)
        obs_scroll = QScrollArea()
        obs_scroll.setWidgetResizable(True)
        obs_scroll.setWidget(obs_container)
        bottom_pane_tabs.addTab(obs_scroll, "OBS")

        # MIDI Mapper Tab
        self.midi_mapper_widget = MidiMapperWidget(self.midi_mapper) if self.midi_mapper else QWidget()
        bottom_pane_tabs.addTab(self.midi_mapper_widget, "MIDI")

        # OSC Mapper Tab
        self.osc_mapper_widget = OSCMapperWidget(self.osc_controller) if self.osc_controller else QWidget()
        bottom_pane_tabs.addTab(self.osc_mapper_widget, "OSC")

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
                          list(self.settings_params.values()) + \
                          list(self.obs_params.values())

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
            Groups.OBS: self.obs_layout,
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

            if param.name == "log_level":
                level = data if isinstance(data, int) else logging.getLevelName(str(data))
                logging.getLogger().setLevel(level)
                log.info(f"Log level changed to {logging.getLevelName(level)}")

            if param.name == "layout":
                self._rebuild_layout()

        except (ValueError, TypeError):
            pass


    def _on_toggle_change(self, param: Param, state):
        """Handle toggle checkbox state changes. Qt.CheckState: Unchecked=0, Checked=2"""
        param.value = 1 if state == Qt.CheckState.Checked.value else 0
        log.info(f"Toggle changed: {param.name} = {param.value}")

        if param.name == "api_enabled":
            self._toggle_api_server(param.value)
        elif param.name == "osc_enabled":
            self._toggle_osc_server(param.value)

    def _toggle_osc_server(self, enabled):
        """Start or stop the OSC server based on the toggle state."""
        if enabled:
            if self.osc_controller is None:
                from osc_controller import OSCController
                param_tables = {}
                for effect_mgr in self.effects:
                    group_name = effect_mgr.group.name.replace("_", " ").title()
                    param_tables[group_name] = (effect_mgr.params, effect_mgr.group)
                param_tables["Mixer"] = (self.mixer.params, None)
                if self.audio_module:
                    param_tables["Audio"] = (self.audio_module.params, None)
                if self.obs_filters:
                    param_tables["OBS"] = (self.obs_filters.params, None)
                self.osc_controller = OSCController(
                    param_tables,
                    host=self.settings.osc_host, port=self.settings.osc_port
                )
            self.osc_controller.start()
            log.info(f"OSC server started on {self.settings.osc_host}:{self.settings.osc_port}")
        else:
            if self.osc_controller is not None:
                self.osc_controller.stop()
                log.info("OSC server stopped")

    def _toggle_api_server(self, enabled):
        """Start or stop the API server based on the toggle state."""
        if enabled:
            if self.api_server is None:
                from api import APIServer
                all_params = ParamTable()
                for effect_mgr in self.effects:
                    all_params.params.update(effect_mgr.params.params)
                all_params.params.update(self.mixer.params.params)
                if self.audio_module:
                    all_params.params.update(self.audio_module.params.params)
                all_params.params.update(self.settings.params.params)
                if self.obs_filters:
                    all_params.params.update(self.obs_filters.params.params)
                self.api_server = APIServer(
                    all_params, mixer=self.mixer,
                    host=self.settings.api_host, port=self.settings.api_port
                )
            self.api_server.start()
            log.info(f"API server started on {self.settings.api_host}:{self.settings.api_port}")
        else:
            if self.api_server is not None:
                self.api_server.stop()
                log.info("API server stopped")

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
