# Video Synth — real-time collaborative visual art synthesizer.
# Copyright (C) 2026 Kyle Henderson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import enum
import logging
import socket
import webbrowser
from param import ParamTable, Param
from common import Groups, MixerSource, Widget, Layout
from audio_reactive import BAND_NAMES
from mixer import MixModes, FileSource
from save import SaveController
from pyqt_widgets import (QTextEditLogger, ColorPickerWidget, VideoWidget, LFOManagerDialog,
                           AudioLinkDialog, SequencerWidget, MidiMapperWidget, OSCMapperWidget,
                           LFO_BUTTON_LINKED_STYLE, LFO_BUTTON_UNLINKED_STYLE,
                           AUD_BUTTON_LINKED_STYLE, AUD_BUTTON_UNLINKED_STYLE)
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox, QRadioButton, QScrollArea, QToolButton, QSizePolicy, QLineEdit, QTabWidget, QComboBox, QDialog, QGridLayout, QListWidget, QColorDialog, QTextEdit, QCheckBox, QStackedWidget
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

    def __init__(self, effects, settings, mixer=None, audio_module=None, obs_filters=None, api_server=None, midi_mapper=None, osc_controller=None, save_controller=None):
        super().__init__()
        self.layout_style = settings.layout.value
        self.effects = effects
        self.settings = settings
        self.mixer = mixer
        self.audio_module = audio_module
        self.obs_filters = obs_filters
        self.api_server = api_server
        self._api_running = api_server is not None
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

        if save_controller is not None:
            self.save_controller = save_controller
        else:
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
        self._crash_recovery_check()

        self.mixer_widgets = {}
        self.param_widgets = {}

        # Section-pane dropdown state (populated by _create_section_scrolls / _build_dropdown_pane)
        self._section_scrolls = {}      # key → QScrollArea
        self._pane_dropdowns = {}       # pane_id → QComboBox
        self._pane_current = {}         # pane_id → current section key
        self._pane_content_layouts = {} # pane_id → QVBoxLayout of content holder

        self.osc_banks = [(effect_manager.oscs, effect_manager.group.name) for effect_manager in effects]

        self.setWindowTitle("PyQt Control Panel")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.showMaximized()

        self._create_layout()
        self.create_ui()
        self._add_web_api_panel()

        # Set up logging to GUI
        self._setup_logging()

        self.mod_refresh_timer = QTimer(self)
        self.mod_refresh_timer.timeout.connect(self._refresh_mod_buttons)
        self.mod_refresh_timer.start(250)

        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.save_controller.autosave)
        self.autosave_timer.start(30_000)  # autosave every 30 seconds

        # Autopilot (auto-cycle patches)
        self.autopilot_active = False
        self.autopilot_interval_ms = 10_000  # default 10 s
        self.autopilot_timer = QTimer(self)
        self.autopilot_timer.timeout.connect(self._autopilot_tick)

        # Performance control button references (set when layout is built)
        self.blackout_btn  = None
        self.freeze_btn    = None
        self.autopilot_btn = None
        self.bpm_label     = None

        # BPM refresh timer
        self.bpm_refresh_timer = QTimer(self)
        self.bpm_refresh_timer.timeout.connect(self._refresh_bpm_label)
        self.bpm_refresh_timer.start(500)

    def _crash_recovery_check(self):
        """On startup, check for a previous crash and offer to restore autosave."""
        from PyQt6.QtWidgets import QMessageBox
        crashed, has_autosave = self.save_controller.check_crash()
        if crashed and has_autosave:
            reply = QMessageBox.question(
                None,
                "Recover from crash?",
                "The previous session ended unexpectedly.\n\nRestore parameters from last autosave?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.save_controller.recover_from_autosave()
                self._refresh_all_widgets()
        elif crashed:
            log.warning("Previous session crashed but no autosave found.")
        self.save_controller.write_lock()

    def closeEvent(self, event):
        """Mark clean exit so crash recovery doesn't trigger on next startup."""
        self.autopilot_timer.stop()
        self.save_controller.clear_lock()
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    #  Keyboard shortcuts                                                  #
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event):
        """Global keyboard shortcuts for live performance."""
        from PyQt6.QtWidgets import QLineEdit, QTextEdit
        if isinstance(self.focusWidget(), (QLineEdit, QTextEdit)):
            super().keyPressEvent(event)
            return

        key  = event.key()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if key == Qt.Key.Key_Space:
            self._toggle_blackout()
        elif key == Qt.Key.Key_F:
            self._toggle_freeze()
        elif key == Qt.Key.Key_A:
            self._toggle_autopilot()
        elif key == Qt.Key.Key_Right and not ctrl:
            self.save_controller.load_next_patch()
        elif key == Qt.Key.Key_Left and not ctrl:
            self.save_controller.load_prev_patch()
        elif key == Qt.Key.Key_R and ctrl:
            self.save_controller.load_random_patch()
        elif key == Qt.Key.Key_S and ctrl:
            self.save_controller.save_patch()
        elif key == Qt.Key.Key_BracketRight:
            # Increase autopilot interval by 5 s
            self.autopilot_interval_ms = min(300_000, self.autopilot_interval_ms + 5_000)
            if self.autopilot_active:
                self.autopilot_timer.setInterval(self.autopilot_interval_ms)
            log.info(f"Autopilot interval: {self.autopilot_interval_ms // 1000}s")
        elif key == Qt.Key.Key_BracketLeft:
            # Decrease autopilot interval by 5 s (min 2 s)
            self.autopilot_interval_ms = max(2_000, self.autopilot_interval_ms - 5_000)
            if self.autopilot_active:
                self.autopilot_timer.setInterval(self.autopilot_interval_ms)
            log.info(f"Autopilot interval: {self.autopilot_interval_ms // 1000}s")
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    #  Performance controls                                                #
    # ------------------------------------------------------------------ #

    def _toggle_blackout(self):
        if self.mixer is None:
            return
        self.mixer.blackout = not self.mixer.blackout
        if self.blackout_btn:
            if self.mixer.blackout:
                self.blackout_btn.setStyleSheet(
                    "background-color: #000; color: #fff; font-weight: bold;")
                self.blackout_btn.setText("■ BLACKOUT")
            else:
                self.blackout_btn.setStyleSheet("")
                self.blackout_btn.setText("Blackout")

    def _toggle_freeze(self):
        if self.mixer is None:
            return
        self.mixer.freeze = not self.mixer.freeze
        if self.freeze_btn:
            if self.mixer.freeze:
                self.freeze_btn.setStyleSheet(
                    "background-color: #1565C0; color: #fff; font-weight: bold;")
                self.freeze_btn.setText("❚❚ FREEZE")
            else:
                self.freeze_btn.setStyleSheet("")
                self.freeze_btn.setText("Freeze")

    def _toggle_autopilot(self):
        self.autopilot_active = not self.autopilot_active
        if self.autopilot_active:
            self.autopilot_timer.start(self.autopilot_interval_ms)
        else:
            self.autopilot_timer.stop()
        if self.autopilot_btn:
            if self.autopilot_active:
                self.autopilot_btn.setStyleSheet(
                    "background-color: #4CAF50; color: #fff; font-weight: bold;")
                self.autopilot_btn.setText("▶ AUTO")
            else:
                self.autopilot_btn.setStyleSheet("")
                self.autopilot_btn.setText("Autopilot")

    def _autopilot_tick(self):
        self.save_controller.load_next_patch()

    def _refresh_bpm_label(self):
        if self.bpm_label is None:
            return
        if self.audio_module and hasattr(self.audio_module, 'beat_detector'):
            bpm = self.audio_module.beat_detector.bpm
            beat = self.audio_module.beat_detector.is_beat
            text = f"BPM: {bpm:.1f}"
            style = "color: #FF9800; font-weight: bold;" if beat else "color: #aaa;"
            self.bpm_label.setText(text)
            self.bpm_label.setStyleSheet(style)

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
        self._section_scrolls.clear()
        self._pane_dropdowns.clear()
        self._pane_current.clear()
        self._pane_content_layouts.clear()

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


    # ------------------------------------------------------------------ #
    #  Shared section-pane helpers (dropdown-driven content areas)        #
    # ------------------------------------------------------------------ #

    _SECTION_KEYS = ['src1_effects', 'src1_animations', 'src2_effects', 'src2_animations', 'post_effects']
    _SECTION_LABELS = {
        'src1_effects':    'Src 1 Effects',
        'src1_animations': 'Src 1 Animations',
        'src2_effects':    'Src 2 Effects',
        'src2_animations': 'Src 2 Animations',
        'post_effects':    'Post Effects',
    }

    def _create_section_scrolls(self):
        """Create the 5 section scroll areas and wire up layout attributes used by create_ui()."""
        self._section_scrolls = {}
        for key in self._SECTION_KEYS:
            container = QWidget()
            QVBoxLayout(container)  # layout owned by container
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(container)
            self._section_scrolls[key] = scroll

        # Assign the layout attributes create_ui() depends on
        self.src1_effects_layout    = self._section_scrolls['src1_effects'].widget().layout()
        self.src1_animations_layout = self._section_scrolls['src1_animations'].widget().layout()
        self.src2_effects_layout    = self._section_scrolls['src2_effects'].widget().layout()
        self.src2_animations_layout = self._section_scrolls['src2_animations'].widget().layout()
        self.post_effects_layout    = self._section_scrolls['post_effects'].widget().layout()
        self.post_effects_container = self._section_scrolls['post_effects'].widget()

        # uncategorized_layout must exist for create_ui() even though it isn't in any pane
        _unc = QWidget()
        self.uncategorized_layout = QVBoxLayout(_unc)
        self._uncategorized_container = _unc  # keep alive

    def _build_dropdown_pane(self, pane_id, initial_key):
        """Return a QWidget containing a section-select dropdown and a content holder."""
        pane = QWidget()
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        pane_layout.setSpacing(2)

        dropdown = QComboBox()
        for key in self._SECTION_KEYS:
            dropdown.addItem(self._SECTION_LABELS[key], key)
        dropdown.setCurrentIndex(self._SECTION_KEYS.index(initial_key))
        pane_layout.addWidget(dropdown)

        content_holder = QWidget()
        holder_layout = QVBoxLayout(content_holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(0)
        holder_layout.addWidget(self._section_scrolls[initial_key])
        pane_layout.addWidget(content_holder, 1)

        self._pane_dropdowns[pane_id]        = dropdown
        self._pane_current[pane_id]          = initial_key
        self._pane_content_layouts[pane_id]  = holder_layout

        dropdown.currentIndexChanged.connect(
            lambda idx, pid=pane_id, cb=dropdown: self._on_section_changed(pid, cb.itemData(idx))
        )
        return pane

    def _sync_pane_exclusion(self):
        """After all panes are built, disable each pane's current selection in every other pane."""
        pane_ids = list(self._pane_dropdowns.keys())
        for pid in pane_ids:
            current_key = self._pane_current[pid]
            for other_pid in pane_ids:
                if other_pid == pid:
                    continue
                other_cb = self._pane_dropdowns[other_pid]
                for i in range(other_cb.count()):
                    if other_cb.itemData(i) == current_key:
                        item = other_cb.model().item(i)
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                        break

    def _on_section_changed(self, pane_id, new_key):
        """Handle a section dropdown change: reparent content and update cross-pane exclusion."""
        if not new_key or new_key == self._pane_current.get(pane_id):
            return
        old_key = self._pane_current[pane_id]
        pane_ids = list(self._pane_dropdowns.keys())

        # Update cross-pane disable/enable in all other panes
        for other_pid in pane_ids:
            if other_pid == pane_id:
                continue
            other_cb = self._pane_dropdowns[other_pid]
            for i in range(other_cb.count()):
                item_key = other_cb.itemData(i)
                item = other_cb.model().item(i)
                if item_key == old_key:
                    # old_key is now free — re-enable unless another pane still uses it
                    still_used = any(
                        self._pane_current[p] == old_key
                        for p in pane_ids if p != pane_id
                    )
                    if not still_used:
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
                elif item_key == new_key:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

        # Swap content widgets
        holder_layout = self._pane_content_layouts[pane_id]
        old_scroll = self._section_scrolls[old_key]
        holder_layout.removeWidget(old_scroll)
        old_scroll.setParent(None)

        new_scroll = self._section_scrolls[new_key]
        holder_layout.addWidget(new_scroll)

        self._pane_current[pane_id] = new_key

    def _create_quad_layout(self):
        self.root_layout = QGridLayout(self.central_widget)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        # --- Left Column: two dropdown-driven section panes ---
        self._create_section_scrolls()

        left_column_widget = QWidget()
        left_column_layout = QVBoxLayout(left_column_widget)

        top_left_pane    = self._build_dropdown_pane('top',    'src1_effects')
        bottom_left_pane = self._build_dropdown_pane('bottom', 'post_effects')
        self._sync_pane_exclusion()

        left_column_layout.addWidget(top_left_pane,    1)
        left_column_layout.addWidget(bottom_left_pane, 1)

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
        All three source/post panes are dropdown-driven (mutually exclusive).
        """
        self.root_layout = QGridLayout(self.central_widget)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        self._create_section_scrolls()

        # --- Top-Left: dropdown pane (default: Src 1 Effects) ---
        self.root_layout.addWidget(self._build_dropdown_pane('top_left', 'src1_effects'), 0, 0)

        # --- Top-Right: reset buttons + dropdown pane (default: Src 2 Effects) ---
        top_right_widget = QWidget()
        top_right_outer = QHBoxLayout(top_right_widget)
        top_right_outer.setContentsMargins(0, 0, 0, 0)
        top_right_outer.setSpacing(0)

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
        top_right_outer.addWidget(self._build_dropdown_pane('top_right', 'src2_effects'), 1)

        self.root_layout.addWidget(top_right_widget, 0, 1)

        # --- Bottom-Left: dropdown pane (default: Post Effects) ---
        self.root_layout.addWidget(self._build_dropdown_pane('bottom_left', 'post_effects'), 1, 0)

        self._sync_pane_exclusion()

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

        self._create_section_scrolls()

        # Top Pane: dropdown-driven section selector
        top_pane = self._build_dropdown_pane('top', 'src1_effects')
        self.root_layout.addWidget(top_pane)

        # Bottom Pane: dropdown-driven section selector + Mixer/Settings/MIDI/OSC tabs
        bottom_pane_tabs = QTabWidget()
        self.root_layout.addWidget(bottom_pane_tabs)

        bottom_section_pane = self._build_dropdown_pane('bottom', 'post_effects')
        self._sync_pane_exclusion()
        bottom_pane_tabs.addTab(bottom_section_pane, "Section")

        # Mixer Tab
        mixer_container = QWidget()
        self.mixer_layout = QVBoxLayout(mixer_container)
        mixer_scroll = QScrollArea()
        mixer_scroll.setWidgetResizable(True)
        mixer_scroll.setWidget(mixer_container)
        bottom_pane_tabs.addTab(mixer_scroll, "Mixer")

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
                    patch_row = QHBoxLayout()
                    for label, slot in [
                        ("Save",     self.save_controller.save_patch),
                        ("← Prev",   self.save_controller.load_prev_patch),
                        ("Random",   self.save_controller.load_random_patch),
                        ("Next →",   self.save_controller.load_next_patch),
                    ]:
                        btn = QPushButton(label)
                        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                        btn.clicked.connect(slot)
                        patch_row.addWidget(btn)
                    target_layout.addLayout(patch_row)

                    # Performance controls row (Blackout / Freeze / Autopilot / BPM)
                    perf_row = QHBoxLayout()

                    self.blackout_btn = QPushButton("Blackout")
                    self.blackout_btn.setToolTip("Cut to black  [Space]")
                    self.blackout_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                    self.blackout_btn.clicked.connect(self._toggle_blackout)
                    perf_row.addWidget(self.blackout_btn)

                    self.freeze_btn = QPushButton("Freeze")
                    self.freeze_btn.setToolTip("Freeze output frame  [F]")
                    self.freeze_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                    self.freeze_btn.clicked.connect(self._toggle_freeze)
                    perf_row.addWidget(self.freeze_btn)

                    self.autopilot_btn = QPushButton("Autopilot")
                    self.autopilot_btn.setToolTip("Auto-cycle patches  [A]  |  [ ] adjust interval")
                    self.autopilot_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                    self.autopilot_btn.clicked.connect(self._toggle_autopilot)
                    perf_row.addWidget(self.autopilot_btn)

                    self.bpm_label = QLabel("BPM: --")
                    self.bpm_label.setStyleSheet("color: #aaa;")
                    self.bpm_label.setToolTip("Detected BPM from audio input")
                    self.bpm_label.setFixedWidth(80)
                    perf_row.addWidget(self.bpm_label)

                    target_layout.addLayout(perf_row)

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
                            s1_combo.currentTextChanged.connect(lambda _: self._update_mixer_visibility())
                            source_layout.addWidget(s1_label)
                            source_layout.addWidget(s1_combo)

                            s2_param = source_params['source_2']
                            s2_label = QLabel(s2_param.name)
                            s2_combo = QComboBox()
                            s2_combo.addItems([str(o) for o in s2_param.options])
                            s2_combo.setCurrentText(str(s2_param.value))
                            s2_combo.currentTextChanged.connect(lambda text, p=s2_param: self._on_dropdown_change(p, text))
                            s2_combo.currentTextChanged.connect(lambda _: self._update_mixer_visibility())
                            source_layout.addWidget(s2_label)
                            source_layout.addWidget(s2_combo)

                            swap_button = QPushButton("Swap")
                            swap_button.clicked.connect(lambda: self._swap_sources(s1_param, s2_param, s1_combo, s2_combo))
                            source_layout.addWidget(swap_button)
                            
                            target_layout.addLayout(source_layout)

                        if blend_mode_param:
                            widget = self._create_param_widget(blend_mode_param)
                            self.mixer_widgets['blend_mode'] = widget
                            target_layout.addWidget(widget)

                        if len(chroma_key_hue_sat_val_params) == 6:
                            chroma_key_dict = {p.name: p for p in chroma_key_hue_sat_val_params}
                            chroma_container = QWidget()
                            color_pickers_layout = QHBoxLayout(chroma_container)
                            color_pickers_layout.setContentsMargins(0, 0, 0, 0)

                            upper_picker = ColorPickerWidget('Upper Color', chroma_key_dict['upper_hue'], chroma_key_dict['upper_sat'], chroma_key_dict['upper_val'])
                            lower_picker = ColorPickerWidget('Lower Color', chroma_key_dict['lower_hue'], chroma_key_dict['lower_sat'], chroma_key_dict['lower_val'])

                            color_pickers_layout.addWidget(upper_picker)
                            color_pickers_layout.addWidget(lower_picker)

                            target_layout.addWidget(chroma_container)

                            self.mixer_widgets['upper_color_picker'] = upper_picker
                            self.mixer_widgets['lower_color_picker'] = lower_picker
                            self.mixer_widgets['chroma_pickers'] = chroma_container

                        for param in remaining_params:
                            widget = self._create_param_widget(param)
                            self.mixer_widgets[param.name] = widget
                            target_layout.addWidget(widget)

                else:
                    for params_in_subgroup in subgroups.values():
                        for param in params_in_subgroup:
                            widget = self._create_param_widget(param)
                            target_layout.addWidget(widget)
                target_layout.addStretch(1)
            else:
                # Create dropdown + stacked widget for subgroups
                subgroup_combo = QComboBox()
                subgroup_stack = QStackedWidget()

                for subgroup_name, params_in_subgroup in subgroups.items():
                    subgroup_widget = QWidget()
                    subgroup_layout = QVBoxLayout(subgroup_widget)

                    for param in params_in_subgroup:
                        widget = self._create_param_widget(param)
                        subgroup_layout.addWidget(widget)
                    subgroup_layout.addStretch(1)

                    subgroup_scroll = QScrollArea()
                    subgroup_scroll.setWidgetResizable(True)
                    subgroup_scroll.setWidget(subgroup_widget)

                    tab_title = subgroup_name.replace("_", " ").title() if isinstance(subgroup_name, str) else subgroup_name.name.replace("_", " ").title()
                    subgroup_combo.addItem(tab_title)
                    subgroup_stack.addWidget(subgroup_scroll)

                # Add sequencer as a stacked page within effects groups
                effect_manager = {
                    Groups.SRC_1_EFFECTS: self.src_1_effects,
                    Groups.SRC_2_EFFECTS: self.src_2_effects,
                    Groups.POST_EFFECTS: self.post_effects,
                }.get(group_enum_or_str)
                if effect_manager:
                    seq_widget = SequencerWidget(effect_manager)
                    subgroup_combo.addItem("Sequence")
                    subgroup_stack.addWidget(seq_widget)

                subgroup_combo.currentIndexChanged.connect(subgroup_stack.setCurrentIndex)
                target_layout.addWidget(subgroup_combo)
                target_layout.addWidget(subgroup_stack, 1)

        self._update_mixer_visibility()


    def _create_param_widget(self, param: Param, register=True, display_name=None):
        widget = QWidget()
        widget.setProperty("param_name", param.name)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label_text = display_name if display_name else param.name.replace("_", " ").title()
        label = QLabel(label_text)
        label.setFixedWidth(100)
        if param.info:
            tooltip = f"{param.info}\n[{param.name}]"
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
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


    def _swap_sources(self, s1_param, s2_param, s1_combo, s2_combo):
        """Swap the values of source_1 and source_2 and update their combos."""
        v1, v2 = s1_param.value, s2_param.value
        s1_param.value = v2
        s2_param.value = v1
        s1_combo.blockSignals(True)
        s2_combo.blockSignals(True)
        s1_combo.setCurrentText(str(v2))
        s2_combo.setCurrentText(str(v1))
        s1_combo.blockSignals(False)
        s2_combo.blockSignals(False)
        if self.mixer:
            self.mixer.start_video(v2, MixerSource.SRC_1)
            self.mixer.start_video(v1, MixerSource.SRC_2)

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
        elif param.name == "lan_enabled":
            if hasattr(self, "lan_checkbox"):
                self.lan_checkbox.blockSignals(True)
                self.lan_checkbox.setChecked(bool(param.value))
                self.lan_checkbox.blockSignals(False)
            # Rebind a running server to the new host.
            if self.api_server is not None:
                was_running = self._api_running
                self.api_server.stop()
                self.api_server = None
                self._api_running = False
                if was_running:
                    self._toggle_api_server(1)
            self._sync_web_api_panel()

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
            host = "0.0.0.0" if self.settings.lan_enabled.value else self.settings.api_host
            if self.api_server is None:
                from api import APIServer
                # Prefix effect params by group so src_1/src_2/post names don't
                # collide (matches the startup path in __main__.py).
                all_params = ParamTable()
                for effect_mgr in self.effects:
                    group_key = effect_mgr.group.name if hasattr(effect_mgr.group, 'name') else str(effect_mgr.group)
                    for k, v in effect_mgr.params.params.items():
                        all_params.params[f"{group_key}.{k}"] = v
                all_params.params.update(self.mixer.params.params)
                if self.audio_module:
                    all_params.params.update(self.audio_module.params.params)
                all_params.params.update(self.settings.params.params)
                if self.obs_filters:
                    all_params.params.update(self.obs_filters.params.params)
                osc_banks = {
                    (em.group.name if hasattr(em.group, 'name') else str(em.group)): em.oscs
                    for em in self.effects
                }
                self.api_server = APIServer(
                    all_params, mixer=self.mixer,
                    save_controller=self.save_controller,
                    midi_mapper=self.midi_mapper,
                    osc_banks=osc_banks,
                    audio_module=self.audio_module,
                    host=host, port=self.settings.api_port
                )
            self.api_server.start()
            self._api_running = True
            log.info(f"API server started on {host}:{self.settings.api_port}")
        else:
            if self.api_server is not None:
                self.api_server.stop()
            self._api_running = False
            log.info("API server stopped")
        self._sync_web_api_panel()

    # ------------------------------------------------------------------
    # Web / API server control panel
    # ------------------------------------------------------------------
    def _build_web_api_panel(self):
        """Build the always-visible Web/API server control box."""
        box = QGroupBox("Web / API Server")
        v = QVBoxLayout(box)

        self.web_api_url_label = QLabel("Server stopped")
        v.addWidget(self.web_api_url_label)

        row = QHBoxLayout()
        self.web_api_toggle_btn = QPushButton("Start Web UI")
        self.web_api_toggle_btn.clicked.connect(self._on_web_api_button)
        row.addWidget(self.web_api_toggle_btn)

        self.web_api_open_btn = QPushButton("Open in Browser")
        self.web_api_open_btn.clicked.connect(self._open_web_ui)
        row.addWidget(self.web_api_open_btn)
        v.addLayout(row)

        self.lan_checkbox = QCheckBox("Allow LAN access (bind 0.0.0.0)")
        self.lan_checkbox.setChecked(bool(self.settings.lan_enabled.value))
        self.lan_checkbox.setToolTip(
            "Expose the server on your local network so phones, tablets or other "
            "machines can connect. Off = localhost only."
        )
        self.lan_checkbox.stateChanged.connect(self._on_lan_toggle)
        v.addWidget(self.lan_checkbox)

        self._sync_web_api_panel()
        return box

    def _add_web_api_panel(self):
        """Insert the Web/API panel at the top of the User Settings tab."""
        if not hasattr(self, "user_settings_layout") or self.user_settings_layout is None:
            return
        self.user_settings_layout.insertWidget(0, self._build_web_api_panel())

    def _lan_ip(self):
        """Best-effort primary LAN IPv4 address for display."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return "127.0.0.1"

    def _web_ui_url(self):
        host = self._lan_ip() if self.settings.lan_enabled.value else "127.0.0.1"
        return f"http://{host}:{self.settings.api_port}/ui/"

    def _on_web_api_button(self):
        self._toggle_api_server(0 if self._api_running else 1)
        self.settings.api_enabled.value = 1 if self._api_running else 0

    def _on_lan_toggle(self, state):
        enabled = 1 if state == Qt.CheckState.Checked.value else 0
        self.settings.lan_enabled.value = enabled
        # Rebinding the host requires a fresh server instance; restart if running.
        if self.api_server is not None:
            was_running = self._api_running
            self.api_server.stop()
            self.api_server = None
            self._api_running = False
            if was_running:
                self._toggle_api_server(1)
        self._sync_web_api_panel()

    def _open_web_ui(self):
        if not self._api_running:
            self._toggle_api_server(1)
            self.settings.api_enabled.value = 1
        webbrowser.open(self._web_ui_url())

    def _sync_web_api_panel(self):
        """Reflect actual server state in the panel widgets."""
        if not hasattr(self, "web_api_toggle_btn"):
            return
        running = self._api_running
        self.web_api_toggle_btn.setText("Stop Web UI" if running else "Start Web UI")
        if running:
            self.web_api_url_label.setText(f"Running at {self._web_ui_url()}")
            self.web_api_url_label.setStyleSheet("color:#4CAF50;")
        else:
            self.web_api_url_label.setText("Server stopped")
            self.web_api_url_label.setStyleSheet("color:#aaa;")

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
        if not self.mixer:
            return
        blend_mode = self.mixer.blend_mode.value
        is_alpha  = blend_mode == MixModes.ALPHA_BLEND.value
        is_luma   = blend_mode == MixModes.LUMA_KEY.value
        is_chroma = blend_mode == MixModes.CHROMA_KEY.value

        # Which params are visible per mode
        _alpha_only  = {'alpha_blend'}
        _luma_only   = {'luma_threshold', 'luma_selection', 'luma_blur'}
        _chroma_only = {'chroma_pickers', 'upper_color_picker', 'lower_color_picker'}
        # Source-conditional params — visible only when the matching source type is selected
        src1 = str(self.mixer.selected_source1.value) if self.mixer else ''
        src2 = str(self.mixer.selected_source2.value) if self.mixer else ''
        _src_conditional = {
            'video_file_src1': src1 == 'VIDEO',
            'video_file_src2': src2 == 'VIDEO',
            'video_pause_src1': src1 == 'VIDEO',
            'video_pause_src2': src2 == 'VIDEO',
            'video_scrub_src1': src1 == 'VIDEO',
            'video_scrub_src2': src2 == 'VIDEO',
            'image_file_src1': src1 == 'IMAGE',
            'image_file_src2': src2 == 'IMAGE',
        }

        for name, widget in self.mixer_widgets.items():
            if name in _alpha_only:
                widget.setVisible(is_alpha)
            elif name in _luma_only:
                widget.setVisible(is_luma)
            elif name in _chroma_only:
                widget.setVisible(is_chroma)
            elif name in _src_conditional:
                widget.setVisible(_src_conditional[name])
            # blend_mode, source_1/2 dropdowns, swap_sources — always visible


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
                if radio_buttons and hasattr(param.value, 'name'):
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
                    else: # Enum or list
                        try:
                            options_list = list(param.options)
                            if options_list and hasattr(options_list[0], 'value'):
                                idx = [o.value for o in options_list].index(param.value)
                            else:
                                idx = options_list.index(param.value)
                            combo_box.setCurrentIndex(idx)
                        except (ValueError, IndexError):
                            pass
