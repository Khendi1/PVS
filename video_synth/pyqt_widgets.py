"""
Custom PyQt widgets used by the video synthesizer GUI.
Extracted from pyqt_gui.py to keep the main GUI file focused on layout and event handling.
"""

import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
                              QGroupBox, QScrollArea, QSizePolicy, QLineEdit, QTabWidget,
                              QComboBox, QDialog, QListWidget, QColorDialog, QTextEdit, QCheckBox)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QTextCursor
from PyQt6.QtCore import Qt, QSize, pyqtSlot, QTimer


log = logging.getLogger(__name__)

# Shared button styles used by LFOManagerDialog, AudioLinkDialog, and PyQTGUI
LFO_BUTTON_UNLINKED_STYLE = "QPushButton { background-color: #607D8B; color: white; }"
LFO_BUTTON_LINKED_STYLE = "QPushButton { background-color: #4CAF50; color: white; }"
AUD_BUTTON_UNLINKED_STYLE = "QPushButton { background-color: #607D8B; color: white; }"
AUD_BUTTON_LINKED_STYLE = "QPushButton { background-color: #FF9800; color: white; }"


class QTextEditLogger(logging.Handler):
    """Custom Qt logging handler that emits log messages to a QTextEdit widget."""
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)
        self.widget.moveCursor(QTextCursor.MoveOperation.End)


class ColorPickerWidget(QWidget):
    """Custom color picker widget for selecting HSV colors."""
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


class VideoWidget(QLabel):
    """Widget to display video frames with aspect ratio preservation."""
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


class LFOManagerDialog(QDialog):
    """Dialog for managing LFO linkage to a parameter and editing LFO settings."""
    def __init__(self, param, osc_bank, gui_instance, mod_button, group=None):
        super().__init__(group)
        self.param = param
        self.osc_bank = osc_bank
        self.gui_instance = gui_instance
        self.mod_button = mod_button
        self.setWindowTitle(f"LFO for {param.name}")
        self.setWindowFlags(Qt.WindowType.Popup)

        self.layout = QVBoxLayout(self)
        self.rebuild_ui()

    def rebuild_ui(self):
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

            self.controls_container = QWidget()
            self.controls_layout = QVBoxLayout(self.controls_container)
            self.layout.addWidget(self.controls_container)

            osc = self.param.linked_oscillator
            for param_name in ['shape', 'frequency', 'amplitude', 'phase', 'seed', 'cutoff_min', 'cutoff_max', 'noise_octaves', 'noise_persistence', 'noise_lacunarity', 'noise_repeat', 'noise_base']:
                if hasattr(osc, param_name):
                    param = getattr(osc, param_name)
                    display_name = param_name.replace("_", " ").title()
                    widget = self.gui_instance._create_param_widget(param, register=False, display_name=display_name)
                    self.controls_layout.addWidget(widget)

    def link_new_lfo(self):
        osc_name = f"{self.param.name}"
        new_osc = self.osc_bank.add_oscillator(name=osc_name)
        new_osc.link_param(self.param)
        self.param.linked_oscillator = new_osc
        self.mod_button.setStyleSheet(LFO_BUTTON_LINKED_STYLE)
        self.rebuild_ui()

    def unlink_lfo(self):
        if self.param.linked_oscillator:
            self.osc_bank.remove_oscillator(self.param.linked_oscillator)
            self.param.linked_oscillator.unlink_param()
            self.param.linked_oscillator = None
            self.mod_button.setStyleSheet(LFO_BUTTON_UNLINKED_STYLE)
            self.rebuild_ui()


class AudioLinkDialog(QDialog):
    """Dialog for managing audio band linkage to a parameter."""
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
        self.aud_button.setStyleSheet(AUD_BUTTON_LINKED_STYLE)
        self.rebuild_ui()

    def unlink_band(self):
        if self.param.linked_audio_band:
            self.audio_module.remove_band(self.param.linked_audio_band)
            self.param.linked_audio_band.unlink_param()
            self.param.linked_audio_band = None
            self.aud_button.setStyleSheet(AUD_BUTTON_UNLINKED_STYLE)
            self.rebuild_ui()


class SequencerWidget(QWidget):
    """Widget for reordering the effects processing chain via drag-and-drop."""
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


class MidiMapperWidget(QWidget):
    """GUI widget for the MidiMapper learn/mapping system."""

    LEARN_IDLE_STYLE = "QPushButton { background-color: #607D8B; color: white; }"
    LEARN_ACTIVE_STYLE = "QPushButton { background-color: #f44336; color: white; }"
    MAPPED_STYLE = "background-color: #2d2d2d; color: #e0e0e0;"

    def __init__(self, midi_mapper, parent=None):
        super().__init__(parent)
        self.midi_mapper = midi_mapper
        self._grouped_params = midi_mapper.get_all_qualified_keys()

        root = QVBoxLayout(self)

        # --- Learn Mode Section ---
        learn_group = QGroupBox("Learn Mode")
        learn_layout = QVBoxLayout(learn_group)

        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("Group:"))
        self.group_combo = QComboBox()
        for group_name in self._grouped_params:
            self.group_combo.addItem(group_name)
        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        group_row.addWidget(self.group_combo, 1)
        learn_layout.addLayout(group_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Param:"))
        self.param_combo = QComboBox()
        param_row.addWidget(self.param_combo, 1)
        learn_layout.addLayout(param_row)

        self._on_group_changed()

        button_row = QHBoxLayout()
        self.learn_button = QPushButton("Learn")
        self.learn_button.setStyleSheet(self.LEARN_IDLE_STYLE)
        self.learn_button.clicked.connect(self._on_learn_click)
        button_row.addWidget(self.learn_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel_click)
        button_row.addWidget(self.cancel_button)
        learn_layout.addLayout(button_row)

        self.learn_status = QLabel("Idle")
        learn_layout.addWidget(self.learn_status)

        root.addWidget(learn_group, 1)

        # --- Mappings Table Section ---
        mappings_group = QGroupBox("Current Mappings")
        mappings_layout = QVBoxLayout(mappings_group)

        self.mappings_list = QListWidget()
        self.mappings_list.setStyleSheet(self.MAPPED_STYLE)
        mappings_layout.addWidget(self.mappings_list)

        actions_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_mappings)
        actions_row.addWidget(refresh_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save_click)
        actions_row.addWidget(save_btn)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self._on_delete_click)
        actions_row.addWidget(delete_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_all_click)
        actions_row.addWidget(clear_btn)

        mappings_layout.addLayout(actions_row)
        mappings_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        root.addWidget(mappings_group, 0)

        # --- Ports Section ---
        ports_group = QGroupBox("Connected Ports")
        ports_layout = QVBoxLayout(ports_group)
        ports_layout.setContentsMargins(4, 4, 4, 4)
        self.ports_list = QListWidget()
        self.ports_list.setStyleSheet(self.MAPPED_STYLE)
        ports_layout.addWidget(self.ports_list)
        ports_group.setMaximumHeight(100)
        root.addWidget(ports_group, 0)

        # Timer to poll learn state and refresh mappings
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_learn_state)
        self._poll_timer.start(250)

        self._refresh_mappings()
        self._refresh_ports()

    def _on_group_changed(self):
        self.param_combo.clear()
        group_name = self.group_combo.currentText()
        params = self._grouped_params.get(group_name, [])
        for name in params:
            self.param_combo.addItem(name.replace("_", " ").title(), name)

    def _get_qualified_key(self):
        group = self.group_combo.currentText()
        param = self.param_combo.currentData()
        if not group or not param:
            return None
        return f"{group}/{param}"

    def _on_learn_click(self):
        qualified_key = self._get_qualified_key()
        if qualified_key is None:
            return
        if self.midi_mapper.start_learn(qualified_key):
            self.learn_button.setStyleSheet(self.LEARN_ACTIVE_STYLE)
            self.learn_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.learn_status.setText(f"Move a knob/fader to map to '{qualified_key}'...")

    def _on_cancel_click(self):
        self.midi_mapper.cancel_learn()
        self._reset_learn_ui()

    def _reset_learn_ui(self):
        self.learn_button.setStyleSheet(self.LEARN_IDLE_STYLE)
        self.learn_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.learn_status.setText("Idle")

    def _poll_learn_state(self):
        state = self.midi_mapper.get_learn_state()
        if not state["learning"]:
            if not self.learn_button.isEnabled():
                self._reset_learn_ui()
                self.learn_status.setText("Mapping saved!")
                self._refresh_mappings()
                self._refresh_ports()

    def _refresh_mappings(self):
        self.mappings_list.clear()
        mappings = self.midi_mapper.get_mappings()
        for port_name, cc_map in mappings.items():
            for cc, qualified_key in sorted(cc_map.items()):
                display = f"{port_name}  |  CC {cc:>3d}  ->  {qualified_key}"
                self.mappings_list.addItem(display)
        if self.mappings_list.count() == 0:
            self.mappings_list.addItem("(no mappings)")

    def _refresh_ports(self):
        self.ports_list.clear()
        for port_name in self.midi_mapper.controllers:
            n = len(self.midi_mapper.controllers[port_name].mappings)
            self.ports_list.addItem(f"{port_name}  ({n} mapping{'s' if n != 1 else ''})")
        if self.ports_list.count() == 0:
            self.ports_list.addItem("(no MIDI ports detected)")

    def _on_save_click(self):
        self.midi_mapper.save_mappings()
        self.learn_status.setText("Mappings saved to YAML")

    def _on_delete_click(self):
        selected = self.mappings_list.currentItem()
        if selected is None or selected.text().startswith("("):
            return
        text = selected.text()
        parts = text.split("|")
        if len(parts) != 2:
            return
        port_name = parts[0].strip()
        cc_param = parts[1].strip()
        cc_str = cc_param.split("->")[0].replace("CC", "").strip()
        try:
            cc = int(cc_str)
        except ValueError:
            return
        self.midi_mapper.clear_mapping(port_name, cc)
        self._refresh_mappings()
        self._refresh_ports()
        self.learn_status.setText(f"Deleted CC {cc} from '{port_name}'")

    def _on_clear_all_click(self):
        self.midi_mapper.clear_all_mappings()
        self._refresh_mappings()
        self._refresh_ports()
        self.learn_status.setText("All mappings cleared")


class OSCMapperWidget(QWidget):
    """GUI widget for the OSCController learn/mapping system."""

    LEARN_IDLE_STYLE = "QPushButton { background-color: #607D8B; color: white; }"
    LEARN_ACTIVE_STYLE = "QPushButton { background-color: #f44336; color: white; }"
    MAPPED_STYLE = "background-color: #2d2d2d; color: #e0e0e0;"

    def __init__(self, osc_controller, parent=None):
        super().__init__(parent)
        self.osc_controller = osc_controller
        self._grouped_params = osc_controller.get_all_qualified_keys()

        root = QVBoxLayout(self)

        # --- Learn Mode Section ---
        learn_group = QGroupBox("OSC Learn Mode")
        learn_layout = QVBoxLayout(learn_group)

        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("Group:"))
        self.group_combo = QComboBox()
        for group_name in self._grouped_params:
            self.group_combo.addItem(group_name)
        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        group_row.addWidget(self.group_combo, 1)
        learn_layout.addLayout(group_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Param:"))
        self.param_combo = QComboBox()
        param_row.addWidget(self.param_combo, 1)
        learn_layout.addLayout(param_row)

        self._on_group_changed()

        button_row = QHBoxLayout()
        self.learn_button = QPushButton("Learn")
        self.learn_button.setStyleSheet(self.LEARN_IDLE_STYLE)
        self.learn_button.clicked.connect(self._on_learn_click)
        button_row.addWidget(self.learn_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel_click)
        button_row.addWidget(self.cancel_button)
        learn_layout.addLayout(button_row)

        self.learn_status = QLabel("Idle")
        learn_layout.addWidget(self.learn_status)

        root.addWidget(learn_group, 1)

        # --- Mappings Section ---
        mappings_group = QGroupBox("Current Mappings")
        mappings_layout = QVBoxLayout(mappings_group)

        self.mappings_list = QListWidget()
        self.mappings_list.setStyleSheet(self.MAPPED_STYLE)
        mappings_layout.addWidget(self.mappings_list)

        actions_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_mappings)
        actions_row.addWidget(refresh_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save_click)
        actions_row.addWidget(save_btn)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self._on_delete_click)
        actions_row.addWidget(delete_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_all_click)
        actions_row.addWidget(clear_btn)

        mappings_layout.addLayout(actions_row)
        mappings_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        root.addWidget(mappings_group, 0)

        # --- Server Info Section ---
        info_group = QGroupBox("Server")
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(4, 4, 4, 4)
        self.info_label = QLabel(f"Listening on {osc_controller.host}:{osc_controller.port}")
        info_layout.addWidget(self.info_label)
        info_group.setMaximumHeight(60)
        root.addWidget(info_group, 0)

        # Timer to poll learn state
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_learn_state)
        self._poll_timer.start(250)

        self._refresh_mappings()

    def _on_group_changed(self):
        self.param_combo.clear()
        group_name = self.group_combo.currentText()
        params = self._grouped_params.get(group_name, [])
        for name in params:
            self.param_combo.addItem(name.replace("_", " ").title(), name)

    def _get_qualified_key(self):
        group = self.group_combo.currentText()
        param = self.param_combo.currentData()
        if not group or not param:
            return None
        return f"{group}/{param}"

    def _on_learn_click(self):
        qualified_key = self._get_qualified_key()
        if qualified_key is None:
            return
        if self.osc_controller.start_learn(qualified_key):
            self.learn_button.setStyleSheet(self.LEARN_ACTIVE_STYLE)
            self.learn_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.learn_status.setText(f"Send an OSC message to map to '{qualified_key}'...")

    def _on_cancel_click(self):
        self.osc_controller.cancel_learn()
        self._reset_learn_ui()

    def _reset_learn_ui(self):
        self.learn_button.setStyleSheet(self.LEARN_IDLE_STYLE)
        self.learn_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.learn_status.setText("Idle")

    def _poll_learn_state(self):
        state = self.osc_controller.get_learn_state()
        if not state["learning"]:
            if not self.learn_button.isEnabled():
                self._reset_learn_ui()
                self.learn_status.setText("Mapping saved!")
                self._refresh_mappings()

    def _refresh_mappings(self):
        self.mappings_list.clear()
        mappings = self.osc_controller.get_mappings()
        for address, qualified_key in sorted(mappings.items()):
            display = f"{address}  ->  {qualified_key}"
            self.mappings_list.addItem(display)
        if self.mappings_list.count() == 0:
            self.mappings_list.addItem("(no mappings)")

    def _on_save_click(self):
        self.osc_controller.save_mappings()
        self.learn_status.setText("Mappings saved to YAML")

    def _on_delete_click(self):
        selected = self.mappings_list.currentItem()
        if selected is None or selected.text().startswith("("):
            return
        text = selected.text()
        parts = text.split("->")
        if len(parts) != 2:
            return
        address = parts[0].strip()
        self.osc_controller.clear_mapping(address)
        self._refresh_mappings()
        self.learn_status.setText(f"Deleted mapping for '{address}'")

    def _on_clear_all_click(self):
        self.osc_controller.clear_all_mappings()
        self._refresh_mappings()
        self.learn_status.setText("All OSC mappings cleared")
