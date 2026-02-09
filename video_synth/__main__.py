"""
Main module for the video synthesizer application.
This module initializes the gui and video mixer, applies effects, and manages the main loop.

All effects classes are initialized, sequenced and applied by the EffectsManager.
The Effects manager is stored in globals.py for easy sharing with the GUI, but is initialized in main.

See the Program Architecture section in README.md for in-depth explainations
of module function and interation.

Author: Kyle Henderson 
"""

import sys
import argparse
import logging
import signal
import threading
import time
import cv2
import numpy as np

from settings import UserSettings
from common import *
from midi import *
from mixer import Mixer
from effects import EffectManager

from PyQt6.QtWidgets import QApplication, QGridLayout
from PyQt6.QtGui import QImage, QPixmap
from pyqt_gui import PyQTGUI


# List of MIDI controller class names to identify and initialize
CONTROLLER_NAMES = [SMC_Mixer.__name__, MidiMix.__name__]


"""Creates ArgumentParser, configures arguments, returns parser"""
def parse_args():
    parser = argparse.ArgumentParser(description='Video Synthesizer initialization arguments')
    parser.add_argument(
        '-l',
        '--log-level',
        default=DEFAULT_LOG_LEVEL,  
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        help='Set the logging level (CRITICAL, ERROR, WARNING, INFO, DEBUG)'
    )
    parser.add_argument(
        '-d',
        '--devices',
        default=DEFAULT_NUM_DEVICES,
        choices=[i for i in range(1,11)],
        type=int,
        help='Number of USB video capture devices to search for on boot. Will safely ignore a extra devices if not found'
    )
    parser.add_argument(
        '-p',
        '--patch',
        default=DEFAULT_PATCH_INDEX,
        type=int,
        help='Initialize program with a saved patch. Defaults to using "saved_values.yaml", but can be changed'
    )
    parser.add_argument(
        '-f',
        '--file',
        default= DEFAULT_SAVE_FILE,
        type=str,
        help='Use an alternate save file. Must still be located in the save directory'
    )
    parser.add_argument(
        '-c',
        '--control-layout',
        default=Layout.QUAD_PREVIEW.name,
        choices=[item.name for item in Layout],
        help='Choose the GUI layout: "tabbed" for 1x2 grid, or "quad" for a 2x2 grid.'
    )
    parser.add_argument(
        '-o',
        '--output-mode',
        default=OutputMode.NONE.name,
        choices=[item.name for item in OutputMode],
        help='Use an external window for video output'
    )
    parser.print_help()
    return parser.parse_args()


""" Global logging module configuration using """
def config_log(log_level):
    logging.basicConfig(
        level=log_level,
        format=f'%(levelname).1s | %(module)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    log = logging.getLogger(__name__)
    return log


"""Video processing loop"""
def video_loop(mixer, effects, should_quit, gui, settings):
    wet_frame = dry_frame = mixer.get_frame()
    if dry_frame is None:
        log.error("Failed to get initial frame from mixer. Exiting video loop.")
        return

    prev_frame = dry_frame.copy()
    frame_count = 0

    # Performance monitoring
    perf_samples = []
    perf_log_interval = 100  # Log every 100 frames

    # CV window setup
    cv_window_active = settings.output_mode.value != OutputMode.NONE.value
    if cv_window_active:
        cv2.namedWindow(VIDEO_OUTPUT_WINDOW_TITLE, cv2.WINDOW_NORMAL)
        if settings.output_mode.value == OutputMode.FULLSCREEN.value:  # Fullscreen mode
            cv2.setWindowProperty(VIDEO_OUTPUT_WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while not should_quit.is_set():
        frame_start = time.perf_counter()
        perf_data = {}

        # Mixer get_frame
        t0 = time.perf_counter()
        dry_frame = mixer.get_frame()
        perf_data['mixer'] = (time.perf_counter() - t0) * 1000

        if mixer.skip or dry_frame is None:
            mixer.skip = False
            log.warning("Skipping frame due to source read failure")
            continue

        # LFO updates
        t0 = time.perf_counter()
        for effect_manager in effects:
            effect_manager.oscs.update()
        perf_data['lfos'] = (time.perf_counter() - t0) * 1000

        # Effects processing
        t0 = time.perf_counter()
        prev_frame, wet_frame = effects[MixerSource.POST.value].get_frames(
            dry_frame, wet_frame, prev_frame, frame_count
        )
        perf_data['effects'] = (time.perf_counter() - t0) * 1000

        frame_count += 1

        # GUI emit
        t0 = time.perf_counter()
        rgb_image = cv2.cvtColor(wet_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        # CRITICAL: Copy data to prevent QImage from referencing temporary/modified memory
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        gui.video_frame_ready.emit(qt_image)
        perf_data['gui_emit'] = (time.perf_counter() - t0) * 1000

        # External window
        if cv_window_active:
            t0 = time.perf_counter()
            cv2.imshow(VIDEO_OUTPUT_WINDOW_TITLE, wet_frame.astype(np.uint8))
            if cv2.waitKey(1) & 0xFF in ESCAPE_KEYS:
                break
            perf_data['cv2_show'] = (time.perf_counter() - t0) * 1000

        # Total frame time
        elapsed = time.perf_counter() - frame_start
        perf_data['total'] = elapsed * 1000
        perf_data['fps'] = 1.0 / elapsed if elapsed > 0 else 0

        perf_samples.append(perf_data)

        # Log performance stats periodically
        if frame_count % perf_log_interval == 0 and perf_samples:
            avg_stats = {}
            for key in perf_samples[0].keys():
                avg_stats[key] = sum(s[key] for s in perf_samples) / len(perf_samples)
            log.info(f"Performance (avg over {len(perf_samples)} frames): "
                    f"FPS={avg_stats['fps']:.1f} | "
                    f"mixer={avg_stats['mixer']:.1f}ms | "
                    f"lfos={avg_stats['lfos']:.1f}ms | "
                    f"effects={avg_stats['effects']:.1f}ms | "
                    f"gui={avg_stats['gui_emit']:.1f}ms | "
                    f"total={avg_stats['total']:.1f}ms")
            perf_samples.clear()

    if cv_window_active:
        cv2.destroyAllWindows()
    log.info("Video loop has gracefully stopped.")


""" Main app setup and loop """
def main(settings, controller_names):

    log.info("Initializing video synthesizer... Press 'q' or 'ESC' to quit")

    # Initialize effect managers for each source and for post-processing
    src_1_effects = EffectManager(Groups.SRC_1_EFFECTS, WIDTH, HEIGHT)
    src_2_effects = EffectManager(Groups.SRC_2_EFFECTS, WIDTH, HEIGHT)
    post_effects = EffectManager(Groups.POST_EFFECTS, WIDTH, HEIGHT)
    effects = (src_1_effects, src_2_effects, post_effects)

    mixer = Mixer(effects, settings.num_devices, WIDTH, HEIGHT)

    # Identify and initialize midi controllers before creating the GUI
    controllers = identify_midi_ports(controller_names, 
                                      src_1_effects.params, 
                                      src_2_effects.params, 
                                      post_effects.params)

    # log.info(f'Starting program with {len(params.keys())} tunable parameters')

    # Handle Ctrl+C from terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    main_window = PyQTGUI(effects, settings, mixer)
    main_window.show() 
    
    should_quit = threading.Event()
    
    # Start main application thread to run the PyQt and video event loop
    video_thread = threading.Thread(
        target=video_loop,
        args=(mixer, effects, should_quit, main_window, settings)
    )
    video_thread.start()
    
    log.info("Starting PyQt event loop.")
    exit_code = app.exec()

    log.info("PyQt event loop finished.")
    should_quit.set()
    video_thread.join()
    sys.exit(exit_code)

    if controllers:
        for c in controllers:
            c.thread_stop = True
            c.thread.join(timeout=5)
            if c.thread.is_alive():
                log.warning("MIDI thread did not terminate gracefully. Forcing exit.")
            else:
                log.info("MIDI thread stopped successfully.")



if __name__ == "__main__":
    args = parse_args()
    log = config_log(args.log_level)
    settings = UserSettings(**args.__dict__)
    main(settings, CONTROLLER_NAMES)
