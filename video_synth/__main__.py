"""
Main module for the video synthesizer application.
This module initializes the gui and video mixer, applies effects, and manages the main loop.
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
from contextlib import contextmanager
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QGridLayout
from PyQt6.QtGui import QImage, QPixmap
from pyqt_gui import PyQTGUI
from settings import UserSettings
from common import *
from midi import *
from mixer import Mixer
from param import ParamTable
from effects_manager import EffectManager
from audio_reactive import AudioReactiveModule
from api import APIServer
from ffmpeg_output import FFmpegOutput, FFmpegStreamOutput


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
        '-nd',
        '--devices',
        default=DEFAULT_NUM_DEVICES,
        choices=[i for i in range(1,11)],
        type=int,
        help='Number of USB video capture devices to search for on boot. Will safely ignore a extra devices if not found'
    )
    parser.add_argument(
        '-pn',
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
    parser.add_argument(
        '-d',
        '--diagnose',
        default=0,
        type=int,
        help='Number of frames to sample for performance diagnosis. Helps to identify bottlenecks causing stuttering.'
    )
    parser.add_argument(
        '--api',
        action='store_true',
        help='Enable REST API server for remote control'
    )
    parser.add_argument(
        '--api-host',
        default='127.0.0.1',
        type=str,
        help='API server host (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--api-port',
        default=8000,
        type=int,
        help='API server port (default: 8000)'
    )
    parser.add_argument(
        '--ffmpeg',
        action='store_true',
        help='Enable FFmpeg output to file or stream'
    )
    parser.add_argument(
        '--ffmpeg-output',
        default='output.mp4',
        type=str,
        help='FFmpeg output path (file path or rtmp:// URL)'
    )
    parser.add_argument(
        '--ffmpeg-preset',
        default='medium',
        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
        help='FFmpeg encoding preset (default: medium)'
    )
    parser.add_argument(
        '--ffmpeg-crf',
        default=23,
        type=int,
        help='FFmpeg CRF quality (0-51, lower=better, default: 23)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without GUI (requires --api or --ffmpeg)'
    )
    parser.print_help()
    return parser.parse_args()


""" Global logging module configuration """
def config_log(log_level):
    logging.basicConfig(
        level=log_level,
        format=f'%(levelname).1s | %(module)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    log = logging.getLogger(__name__)
    return log


@contextmanager
def perf_timer(perf_data, key, enabled):
    if enabled:
        t0 = time.perf_counter()
        yield
        perf_data[key] = (time.perf_counter() - t0) * 1000
    else:
        yield


"""Video processing loop"""
def video_loop(mixer, effects, should_quit, gui, settings, audio_module=None, ffmpeg_output=None):
    import gc  # For explicit garbage collection

    wet_frame = dry_frame = mixer.get_frame()
    if dry_frame is None:
        log.error("Failed to get initial frame from mixer. Exiting video loop.")
        return

    prev_frame = dry_frame.copy()
    frame_count = 0

    # Performance monitoring
    perf_samples = []
    perf_log_interval = settings.diagnose_frames.value  # Log every n frames
    DEBUG = settings.diagnose_frames.value > 0
    gc_interval = 600  # Run garbage collection every 600 frames to prevent memory buildup

    # CV window setup
    cv_window_active = settings.output_mode.value != OutputMode.NONE.value
    if cv_window_active:
        cv2.namedWindow(VIDEO_OUTPUT_WINDOW_TITLE, cv2.WINDOW_NORMAL)
        if settings.output_mode.value == OutputMode.FULLSCREEN.value:  # Fullscreen mode
            cv2.setWindowProperty(VIDEO_OUTPUT_WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Main loop
    while not should_quit.is_set():
        if DEBUG:
            frame_start = time.perf_counter()
        perf_data = {}

        # Mixer get_frame
        with perf_timer(perf_data, 'mixer', DEBUG):
            dry_frame = mixer.get_frame()

        if mixer.skip or dry_frame is None:
            mixer.skip = False
            log.warning("Skipping frame due to source read failure")
            continue

        # --- LFO updates ---
        with perf_timer(perf_data, 'lfos', DEBUG):
            for effect_manager in effects:
                effect_manager.oscs.update()

        # --- Audio reactive updates ---
        with perf_timer(perf_data, 'audio', DEBUG):
            if audio_module is not None:
                audio_module.analyze()

        # Effects processing
        with perf_timer(perf_data, 'effects', DEBUG):
            prev_frame, wet_frame = effects[MixerSource.POST.value].get_frames(
                dry_frame, wet_frame, prev_frame, frame_count
            )
            frame_count += 1

        # Store current frame for API snapshot endpoint
        with perf_timer(perf_data, 'api_copy', DEBUG):
            mixer.current_frame = wet_frame.astype(np.uint8)

        # FFmpeg output
        if ffmpeg_output is not None:
            with perf_timer(perf_data, 'ffmpeg', DEBUG):
                ffmpeg_output.write_frame(mixer.current_frame)

        # GUI emit
        if gui is not None:
            with perf_timer(perf_data, 'gui_emit', DEBUG):
                rgb_image = cv2.cvtColor(wet_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                # CRITICAL: Copy data to prevent QImage from referencing temporary/modified memory
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                gui.video_frame_ready.emit(qt_image)

        # External window
        if cv_window_active:
            with perf_timer(perf_data, 'cv2_show', DEBUG):
                cv2.imshow(VIDEO_OUTPUT_WINDOW_TITLE, wet_frame.astype(np.uint8))
                # PERFORMANCE FIX: Only check for key presses every X frames to reduce blocking
                if frame_count % 20 == 0:
                    key = cv2.waitKey(1) & 0xFF
                    if key in ESCAPE_KEYS:
                        break

        if DEBUG:
            # Total frame time
            elapsed = time.perf_counter() - frame_start
            perf_data['total'] = elapsed * 1000
            perf_data['fps'] = 1.0 / elapsed if elapsed > 0 else 0

            # Collect effects_manager breakdown
            em_perf = effects[MixerSource.POST.value].perf_data
            if em_perf:
                perf_data['em_alpha_blend'] = em_perf.get('alpha_blend', 0)
                perf_data['em_effect_chain'] = em_perf.get('effect_chain', 0)
                perf_data['em_feedback'] = em_perf.get('feedback', 0)

                # Identify the top 3 slowest individual effects
                effect_timings = em_perf.get('effects', {})
                if effect_timings:
                    top_effects = sorted(effect_timings.items(), key=lambda x: x[1], reverse=True)[:3]
                    perf_data['top_effects'] = top_effects

            perf_samples.append(perf_data)

            # Log performance stats periodically
            if frame_count % perf_log_interval == 0 and perf_samples:
                n = len(perf_samples)

                # Average the core numeric metrics
                core_keys = ['fps', 'mixer', 'lfos', 'audio', 'effects', 'api_copy',
                             'ffmpeg', 'gui_emit', 'cv2_show',
                             'em_alpha_blend', 'em_effect_chain', 'em_feedback', 'total']
                avg = {}
                for key in core_keys:
                    vals = [s[key] for s in perf_samples if key in s]
                    if vals:
                        avg[key] = sum(vals) / len(vals)

                # Build the log message dynamically
                parts = [f"FPS={avg.get('fps', 0):.1f}"]

                stage_order = [
                    ('mixer', 'mixer'), ('lfos', 'lfos'), ('audio', 'audio'),
                    ('effects', 'effects'), ('api_copy', 'api_copy'),
                    ('ffmpeg', 'ffmpeg'), ('gui_emit', 'gui'), ('cv2_show', 'cv2'),
                    ('total', 'TOTAL'),
                ]
                for key, label in stage_order:
                    if key in avg:
                        parts.append(f"{label}={avg[key]:.1f}ms")

                # Effects breakdown line
                em_parts = []
                for key, label in [('em_alpha_blend', 'alpha'), ('em_effect_chain', 'chain'), ('em_feedback', 'fb')]:
                    if key in avg:
                        em_parts.append(f"{label}={avg[key]:.1f}ms")

                # Top slow effects from last sample
                top_str = ""
                last_top = perf_samples[-1].get('top_effects', [])
                if last_top:
                    top_str = " | top: " + ", ".join(f"{name}={t:.1f}ms" for name, t in last_top if t > 0.5)

                msg = f"Performance (avg/{n} frames): \n\t" + " | ".join(parts)
                if em_parts:
                    msg += f"\n\t  effects breakdown: " + " | ".join(em_parts) + top_str

                log.info(msg)
                perf_samples.clear()

        # PERFORMANCE FIX: Explicit garbage collection to prevent NumPy and OpenCV mem accumulation
        if frame_count % gc_interval == 0:
            gc.collect()

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

    # Initialize audio reactive module
    audio_params = ParamTable(group="AudioReactive")
    audio_module = AudioReactiveModule(audio_params)
    audio_module.start()

    # Identify and initialize midi controllers before creating the GUI
    controllers = identify_midi_ports(controller_names,
                                      src_1_effects.params,
                                      src_2_effects.params,
                                      post_effects.params,
                                      mixer.params,)

    # Initialize API server if requested
    api_server = None
    if settings.api:
        # Collect all params for API
        all_params = ParamTable()
        for effect_mgr in effects:
            all_params.table.update(effect_mgr.params.table)
        all_params.table.update(mixer.params.table)
        all_params.table.update(audio_params.table)

        api_server = APIServer(all_params, mixer=mixer, host=settings.api_host, port=settings.api_port)
        api_server.start()

    # Initialize FFmpeg output if requested
    ffmpeg_output = None
    if settings.ffmpeg:
        if settings.ffmpeg_output.startswith('rtmp://'):
            log.info("Using RTMP streaming mode")
            ffmpeg_output = FFmpegStreamOutput(
                WIDTH, HEIGHT, fps=30,
                rtmp_url=settings.ffmpeg_output,
                preset=settings.ffmpeg_preset
            )
        else:
            log.info(f"Using file output mode: {settings.ffmpeg_output}")
            ffmpeg_output = FFmpegOutput(
                WIDTH, HEIGHT, fps=30,
                output=settings.ffmpeg_output,
                preset=settings.ffmpeg_preset,
                crf=settings.ffmpeg_crf
            )
        ffmpeg_output.start()

    # Validate headless mode
    if settings.headless and not (settings.api or settings.ffmpeg):
        log.error("Headless mode requires --api or --ffmpeg to be enabled")
        sys.exit(1)

    # Handle Ctrl+C from terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Create and start the control panel GUI (unless headless)
    app = None
    main_window = None
    if not settings.headless:
        app = QApplication(sys.argv)
        main_window = PyQTGUI(effects, settings, mixer, audio_module=audio_module)
        main_window.show()
    else:
        log.info("Running in headless mode (no GUI)")

    should_quit = threading.Event()

    # Start main application thread to run the video event loop
    video_thread = threading.Thread(
        target=video_loop,
        args=(mixer, effects, should_quit, main_window, settings, audio_module, ffmpeg_output)
    )
    video_thread.start()

    if not settings.headless:
        log.info("Starting PyQt event loop.")
        exit_code = app.exec()
        log.info("PyQt event loop finished.")
        should_quit.set()
    else:
        # In headless mode, wait for Ctrl+C
        log.info("Headless mode active. Press Ctrl+C to stop.")
        try:
            video_thread.join()
        except KeyboardInterrupt:
            log.info("Received keyboard interrupt, shutting down...")
            should_quit.set()
        exit_code = 0

    video_thread.join()
    audio_module.stop()

    if ffmpeg_output is not None:
        ffmpeg_output.stop()

    if api_server is not None:
        api_server.stop()

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
