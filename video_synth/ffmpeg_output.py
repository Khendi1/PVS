"""
FFmpeg output module for video synthesizer.
Pipes frames to ffmpeg for encoding to file or streaming.
"""

import subprocess
import logging
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)


class FFmpegOutput:
    """Manages ffmpeg subprocess for video output."""

    def __init__(self, width: int, height: int, fps: int = 30, output: str = "output.mp4",
                 preset: str = "medium", crf: int = 23, format: str = "mp4"):
        """
        Initialize FFmpeg output.

        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
            output: Output path or URL (file path, rtmp://, etc.)
            preset: FFmpeg encoding preset (ultrafast, fast, medium, slow, veryslow)
            crf: Constant Rate Factor for quality (0-51, lower = better quality, 23 is default)
            format: Output format (mp4, flv, matroska, etc.)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.output = output
        self.preset = preset
        self.crf = crf
        self.format = format

        self.process: Optional[subprocess.Popen] = None
        self.frame_count = 0

    def start(self):
        """Start the ffmpeg process."""
        if self.process is not None:
            log.warning("FFmpeg process already running")
            return

        # Build ffmpeg command
        # Input: raw BGR24 video from stdin
        # Output: encoded video to file or stream
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',  # OpenCV uses BGR
            '-r', str(self.fps),
            '-i', '-',  # Read from stdin
            '-an',  # No audio
            '-vcodec', 'libx264',
            '-preset', self.preset,
            '-crf', str(self.crf),
            '-pix_fmt', 'yuv420p',  # For compatibility
            '-f', self.format,
            self.output
        ]

        log.info(f"Starting FFmpeg: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            log.info(f"FFmpeg started, writing to {self.output}")
        except FileNotFoundError:
            log.error("FFmpeg not found. Please install ffmpeg and add it to PATH.")
            self.process = None
        except Exception as e:
            log.error(f"Failed to start FFmpeg: {e}")
            self.process = None

    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to ffmpeg.

        Args:
            frame: numpy array (height, width, 3) in BGR format (OpenCV format)
        """
        if self.process is None or self.process.stdin is None:
            return

        try:
            # Ensure frame is correct shape and dtype
            if frame.shape != (self.height, self.width, 3):
                log.warning(f"Frame shape mismatch: expected ({self.height}, {self.width}, 3), got {frame.shape}")
                frame = frame[:self.height, :self.width, :]

            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # Write frame as raw bytes
            self.process.stdin.write(frame.tobytes())
            self.frame_count += 1

            # Log progress periodically
            if self.frame_count % 300 == 0:  # Every 10 seconds at 30fps
                log.info(f"FFmpeg: {self.frame_count} frames written")

        except BrokenPipeError:
            log.error("FFmpeg pipe broken. Process may have terminated.")
            self.stop()
        except Exception as e:
            log.error(f"Error writing frame to FFmpeg: {e}")

    def stop(self):
        """Stop the ffmpeg process and finalize the output."""
        if self.process is None:
            return

        log.info(f"Stopping FFmpeg after {self.frame_count} frames")

        try:
            if self.process.stdin:
                self.process.stdin.close()

            # Wait for process to finish
            self.process.wait(timeout=10)

            # Log any errors
            if self.process.stderr:
                stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                if stderr_output:
                    log.debug(f"FFmpeg stderr: {stderr_output}")

        except subprocess.TimeoutExpired:
            log.warning("FFmpeg process did not terminate, killing...")
            self.process.kill()
        except Exception as e:
            log.error(f"Error stopping FFmpeg: {e}")
        finally:
            self.process = None
            log.info(f"FFmpeg stopped. Output saved to {self.output}")


class FFmpegStreamOutput(FFmpegOutput):
    """FFmpeg output configured for low-latency streaming via MPEG-TS/UDP, SRT, or RTMP."""

    def __init__(self, width: int, height: int, fps: int = 30,
                 url: str = "udp://127.0.0.1:1234",
                 preset: str = "veryfast", bitrate: str = "2500k"):
        """
        Initialize FFmpeg for live streaming.

        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
            url: Stream URL (udp://, srt://, or rtmp://)
            preset: FFmpeg encoding preset (use ultrafast/veryfast for streaming)
            bitrate: Target bitrate (e.g., "2500k")
        """
        self.bitrate = bitrate

        # Determine container format from URL protocol
        if url.startswith('rtmp://'):
            fmt = 'flv'
        else:
            # MPEG-TS for UDP and SRT (lower latency than FLV/RTMP)
            fmt = 'mpegts'

        super().__init__(width, height, fps, url, preset, format=fmt)

    def start(self):
        """Start ffmpeg for live streaming."""
        if self.process is not None:
            log.warning("FFmpeg process already running")
            return

        cmd = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',
            '-an',
            '-vcodec', 'libx264',
            '-preset', self.preset,
            '-tune', 'zerolatency',
            '-b:v', self.bitrate,
            '-maxrate', self.bitrate,
            '-bufsize', f'{int(self.bitrate[:-1]) * 2}k',
            '-pix_fmt', 'yuv420p',
            '-g', str(self.fps),  # Keyframe every 1 second for low latency
            '-f', self.format,
            self.output
        ]

        log.info(f"Starting FFmpeg stream: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            log.info(f"FFmpeg streaming to {self.output}")
        except FileNotFoundError:
            log.error("FFmpeg not found. Please install ffmpeg and add it to PATH.")
            self.process = None
        except Exception as e:
            log.error(f"Failed to start FFmpeg: {e}")
            self.process = None
