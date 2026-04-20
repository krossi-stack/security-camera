"""Main pipeline: capture -> detect -> act -> serve.

Combines all phases into a single entry point. Runs the detection loop
and optionally starts the web server for remote viewing.

Usage:
    # Detection + local OpenCV window only:
    python pipeline.py

    # Detection + web server (view from browser):
    python pipeline.py --serve
"""

import argparse
import cv2
import logging
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from camera import CameraStream
from detector import PersonDetector, Detection

try:
    from actions import ActionManager
except ImportError:
    ActionManager = None
from config import (
    CAMERA_RTSP_URL,
    CAMERA_FPS,
    DISPLAY_WINDOW_NAME,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    DETECT_EVERY_N_FRAMES,
    YOLO_MODEL,
    DETECTION_CONFIDENCE,
    WEB_HOST,
    WEB_PORT,
    HLS_DIR,
    HLS_SEGMENT_DURATION,
    HLS_PLAYLIST_SIZE,
    EVENT_DIR,
    EVENT_PRE_ROLL,
    EVENT_POST_ROLL,
    EVENT_MAX_DAYS,
    EVENT_CONFIDENCE,
    EVENT_SUSTAIN_SECONDS,
    FFMPEG_ENCODER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_encoder() -> str:
    """Pick the ffmpeg H.264 encoder for this platform.

    If FFMPEG_ENCODER is set to something other than "auto", use it directly.
    Otherwise auto-detect: Jetson Tegra -> h264_nvmpi, else -> libx264.
    """
    if FFMPEG_ENCODER != "auto":
        return FFMPEG_ENCODER
    if Path("/etc/nv_tegra_release").exists():
        return "h264_nvmpi"
    return "libx264"


# ======================================================================
# Event Recorder — ring buffer + state machine for clip-based recording
# ======================================================================

class _RecState(Enum):
    IDLE = "idle"
    RECORDING = "recording"


class EventRecorder:
    """Records short MP4 clips around person-detection events.

    Keeps a rolling buffer of the last PRE_ROLL seconds.  Recording
    only starts when at least one detection exceeds EVENT_CONFIDENCE
    for EVENT_SUSTAIN_SECONDS continuously.  Once recording, it
    continues until POST_ROLL seconds after the last high-confidence
    detection.
    """

    def __init__(
        self,
        event_dir: str,
        pre_roll: float,
        post_roll: float,
        fps: float,
        min_confidence: float,
        sustain_seconds: float,
        encoder: str = "libx264",
    ):
        self.event_dir = Path(event_dir)
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self.pre_roll = pre_roll
        self.post_roll = post_roll
        self.fps = fps
        self.min_confidence = min_confidence
        self.sustain_seconds = sustain_seconds
        self._encoder = encoder

        self._frame_interval = 1.0 / fps
        self.buffer: deque = deque(maxlen=int(pre_roll * fps))
        self._buffer_times: deque = deque(maxlen=int(pre_roll * fps))
        self.state = _RecState.IDLE
        self.last_confident_time = 0.0
        self._confident_since = 0.0  # when continuous high-conf streak started
        self._writer = None
        self._clip_path = None
        self._clip_start = 0.0
        self._last_write_time = 0.0  # wall-clock time of last frame write

    def _has_confident_detection(self, detections: list[Detection]) -> bool:
        return any(d.confidence >= self.min_confidence for d in detections)

    def push_frame(self, frame, detections: list[Detection]):
        """Feed every frame from the main loop.

        Args:
            frame: Raw BGR numpy array.
            detections: List of Detection objects from YOLO.
        """
        confident = self._has_confident_detection(detections)
        now = time.time()

        if confident:
            self.last_confident_time = now
            if self._confident_since == 0.0:
                self._confident_since = now
        else:
            self._confident_since = 0.0

        # How long the current streak of high-confidence detections has lasted
        sustained = (
            confident and (now - self._confident_since) >= self.sustain_seconds
        )

        if self.state == _RecState.IDLE:
            # Rate-limit buffer to target FPS so pre-roll covers the
            # intended duration even when the main loop is faster.
            if (not self._buffer_times
                    or (now - self._buffer_times[-1]) >= self._frame_interval):
                self.buffer.append(frame.copy())
                self._buffer_times.append(now)
            if sustained:
                self._start_clip(frame.shape)
                for buffered in self.buffer:
                    self._write_frame(buffered)
                self.buffer.clear()
                self._buffer_times.clear()
                self._last_write_time = now
                self.state = _RecState.RECORDING

        elif self.state == _RecState.RECORDING:
            # Write exactly one frame per interval; duplicate on slow loops.
            elapsed = now - self._last_write_time
            n = int(elapsed * self.fps)
            if n > 0:
                for _ in range(n):
                    self._write_frame(frame)
                self._last_write_time += n * self._frame_interval
            elapsed_since_det = now - self.last_confident_time
            if not confident and elapsed_since_det >= self.post_roll:
                self._finish_clip()
                self.state = _RecState.IDLE

    def _start_clip(self, shape):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._clip_path = self.event_dir / f"event_{timestamp}.mp4"
        h, w = shape[:2]
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{w}x{h}",
            "-framerate", str(self.fps),
            "-i", "pipe:0",
            "-pix_fmt", "yuv420p",
            "-c:v", self._encoder,
        ]
        if self._encoder == "libx264":
            cmd += ["-preset", "ultrafast"]
        cmd += ["-movflags", "+faststart", str(self._clip_path)]
        self._writer = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._clip_start = time.time()
        logger.info("Event recording started: %s", self._clip_path.name)

    def _write_frame(self, frame):
        if self._writer is None or self._writer.poll() is not None:
            return
        try:
            self._writer.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            pass

    def _finish_clip(self):
        if self._writer is None:
            return
        try:
            self._writer.stdin.close()
        except OSError:
            pass
        try:
            self._writer.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._writer.kill()
        duration = time.time() - self._clip_start
        logger.info("Saved event clip: %s (%.1fs)", self._clip_path.name, duration)
        self._writer = None
        self._clip_path = None
        self._last_write_time = 0.0

    def close(self):
        """Finalize any in-progress clip (call on shutdown)."""
        if self.state == _RecState.RECORDING:
            self._finish_clip()
            self.state = _RecState.IDLE


# ======================================================================
# Cleanup — delete event clips older than EVENT_MAX_DAYS
# ======================================================================

def cleanup_old_events(event_dir: str, max_days: int):
    """Delete .mp4 clips older than max_days."""
    cutoff = datetime.now() - timedelta(days=max_days)
    removed = 0
    for path in Path(event_dir).glob("event_*.mp4"):
        # Parse timestamp from filename: event_YYYYMMDD_HHMMSS.mp4
        try:
            ts_str = path.stem.replace("event_", "")
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            if ts < cutoff:
                path.unlink()
                removed += 1
        except ValueError:
            continue
    if removed:
        logger.info("Cleaned up %d old event clip(s)", removed)


# ======================================================================
# HLS Streamer — pipe annotated frames to FFmpeg -> .m3u8 + .ts
# ======================================================================

class HLSStreamer:
    """Runs FFmpeg to read the RTSP stream directly and output HLS segments.

    This avoids decoding/re-encoding in Python entirely. FFmpeg copies the
    camera's native H.264 stream into .ts segments, giving perfectly smooth
    playback with near-zero CPU usage.
    """

    def __init__(self, hls_dir: str, rtsp_url: str):
        self.hls_dir = Path(hls_dir)
        self.hls_dir.mkdir(parents=True, exist_ok=True)
        # Wipe stale segments from previous run
        for f in self.hls_dir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass

        self.playlist = str(self.hls_dir / "stream.m3u8")
        self.rtsp_url = rtsp_url
        self._proc = None

    def start(self):
        cmd = [
            "ffmpeg",
            "-y",
            # RTSP input — use TCP for reliability
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            # Copy the H.264 stream as-is (no re-encoding)
            "-c:v", "copy",
            "-an",  # drop audio
            # HLS output
            "-f", "hls",
            "-hls_time", str(HLS_SEGMENT_DURATION),
            "-hls_list_size", str(HLS_PLAYLIST_SIZE),
            "-hls_flags", "delete_segments+append_list",
            "-hls_segment_filename", str(self.hls_dir / "seg_%03d.ts"),
            self.playlist,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("HLS streamer started (RTSP direct) -> %s", self.playlist)

    def stop(self):
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None
        logger.info("HLS streamer stopped")


# ======================================================================
# Web server — HLS live feed + event replay
# ======================================================================

def start_web_server(event_dir: str, hls_dir: str):
    """Start the FastAPI server in a background thread."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, FileResponse

    web_app = FastAPI(title="Live Feed")

    @web_app.get("/", response_class=HTMLResponse)
    async def index():
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Feed</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #111; color: #eee;
            font-family: system-ui, sans-serif;
            display: flex; flex-direction: column;
            align-items: center; min-height: 100vh;
        }
        header {
            padding: 1rem; text-align: center; width: 100%;
            background: #1a1a1a; border-bottom: 1px solid #333;
        }
        header h1 { font-size: 1.2rem; font-weight: 500; }
        nav { margin-top: 0.5rem; }
        nav a { color: #7af; text-decoration: none; margin: 0 0.5rem; }
        nav a:hover { text-decoration: underline; }
        .feed-container {
            flex: 1; display: flex; align-items: center;
            justify-content: center; width: 100%; padding: 1rem;
        }
        .feed-container video {
            max-width: 100%; max-height: 85vh; border-radius: 4px;
            background: #000;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
</head>
<body>
    <header>
        <h1>Live Feed</h1>
        <nav><a href="/">Live</a> | <a href="/events">Events</a></nav>
    </header>
    <div class="feed-container">
        <video id="live" muted autoplay></video>
    </div>
    <script>
        const video = document.getElementById('live');
        if (Hls.isSupported()) {
            const hls = new Hls({
                liveSyncDurationCount: 1,
                liveMaxLatencyDurationCount: 2,
                lowLatencyMode: true,
                enableWorker: true,
            });
            hls.loadSource('/hls/stream.m3u8');
            hls.attachMedia(video);
            hls.on(Hls.Events.MANIFEST_PARSED, () => video.play());
        } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
            // Safari native HLS
            video.src = '/hls/stream.m3u8';
            video.addEventListener('loadedmetadata', () => video.play());
        }
    </script>
</body>
</html>"""

    @web_app.get("/hls/{filename}")
    async def serve_hls(filename: str):
        path = Path(hls_dir) / filename
        if not path.exists():
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "not found"}, status_code=404)
        if filename.endswith(".m3u8"):
            media_type = "application/vnd.apple.mpegurl"
        else:
            media_type = "video/mp2t"
        return FileResponse(
            str(path), media_type=media_type,
            headers={"Cache-Control": "no-cache, no-store"},
        )

    @web_app.get("/events", response_class=HTMLResponse)
    async def events_page():
        clips = sorted(Path(event_dir).glob("event_*.mp4"), reverse=True)
        rows = ""
        for clip in clips:
            try:
                ts_str = clip.stem.replace("event_", "")
                ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                display_time = ts.strftime("%b %d, %Y  %I:%M:%S %p")
            except ValueError:
                display_time = clip.stem
            size_mb = clip.stat().st_size / (1024 * 1024)
            rows += f"""
            <tr>
                <td>{display_time}</td>
                <td>{size_mb:.1f} MB</td>
                <td>
                    <a href="#" onclick="playClip('/events/{clip.name}'); return false;">Play</a>
                    &nbsp;|&nbsp;
                    <a href="/events/{clip.name}" download>Download</a>
                </td>
            </tr>"""

        if not rows:
            rows = '<tr><td colspan="3" style="text-align:center; padding:2rem; color:#888;">No events recorded yet</td></tr>'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Events</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #111; color: #eee;
            font-family: system-ui, sans-serif;
            display: flex; flex-direction: column;
            align-items: center; min-height: 100vh;
        }}
        header {{
            padding: 1rem; text-align: center; width: 100%;
            background: #1a1a1a; border-bottom: 1px solid #333;
        }}
        header h1 {{ font-size: 1.2rem; font-weight: 500; }}
        nav {{ margin-top: 0.5rem; }}
        nav a {{ color: #7af; text-decoration: none; margin: 0 0.5rem; }}
        nav a:hover {{ text-decoration: underline; }}
        .content {{ width: 100%; max-width: 900px; padding: 1rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: #999; font-size: 0.85rem; text-transform: uppercase; }}
        td a {{ color: #7af; text-decoration: none; }}
        td a:hover {{ text-decoration: underline; }}
        .player {{
            margin-top: 1rem; text-align: center;
            display: none;
        }}
        .player video {{
            max-width: 100%; max-height: 60vh; border-radius: 4px;
            background: #000;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Events</h1>
        <nav><a href="/">Live</a> | <a href="/events">Events</a></nav>
    </header>
    <div class="content">
        <div class="player" id="player">
            <video id="video" controls autoplay></video>
        </div>
        <table>
            <thead><tr><th>Time</th><th>Size</th><th>Actions</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    <script>
        function playClip(url) {{
            const player = document.getElementById('player');
            const video = document.getElementById('video');
            video.src = url;
            player.style.display = 'block';
            video.play();
            player.scrollIntoView({{ behavior: 'smooth' }});
        }}
    </script>
</body>
</html>"""

    @web_app.get("/events/{filename}")
    async def serve_event(filename: str):
        path = Path(event_dir) / filename
        if not path.exists() or not path.name.startswith("event_"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(path), media_type="video/mp4", filename=filename)

    thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": web_app, "host": WEB_HOST, "port": WEB_PORT, "log_level": "warning"},
        daemon=True,
    )
    thread.start()
    logger.info("Web server started at http://%s:%d", WEB_HOST, WEB_PORT)


# ======================================================================
# Main entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Security camera detection pipeline")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Also start the MJPEG web server for remote viewing",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the local OpenCV window (headless mode for servers)",
    )
    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="Disable smart plug / action triggers",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable event clip recording",
    )
    args = parser.parse_args()

    # Clean up old event clips on startup
    cleanup_old_events(EVENT_DIR, EVENT_MAX_DAYS)

    # Initialize components — use threaded capture so we always get the
    # latest frame instead of reading from a growing RTSP buffer.
    stream = CameraStream(CAMERA_RTSP_URL)
    stream.start_capture_thread()

    detector = PersonDetector(model_name=YOLO_MODEL, confidence=DETECTION_CONFIDENCE)

    encoder = _resolve_encoder()
    logger.info("Using ffmpeg encoder: %s", encoder)

    recorder = None
    if not args.no_record:
        recorder = EventRecorder(
            EVENT_DIR, EVENT_PRE_ROLL, EVENT_POST_ROLL, CAMERA_FPS,
            EVENT_CONFIDENCE, EVENT_SUSTAIN_SECONDS,
            encoder=encoder,
        )

    action_manager = None
    if not args.no_actions and ActionManager is not None:
        action_manager = ActionManager()

    hls_streamer = None
    if args.serve:
        hls_streamer = HLSStreamer(HLS_DIR, CAMERA_RTSP_URL)
        hls_streamer.start()
        start_web_server(EVENT_DIR, HLS_DIR)

    if not args.no_display:
        cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DISPLAY_WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    logger.info("Pipeline running. Press Ctrl+C to quit.")

    last_detections: list[Detection] = []
    last_detect_time = 0.0
    detect_interval = DETECT_EVERY_N_FRAMES / CAMERA_FPS  # seconds between detections

    try:
        while True:
            frame = stream.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Run detection at a fixed time interval
            now = time.time()
            if now - last_detect_time >= detect_interval:
                last_detect_time = now
                last_detections = detector.detect(frame)

                if last_detections:
                    logger.info("Detected %d person(s)", len(last_detections))

                    if action_manager is not None:
                        action_manager.on_person_detected(
                            num_persons=len(last_detections),
                        )

            # Feed the event recorder
            if recorder is not None:
                recorder.push_frame(frame, detections=last_detections)

            # Annotate frame with latest detections
            annotated = detector.annotate(frame, last_detections)

            # Show local window
            if not args.no_display:
                cv2.imshow(DISPLAY_WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Ctrl+C received.")
    finally:
        logger.info("Shutting down...")
        if hls_streamer is not None:
            hls_streamer.stop()
        if recorder is not None:
            recorder.close()
        stream.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
