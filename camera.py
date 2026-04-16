"""Camera capture module with RTSP connection and auto-reconnect."""

import cv2
import time
import logging
import threading

logger = logging.getLogger(__name__)


class CameraStream:
    """Manages an RTSP connection with automatic reconnection.

    Provides both synchronous frame reading (for viewer.py) and a
    threaded capture mode (for server.py / pipeline.py) that always
    holds the most recent frame, avoiding buffer-lag.
    """

    def __init__(self, rtsp_url: str, reconnect_delay: float = 5.0):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.cap = None

        # Threaded capture state
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish connection to the RTSP stream."""
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # Minimize internal buffer to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            logger.info("Connected to %s", self.rtsp_url)
            return True

        logger.error("Failed to connect to %s", self.rtsp_url)
        return False

    def reconnect_loop(self):
        """Block until a connection is successfully established."""
        while not self.connect():
            logger.info("Retrying in %.1fs...", self.reconnect_delay)
            time.sleep(self.reconnect_delay)

    def release(self):
        """Stop the capture thread (if running) and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ------------------------------------------------------------------
    # Synchronous frame reading (Phase 1 viewer)
    # ------------------------------------------------------------------

    def read_frame(self):
        """Read a single frame. Returns (success, frame)."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    # ------------------------------------------------------------------
    # Threaded capture (Phase 2+ server / pipeline)
    # ------------------------------------------------------------------

    def start_capture_thread(self):
        """Start a background thread that continuously grabs frames.

        Use `get_latest_frame()` to retrieve the most recent frame
        without RTSP buffer lag.
        """
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Internal loop that keeps grabbing frames."""
        self.reconnect_loop()
        while self._running:
            success, frame = self.cap.read()
            if not success:
                logger.warning("Frame read failed, reconnecting...")
                self.reconnect_loop()
                continue
            with self._frame_lock:
                self._latest_frame = frame

    def get_latest_frame(self):
        """Return the most recently captured frame (or None)."""
        with self._frame_lock:
            return self._latest_frame
