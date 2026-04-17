# =============================================================================
# Security Camera System - Configuration
# =============================================================================
# Values can be overridden via environment variables or a .env file.
# See .env.example for all available settings.
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default, cast=str):
    """Read an env var with type casting and a fallback default."""
    val = os.environ.get(key)
    if val is None:
        return default
    return cast(val)


# --- Camera Settings ---
# Default IP: 192.168.100.150 (if DHCP fails after 30s)
# Default login: admin / 9999
# Web UI: http://<camera-ip>:80
# RTSP port: 554 (configurable in web UI under System > Network)
# The RTSP address is shown in the web UI under Stream > Encoding.
# DHCP is enabled by default — check your router's client list first.
CAMERA_RTSP_URL = _env("CAMERA_RTSP_URL", "rtsp://admin:9999@192.168.100.150:554/stream/main")
CAMERA_FPS = _env("CAMERA_FPS", 15, int)

# --- Display Settings (Phase 1: OpenCV viewer) ---
DISPLAY_WINDOW_NAME = "Live Feed"
DISPLAY_WIDTH = _env("DISPLAY_WIDTH", 1280, int)
DISPLAY_HEIGHT = _env("DISPLAY_HEIGHT", 720, int)

# --- Web Server Settings (Phase 2) ---
WEB_HOST = _env("WEB_HOST", "0.0.0.0")
WEB_PORT = _env("WEB_PORT", 8000, int)

# --- HLS Streaming Settings ---
HLS_DIR = _env("HLS_DIR", "logs/hls")
HLS_SEGMENT_DURATION = _env("HLS_SEGMENT_DURATION", 1, int)
HLS_PLAYLIST_SIZE = _env("HLS_PLAYLIST_SIZE", 3, int)

# --- Detection Settings (Phase 3) ---
YOLO_MODEL = _env("YOLO_MODEL", "yolov8n.pt")
DETECTION_CONFIDENCE = _env("DETECTION_CONFIDENCE", 0.45, float)
DETECT_EVERY_N_FRAMES = _env("DETECT_EVERY_N_FRAMES", 3, int)

# --- Event Recording Settings (clip-based recording) ---
EVENT_DIR = _env("EVENT_DIR", "logs/events")
EVENT_PRE_ROLL = _env("EVENT_PRE_ROLL", 5, int)
EVENT_POST_ROLL = _env("EVENT_POST_ROLL", 10, int)
EVENT_MAX_DAYS = _env("EVENT_MAX_DAYS", 30, int)
EVENT_CONFIDENCE = _env("EVENT_CONFIDENCE", 0.75, float)
EVENT_SUSTAIN_SECONDS = _env("EVENT_SUSTAIN_SECONDS", 1.0, float)

# --- Encoding ---
# "auto" = libx264 on x86, h264_nvmpi on Jetson. Override to force a specific encoder.
FFMPEG_ENCODER = _env("FFMPEG_ENCODER", "auto")

# --- Action Trigger Settings (Phase 4) ---
SMART_PLUG_IP = _env("SMART_PLUG_IP", "192.168.1.150")
LIGHT_ON_DURATION = _env("LIGHT_ON_DURATION", 120, int)
ACTION_COOLDOWN = _env("ACTION_COOLDOWN", 30, int)

# --- Schedule (Phase 4) ---
# Actions only fire during these hours (24h format)
ALERT_START_HOUR = _env("ALERT_START_HOUR", 23, int)
ALERT_END_HOUR = _env("ALERT_END_HOUR", 6, int)

# --- MJPEG (standalone server.py) ---
MJPEG_QUALITY = _env("MJPEG_QUALITY", 80, int)
