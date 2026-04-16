# =============================================================================
# Security Camera System - Configuration
# =============================================================================

# --- Camera Settings ---
# Default IP: 192.168.100.150 (if DHCP fails after 30s)
# Default login: admin / 9999
# Web UI: http://<camera-ip>:80
# RTSP port: 554 (configurable in web UI under System > Network)
# The RTSP address is shown in the web UI under Stream > Encoding.
# DHCP is enabled by default — check your router's client list first.
CAMERA_RTSP_URL = "rtsp://admin:9999@192.168.100.150:554/stream/main"
CAMERA_FPS = 15

# --- Display Settings (Phase 1: OpenCV viewer) ---
DISPLAY_WINDOW_NAME = "Live Feed"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# --- Web Server Settings (Phase 2) ---
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000

# --- HLS Streaming Settings ---
HLS_DIR = "logs/hls"                # Temp directory for .m3u8 and .ts segments
HLS_SEGMENT_DURATION = 1            # Seconds per .ts segment (lower = less latency)
HLS_PLAYLIST_SIZE = 3               # Number of segments in the live playlist

# --- Detection Settings (Phase 3) ---
YOLO_MODEL = "yolov8n.pt"  # nano model; use "yolov8s.pt" with GPU
DETECTION_CONFIDENCE = 0.45
DETECT_EVERY_N_FRAMES = 3  # Run inference every Nth frame (5 Hz at 15fps)
# --- Event Recording Settings (clip-based recording) ---
EVENT_DIR = "logs/events"           # Where event clips are saved
EVENT_PRE_ROLL = 5                  # Seconds of footage to keep before detection
EVENT_POST_ROLL = 10                # Seconds to keep recording after last detection
EVENT_MAX_DAYS = 30                 # Auto-delete clips older than this
EVENT_CONFIDENCE = 0.75             # Min confidence to trigger recording (display still uses 0.45)
EVENT_SUSTAIN_SECONDS = 1.0         # High-confidence detections must persist this long to start recording

# --- Action Trigger Settings (Phase 4) ---
SMART_PLUG_IP = "192.168.1.150"  # Your TP-Link Kasa/Tapo plug IP
LIGHT_ON_DURATION = 120  # Seconds to keep light on after detection
ACTION_COOLDOWN = 30  # Min seconds between triggers

# --- Schedule (Phase 4) ---
# Actions only fire during these hours (24h format)
ALERT_START_HOUR = 23  # 11 PM
ALERT_END_HOUR = 6     # 6 AM
