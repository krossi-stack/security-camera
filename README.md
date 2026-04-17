# HUD Camera

Person detection security camera system using a Marshall CV574 with RTSP streaming and YOLOv8.

## Network Setup

```
[Marshall CV574]                        [Your PC]
IP: 192.168.100.150        Ethernet     Ethernet: 192.168.100.x
RTSP: port 554          <----------->   WiFi: 10.1.21.172
Web UI: port 80                         Web server: port 8000
```

- **Camera web UI:** http://192.168.100.150 (login: `admin` / `9999`)
- **Camera RTSP stream:** `rtsp://admin:9999@192.168.100.150:554/stream/main`
- **HUD Camera web view (from this PC):** http://localhost:8000
- **HUD Camera web view (from other devices on WiFi):** http://10.1.21.172:8000

The PC connects to the camera over Ethernet and re-serves the annotated feed over WiFi via a web server. Other devices on the WiFi network can view the stream in a browser.

## Hardware

- **Marshall CV574** — 4K NDI|HX3 POV camera (we use RTSP at 1080p)
- **PoE injector or PoE switch** (802.3af/at) — powers the camera over Ethernet
- **Cat6 Ethernet cable** — connects camera to PC (or to network switch)

### For outdoor deployment
- **Weatherproof camera housing** (IP66/IP67) — the CV574 is not outdoor-rated
- **Outdoor-rated Cat6 cable** — direct burial or in conduit
- **IR illuminator** (e.g., Tendelux AI4, ~850nm) — the CV574 has no built-in IR for night vision
- **TP-Link Kasa or Tapo smart plug** — for automated light triggers

## Software Prerequisites

- Python 3.12+
- FFmpeg (required for HLS web streaming)
- Windows 11 (tested on PC), Jetson Orin Nano with JetPack 6 (tested)

### Installing FFmpeg

```bash
winget install ffmpeg
```

Restart your terminal after installing so it's on PATH.

## Installation

### Windows

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies (phase 4 includes everything)
pip install -r requirements/phase4.txt
```

**GPU Support (Optional)** — for faster detection with an NVIDIA GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/phase3.txt
```

### Jetson Orin Nano (JetPack 6)

The pipeline runs on Jetson Orin Nano for low-power 24/7 operation with hardware-accelerated detection and encoding.

**Prerequisites:**
- JetPack 6 (L4T 36.x) flashed via SDK Manager
- FFmpeg with NVIDIA codec support (included in JetPack)

```bash
# 1. Create a virtual env (--system-site-packages keeps NVIDIA's CUDA bindings visible)
python3 -m venv --system-site-packages venv
source venv/bin/activate

# 2. Install PyTorch from NVIDIA's ARM64 wheel index
pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/

# 3. Install remaining dependencies
pip install -r requirements/jetson.txt

# 4. Create a .env file (see .env.example for all options)
cp .env.example .env
# Edit .env with your camera IP, plug IP, etc.
```

Run headless:

```bash
source venv/bin/activate
python pipeline.py --serve --no-display
```

The pipeline auto-detects the Jetson and uses the hardware H.264 encoder (`h264_nvmpi`) for event recording. Override with `FFMPEG_ENCODER` in `.env` if needed.

## Camera Setup

1. **Connect the camera** via PoE Ethernet. It powers on automatically.

2. **Access the web UI** at http://192.168.100.150
   - DHCP is enabled by default. If your router assigns a different IP, check the router's DHCP client list.
   - If DHCP fails, the camera falls back to `192.168.100.150` after 30 seconds.
   - Default login: `admin` / `9999`

3. **Configure the stream** in the web UI:
   - **Stream > Encoding:** Set main stream to 1080p, H.264, bitrate 4096-8192
   - **Stream > Encoding:** Confirm the RTSP address is `rtsp://192.168.100.150:554/stream/main`
   - **System > Network:** RTSP port 554, RTSP Encrypt **unchecked**

4. **Update `config.py`** if your camera IP differs from the default.

## Usage

### Person detection + web streaming (recommended)

```bash
python pipeline.py --serve
```

Opens a local OpenCV window with detection overlays and starts an HLS web server at `http://localhost:8000`. The web UI has a **Live Feed** page and an **Events** page for reviewing recorded detection clips. Press `q` in the OpenCV window to quit.

### Headless mode (no monitor)

```bash
python pipeline.py --serve --no-display
```

Runs detection and web streaming without an OpenCV window. Press `Ctrl+C` to quit.

### Detection only (no web server)

```bash
python pipeline.py
```

Opens the local OpenCV window with detection overlays. No web server. Press `q` to quit.

### View raw feed (no detection)

```bash
python viewer.py
```

Opens an OpenCV window showing the raw camera feed. Press `q` to quit.

### Raw feed web server (no detection)

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Serves the raw camera feed via MJPEG without detection overlays.

### Pipeline flags

| Flag | Description |
|---|---|
| `--serve` | Start the HLS web server for remote viewing |
| `--no-display` | Headless mode — no OpenCV window |
| `--no-actions` | Disable smart plug triggers |
| `--no-record` | Disable event clip recording |

## How It Works

1. **camera.py** connects to the camera via RTSP and runs a background thread that continuously grabs frames, keeping only the latest one to avoid buffer lag.

2. **detector.py** wraps YOLOv8n (nano model). It runs inference on frames and returns bounding boxes for detected people (COCO class 0, confidence >= 0.45).

3. **pipeline.py** ties it together: grabs the latest frame, runs detection every ~0.2 seconds, annotates the frame with bounding boxes, records event clips around detections, and streams via HLS.

4. **actions.py** controls a smart plug to turn on a light when a person is detected during configured alert hours, with cooldown and auto-off.

5. **server.py** is a standalone FastAPI MJPEG server for viewing the raw feed without detection.

### Event Recording

When a person is detected with high confidence (>= 75%) for at least 1 second, the pipeline records a video clip including 5 seconds of pre-roll and 10 seconds of post-roll after the last detection. Clips are saved to `logs/events/` and viewable from the `/events` page on the web UI. Clips older than 30 days are automatically deleted.

### HLS Streaming

The web feed uses HLS (HTTP Live Streaming) instead of MJPEG. Annotated frames are piped to FFmpeg which encodes H.264 video into 2-second `.ts` segments. This uses ~10-20x less bandwidth than MJPEG and works smoothly in any browser. There is ~4-6 seconds of latency.

## Configuration

All settings are in `config.py` and can be overridden via environment variables or a `.env` file (see `.env.example`):

| Setting | Default | Description |
|---|---|---|
| `CAMERA_RTSP_URL` | `rtsp://admin:9999@192.168.100.150:554/stream/main` | Camera RTSP stream URL |
| `CAMERA_FPS` | `15` | Expected camera framerate |
| `DISPLAY_WIDTH` | `1280` | Local viewer window width |
| `DISPLAY_HEIGHT` | `720` | Local viewer window height |
| `WEB_PORT` | `8000` | Web server port |
| `HLS_SEGMENT_DURATION` | `2` | Seconds per HLS segment (lower = less latency) |
| `HLS_PLAYLIST_SIZE` | `5` | Number of segments in the live playlist |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model size |
| `DETECTION_CONFIDENCE` | `0.45` | Min confidence for display overlays |
| `DETECT_EVERY_N_FRAMES` | `3` | Controls detection frequency |
| `FFMPEG_ENCODER` | `auto` | H.264 encoder (`auto`, `libx264`, `h264_nvmpi`, etc.) |
| `MJPEG_QUALITY` | `80` | JPEG quality for standalone MJPEG server |
| `EVENT_DIR` | `logs/events` | Where event clips are saved |
| `EVENT_PRE_ROLL` | `5` | Seconds of footage before detection in clips |
| `EVENT_POST_ROLL` | `10` | Seconds after last detection in clips |
| `EVENT_MAX_DAYS` | `30` | Auto-delete clips older than this |
| `EVENT_CONFIDENCE` | `0.75` | Min confidence to trigger recording |
| `EVENT_SUSTAIN_SECONDS` | `1.0` | Detection must persist this long to start recording |
| `SMART_PLUG_IP` | `192.168.1.150` | Smart plug IP |
| `LIGHT_ON_DURATION` | `120` | Seconds to keep light on |
| `ACTION_COOLDOWN` | `30` | Min seconds between triggers |
| `ALERT_START_HOUR` | `23` | Alert window start (24h) |
| `ALERT_END_HOUR` | `6` | Alert window end (24h) |

## Project Structure

```
computer_vision/
  config.py          — All settings (camera IP, detection thresholds, schedule)
  camera.py          — RTSP connection with threaded capture and auto-reconnect
  viewer.py          — View live feed in an OpenCV window
  server.py          — MJPEG web server (FastAPI) for raw feed
  detector.py        — YOLOv8 person detection wrapper
  actions.py         — Smart plug control with cooldown and scheduling
  pipeline.py        — Main orchestrator: capture + detect + record + stream
  .env.example       — Template for environment variable overrides
  requirements/
    base.txt         — opencv, numpy, python-dotenv
    phase2.txt       — + fastapi, uvicorn
    phase3.txt       — + ultralytics (YOLO + PyTorch)
    phase4.txt       — + python-kasa
    jetson.txt       — Flat requirements for Jetson Orin Nano
  logs/
    events/          — Detection event clips saved here
    hls/             — Temporary HLS segments (auto-cleaned)
```

## Troubleshooting

### Can't connect to the camera
- Verify the camera and PC are on the same subnet (`192.168.100.x`)
- Ping the camera: `ping 192.168.100.150`
- If your PC only has WiFi on a different subnet, connect an Ethernet cable directly to the camera or use a PoE switch on the same network

### RTSP stream doesn't connect
- Test in VLC first: Media > Open Network Stream > `rtsp://admin:9999@192.168.100.150:554/stream/main`
- Confirm the RTSP address in the camera web UI under Stream > Encoding
- Make sure RTSP Encrypt is **unchecked**
- The camera only supports one RTSP client at a time — make sure no other viewer is connected
- If the stream was interrupted ungracefully, reboot the camera from web UI: Maintenance > Reboot

### Video feed is laggy
- The pipeline uses threaded capture which discards stale frames — if still laggy, reduce resolution to 720p in the camera web UI
- Detection runs on a time interval (~0.2s) to prevent CPU overload
- For faster detection, use a GPU with CUDA or deploy on an NVIDIA Jetson

### Web feed not loading
- Make sure FFmpeg is installed and on PATH (`ffmpeg -version`)
- Wait a few seconds after startup for the first HLS segments to be generated
- Check the terminal for FFmpeg errors

### Detection misses people or has false positives
- Raise `DETECTION_CONFIDENCE` to 0.55+ to reduce false positives
- Lower to 0.35 if the camera is far from subjects
- Upgrade to `yolov8s.pt` for better accuracy (needs GPU for real-time)
- Ensure adequate lighting — the CV574 has no IR; use an external IR illuminator for night

## Future Work

- **Notifications:** Push notifications when a person is detected during alert hours
