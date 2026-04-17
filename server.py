"""Phase 2: Serve the camera feed as MJPEG over HTTP.

Usage:
    pip install -r requirements/windows.txt
    uvicorn server:app --host 0.0.0.0 --port 8000

Then open http://<your-pc-ip>:8000 from any device on the network.
"""

import cv2
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from camera import CameraStream
from config import CAMERA_RTSP_URL, MJPEG_QUALITY

app = FastAPI(title="HUD Camera")

# Mount static files if the directory exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Shared camera stream — started once on app startup
stream = CameraStream(CAMERA_RTSP_URL)


@app.on_event("startup")
def startup():
    stream.start_capture_thread()


@app.on_event("shutdown")
def shutdown():
    stream.release()


def generate_mjpeg():
    """Yield MJPEG frames for the streaming response."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY]
    while True:
        frame = stream.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        _, jpeg = cv2.imencode(".jpg", frame, encode_params)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HUD Camera</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #111;
            color: #eee;
            font-family: system-ui, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        header {
            padding: 1rem;
            text-align: center;
            width: 100%;
            background: #1a1a1a;
            border-bottom: 1px solid #333;
        }
        header h1 { font-size: 1.2rem; font-weight: 500; }
        .feed-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 1rem;
        }
        .feed-container img {
            max-width: 100%;
            max-height: 85vh;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <header>
        <h1>HUD Camera</h1>
    </header>
    <div class="feed-container">
        <img src="/feed" alt="Live camera feed">
    </div>
</body>
</html>"""


@app.get("/feed")
async def video_feed():
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
