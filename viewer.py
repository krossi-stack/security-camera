"""View the camera feed in an OpenCV window.

Usage:
    1. Edit config.py with your camera's RTSP URL.
    2. pip install -r requirements/windows.txt
    3. python viewer.py

Press 'q' to quit.
"""

import cv2
import logging

from camera import CameraStream
from config import (
    CAMERA_RTSP_URL,
    DISPLAY_WINDOW_NAME,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    stream = CameraStream(CAMERA_RTSP_URL)
    stream.reconnect_loop()

    print(f"Streaming from {CAMERA_RTSP_URL}")
    print("Press 'q' to quit.")

    while True:
        success, frame = stream.read_frame()

        if not success:
            print("Lost connection. Reconnecting...")
            stream.reconnect_loop()
            continue

        display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow(DISPLAY_WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
