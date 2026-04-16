"""Phase 3: YOLOv8 person detection wrapper."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO dataset class 0 = "person"


@dataclass
class Detection:
    """A single person detection with bounding box and confidence."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class PersonDetector:
    """Wraps a YOLOv8 model to detect people in frames."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.45):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a frame. Returns list of person detections."""
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            detections.append(Detection(x1, y1, x2, y2, conf))

        return detections

    def annotate(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame."""
        annotated = frame.copy()
        for det in detections:
            cv2.rectangle(
                annotated, (det.x1, det.y1), (det.x2, det.y2), (0, 0, 255), 2
            )
            label = f"Person {det.confidence:.0%}"
            cv2.putText(
                annotated,
                label,
                (det.x1, det.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        return annotated
