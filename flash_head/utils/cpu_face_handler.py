import mediapipe as mp
import numpy as np
from typing import Tuple, List


class CPUFaceHandler:
    """Handler for CPU-based face detection using MediaPipe.
    (2 ms/frame)
    This handler provides a simple interface for face detection using MediaPipe's
    face detection model. It's optimized for CPU usage and provides basic face
    detection functionality.
    """

    def __init__(self, model_selection: int = 1, min_detection_confidence: float = 0.0):
        """Initialize the face detection handler."""
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, image: np.ndarray) -> Tuple[int, List[int]]:
        """Detect faces in the given image.

        Args:
            image (np.ndarray): RGB image array.

        Returns:
            Tuple[int, List[int]]: A tuple containing:
                - Number of faces detected (int)
                - Bounding box coordinates [x1, y1, x2, y2] if exactly one face is detected,
                  empty list otherwise
        """
        bboxs, scores = [], []
        results = self.detector.process(image)
        detection_result = results.detections
        if detection_result is None:
            return bboxs, scores
        for detection in detection_result:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height
            x1, y1, x2, y2 = x, y, x + w, y + h
            bboxs.append([x1, y1, x2, y2])
            scores.append(detection.score[0])
        return bboxs, scores

    def __call__(self, image: np.ndarray) -> Tuple[int, List[int]]:
        """Make the handler callable.

        Args:
            image (np.ndarray): RGB image array.

        Returns:
            Tuple[int, List[int]]: Same as detect() method.
        """
        return self.detect(image)
