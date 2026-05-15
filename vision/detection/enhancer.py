"""Frame enhancer — CLAHE-based low-light improvement."""
import cv2
import numpy as np


class Enhancer:
    """Apply CLAHE to dark frames; skip bright frames (no-op above mean=90)."""

    def __init__(self):
        self.clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.enabled = True

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or np.mean(frame) > 90:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge([self.clahe.apply(l), a, b]),
                            cv2.COLOR_LAB2BGR)
