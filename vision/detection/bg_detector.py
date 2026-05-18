"""Background-subtraction detector (MOG2) — YOLO-free fallback."""
import cv2
import numpy as np


class BGDetector:
    """MOG2-based foreground extractor, returns bounding boxes of moving objects."""

    def __init__(self, min_a: int = 1500, max_a: int = 150_000):
        self.min_a  = min_a
        self.max_a  = max_a
        self.bg       = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True)
        self.k_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
        self.k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        self.k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,    (11, 11))

    def detect(self, frame: np.ndarray) -> list:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        fg   = self.bg.apply(gray, learningRate=0.003)
        fg   = np.where(fg == 255, 255, 0).astype(np.uint8)
        fg   = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.k_open)
        fg   = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)
        fg   = cv2.dilate(fg, self.k_dilate, iterations=2)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            a = w * h
            if not (self.min_a < a < self.max_a):
                continue
            if not (0.2 < w / max(h, 1) < 5.5):
                continue
            boxes.append((x, y, w, h))
        return boxes

    def update_area(self, mn: int, mx: int) -> None:
        self.min_a, self.max_a = mn, mx
