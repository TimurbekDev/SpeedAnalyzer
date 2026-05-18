"""YOLO detector wrapper — runs inference, filters by vehicle classes."""
import numpy as np
from config.defaults import VEHICLE_CLASSES, CONF_DEFAULT


class YOLODetector:
    def __init__(self, model, conf: float = CONF_DEFAULT):
        self.model  = model
        self.conf   = conf
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"

    def detect(self, frame: np.ndarray) -> list:
        if self.model is None:
            return []
        res = self.model(
            frame, verbose=False, conf=self.conf,
            classes=list(VEHICLE_CLASSES),
            half=(self.device == "cuda"),
        )[0]
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            w, h = x2 - x1, y2 - y1
            boxes.append((x1, y1, w, h))
        return boxes
