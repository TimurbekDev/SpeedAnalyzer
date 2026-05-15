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

    def detect(self, frame: np.ndarray, roi_mask=None) -> list:
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
            if roi_mask is not None:
                vm   = np.zeros(roi_mask.shape, np.uint8)
                vy1_ = max(0, y1); vy2_ = min(roi_mask.shape[0], y2)
                vx1_ = max(0, x1); vx2_ = min(roi_mask.shape[1], x2)
                vm[vy1_:vy2_, vx1_:vx2_] = 255
                overlap = (np.count_nonzero(np.bitwise_and(vm, roi_mask))
                           / max(w * h, 1))
                if overlap < 0.3:
                    continue
            boxes.append((x1, y1, w, h))
        return boxes
