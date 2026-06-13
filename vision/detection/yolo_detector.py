"""
YOLO detector + ByteTrack multi-object tracker.

`track()` runs detection AND tracking inside Ultralytics (ByteTrack), returning
boxes already tagged with a stable track ID.  ByteTrack keeps the same ID across
brief occlusions / missed detections, which fixes the "one car gets several
track IDs → duplicated/!wrong data" problem of the old hand-rolled tracker.

It tracks every vehicle in the whole frame; the pipeline downstream keeps only
the ones inside the lane ROI.
"""
import numpy as np
from config.defaults import VEHICLE_CLASSES, CONF_DEFAULT, TRACKER_CFG


class YOLODetector:
    def __init__(self, model, conf: float = CONF_DEFAULT):
        self.model = model
        self.conf  = conf
        self._started = False        # False → next track() resets ByteTrack state
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"

    def reset(self) -> None:
        """Forget all tracks — call when a new video/run starts."""
        self._started = False

    def track(self, frame: np.ndarray) -> list:
        """
        Detect + track vehicles. Returns [(track_id, x, y, w, h), ...].

        persist=False on the first frame of a run reinitialises ByteTrack so IDs
        restart from 1; True afterwards keeps IDs stable across frames.
        """
        if self.model is None:
            return []
        res = self.model.track(
            frame, persist=self._started, tracker=TRACKER_CFG,
            classes=list(VEHICLE_CLASSES), conf=self.conf,
            verbose=False, half=(self.device == "cuda"),
        )[0]
        self._started = True

        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids  = boxes.id.cpu().numpy().astype(int)
        out = []
        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            out.append((int(tid), int(x1), int(y1),
                        int(x2 - x1), int(y2 - y1)))
        return out
