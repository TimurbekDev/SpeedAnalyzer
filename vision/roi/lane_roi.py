"""
LaneROI — a single road-lane polygon, stored in *video* pixel coordinates.

Responsibilities are deliberately small: hold the polygon, test membership of a
point (a vehicle's ground-contact point), and support interactive editing
(move / delete / hit-test vertices).  All distance/perspective logic lives in
LaneCalibration; all detection/tracking lives in the worker.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional

from config.defaults import ROI_MIN_POINTS


class LaneROI:
    def __init__(self):
        self.points: List[list] = []
        self._contour: Optional[np.ndarray] = None

    # ── Geometry ──────────────────────────────────────────────────────────────

    @property
    def valid(self) -> bool:
        return len(self.points) >= ROI_MIN_POINTS

    def set_points(self, points: List[list]) -> None:
        self.points = [[int(p[0]), int(p[1])] for p in points]
        self._rebuild()

    def add_point(self, x: int, y: int) -> None:
        self.points.append([int(x), int(y)])
        self._rebuild()

    def move_point(self, idx: int, x: int, y: int) -> None:
        if 0 <= idx < len(self.points):
            self.points[idx] = [int(x), int(y)]
            self._rebuild()

    def delete_point(self, idx: int) -> bool:
        if len(self.points) <= ROI_MIN_POINTS or not (0 <= idx < len(self.points)):
            return False
        self.points.pop(idx)
        self._rebuild()
        return True

    def clear(self) -> None:
        self.points = []
        self._contour = None

    def hit_point(self, x: float, y: float, radius: float) -> Optional[int]:
        best_i, best_d = None, radius * radius
        for i, (px, py) in enumerate(self.points):
            d = (px - x) ** 2 + (py - y) ** 2
            if d <= best_d:
                best_i, best_d = i, d
        return best_i

    def contains(self, x: float, y: float) -> bool:
        if self._contour is None:
            return False
        return cv2.pointPolygonTest(self._contour, (float(x), float(y)), False) >= 0

    def _rebuild(self) -> None:
        self._contour = (np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)
                         if self.valid else None)
