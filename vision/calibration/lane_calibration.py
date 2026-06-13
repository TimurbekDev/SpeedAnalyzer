"""
Lane perspective calibration — converts image pixels to real-world metres.

Why not pixel distance?
───────────────────────
A road photographed from a tilted camera is foreshortened: 1 metre near the
camera spans many pixels, the same metre far away spans few.  Measuring a gap
in raw pixels is therefore wrong by the perspective factor.

How this fixes it
─────────────────
The lane ROI is (approximately) a rectangle on the ground plane seen in
perspective — a trapezoid in the image.  We map the four lane corners to a real
metric rectangle of  width_m × length_m  with a homography H
(cv2.getPerspectiveTransform).  Any image point projected through H lands in a
top-down metric frame where the X axis is across the lane and the Y axis is
along it, both in metres.  Euclidean distance in that frame is the true
ground-plane distance, independent of where in the image the vehicles are.

Use the vehicle's *ground contact point* (bottom-centre of the bbox), since the
homography is only valid on the road plane, not for points up in the air.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Tuple


class LaneCalibration:
    def __init__(self,
                 width_m:  float = 3.5,
                 length_m: float = 20.0):
        self.width_m  = float(width_m)
        self.length_m = float(length_m)
        self.H:     Optional[np.ndarray] = None     # image px → metres
        self.H_inv: Optional[np.ndarray] = None     # metres → image px
        self.src:   Optional[np.ndarray] = None     # ordered lane corners (px)
        self.ready = False

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, points: List[list],
                  width_m: float, length_m: float) -> bool:
        """
        Build the homography from the ROI polygon and real lane dimensions.

        points   — ROI polygon vertices in video pixels (>= 3; 4 is ideal).
        width_m  — real lane width  (across the lane).
        length_m — real lane length (the distance the ROI spans along the lane).
        """
        self.width_m  = max(0.1, float(width_m))
        self.length_m = max(0.1, float(length_m))

        quad = self._quad_from_polygon(points)
        if quad is None:
            self.ready = False
            return False

        src = self._order_corners(quad).astype(np.float32)
        dst = np.array([[0.0,          0.0],            # top-left  (far-left)
                        [self.width_m, 0.0],            # top-right (far-right)
                        [self.width_m, self.length_m],  # bot-right (near-right)
                        [0.0,          self.length_m]], # bot-left  (near-left)
                       dtype=np.float32)
        self.H     = cv2.getPerspectiveTransform(src, dst)
        self.H_inv = cv2.getPerspectiveTransform(dst, src)
        self.src   = src
        self.ready = True
        return True

    def reset(self) -> None:
        self.H = self.H_inv = self.src = None
        self.ready = False

    # ── Measurement ───────────────────────────────────────────────────────────

    def to_metric(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """Project an image point to ground-plane metres (x_across, y_along)."""
        p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        m = cv2.perspectiveTransform(p, self.H)[0, 0]
        return float(m[0]), float(m[1])

    def distance_m(self, pt_a: Tuple[float, float],
                   pt_b: Tuple[float, float]) -> float:
        """True ground-plane distance in metres between two image points."""
        ax, ay = self.to_metric(pt_a)
        bx, by = self.to_metric(pt_b)
        return float(np.hypot(ax - bx, ay - by))

    def to_image(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Project a ground-plane metric point back to image pixels."""
        p = np.array([[[float(x_m), float(y_m)]]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, self.H_inv)[0, 0]
        return int(round(q[0])), int(round(q[1]))

    def grid_segments(self, step_m: float = 5.0) -> List[tuple]:
        """
        Image-space line segments for a metric grid over the lane — lets the
        operator visually check the calibration against real road markings.
        Returns [((x1,y1),(x2,y2), is_major), ...]; is_major flags lane edges.
        """
        if not self.ready:
            return []
        segs: list = []
        # Longitudinal lines (along the lane) at each metre-X step.
        x = 0.0
        while x <= self.width_m + 1e-6:
            major = x <= 1e-6 or x >= self.width_m - 1e-6
            segs.append((self.to_image(x, 0.0),
                         self.to_image(x, self.length_m), major))
            x += step_m
        # Lateral lines (across the lane) at each metre-Y step.
        y = 0.0
        while y <= self.length_m + 1e-6:
            major = y <= 1e-6 or y >= self.length_m - 1e-6
            segs.append((self.to_image(0.0, y),
                         self.to_image(self.width_m, y), major))
            y += step_m
        return segs

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _quad_from_polygon(points: List[list]) -> Optional[np.ndarray]:
        """Reduce any ROI polygon to the 4 corners used for the homography."""
        pts = np.array(points, dtype=np.float32)
        if len(pts) < 3:
            return None
        if len(pts) == 4:
            return pts
        if len(pts) == 3:
            # Degenerate lane: synthesise a 4th corner so a homography exists.
            return np.array([pts[0], pts[1], pts[2],
                             pts[0] + (pts[2] - pts[1])], dtype=np.float32)

        # > 4 points: approximate the polygon down to a quadrilateral.
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        peri = cv2.arcLength(hull, True)
        for k in range(1, 20):
            approx = cv2.approxPolyDP(hull, 0.01 * k * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        # Fallback: minimum-area rectangle corners.
        box = cv2.boxPoints(cv2.minAreaRect(pts))
        return box.astype(np.float32)

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 corners as [far-left, far-right, near-right, near-left].

        For a down-the-road camera the lane's far edge is higher in the image
        (smaller y) and the near edge is lower (larger y).  Split by y, then
        order each pair left→right by x.  This is more robust for road
        trapezoids than the classic sum/diff ordering.
        """
        pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
        by_y = pts[np.argsort(pts[:, 1])]
        top2 = by_y[:2][np.argsort(by_y[:2, 0])]    # far  edge, left→right
        bot2 = by_y[2:][np.argsort(by_y[2:, 0])]    # near edge, left→right
        tl, tr = top2[0], top2[1]
        bl, br = bot2[0], bot2[1]
        return np.array([tl, tr, br, bl], dtype=np.float32)
