"""
Per-lane following-distance calculator.

For each tracked vehicle, finds the nearest vehicle behind it in the same
lane and returns the real-world gap in metres.

Lane assignment:
    Lateral position in metric space is quantised into lane bins of
    `lane_width_m` metres.  Bird's-eye transform (when calibrated) gives
    a true top-view x-axis, making lane bins accurate.  Without it the
    raw horizontal pixel / px_per_m approximation is used.

"Behind" determination:
    Each vehicle's alpha-beta filtered velocity direction (dvx, dvy) defines
    its forward hemisphere.  A candidate C is considered "behind" vehicle A
    when dot((C - A), vel_dir_A) < 0.  Stationary or freshly spawned tracks
    (|vel| < threshold) fall back to screen-y ordering (larger y = rear).
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FollowingResult:
    vehicle_id:      int
    lane_id:         int
    rear_vehicle_id: Optional[int]    # None when no rear vehicle in lane
    distance_m:      Optional[float]  # None when no rear vehicle in lane


class FollowingDistanceCalculator:
    """
    Computes per-vehicle following distance to the nearest rear vehicle
    in the same lane.

    API mirrors DistanceCalculator so the same BirdseyeTransform calibration
    can be applied with set_birdseye().
    """

    # Minimum velocity magnitude (px/frame) to trust the direction vector.
    _VEL_THR: float = 0.3

    def __init__(self, lane_width_m: float = 3.5, px_per_m: float = 20.0):
        self.lane_width_m  = max(0.5, float(lane_width_m))
        self.px_per_m      = max(0.1, float(px_per_m))
        self._M:           Optional[np.ndarray] = None
        self._be_px_per_m: Optional[float]      = None

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_birdseye(self, M: Optional[np.ndarray], be_px_per_m: float) -> None:
        """Mirror of DistanceCalculator.set_birdseye — call after calibration."""
        self._M           = M
        self._be_px_per_m = max(0.1, be_px_per_m) if M is not None else None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_metric(self, raw: np.ndarray) -> np.ndarray:
        """Project N×2 pixel centroids to metric coordinates (metres)."""
        if self._M is not None and self._be_px_per_m is not None:
            warped = cv2.perspectiveTransform(
                raw.reshape(-1, 1, 2), self._M).reshape(-1, 2)
            return warped / self._be_px_per_m
        return raw / self.px_per_m

    def _lane_id(self, metric_x: float) -> int:
        return int(metric_x / self.lane_width_m)

    # ── Core computation ──────────────────────────────────────────────────────

    def compute(self, tracks) -> List[FollowingResult]:
        """
        Return one FollowingResult per track.

        O(n²) over tracks in the same lane — fast enough for typical scene
        sizes (< 30 vehicles).  No memory allocation per pair beyond the
        result list.
        """
        if not tracks:
            return []

        raw    = np.array([[t.cx, t.cy] for t in tracks], dtype=np.float32)
        metric = self._to_metric(raw)
        n      = len(tracks)

        lane_ids = [self._lane_id(float(metric[i, 0])) for i in range(n)]

        results: List[FollowingResult] = []

        for i, t in enumerate(tracks):
            dvx, dvy = t.vel_dir
            has_dir  = (dvx != 0.0 or dvy != 0.0)

            best_id:   Optional[int] = None
            best_dist: float         = float('inf')

            for j in range(n):
                if i == j or lane_ids[j] != lane_ids[i]:
                    continue

                other = tracks[j]

                # Determine if `other` is in the rear hemisphere of `t`.
                if has_dir:
                    # Negative dot → other is behind t's forward direction.
                    dot = (other.cx - t.cx) * dvx + (other.cy - t.cy) * dvy
                    is_rear = dot < 0.0
                else:
                    # Fallback for stationary/new tracks: larger screen-y = rear.
                    is_rear = other.cy > t.cy

                if not is_rear:
                    continue

                dx_m = float(metric[j, 0] - metric[i, 0])
                dy_m = float(metric[j, 1] - metric[i, 1])
                dist = (dx_m * dx_m + dy_m * dy_m) ** 0.5

                if dist < best_dist:
                    best_dist = dist
                    best_id   = other.id

            results.append(FollowingResult(
                vehicle_id=t.id,
                lane_id=lane_ids[i],
                rear_vehicle_id=best_id,
                distance_m=best_dist if best_id is not None else None,
            ))

        return results
