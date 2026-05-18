"""
Per-lane following-distance calculator — optimised.

Complexity improvement: O(n²) → O(n log n)

Original approach: for each vehicle, scan all same-lane vehicles to find rear.
Optimised approach:
  1. Batch-transform all centroids to metric space in one call.
  2. Group track indices by lane ID.
  3. Sort each lane by metric Y (ascending = further ahead).
  4. For each vehicle, check only its two immediate sorted neighbours (O(1)).
     The immediate rear is always adjacent in the sorted order — no full scan.

"Behind" determination:
  Uses alpha-beta filtered velocity direction (vel_dir).
  Stationary/new tracks (|vel| < threshold) fall back to screen-Y ordering.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FollowingResult:
    vehicle_id:      int
    lane_id:         int
    rear_vehicle_id: Optional[int]
    distance_m:      Optional[float]


class FollowingDistanceCalculator:
    """
    Per-frame following distance to the nearest rear vehicle in the same lane.

    API matches DistanceCalculator so the same BirdseyeTransform calibration
    applies via set_birdseye().
    """

    def __init__(self, lane_width_m: float = 3.5, px_per_m: float = 20.0):
        self.lane_width_m  = max(0.5, float(lane_width_m))
        self.px_per_m      = max(0.1, float(px_per_m))
        self._M:           Optional[np.ndarray] = None
        self._be_px_per_m: Optional[float]      = None

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_birdseye(self, M: Optional[np.ndarray], be_px_per_m: float) -> None:
        self._M           = M
        self._be_px_per_m = max(0.1, be_px_per_m) if M is not None else None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _batch_to_metric(self, raw: np.ndarray) -> np.ndarray:
        """Vectorized: N×2 pixel centroids → N×2 metric [x_m, y_m]."""
        if self._M is not None and self._be_px_per_m is not None:
            warped = cv2.perspectiveTransform(
                raw.reshape(-1, 1, 2), self._M).reshape(-1, 2)
            return warped / self._be_px_per_m
        return raw / self.px_per_m

    # ── Core computation ──────────────────────────────────────────────────────

    def compute(self, tracks) -> List[FollowingResult]:
        """
        Return one FollowingResult per track.

        Steps:
          1. Batch centroid → metric (one transform call).
          2. Group by lane_id.
          3. Per lane: sort by metric Y → check only adjacent pair for rear.
        """
        if not tracks:
            return []

        n = len(tracks)

        # ── 1. Batch transform ─────────────────────────────────────────────────
        raw    = np.array([[t.cx, t.cy] for t in tracks], dtype=np.float32)
        metric = self._batch_to_metric(raw)            # N×2

        # ── 2. Lane assignment ─────────────────────────────────────────────────
        lane_ids = np.maximum(0, (metric[:, 0] / self.lane_width_m).astype(int))

        # ── 3. Group track indices by lane ─────────────────────────────────────
        lanes: Dict[int, List[int]] = {}
        for i in range(n):
            lid = int(lane_ids[i])
            if lid not in lanes:
                lanes[lid] = []
            lanes[lid].append(i)

        # ── 4. Per lane: sort by metric Y, check adjacent neighbours ──────────
        results: List[Optional[FollowingResult]] = [None] * n

        for lid, indices in lanes.items():
            # Single vehicle in lane → no rear possible.
            if len(indices) == 1:
                i = indices[0]
                results[i] = FollowingResult(
                    vehicle_id=tracks[i].id, lane_id=lid,
                    rear_vehicle_id=None, distance_m=None)
                continue

            # Sort ascending by metric Y (lower Y = further ahead in scene).
            indices.sort(key=lambda i: float(metric[i, 1]))

            for pos, i in enumerate(indices):
                t        = tracks[i]
                dvx, dvy = t.vel_dir
                has_dir  = dvx != 0.0 or dvy != 0.0

                best_id:   Optional[int] = None
                best_dist: float         = float("inf")

                # Only the two immediate neighbours need checking —
                # the immediate rear is always adjacent in the sorted list.
                for nb_pos in (pos - 1, pos + 1):
                    if nb_pos < 0 or nb_pos >= len(indices):
                        continue
                    j     = indices[nb_pos]
                    other = tracks[j]

                    # Rear-hemisphere test.
                    if has_dir:
                        dot     = ((other.cx - t.cx) * dvx
                                   + (other.cy - t.cy) * dvy)
                        is_rear = dot < 0.0
                    else:
                        # Stationary fallback: higher screen Y = rear.
                        is_rear = other.cy > t.cy

                    if not is_rear:
                        continue

                    dx_m = float(metric[j, 0] - metric[i, 0])
                    dy_m = float(metric[j, 1] - metric[i, 1])
                    dist = (dx_m * dx_m + dy_m * dy_m) ** 0.5

                    if dist < best_dist:
                        best_dist = dist
                        best_id   = other.id

                results[i] = FollowingResult(
                    vehicle_id=t.id,
                    lane_id=lid,
                    rear_vehicle_id=best_id,
                    distance_m=best_dist if best_id is not None else None,
                )

        return results
