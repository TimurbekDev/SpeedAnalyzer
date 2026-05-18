"""
Crossing-triggered rear-distance calculator — optimised.

Called ONCE per crossing event (not every frame).

Algorithm
─────────
1. Batch-transform all track bottom-center points to birdseye metric in one
   cv2.perspectiveTransform call — no per-candidate reprojection.
2. Assign lane IDs from birdseye X / lane_width_m (correct lateral bins;
   the CrossLine corridor width is a counting corridor, NOT a lane boundary).
3. Keep only tracks in the same lane as the crossed vehicle.
4. Apply CrossLine _side() sign to keep only rear-side candidates
   (consistent with the counter's side convention).
5. Among rear candidates, select the one with the smallest longitudinal
   (birdseye Y) gap — the immediate rear neighbour.
6. Return anisotropic euclidean metric distance (separate x/y scales).

Complexity: O(n) transform + O(k) filter + O(k) min-scan  where k = lane size.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class CrossingDistanceResult:
    timestamp:     str
    crossed_id:    int
    rear_id:       Optional[int]
    lane_id:       int
    distance_m:    Optional[float]
    crossed_speed: float
    rear_speed:    float


class CrossingDistanceCalculator:
    """
    Single-shot: call find_rear() once per crossing event (from Worker thread).

    set_birdseye() must be called whenever BirdseyeTransform is (re)calibrated.
    Accepts separate per-axis scales to avoid the averaging error of px_per_m_out.
    """

    def __init__(self, px_per_m: float = 20.0, lane_width_m: float = 3.5):
        self.px_per_m     = max(0.1, float(px_per_m))
        self.lane_width_m = max(0.5, float(lane_width_m))
        self._M:          Optional[np.ndarray] = None
        self._px_per_m_x: Optional[float]      = None
        self._px_per_m_y: Optional[float]      = None

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_birdseye(self, M: Optional[np.ndarray],
                     px_per_m_x: float, px_per_m_y: float) -> None:
        self._M          = M
        self._px_per_m_x = max(0.1, px_per_m_x) if M is not None else None
        self._px_per_m_y = max(0.1, px_per_m_y) if M is not None else None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_bottom_pts(self, tracks) -> np.ndarray:
        """Build N×2 float32 array of bottom-center bounding-box points."""
        pts = np.empty((len(tracks), 2), dtype=np.float32)
        for i, t in enumerate(tracks):
            x, y, w, h = t.bbox
            pts[i, 0] = x + w * 0.5
            pts[i, 1] = y + float(h)
        return pts

    def _batch_to_metric(self, pts: np.ndarray) -> np.ndarray:
        """
        Vectorized projection: N×2 pixel points → N×2 metric [x_m, y_m].

        One perspectiveTransform call for the whole track set.
        In-place division avoids an extra allocation.
        """
        if self._M is not None:
            be = cv2.perspectiveTransform(
                pts.reshape(-1, 1, 2), self._M).reshape(-1, 2).copy()
            be[:, 0] /= self._px_per_m_x
            be[:, 1] /= self._px_per_m_y
            return be
        return pts / self.px_per_m

    # ── Core API ──────────────────────────────────────────────────────────────

    def find_rear(self,
                  ev:         dict,
                  all_tracks: list,
                  cross_line) -> 'CrossingDistanceResult':
        """
        Find the immediate rear vehicle in the same lane at a crossing event.

        ev         — crossing event dict  {id, dir, cx, cy, speed}
        all_tracks — trk.tracks at the crossing frame (includes crossed track)
        cross_line — CrossLine instance (rear-side geometry)
        """
        crossed_id = ev["id"]
        ts         = datetime.now().strftime("%H:%M:%S")

        def _empty(lane_id: int = 0) -> CrossingDistanceResult:
            return CrossingDistanceResult(
                timestamp=ts, crossed_id=crossed_id,
                rear_id=None, lane_id=lane_id,
                distance_m=None,
                crossed_speed=ev["speed"], rear_speed=0.0)

        n = len(all_tracks)
        if n == 0:
            return _empty()

        # ── 1. Batch birdseye transform (single GPU/SIMD call) ────────────────
        pts  = self._build_bottom_pts(all_tracks)
        be_m = self._batch_to_metric(pts)          # N×2: [x_m, y_m]

        # ── 2. Lane IDs from lateral birdseye position ────────────────────────
        lane_v = np.maximum(0, (be_m[:, 0] / self.lane_width_m).astype(np.int32))

        # ── 3. Locate the crossed vehicle in the track list ───────────────────
        crossed_idx = next((i for i, t in enumerate(all_tracks)
                            if t.id == crossed_id), None)
        if crossed_idx is None:
            return _empty()

        crossed_lane = int(lane_v[crossed_idx])
        cx_m, cy_m   = float(be_m[crossed_idx, 0]), float(be_m[crossed_idx, 1])

        # ── 4. Same-lane candidates (exclude self) ────────────────────────────
        same_lane = [i for i in range(n)
                     if i != crossed_idx and lane_v[i] == crossed_lane]
        if not same_lane:
            return _empty(crossed_lane)

        # ── 5. Rear-side filter via CrossLine sign ────────────────────────────
        # "IN"  → vehicle is now on A-side (sign > 0) → came from B → rear sign < 0
        # "OUT" → vehicle is now on B-side (sign < 0) → came from A → rear sign > 0
        x1, y1   = float(cross_line.p1[0]), float(cross_line.p1[1])
        cl_dx    = cross_line._dx
        cl_dy    = cross_line._dy
        rear_sgn = -1.0 if ev["dir"] == "IN" else 1.0

        rear: list[int] = []
        for i in same_lane:
            t  = all_tracks[i]
            rx = t.cx - x1
            ry = t.cy - y1
            cv = cl_dx * ry - cl_dy * rx          # CrossLine signed distance
            if cv * rear_sgn > 0.0:
                rear.append(i)

        if not rear:
            return _empty(crossed_lane)

        # ── 6. Immediate rear = smallest longitudinal (Y) gap ─────────────────
        # In birdseye space the Y axis is longitudinal (along traffic flow).
        # The closest vehicle in Y among rear-side candidates is the one
        # directly behind — no need to scan all pairs.
        best_i = min(rear, key=lambda i: abs(float(be_m[i, 1]) - cy_m))

        # ── 7. Euclidean metric distance (anisotropic birdseye) ───────────────
        dx_m   = float(be_m[best_i, 0]) - cx_m
        dy_m   = float(be_m[best_i, 1]) - cy_m
        dist_m = round(float(np.hypot(dx_m, dy_m)), 2)

        return CrossingDistanceResult(
            timestamp=ts,
            crossed_id=crossed_id,
            rear_id=all_tracks[best_i].id,
            lane_id=crossed_lane,
            distance_m=dist_m,
            crossed_speed=ev["speed"],
            rear_speed=round(all_tracks[best_i].speed_kmh, 1),
        )
