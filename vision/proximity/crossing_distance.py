"""
Crossing-triggered following-distance calculator.

At the exact frame a vehicle crosses the counting line this module:
  1. filters all current tracks to those in the SAME lane
     (CrossLine corridor perpendicular-distance guard, no bin bucketing)
  2. keeps only tracks on the REAR side of the line
     (CrossLine._side() sign convention — exact same geometry as the counter)
  3. selects the nearest rear vehicle by metric distance in bird's-eye space
  4. returns a CrossingDistanceResult ready for export

Design decisions vs FollowingDistanceCalculator
───────────────────────────────────────────────
• Same-lane filter: perpendicular distance to the CrossLine ≤ half_width
  (reuses the exact corridor geometry that gatekeeps crossings, so "same lane"
  is always consistent with "can cross here")
• Rear-side filter: CrossLine._side() sign — not velocity dot-product
  (velocity can be unstable at the crossing instant)
• Anchor point: BOTTOM-CENTER of bounding box — contacts the ground plane,
  so the bird's-eye transform maps it to the correct metric position
• Distance: anisotropic bird's-eye metric space — divides x and y by their
  individual px_per_m scales, correcting the averaging error that occurs
  when a single px_per_m_out is used for a non-square calibration area
"""

import cv2
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


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
    Single-shot: call find_rear() once per crossing event from the Worker thread.

    set_birdseye() must be called whenever the BirdseyeTransform is (re)calibrated.
    Pass the two per-axis scales separately so distance is computed without
    the averaging error present in px_per_m_out.
    """

    def __init__(self, px_per_m: float = 20.0, lane_width_m: float = 3.5):
        self.px_per_m     = max(0.1, float(px_per_m))
        self.lane_width_m = max(0.5, float(lane_width_m))
        self._M:           Optional[np.ndarray] = None
        self._px_per_m_x:  Optional[float]      = None
        self._px_per_m_y:  Optional[float]      = None

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_birdseye(self, M: Optional[np.ndarray],
                     px_per_m_x: float, px_per_m_y: float) -> None:
        """
        Provide the bird's-eye matrix and the two per-axis metre scales.

        px_per_m_x = OUT_W / real_width_m   (lateral axis)
        px_per_m_y = OUT_H / real_height_m  (longitudinal axis)

        When M is None the calculator falls back to flat-ground px_per_m.
        """
        self._M          = M
        self._px_per_m_x = max(0.1, px_per_m_x) if M is not None else None
        self._px_per_m_y = max(0.1, px_per_m_y) if M is not None else None

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _bottom_center(self, track) -> np.ndarray:
        """Bottom-center of bounding box as 1×2 float32 — contacts ground plane."""
        x, y, w, h = track.bbox
        return np.array([[x + w * 0.5, y + float(h)]], dtype=np.float32)

    def _to_metric(self, pt_1x2: np.ndarray) -> np.ndarray:
        """
        Project a 1×2 pixel point to metric coordinates (metres).

        With bird's-eye  → warp, then divide x and y by their own scale.
        Without bird's-eye → flat-ground: divide by px_per_m.

        Returns a 1-D array [x_m, y_m].
        """
        if self._M is not None:
            warped = cv2.perspectiveTransform(
                pt_1x2.reshape(1, 1, 2), self._M).reshape(2)
            return np.array([warped[0] / self._px_per_m_x,
                             warped[1] / self._px_per_m_y])
        return pt_1x2.reshape(2) / self.px_per_m

    def _metric_dist(self, track_a, track_b) -> float:
        ma = self._to_metric(self._bottom_center(track_a))
        mb = self._to_metric(self._bottom_center(track_b))
        return float(np.hypot(ma[0] - mb[0], ma[1] - mb[1]))

    def _lane_id_for(self, track) -> int:
        m = self._to_metric(self._bottom_center(track))
        return max(0, int(m[0] / self.lane_width_m))

    # ── Core API ──────────────────────────────────────────────────────────────

    def find_rear(self,
                  ev:         dict,
                  all_tracks: list,
                  cross_line) -> CrossingDistanceResult:
        """
        Find the nearest rear vehicle in the same lane at a crossing event.

        Parameters
        ──────────
        ev          — crossing event dict (keys: id, dir, cx, cy, speed)
        all_tracks  — self.trk.tracks at the crossing frame (includes crossed track)
        cross_line  — CrossLine instance (geometry source for lane + side filters)

        Algorithm
        ─────────
        1. Same-lane: perpendicular distance to line ≤ half_width  (no [0,1] clip)
        2. Rear-side: _side() sign matches the "came-from" direction
        3. Nearest by anisotropic bird's-eye metric distance (bottom-center anchors)
        """
        crossed_id = ev["id"]
        crossed_tr = next((t for t in all_tracks if t.id == crossed_id), None)

        # ── Rear-side sign ────────────────────────────────────────────────────
        # check_track sets direction based on curr_side AFTER crossing:
        #   "IN"  → curr_side > 0 (now on A-side) → came from B-side (sign < 0)
        #   "OUT" → curr_side < 0 (now on B-side) → came from A-side (sign > 0)
        # Rear vehicles are still on the "came-from" side.
        rear_sign = -1.0 if ev["dir"] == "IN" else 1.0

        x1  = float(cross_line.p1[0])
        y1  = float(cross_line.p1[1])
        dx  = cross_line._dx
        dy  = cross_line._dy
        seg_sq = cross_line._seg_len_sq
        hw_sq  = cross_line._hw_sq

        # ── Filter candidates ─────────────────────────────────────────────────
        candidates: List = []
        for t in all_tracks:
            if t.id == crossed_id:
                continue

            rx = t.cx - x1
            ry = t.cy - y1

            # cross_val ≡ CrossLine._side(t.cx, t.cy, ...) — reuse single computation
            cross_val = dx * ry - dy * rx

            # Same-lane: perpendicular distance² ≤ half_width²
            # perp_dist² = cross_val² / seg_sq  ≤  hw²
            # → cross_val² ≤ hw² * seg_sq
            if cross_val * cross_val > hw_sq * seg_sq:
                continue

            # Rear-side: cross_val and rear_sign must agree in sign (strictly)
            if cross_val * rear_sign <= 0.0:
                continue

            candidates.append(t)

        # ── Lane id (from crossed vehicle bottom-center in metric space) ───────
        if crossed_tr is not None:
            lane_id = self._lane_id_for(crossed_tr)
        else:
            pt = np.array([[ev["cx"], ev["cy"]]], dtype=np.float32)
            m  = self._to_metric(pt)
            lane_id = max(0, int(m[0] / self.lane_width_m))

        if not candidates:
            return CrossingDistanceResult(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                crossed_id=crossed_id,
                rear_id=None,
                lane_id=lane_id,
                distance_m=None,
                crossed_speed=ev["speed"],
                rear_speed=0.0,
            )

        # ── Nearest rear by metric distance ───────────────────────────────────
        best_tr:   Optional[object] = None
        best_dist: float            = float("inf")

        for t in candidates:
            if crossed_tr is not None:
                dist = self._metric_dist(crossed_tr, t)
            else:
                pt_c = np.array([[ev["cx"], ev["cy"]]], dtype=np.float32)
                mc   = self._to_metric(pt_c)
                mr   = self._to_metric(self._bottom_center(t))
                dist = float(np.hypot(mc[0] - mr[0], mc[1] - mr[1]))

            if dist < best_dist:
                best_dist = dist
                best_tr   = t

        return CrossingDistanceResult(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            crossed_id=crossed_id,
            rear_id=best_tr.id if best_tr else None,
            lane_id=lane_id,
            distance_m=round(best_dist, 2) if best_tr else None,
            crossed_speed=ev["speed"],
            rear_speed=round(best_tr.speed_kmh, 1) if best_tr else 0.0,
        )
