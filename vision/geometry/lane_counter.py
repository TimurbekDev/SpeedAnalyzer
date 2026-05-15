"""
Lane-Limited CrossLine — drop-in replacement for the CrossLine in speed_analyzer_pro.py.

Root cause of old bug: _side() uses an INFINITE mathematical line, so vehicles
in neighboring lanes that cross its extension still trigger counting.

Fix: two O(1) geometric guards before the side-change test.
  1. Projection guard  — t = dot(C-P1, P2-P1) / |P2-P1|²  must be in [0, 1]
     → rejects vehicles past the segment endpoints (neighboring lanes beside the zone)
  2. Corridor guard    — perp_dist² ≤ half_width²  (no sqrt needed)
     → rejects vehicles in parallel lanes whose projection does fall on the segment
     → the corridor width == the lane width you want to monitor

Both checks are pure integer/float arithmetic, no numpy, no sqrt in hot path.
"""

import cv2
import numpy as np


class CrossLine:
    """
    Lane-limited virtual counting line.

    Only vehicles whose centroid satisfies BOTH:
      • projection onto segment: t ∈ [0, 1]   (between p1 and p2)
      • perpendicular distance  ≤ half_width   (inside the lane corridor)
    can trigger a crossing event.

    Side / direction convention (unchanged from original):
      sign = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
      sign > 0  → side A → "IN"   (moving in the direction of the left-normal)
      sign < 0  → side B → "OUT"
    """

    DEFAULT_HW: int = 80   # default corridor half-width in pixels

    def __init__(self):
        self.p1: list | None = None
        self.p2: list | None = None
        self.active    = False
        self.count_in  = 0
        self.count_out = 0
        self.events: list[dict] = []
        self.half_width: int = self.DEFAULT_HW

        # cached per define()
        self._dx:       float = 0.0
        self._dy:       float = 0.0
        self._seg_len_sq: float = 0.0   # |P2-P1|²  — used in both guards
        self._hw_sq:    float = float(self.DEFAULT_HW ** 2)

    # ── Configuration ─────────────────────────────────────────────────────────

    def define(self, p1: list, p2: list) -> None:
        self.p1 = p1[:]
        self.p2 = p2[:]
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1.0:
            self.active = False
            return
        self._dx          = dx
        self._dy          = dy
        self._seg_len_sq  = seg_len_sq
        self.active       = True

    def set_width(self, half_width: int) -> None:
        """Corridor half-width in pixels.  Tune to ~50 % of the visible lane width."""
        self.half_width = max(10, int(half_width))
        self._hw_sq     = float(self.half_width ** 2)

    def reset_counts(self) -> None:
        self.count_in = self.count_out = 0
        self.events.clear()

    def clear(self) -> None:
        self.p1 = self.p2 = None
        self.active = False
        self._dx = self._dy = self._seg_len_sq = 0.0
        self.reset_counts()

    # ── Core geometry ─────────────────────────────────────────────────────────

    @staticmethod
    def _side(px: float, py: float,
              x1: float, y1: float,
              x2: float, y2: float) -> float:
        """Signed cross product — same formula as original."""
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def _in_corridor(self, cx: float, cy: float) -> bool:
        """
        True iff (cx, cy) is inside the lane corridor.

        Guard 1 — projection t ∈ [0, 1]:
            t = dot(C - P1,  P2 - P1) / |P2 - P1|²
            t < 0  → beyond P1 endpoint
            t > 1  → beyond P2 endpoint
            Both cases mean the vehicle is in a lane outside this segment's span.

        Guard 2 — perpendicular distance² ≤ half_width²:
            dist = |cross(P2-P1, P1-C)| / |P2-P1|
            dist² = cross² / seg_len_sq  ≤  hw²
            No sqrt needed — compare squared values.

        Cost: 6 multiplications + 4 additions + 2 comparisons.  Runs every
        frame for every track — kept branch-free and allocation-free.
        """
        x1, y1 = self.p1
        dx, dy  = self._dx, self._dy
        rx = cx - x1
        ry = cy - y1

        # Guard 1: projection
        t = (rx * dx + ry * dy) / self._seg_len_sq
        if not (0.0 <= t <= 1.0):
            return False

        # Guard 2: perpendicular distance² (cross product magnitude²)
        # cross = dx*(cy-y1) - dy*(cx-x1) = dx*ry - dy*rx
        cross = dx * ry - dy * rx
        return (cross * cross) <= (self._hw_sq * self._seg_len_sq)

    # ── Per-track crossing check ───────────────────────────────────────────────

    def check_track(self, track, fid: int, cam=None) -> dict | None:
        """
        Called inside Worker for every active track, every frame.

        Returns crossing event dict or None.
        Direction counts NOT modified here — caller (_poll) accumulates so that
        events from queue-drained frames are never lost.
        Each track is counted only ONCE via track.crossed flag.

        cam — CameraModel (optional). When provided, speed is computed from the
              track's centroid history at the exact crossing frame via
              track.lock_crossing_speed(cam). Without it, track.speed_kmh is used
              as-is (may be 0 if no ROI was set).
        """
        if not self.active or track.crossed or track.hits < 2:
            return None

        # ── Lane corridor gate ─────────────────────────────────────────────────
        # ONLY vehicles physically inside this corridor can cross.
        # Neighbors, oncoming opposite lane, vehicles past endpoints: all filtered.
        if not self._in_corridor(track.cx, track.cy):
            return None

        x1, y1 = self.p1
        x2, y2 = self.p2

        curr_side = self._side(track.cx, track.cy, x1, y1, x2, y2)

        # First appearance inside corridor — initialise side, no crossing yet.
        # Skip if centroid is exactly on the line (curr_side == 0) to avoid
        # storing 0 as a reference side, which would break the crossing test.
        if track.line_side is None:
            if curr_side != 0:
                track.line_side = curr_side
            return None

        # Suppress jitter (require real centroid movement)
        motion = np.hypot(track.cx - track.prev_cx, track.cy - track.prev_cy)
        if motion < 2.0:
            if curr_side != 0:
                track.line_side = curr_side
            return None

        # True crossing: curr_side flipped relative to last stored nonzero side.
        # We compare against track.line_side (not the raw prev centroid's side)
        # so that a centroid landing exactly on the line (curr_side == 0) in one
        # frame does not break detection in the next frame.
        if curr_side != 0 and track.line_side * curr_side < 0:
            direction       = "IN" if curr_side > 0 else "OUT"
            track.crossed   = True
            track.cross_dir = direction
            track.line_side = curr_side

            # Lock speed at crossing moment using full centroid history.
            # lock_crossing_speed() runs the alpha-beta estimator regression
            # and is a no-op on subsequent calls (speed_reported guard).
            speed = track.lock_crossing_speed(cam)

            return {
                "id":    track.id,
                "dir":   direction,
                "frame": fid,
                "speed": round(speed, 1),
                "cx":    track.cx,
                "cy":    track.cy,
            }

        if curr_side != 0:
            track.line_side = curr_side
        return None

    # ── Drawing ────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, scale: float = 1.0) -> None:
        """
        Draw:
          • semi-transparent corridor polygon  (blue tint)
          • dashed corridor edges              (dim cyan)
          • solid counting line                (bright cyan)
          • direction arrow
          • IN/OUT count badge
        """
        if not self.active or self.p1 is None or self.p2 is None:
            return

        x1 = int(self.p1[0] * scale);  y1 = int(self.p1[1] * scale)
        x2 = int(self.p2[0] * scale);  y2 = int(self.p2[1] * scale)
        hw = int(self.half_width * scale)

        line_len = float(np.hypot(x2 - x1, y2 - y1))
        if line_len < 1:
            return

        # Perpendicular unit vector (left normal of P1→P2)
        nx = -(y2 - y1) / line_len
        ny =  (x2 - x1) / line_len

        # Corridor corners (parallelogram aligned with the line)
        corners = np.array([
            [int(x1 + nx * hw),  int(y1 + ny * hw)],
            [int(x2 + nx * hw),  int(y2 + ny * hw)],
            [int(x2 - nx * hw),  int(y2 - ny * hw)],
            [int(x1 - nx * hw),  int(y1 - ny * hw)],
        ], dtype=np.int32)

        # Semi-transparent corridor fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [corners], (0, 80, 160))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # Dashed corridor boundary
        cv2.polylines(frame, [corners], True, (0, 120, 200), 1, cv2.LINE_AA)

        # Solid counting line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0),    5, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

        # Endpoint dots
        cv2.circle(frame, (x1, y1), 5, (0, 200, 255), -1)
        cv2.circle(frame, (x2, y2), 5, (0, 200, 255), -1)

        # Direction arrow (left normal → IN direction)
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        arlen  = 22
        ax = int(mx + arlen * nx)
        ay = int(my + arlen * ny)
        cv2.arrowedLine(frame, (mx, my), (ax, ay),
                        (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)

        # IN / OUT count badge
        txt = f"IN:{self.count_in}  OUT:{self.count_out}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bx = mx - tw // 2
        by = my - 22
        cv2.rectangle(frame, (bx - 5, by - th - 4), (bx + tw + 5, by + 4),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (bx - 5, by - th - 4), (bx + tw + 5, by + 4),
                      (0, 200, 255), 1)
        cv2.putText(frame, txt, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Corridor width label (bottom-right corner of corridor)
        hw_txt = f"w:{self.half_width * 2}px"
        cv2.putText(frame, hw_txt,
                    (int(x2 - nx * hw) + 4, int(y2 - ny * hw) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 200), 1, cv2.LINE_AA)
