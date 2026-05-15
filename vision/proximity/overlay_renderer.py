"""
OpenCV proximity overlay renderer.

render_proximity_overlay — colour-coded distance lines + labels between vehicles.
HeatmapAccumulator       — low-res grid heatmap with temporal decay.

All drawing is in-place on BGR frames.
"""

import cv2
import numpy as np
from typing import List
from vision.proximity.distance_calculator import VehiclePair

# BGR palette for the three safety zones
_GREEN  = (0, 220,  80)    # safe
_YELLOW = (0, 200, 240)    # warning  (looks yellow in BGR preview)
_RED    = (0,  60, 255)    # danger


def _dist_color(m_dist: float, safe_m: float, warn_m: float) -> tuple:
    """Smooth interpolated BGR colour based on distance vs thresholds."""
    if m_dist >= safe_m:
        return _GREEN
    if m_dist >= warn_m:
        t = (m_dist - warn_m) / max(safe_m - warn_m, 1e-4)
        return tuple(int(_YELLOW[c] + t * (_GREEN[c] - _YELLOW[c])) for c in range(3))
    t = m_dist / max(warn_m, 1e-4)
    return tuple(int(_RED[c] + t * (_YELLOW[c] - _RED[c])) for c in range(3))


def render_proximity_overlay(
        frame:       np.ndarray,
        pairs:       List[VehiclePair],
        safe_m:      float,
        warn_m:      float,
        show_labels: bool = True,
        max_pairs:   int  = 24,
) -> None:
    """
    Draw distance lines between vehicle pairs, colour-coded by safety zone.
    Only the closest max_pairs pairs are drawn to avoid clutter.
    """
    if not pairs:
        return
    drawn = sorted(pairs, key=lambda p: p.m_dist)[:max_pairs]
    for pair in drawn:
        color = _dist_color(pair.m_dist, safe_m, warn_m)
        p1 = (int(pair.cx_a), int(pair.cy_a))
        p2 = (int(pair.cx_b), int(pair.cy_b))
        cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
        if show_labels:
            mx = (p1[0] + p2[0]) // 2
            my = (p1[1] + p2[1]) // 2
            txt = f"{pair.m_dist:.1f}m"
            # dark stroke for legibility on any background
            cv2.putText(frame, txt, (mx + 2, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0),  2, cv2.LINE_AA)
            cv2.putText(frame, txt, (mx + 2, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color,       1, cv2.LINE_AA)


class HeatmapAccumulator:
    """
    Accumulates vehicle centroid positions into a low-resolution grid,
    applies temporal decay, then renders as a COLORMAP_JET overlay.

    Runs entirely on the main thread (_poll → update, _render → render).
    No locking required.
    """

    def __init__(self, grid_w: int = 80, grid_h: int = 45):
        self.gw     = grid_w
        self.gh     = grid_h
        self._grid  = np.zeros((grid_h, grid_w), dtype=np.float32)
        self._decay = 0.92

    def update(self, tracks, frame_w: int, frame_h: int) -> None:
        self._grid *= self._decay
        fw = max(frame_w, 1)
        fh = max(frame_h, 1)
        for t in tracks:
            gx = int(t.cx / fw * self.gw)
            gy = int(t.cy / fh * self.gh)
            gx = max(0, min(self.gw - 1, gx))
            gy = max(0, min(self.gh - 1, gy))
            self._grid[gy, gx] += 1.0

    def render(self, frame: np.ndarray, alpha: float = 0.30) -> None:
        if self._grid.max() < 0.3:
            return
        fh, fw = frame.shape[:2]
        norm    = cv2.normalize(self._grid, None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)
        blurred = cv2.GaussianBlur(norm, (7, 7), 0)
        colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        resized = cv2.resize(colored, (fw, fh), interpolation=cv2.INTER_LINEAR)

        # Soft mask: only paint where the heatmap has meaningful intensity
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (31, 31), 0)
        mask = np.clip(gray * alpha, 0.0, alpha)

        for c in range(3):
            frame[:, :, c] = np.clip(
                frame[:, :, c].astype(np.float32) * (1.0 - mask)
                + resized[:, :, c].astype(np.float32) * mask,
                0, 255,
            ).astype(np.uint8)

    def reset(self) -> None:
        self._grid[:] = 0.0
