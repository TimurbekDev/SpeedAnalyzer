"""
Pairwise inter-vehicle distance calculator.

For n tracks computes all n(n-1)/2 pairs.
Distances are reported in pixels AND meters via px_per_m calibration.
Optional bird's-eye perspective transform pre-warps centroids before
computing distances so camera angle is corrected.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VehiclePair:
    id_a:    int
    id_b:    int
    cx_a:    float
    cy_a:    float
    cx_b:    float
    cy_b:    float
    px_dist: float
    m_dist:  float


class DistanceCalculator:
    """
    Computes pairwise Euclidean distances between vehicle centroids.

    px_per_m  — pixels per metre (simple flat-ground calibration).
    If a bird's-eye transform M is set, centroids are first warped to a
    top-view metric space before distance is measured; px_per_m is then
    the scale of that warped space (set via set_birdseye).
    """

    def __init__(self, px_per_m: float = 20.0):
        self.px_per_m       = max(0.1, float(px_per_m))
        self._M:            Optional[np.ndarray] = None   # 3×3 perspective
        self._be_px_per_m:  Optional[float]      = None   # scale in warped space

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_birdseye(self, M: Optional[np.ndarray],
                     be_px_per_m: float) -> None:
        self._M           = M
        self._be_px_per_m = max(0.1, be_px_per_m) if M is not None else None

    # ── Core calculation ──────────────────────────────────────────────────────

    def compute(self, tracks) -> List[VehiclePair]:
        if len(tracks) < 2:
            return []

        raw = np.array([[t.cx, t.cy] for t in tracks], dtype=np.float32)

        if self._M is not None and self._be_px_per_m is not None:
            warped = cv2.perspectiveTransform(
                raw.reshape(-1, 1, 2), self._M).reshape(-1, 2)
            metric = warped / self._be_px_per_m      # → metres in top-view
        else:
            metric = raw / self.px_per_m             # simple flat-ground

        pairs: List[VehiclePair] = []
        n = len(tracks)
        for i in range(n):
            for j in range(i + 1, n):
                dx_m  = float(metric[i, 0] - metric[j, 0])
                dy_m  = float(metric[i, 1] - metric[j, 1])
                m_dist = (dx_m * dx_m + dy_m * dy_m) ** 0.5

                dx_px  = float(raw[i, 0] - raw[j, 0])
                dy_px  = float(raw[i, 1] - raw[j, 1])
                px_dist = (dx_px * dx_px + dy_px * dy_px) ** 0.5

                pairs.append(VehiclePair(
                    id_a=tracks[i].id, id_b=tracks[j].id,
                    cx_a=raw[i, 0],    cy_a=raw[i, 1],
                    cx_b=raw[j, 0],    cy_b=raw[j, 1],
                    px_dist=px_dist,   m_dist=m_dist,
                ))
        return pairs
