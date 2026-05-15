"""
4-point perspective (bird's-eye) transform for accurate distance measurement.

Usage:
    bt = BirdseyeTransform()
    bt.calibrate(src_pts_4x2, real_width_m, real_height_m)
    calc.set_birdseye(bt.M, bt.px_per_m_out)

src_pts order: top-left → top-right → bottom-right → bottom-left
"""

import cv2
import numpy as np
from typing import Optional


class BirdseyeTransform:
    OUT_W = 500
    OUT_H = 500

    def __init__(self):
        self.M:            Optional[np.ndarray] = None
        self.px_per_m_out: float = 1.0
        self.active        = False

    def calibrate(self,
                  src_pts:  np.ndarray,   # 4×2 float32 image points (TL,TR,BR,BL)
                  width_m:  float,        # real-world width of rectangle (m)
                  height_m: float) -> None:
        dst = np.array([
            [0,             0            ],
            [self.OUT_W - 1, 0            ],
            [self.OUT_W - 1, self.OUT_H - 1],
            [0,             self.OUT_H - 1],
        ], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst)
        px_per_m_x = self.OUT_W / max(width_m,  1e-3)
        px_per_m_y = self.OUT_H / max(height_m, 1e-3)
        self.px_per_m_out = (px_per_m_x + px_per_m_y) / 2.0
        self.active = True

    def clear(self) -> None:
        self.M            = None
        self.px_per_m_out = 1.0
        self.active       = False
