"""
Camera projection model — converts pixel-space velocity to km/h.

Uses direction-aware scales derived from camera height, tilt, and focal length:
  horizontal component → lateral scale   Z/f
  vertical component   → depth scale     Z²/(f·H)
where Z is camera-frame depth to the ground point at the reference image row.
"""

import numpy as np
import json


class CameraModel:
    def __init__(self):
        self.enabled      = False
        self.H            = 6.0     # camera height above road (m)
        self.tilt         = 15.0    # camera tilt/pitch (degrees, downward positive)
        self.pan          = 0.0     # camera pan (degrees, unused in speed calc)
        self.f            = 800.0   # focal length (pixels)
        self.cx           = 640.0   # principal point x
        self.cy           = 360.0   # principal point y
        self.px_per_m     = 20.0    # fallback scale when camera model is disabled
        self.speed_factor = 1.0     # user-adjustable output multiplier

    def setup(self, H: float, tilt: float, pan: float,
              f: float, fw: int, fh: int) -> None:
        self.H, self.tilt, self.pan, self.f = H, tilt, pan, f
        self.cx, self.cy = fw / 2.0, fh / 2.0
        self.enabled = True

    def load_json(self, path: str) -> None:
        with open(path) as fp:
            d = json.load(fp)
        self.H    = d.get("camera_height_m",  6.0)
        self.tilt = d.get("camera_tilt_deg", d.get("camera_pitch_deg", 15.0))
        self.pan  = d.get("camera_pan_deg",  0.0)
        self.f    = d.get("focal_length_px", 800.0)
        self.cx   = d.get("frame_width",  1280) / 2.0
        self.cy   = d.get("frame_height",  720) / 2.0
        self.enabled = True

    def pixel_speed_to_kmh(self, vx_px_s: float, vy_px_s: float,
                           ref_v: float) -> float:
        sf = max(0.01, self.speed_factor)
        if self.enabled:
            tilt_r  = np.radians(self.tilt)
            alpha_v = np.arctan((self.cy - ref_v) / self.f)
            beta    = tilt_r - alpha_v
            if beta < 0.015:
                return 0.0
            Z = self.H * np.cos(alpha_v) / np.sin(beta)
            if Z < 0.1:
                return 0.0
            m_per_px_x = Z / self.f
            m_per_px_y = (Z * Z) / (self.f * self.H)
            speed_m_s  = np.sqrt((vx_px_s * m_per_px_x) ** 2 +
                                 (vy_px_s * m_per_px_y) ** 2)
            return speed_m_s * 3.6 * sf
        px_per_s = np.sqrt(vx_px_s ** 2 + vy_px_s ** 2)
        return px_per_s / self.px_per_m * 3.6 * sf
