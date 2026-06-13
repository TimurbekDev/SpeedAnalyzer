"""
Central defaults and tuning constants for the lane-distance app.
All magic numbers live here — change once, affects the whole app.
"""

import os

# ── YOLO / detection ──────────────────────────────────────────────────────────
# COCO: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES: set  = {2, 3, 5, 7}
CONF_DEFAULT:    float = 0.35       # minimum detection confidence (real footage)

# ── Tracking (Ultralytics ByteTrack) ──────────────────────────────────────────
# Absolute path to the repo's tuned ByteTrack config (longer track_buffer →
# fewer ID switches → no duplicate log rows for one physical vehicle).
TRACKER_CFG: str = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "bytetrack.yaml")

# ── Thread queue sizes ────────────────────────────────────────────────────────
CAP_Q_SIZE: int = 8   # capture → worker queue
RES_Q_SIZE: int = 8   # worker → UI queue

# ── ROI / lane calibration ────────────────────────────────────────────────────
ROI_MIN_POINTS: int = 3       # a polygon needs at least 3 vertices
ROI_HANDLE_PX:  int = 10      # vertex grab radius (video px) while editing
ROI_COLOR:    tuple = (255, 180, 0)   # BGR default lane-ROI outline/fill (draft)

# Per-ROI colours (BGR) cycled as ROIs are added, so each lane is visually
# distinct on the video.
ROI_PALETTE: list = [
    (255, 180,   0),   # amber
    ( 80, 220,   0),   # green
    (  0, 170, 255),   # orange
    (255,  90, 200),   # magenta
    (255, 255,   0),   # cyan
    ( 90,  90, 255),   # red
    (200, 120, 255),   # pink
    (  0, 230, 230),   # yellow
]

LANE_LENGTH_M_DEFAULT: float = 20.0   # real length the ROI spans along the lane
LANE_WIDTH_M_DEFAULT:  float = 3.5    # standard road-lane width (m)
GRID_STEP_M:           float = 5.0    # spacing of the calibration check grid (m)

# ── Following-distance safety thresholds (for gap colour coding) ──────────────
SAFE_GAP_M: float = 15.0   # ≥ safe  → green
WARN_GAP_M: float =  8.0   # ≥ warn  → amber ; below → red

# ── Speed estimation (metric / homography based) ──────────────────────────────
SPEED_WINDOW_S: float = 0.6     # regression window for ground-plane displacement
SPEED_EWMA:     float = 0.4     # output smoothing (lower = smoother, laggier)
SPEED_MIN_DT:   float = 0.15    # minimum time span before a speed is reported
SPEED_MAX_KMH:  float = 250.0   # physical sanity cap
# Wait this long for a valid speed on both vehicles before logging the pair;
# measure anyway after the grace window so stopped/jam traffic is still recorded.
MEASURE_SPEED_GRACE_S: float = 1.0

# ── Excel export ──────────────────────────────────────────────────────────────
EXPORT_DIR: str = "."   # where auto-saved .xlsx files are written
