"""
Central defaults and tuning constants.
All magic numbers live here — change once, affects the whole app.
"""

# ── YOLO / detection ──────────────────────────────────────────────────────────
VEHICLE_CLASSES: set  = {2, 5, 7}   # COCO: car=2, bus=5, truck=7
CONF_DEFAULT:    float = 0.35        # minimum detection confidence
FRAME_SKIP:      int   = 2           # run YOLO every N frames (1 = every frame)

# ── Thread queue sizes ────────────────────────────────────────────────────────
CAP_Q_SIZE: int = 4   # capture → worker queue
RES_Q_SIZE: int = 6   # worker → UI queue

# ── Speed display bracketing ──────────────────────────────────────────────────
# (upper_km/h, BGR_color)
SPEED_COLORS: list = [
    (60,  (0, 220,  80)),   # green  — under 60
    (110, (0, 200, 255)),   # yellow — 60-110
    (999, (0,  60, 255)),   # red    — above 110
]

# ── Screenshot output folder ──────────────────────────────────────────────────
SCREENSHOT_DIR: str = "speed_screenshots"

# ── Tracker defaults ──────────────────────────────────────────────────────────
TRACKER_MAX_AGE:  int   = 20    # frames before dropping a lost track
TRACKER_MIN_HITS: int   = 3     # detections before track is reported
TRACKER_IOU_THR:  float = 0.20  # IoU overlap threshold for matching
TRACKER_DIST_THR: int   = 120   # centroid distance threshold for matching
