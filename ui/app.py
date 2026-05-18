"""Main application class — wires backend to modern UI."""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import time
import os
from datetime import datetime
from collections import deque

from config.defaults import (CONF_DEFAULT, CAP_Q_SIZE, RES_Q_SIZE,
                             PROX_SAFE_DIST_M, PROX_WARN_DIST_M,
                             PROX_PX_PER_M, PROX_LOG_INTERVAL)
from core.entities.track import Track
from vision.detection.loader        import load_yolo
from vision.detection.bg_detector   import BGDetector
from vision.detection.enhancer      import Enhancer
from vision.tracking.stabilizer     import VideoStabilizer
from vision.tracking.tracker        import Tracker
from vision.geometry.camera_model   import CameraModel
from vision.geometry.lane_counter   import CrossLine
from vision.rendering.frame_renderer import draw_tracks, draw_cross_flash
from vision.pipeline.worker         import Worker
from vision.proximity import (DistanceCalculator, ProximityAnalytics,
                               render_proximity_overlay, HeatmapAccumulator,
                               BirdseyeTransform,
                               FollowingDistanceCalculator, FollowingResult,
                               CrossingDistanceCalculator)
from vision.export import TrafficReportExporter, make_filename
from ui.theme.dark          import Theme
from ui.widgets.components  import TopBar, SidebarNav, PlaybackBar, ToastManager
from ui.dialogs.calibrator  import CalibratorWindow
from ui.dialogs.birdseye_dialog import BirdseyeDialog
from ui.panels.dashboard    import DashboardPanel
from ui.panels.detection    import DetectionPanel
from ui.panels.analytics    import AnalyticsPanel
from ui.panels.proximity_panel import ProximityPanel
from ui.panels.settings_panel import SettingsPanel
from ui.panels.export_panel import ExportPanel


class SpeedAnalyzerModern:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._session_start = time.time()
        self._init_state()
        self._load_backend()
        self._build_ui()
        self._ui_loop()

    # ── State ─────────────────────────────────────────────────────────────────
    def _init_state(self):
        self.cap        = None
        self.fps        = 30.0
        self.video_name = ""
        self.playing    = False
        self.worker     = None
        self.cap_thread = None

        self.in_q:  queue.Queue = queue.Queue(maxsize=CAP_Q_SIZE)
        self.out_q: queue.Queue = queue.Queue(maxsize=RES_Q_SIZE)

        self._last_frame:  np.ndarray | None = None
        self._last_tracks: list = []
        self._ftimes = deque(maxlen=30)

        self.sx = self.sy = 1.0
        self.ox = self.oy = 0

        self.results:      list = []
        self.recorded_ids: set  = set()
        self.ssdir = "speed_screenshots"
        os.makedirs(self.ssdir, exist_ok=True)
        self._slider_busy = False

        self.cross_line    = CrossLine()
        self._drawing_line = False
        self._line_p1:  list | None = None
        self._line_tmp: list | None = None
        self._cross_flash: dict = {}

        # Proximity zone (second drawable zone for distance analytics)
        self._prox_zone:        list              = []
        self._drawing_prox:     bool              = False
        self._prox_tmp:         list | None       = None
        self._prox_zone_mask:   np.ndarray | None = None
        self._prox_zone_area_m2: float            = 5000.0
        self._last_prox_snap:   dict              = {
            "pairs": [], "count": 0, "min_m": 0.0,
            "max_m": 0.0, "avg_m": 0.0, "congestion": 0.0, "density": 0.0,
        }
        self._prox_refresh_t:   float = 0.0
        self._prox_log:         list  = []
        self._prox_log_counter: int   = 0

        self._last_following:   list  = []  # List[FollowingResult], updated each frame

        self._speed_event = threading.Event()
        self._dist_event  = threading.Event()
        self._speed_event.set()
        self._dist_event.set()

        self._stab_dx = self._stab_dy = 0.0
        self._violations = 0
        self._total      = 0

        self.var_limit     = tk.DoubleVar(value=60.0)
        self.var_stab      = tk.BooleanVar(value=True)
        self.var_stabview  = tk.BooleanVar(value=True)
        self.var_cam_on    = tk.BooleanVar(value=False)
        self.var_cam_H     = tk.DoubleVar(value=6.0)
        self.var_cam_T     = tk.DoubleVar(value=15.0)
        self.var_cam_P     = tk.DoubleVar(value=0.0)
        self.var_cam_F     = tk.DoubleVar(value=800.0)
        self.var_min_a     = tk.IntVar(value=1500)
        self.var_max_a     = tk.IntVar(value=150000)
        self.var_trail     = tk.BooleanVar(value=True)
        self.var_boxes     = tk.BooleanVar(value=True)
        self.var_speed_lbl = tk.BooleanVar(value=True)
        self.var_autosave  = tk.BooleanVar(value=True)
        self.var_speed_mul = tk.DoubleVar(value=1.0)
        self.var_speed_k   = tk.DoubleVar(value=1.0)
        self.var_conf      = tk.DoubleVar(value=CONF_DEFAULT)
        self.var_lane_hw   = tk.IntVar(value=CrossLine.DEFAULT_HW)

        # Proximity analytics controls
        self.var_safe_dist    = tk.DoubleVar(value=PROX_SAFE_DIST_M)
        self.var_warn_dist    = tk.DoubleVar(value=PROX_WARN_DIST_M)
        self.var_prox_px_m    = tk.DoubleVar(value=PROX_PX_PER_M)
        self.var_show_dist    = tk.BooleanVar(value=True)
        self.var_show_labels  = tk.BooleanVar(value=True)
        self.var_show_heatmap = tk.BooleanVar(value=False)
        self.var_use_birdseye = tk.BooleanVar(value=False)

    def _load_backend(self):
        self.yolo_det, self.model_name = load_yolo()
        self.bg_det    = BGDetector()
        self.enhancer  = Enhancer()
        self.stabilizer = VideoStabilizer()
        self.cam       = CameraModel()
        self.tracker   = Tracker(30.0, self.cam)
        self.prox_calc      = DistanceCalculator(px_per_m=PROX_PX_PER_M)
        self.prox_analytics = ProximityAnalytics(
            safe_m=PROX_SAFE_DIST_M, warn_m=PROX_WARN_DIST_M)
        self.birdseye          = BirdseyeTransform()
        self.heatmap           = HeatmapAccumulator()
        self.following_calc    = FollowingDistanceCalculator(
            px_per_m=PROX_PX_PER_M)
        self.crossing_calc = CrossingDistanceCalculator(
            px_per_m=PROX_PX_PER_M)
        self._exporter     = TrafficReportExporter()

    def reload_yolo(self, model_name: str):
        """Load a different YOLO model (e.g. 'yolov8s', 'yolov8m')."""
        print(f"[APP] Switching to {model_name}...")
        det, name = load_yolo(model_name=model_name)
        if det is None:
            self.toast.show(f"Failed to load {model_name}", "error")
            return
        self.yolo_det = det
        self.model_name = name
        self.topbar.update_model_name(name)
        self.toast.show(f"Switched to {name}", "info")

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("Speed Analyzer Pro")
        self.root.geometry("1480x900")
        self.root.minsize(1200, 700)
        self.root.configure(bg=Theme.BG)

        self.topbar = TopBar(self.root, model_name=self.model_name)
        self.topbar.pack(fill="x")
        tk.Frame(self.root, bg=Theme.DIVIDER, height=1).pack(fill="x")

        self.toast = ToastManager(self.root)

        middle = tk.Frame(self.root, bg=Theme.BG)
        middle.pack(fill="both", expand=True)

        self.sidebar = SidebarNav(middle, self._navigate)
        self.sidebar.pack(side="left", fill="y")
        tk.Frame(middle, bg=Theme.DIVIDER, width=1).pack(side="left", fill="y")

        self.content = tk.Frame(middle, bg=Theme.BG)
        self.content.pack(side="left", fill="both", expand=True)

        tk.Frame(self.root, bg=Theme.DIVIDER, height=1).pack(fill="x", side="bottom")
        self.playbar = PlaybackBar(self.root, self)
        self.playbar.pack(fill="x", side="bottom")

        self._panels: dict = {
            "dashboard": DashboardPanel(self.content, self),
            "detection": DetectionPanel(self.content, self),
            "analytics": AnalyticsPanel(self.content, self),
            "proximity": ProximityPanel(self.content, self),
            "settings":  SettingsPanel(self.content, self),
            "export":    ExportPanel(self.content, self),
        }
        for panel in self._panels.values():
            panel.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._navigate("dashboard")

        self.canvas = self._panels["dashboard"].video.canvas
        self.canvas.bind("<ButtonPress-1>",  self._canvas_click)
        self.canvas.bind("<Motion>",         self._canvas_motion)
        self.canvas.bind("<ButtonPress-3>",  self._canvas_rclick)
        self.canvas.bind("<Configure>",
                         lambda _: self._render() if self._last_frame is not None else None)

    def _navigate(self, key: str):
        self._active_panel = key
        for k, p in self._panels.items():
            (p.lift if k == key else p.lower)()
        if key == "analytics":
            self._panels["analytics"].refresh(
                self.results, self._session_start, self.var_limit.get())
        if key == "export":
            self._panels["export"].refresh(self.results, self.var_limit.get())

    # ── 60-fps UI refresh ─────────────────────────────────────────────────────
    def _ui_loop(self):
        if self._last_tracks:
            speeds = [t.speed_kmh for t in self._last_tracks if t.speed_kmh > 1]
            avg = np.mean(speeds) if speeds else 0
            self._panels["dashboard"].update_cards(
                self._total, avg, self._violations, self._current_fps())
            self.topbar.set_tracks(len(self._last_tracks))
        self.root.after(33, self._ui_loop)

    def _current_fps(self) -> float:
        if len(self._ftimes) < 2:
            return 0.0
        dts = [self._ftimes[i+1] - self._ftimes[i]
               for i in range(len(self._ftimes) - 1)
               if self._ftimes[i+1] > self._ftimes[i]]
        return (1.0 / (sum(dts) / len(dts))) if dts else 0.0

    # ── Video open ────────────────────────────────────────────────────────────
    def _open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.ts"), ("All", "*.*")])
        if not path:
            return
        self._stop_play()
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_name = os.path.basename(path)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.playbar.slider.configure(to=max(1, total - 1))
        self.playbar.set_time(0, total / self.fps)
        self.tracker   = Tracker(self.fps, self.cam)
        self.results   = []; self.recorded_ids = set(); self._cross_flash = {}
        self._total    = 0;  self._violations = 0
        self._prox_zone      = []; self._prox_zone_mask = None
        self._prox_log       = []; self._prox_log_counter = 0
        self._last_following = []
        self._exporter.reset()
        self.prox_analytics.reset(); self.heatmap.reset()
        self._panels["dashboard"].chart.reset()
        ok, f = self.cap.read()
        if ok:
            self._last_frame  = f
            self._last_tracks = []
            self._panels["dashboard"].video.set_has_frame()
            self._render()
            # Auto-define default crossline at 60% frame height if none active
            if not self.cross_line.active:
                fh, fw = f.shape[:2]
                y0 = int(fh * 0.60)
                self.cross_line.define([0, y0], [fw, y0])
                self.cross_line.set_width(self.var_lane_hw.get())
        self.root.title(f"Speed Analyzer Pro — {self.video_name}")
        self.topbar.set_status("IDLE")
        self.topbar.set_model(self.model_name or "BG MODE")
        self.toast.show(f"Opened: {self.video_name}", "info")

    # ── Playback ──────────────────────────────────────────────────────────────
    def _toggle_play(self):
        if not self.cap:
            self.toast.show("Open a video file first.", "warning"); return
        if self.playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        self.playing = True
        self.playbar.set_playing(True)
        self.topbar.set_status("PLAYING")
        for q in (self.in_q, self.out_q):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break
        if self.tracker is None:
            self.tracker = Tracker(self.fps, self.cam)
        self.tracker.reset()
        self.stabilizer.reset()
        self.bg_det = BGDetector(self.var_min_a.get(), self.var_max_a.get())
        self.worker = Worker(self.yolo_det, self.bg_det, self.enhancer,
                             self.tracker, self.in_q, self.out_q,
                             cross_line=self.cross_line,
                             prox_zone=self._prox_zone_mask,
                             prox_calc=self.prox_calc,
                             prox_analytics=self.prox_analytics,
                             prox_area_m2=self._prox_zone_area_m2,
                             following_calc=self.following_calc,
                             crossing_calc=self.crossing_calc,
                             speed_event=self._speed_event,
                             dist_event=self._dist_event)
        self.worker.start()
        self.cap_thread = threading.Thread(target=self._cap_loop, daemon=True)
        self.cap_thread.start()
        self._schedule_poll()

    def _stop_play(self):
        if not self.playing:
            return
        self.playing = False
        self.playbar.set_playing(False)
        self.topbar.set_status("IDLE")
        try:
            self.in_q.put_nowait(None)
        except queue.Full:
            pass
        if self.worker:
            self.worker.stop()
            self.worker = None
        if self.var_autosave.get() and self.results:
            self._export_unified(auto=True)

    def _cap_loop(self):
        interval = 1.0 / self.fps
        t_next   = time.perf_counter() + interval
        while self.playing:
            ok, frame = self.cap.read()
            if not ok:
                try: self.in_q.put(None, timeout=1.0)
                except queue.Full: pass
                break
            stab_frame, dx, dy, _ = self.stabilizer.process(frame)
            self._stab_dx, self._stab_dy = dx, dy
            try:
                fid = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.in_q.put_nowait((fid, stab_frame))
            except queue.Full:
                pass
            sleep_t = t_next - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            t_next += interval

    def _schedule_poll(self):
        delay = max(8, int(1000.0 / self.fps))
        self.root.after(delay, self._poll)

    def _poll(self):
        if not self.playing:
            return
        latest_frame  = None
        latest_tracks = None
        all_finished:  list = []
        all_crossings: list = []

        try:
            while True:
                item = self.out_q.get_nowait()
                if item is None:
                    self._stop_play()
                    self.toast.show("Video finished.", "info")
                    return
                frame_d, tracks_d, finished_d, crossings_d, following_d = item
                latest_frame  = frame_d
                latest_tracks = tracks_d
                all_finished.extend(finished_d)
                all_crossings.extend(crossings_d)
                self._last_following = following_d
        except queue.Empty:
            pass

        if latest_frame is not None:
            self._last_frame  = latest_frame
            self._last_tracks = latest_tracks

            # Heatmap update (all tracks, every frame, main thread only)
            if self._last_tracks:
                fh, fw = self._last_frame.shape[:2]
                self.heatmap.update(self._last_tracks, fw, fh)

            # Proximity panel refresh at ~5 fps to avoid TK widget churn
            now_t = time.perf_counter()
            if now_t - self._prox_refresh_t > 0.20:
                self._last_prox_snap = self.prox_analytics.snapshot
                self._panels["proximity"].refresh(self._last_prox_snap)
                self._prox_refresh_t = now_t

            # Proximity log entry every PROX_LOG_INTERVAL frames
            self._prox_log_counter += 1
            if (self._prox_log_counter % PROX_LOG_INTERVAL == 0
                    and self._last_prox_snap["count"] > 0):
                entry = {
                    "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "vehicles":    self._last_prox_snap["count"],
                    "min_m":       round(self._last_prox_snap["min_m"],  1),
                    "max_m":       round(self._last_prox_snap["max_m"],  1),
                    "avg_m":       round(self._last_prox_snap["avg_m"],  1),
                    "congestion":  round(self._last_prox_snap["congestion"], 1),
                    "density":     round(self._last_prox_snap["density"],    2),
                }
                self._prox_log.append(entry)

            if all_crossings:
                for ev in all_crossings:
                    if ev["dir"] == "IN":
                        self.cross_line.count_in  += 1
                    else:
                        self.cross_line.count_out += 1
                    self.cross_line.events.append(ev)
                    self._record_crossing(ev)
                    self._cross_flash[ev["id"]] = {
                        "speed": ev["speed"], "dir": ev["dir"],
                        "cx":    ev["cx"],    "cy":  ev["cy"],
                        "t0":    time.perf_counter(),
                    }
                self._panels["dashboard"].update_crossing(
                    self.cross_line.count_in, self.cross_line.count_out)

            if latest_tracks:
                spd = max((t.speed_kmh for t in latest_tracks), default=0)
                if spd > 1:
                    self._panels["dashboard"].chart.push(spd, self.var_limit.get())

            self._update_fps()
            self._panels["dashboard"].video.set_has_frame()

        if self._last_frame is not None:
            self._render()
            if self.cap:
                pos   = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._slider_busy = True
                try:
                    self.playbar.slider.set(pos)
                    self.playbar.set_time(pos / self.fps, total / self.fps)
                finally:
                    self._slider_busy = False

        self._schedule_poll()

    def _update_fps(self):
        self._ftimes.append(time.perf_counter())
        self.topbar.set_fps(self._current_fps())

    def _record_crossing(self, ev: dict):
        track_id = ev["id"]
        if track_id in self.recorded_ids:
            return
        self.recorded_ids.add(track_id)
        speed = ev["speed"]
        limit = self.var_limit.get()
        self._total += 1
        if speed > limit:
            self._violations += 1
            if speed > limit * 1.3:
                self.toast.show(f"Speed Alert: #{track_id}  {speed:.0f} km/h",
                                "error", duration=4000)
        record = {
            "id":        track_id,
            "speed_kmh": round(speed, 1),
            "cross_dir": ev["dir"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video":     self.video_name,
        }
        self.results.append(record)
        self._panels["dashboard"].log_detection(track_id, speed, ev["dir"], limit)

        dist_result = ev.get("crossing_dist")
        self._exporter.append(record, dist_result, limit)

    # ── Step / seek ───────────────────────────────────────────────────────────
    def _step(self, n: int):
        if not self.cap:
            return
        cur   = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, cur + n)))
        ok, f = self.cap.read()
        if ok:
            self._last_frame = f
            self._panels["dashboard"].video.set_has_frame()
            self._render()
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._slider_busy = True
        try:
            self.playbar.slider.set(pos)
            self.playbar.set_time(pos / self.fps, total / self.fps)
        finally:
            self._slider_busy = False

    def _seek(self, v):
        if not self.cap or self._slider_busy:
            return
        pos = int(float(v))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, f = self.cap.read()
        if ok:
            self._last_frame = f
            self._panels["dashboard"].video.set_has_frame()
            self._render()
        if self.cap:
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.playbar.set_time(pos / self.fps, total / self.fps)

    # ── Canvas / line ────────────────────────────────────────────────────────
    def _line_start(self):
        self._drawing_line = True
        self._line_p1      = None
        self._line_tmp     = None
        self.cross_line.clear()
        self.cross_line.set_width(self.var_lane_hw.get())
        self._panels["dashboard"].update_crossing(0, 0)
        if self._last_frame is not None:
            self._render()
        self.toast.show("Click two points to define the counting line.", "info")

    def _clear_line(self):
        self.cross_line.clear()
        self._drawing_line = False
        self._line_p1 = self._line_tmp = None
        self._panels["dashboard"].update_crossing(0, 0)
        if self._last_frame is not None:
            self._render()

    def _canvas_click(self, e):
        if self._last_frame is None:
            return
        vx, vy = self._c2v(e.x, e.y)
        if self._drawing_line:
            if self._line_p1 is None:
                self._line_p1 = [vx, vy]
            else:
                self.cross_line.define(self._line_p1, [vx, vy])
                self.cross_line.set_width(self.var_lane_hw.get())
                self._drawing_line = False
                self._line_p1      = None
                self._line_tmp     = None
                if self.worker:
                    self.worker.set_cross_line(self.cross_line)
                self._render()
                hw = self.var_lane_hw.get()
                self.toast.show(f"Counting line set  |  corridor ±{hw}px", "success")
            return
        if self._drawing_prox:
            self._prox_zone.append([vx, vy])
            self._render()

    def _canvas_rclick(self, e):
        if self._drawing_line:
            self._drawing_line = False
            self._line_p1 = self._line_tmp = None
            self._render()
            return
        if self._drawing_prox and len(self._prox_zone) >= 3:
            self._drawing_prox = False
            self._prox_tmp     = None
            fh, fw = self._last_frame.shape[:2]
            mask = np.zeros((fh, fw), np.uint8)
            cv2.fillPoly(mask, [np.array(self._prox_zone, np.int32)], 255)
            self._prox_zone_mask = mask
            # Estimate real-world area from polygon area in px → m²
            area_px = float(cv2.contourArea(
                np.array(self._prox_zone, np.int32).reshape(-1, 1, 2)))
            px_m = max(self.var_prox_px_m.get(), 0.1)
            self._prox_zone_area_m2 = area_px / (px_m * px_m)
            if self.worker:
                self.worker.set_prox_zone(mask, self._prox_zone_area_m2)
            self._render()
            self._navigate("proximity")
            self.toast.show(
                "Proximity zone set — play video to see analytics.", "success")

    def _canvas_motion(self, e):
        if self._last_frame is None:
            return
        vx, vy = self._c2v(e.x, e.y)
        if self._drawing_line:
            self._line_tmp = [vx, vy]
            self._render()
        elif self._drawing_prox:
            self._prox_tmp = [vx, vy]
            self._render()

    def _c2v(self, ex: int, ey: int):
        return (max(0, min(int((ex - self.ox) * self.sx), 9999)),
                max(0, min(int((ey - self.oy) * self.sy), 9999)))

    # ── Proximity zone drawing ────────────────────────────────────────────────

    def _prox_start(self):
        self._drawing_prox = True
        self._drawing_line = False
        self._prox_zone    = []
        self._prox_tmp     = None
        self.toast.show(
            "Left-click to add proximity zone points. Right-click to finish.", "info")

    def _clear_prox(self):
        self._prox_zone      = []
        self._prox_zone_mask = None
        self._drawing_prox   = False
        self._prox_tmp       = None
        self._last_prox_snap = {
            "pairs": [], "count": 0, "min_m": 0.0,
            "max_m": 0.0, "avg_m": 0.0, "congestion": 0.0, "density": 0.0,
        }
        self.prox_analytics.reset()
        if self.worker:
            self.worker.set_prox_zone(None)
        if self._last_frame is not None:
            self._render()
        self.toast.show("Proximity zone cleared.", "warning")

    # ── Proximity settings ────────────────────────────────────────────────────

    def _apply_prox_settings(self):
        px_m = max(0.1, self.var_prox_px_m.get())
        self.prox_calc.px_per_m      = px_m
        self.following_calc.px_per_m = px_m
        self.crossing_calc.px_per_m  = px_m
        self.prox_analytics.safe_m    = self.var_safe_dist.get()
        self.prox_analytics.warn_m    = self.var_warn_dist.get()
        if self.var_use_birdseye.get() and self.birdseye.active:
            self.prox_calc.set_birdseye(self.birdseye.M,
                                        self.birdseye.px_per_m_out)
            self.following_calc.set_birdseye(self.birdseye.M,
                                             self.birdseye.px_per_m_out)
            self.crossing_calc.set_birdseye(self.birdseye.M,
                                            self.birdseye.px_per_m_x,
                                            self.birdseye.px_per_m_y)
        else:
            self.prox_calc.set_birdseye(None, 0.0)
            self.following_calc.set_birdseye(None, 0.0)
            self.crossing_calc.set_birdseye(None, 1.0, 1.0)
        # Recompute zone area with new px_m if zone exists
        if len(self._prox_zone) >= 3:
            area_px = float(cv2.contourArea(
                np.array(self._prox_zone, np.int32).reshape(-1, 1, 2)))
            self._prox_zone_area_m2 = area_px / (px_m * px_m)
            if self.worker:
                self.worker.set_prox_zone(self._prox_zone_mask,
                                          self._prox_zone_area_m2)
        self.toast.show("Proximity settings applied.", "success")

    # ── Feature toggles ──────────────────────────────────────────────────────

    def _toggle_speed(self):
        if self._speed_event.is_set():
            self._speed_event.clear()
            self.playbar.btn_speed.set_active(False)
        else:
            self._speed_event.set()
            self.playbar.btn_speed.set_active(True)

    def _toggle_dist(self):
        if self._dist_event.is_set():
            self._dist_event.clear()
            self.playbar.btn_dist.set_active(False)
        else:
            self._dist_event.set()
            self.playbar.btn_dist.set_active(True)

    # ── Bird's-eye calibration ────────────────────────────────────────────────

    def _open_birdseye_calib(self):
        if self._last_frame is None:
            self.toast.show("Open a video file first.", "warning")
            return
        BirdseyeDialog(self.root, self._last_frame,
                       callback=self._on_birdseye_calibrated)

    def _on_birdseye_calibrated(self, src_pts, width_m: float, height_m: float):
        self.birdseye.calibrate(src_pts, width_m, height_m)
        self.var_use_birdseye.set(True)
        self._apply_prox_settings()
        self.toast.show(
            f"Bird's-eye calibrated  {width_m}m × {height_m}m", "success")

    # ── Proximity Excel export ────────────────────────────────────────────────

    def _export_proximity_excel(self):
        if not EXCEL_OK:
            messagebox.showerror("Error", "openpyxl not installed."); return
        if not self._prox_log:
            self.toast.show("No proximity data recorded yet.", "warning"); return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile=f"proximity_{datetime.now():%Y%m%d_%H%M%S}.xlsx")
        if not path:
            return
        try:
            wb = Workbook()
            ws = wb.active
            ws.title = "Proximity Analytics"

            hdr_fill = PatternFill("solid", fgColor="0D1526")
            hdr_font = Font(bold=True, color="7C3AED", name="Consolas", size=10)
            def_font = Font(name="Consolas", size=9)
            center   = Alignment(horizontal="center")
            thin     = Side(style="thin", color="1E3A5F")
            border   = Border(left=thin, right=thin, top=thin, bottom=thin)

            headers = ["Timestamp", "Vehicles", "Min Dist (m)",
                       "Max Dist (m)", "Avg Dist (m)", "Congestion %", "Density"]
            col_w   = [22, 10, 14, 14, 14, 14, 14]
            for ci, (h, w) in enumerate(zip(headers, col_w), 1):
                c = ws.cell(row=1, column=ci, value=h)
                c.fill = hdr_fill; c.font = hdr_font
                c.alignment = center; c.border = border
                ws.column_dimensions[get_column_letter(ci)].width = w

            for ri, row in enumerate(self._prox_log, 2):
                vals = [row["timestamp"], row["vehicles"],
                        row["min_m"], row["max_m"], row["avg_m"],
                        row["congestion"], row["density"]]
                for ci, val in enumerate(vals, 1):
                    c = ws.cell(row=ri, column=ci, value=val)
                    c.font = def_font; c.alignment = center; c.border = border

            wb.save(path)
            self.toast.show(f"Proximity exported: {os.path.basename(path)}", "success")
        except Exception as ex:
            messagebox.showerror("Export Error", str(ex))

    def _clear_all(self):
        self._stop_play()
        self._drawing_line = False
        self._line_tmp     = None; self._line_p1 = None
        self.cross_line.clear()
        self._panels["dashboard"].update_crossing(0, 0)
        self.tracker       = Tracker(self.fps, self.cam)
        self.stabilizer.reset()
        self.results       = []; self.recorded_ids = set()
        self._cross_flash  = {}; self._total = 0; self._violations = 0
        Track._next        = 1
        self._panels["dashboard"].chart.reset()
        # Proximity reset
        self._prox_zone    = []; self._prox_zone_mask = None
        self._drawing_prox = False; self._prox_tmp = None
        self._prox_log        = []; self._prox_log_counter = 0
        self._last_following = []
        self._exporter.reset()
        self.prox_analytics.reset()
        self.heatmap.reset()
        self._last_prox_snap = {
            "pairs": [], "count": 0, "min_m": 0.0,
            "max_m": 0.0, "avg_m": 0.0, "congestion": 0.0, "density": 0.0,
        }
        if self._last_frame is not None:
            self._render()
        self.toast.show("All data cleared.", "warning")

    # ── Render ────────────────────────────────────────────────────────────────
    def _render(self):
        if self._last_frame is None:
            return
        d  = self._last_frame.copy()
        fh, fw = d.shape[:2]
        cw = max(self.canvas.winfo_width(),  100)
        ch = max(self.canvas.winfo_height(), 100)
        scale = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        self.sx = fw / max(nw, 1)
        self.sy = fh / max(nh, 1)
        self.ox = (cw - nw) // 2
        self.oy = (ch - nh) // 2

        # Proximity zone (purple)
        if len(self._prox_zone) >= 3:
            ov = d.copy()
            cv2.fillPoly(ov, [np.array(self._prox_zone, np.int32)], (100, 0, 160))
            cv2.addWeighted(ov, 0.15, d, 0.85, 0, d)
            cv2.polylines(d, [np.array(self._prox_zone, np.int32)],
                          True, (180, 0, 255), 2)
            cv2.putText(d, "PROX ZONE",
                        (self._prox_zone[0][0] + 5, self._prox_zone[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 0, 255), 1, cv2.LINE_AA)
        elif self._prox_zone and self._drawing_prox:
            for i, pt in enumerate(self._prox_zone):
                cv2.circle(d, tuple(pt), 5, (180, 0, 255), -1)
                if i > 0:
                    cv2.line(d, tuple(self._prox_zone[i-1]), tuple(pt),
                             (180, 0, 255), 2)
            if self._prox_tmp:
                cv2.line(d, tuple(self._prox_zone[-1]), tuple(self._prox_tmp),
                         (180, 0, 255), 1)

        # Heatmap (all tracks, blended under other overlays)
        if self.var_show_heatmap.get():
            self.heatmap.render(d)

        self.cross_line.draw(d)
        if self._drawing_line and self._line_p1 and self._line_tmp:
            cv2.line(d, tuple(self._line_p1), tuple(self._line_tmp),
                     (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(d, tuple(self._line_p1), 5, (0, 200, 255), -1)

        draw_tracks(d, self._last_tracks, 1.0,
                    show_boxes=self.var_boxes.get(),
                    show_speed=self.var_speed_lbl.get())

        # Proximity distance lines
        if self.var_show_dist.get() and self._last_prox_snap["pairs"]:
            render_proximity_overlay(
                d,
                self._last_prox_snap["pairs"],
                safe_m=self.var_safe_dist.get(),
                warn_m=self.var_warn_dist.get(),
                show_labels=self.var_show_labels.get(),
            )

        # Following-distance overlay: dashed arrow + gap label per track
        if self._last_following:
            track_pos = {t.id: (int(t.cx), int(t.cy)) for t in self._last_tracks}
            for fr in self._last_following:
                if fr.rear_vehicle_id is None or fr.distance_m is None:
                    continue
                pa = track_pos.get(fr.vehicle_id)
                pb = track_pos.get(fr.rear_vehicle_id)
                if pa is None or pb is None:
                    continue
                safe_m = self.var_safe_dist.get()
                warn_m = self.var_warn_dist.get()
                if fr.distance_m >= safe_m:
                    color = (0, 220, 80)
                elif fr.distance_m >= warn_m:
                    color = (0, 200, 240)
                else:
                    color = (0, 60, 255)
                cv2.arrowedLine(d, pb, pa, color, 1, cv2.LINE_AA, tipLength=0.2)
                mx, my = (pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2
                txt = f"{fr.distance_m:.1f}m"
                cv2.putText(d, txt, (mx + 2, my - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(d, txt, (mx + 2, my - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

        if self.var_speed_lbl.get() and self._cross_flash:
            now = time.perf_counter()
            draw_cross_flash(d, self._cross_flash, 1.0,
                             speed_limit=self.var_limit.get(), now=now)
            self._cross_flash = {k: v for k, v in self._cross_flash.items()
                                 if now - v["t0"] < 2.5}

        stab_txt = (f"Stab dx={self._stab_dx:.1f} dy={self._stab_dy:.1f}"
                    if self.stabilizer.enabled else "Stab: OFF")
        cv2.putText(d, stab_txt, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 160, 200), 1, cv2.LINE_AA)

        # Proximity zone HUD (always visible on video when zone is set)
        if self._prox_zone_mask is not None:
            sn  = self._last_prox_snap
            cnt = sn.get("count", 0)
            avg = sn.get("avg_m", 0.0)
            cng = sn.get("congestion", 0.0)
            if cnt >= 2:
                hud_color = (0, 220, 80) if avg >= self.var_safe_dist.get() else \
                            (0, 200, 240) if avg >= self.var_warn_dist.get() else \
                            (0, 60, 255)
            else:
                hud_color = (180, 0, 255)
            hud_lines = [
                f"ZONE  {cnt} veh",
                f"Avg {avg:.1f}m" if cnt >= 2 else "Need 2+ veh",
                f"Cong {cng:.0f}%"  if cnt >= 2 else "",
            ]
            for i, ln in enumerate(hud_lines):
                if not ln:
                    continue
                yy = 42 + i * 17
                cv2.putText(d, ln, (8, yy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 0, 0),   2, cv2.LINE_AA)
                cv2.putText(d, ln, (8, yy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, hud_color,   1, cv2.LINE_AA)

        if nw > 0 and nh > 0:
            disp = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_LINEAR)
            img  = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
            self.canvas.delete("frame")
            self.canvas.create_image(self.ox, self.oy, anchor="nw",
                                     image=img, tags="frame")
            self.canvas._img_ref = img

    # ── Camera model ──────────────────────────────────────────────────────────
    def _apply_camera(self):
        self.cam.speed_factor = max(0.01, self.var_speed_k.get())
        if self.var_cam_on.get():
            fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  if self.cap else 1280
            fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else  720
            self.cam.setup(self.var_cam_H.get(), self.var_cam_T.get(),
                           self.var_cam_P.get(), self.var_cam_F.get(), fw, fh)
            self.toast.show("Camera model enabled.", "success")
        else:
            self.cam.enabled = False

    def _load_cam_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.cam.load_json(path)
            self.var_cam_H.set(self.cam.H)
            self.var_cam_T.set(self.cam.tilt)
            self.var_cam_P.set(self.cam.pan)
            self.var_cam_F.set(self.cam.f)
            self.var_cam_on.set(True)
            self.toast.show("Camera JSON loaded.", "success")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    def _open_calibrator(self):
        CalibratorWindow(self.root, callback=self._on_calib_result,
                         frame_ref=self._last_frame)

    def _on_calib_result(self, params: dict):
        self.cam.H    = params.get("camera_height_m",  self.cam.H)
        self.cam.tilt = params.get("camera_pitch_deg", self.cam.tilt)
        self.cam.f    = params.get("focal_length_px",  self.cam.f)
        fw = params.get("frame_width",  1280)
        fh = params.get("frame_height",  720)
        self.cam.cx, self.cam.cy = fw / 2.0, fh / 2.0
        if "px_per_m" in params:
            self.cam.px_per_m = params["px_per_m"]
        self.cam.enabled = True
        self.var_cam_on.set(True)
        self.var_cam_H.set(self.cam.H)
        self.var_cam_T.set(self.cam.tilt)
        self.var_cam_F.set(self.cam.f)
        self.toast.show("Camera calibrated.", "success")

    def _load_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("Model", "*.pt *.onnx *.engine"), ("All", "*.*")])
        if not path:
            return
        self._stop_play()
        det, name = load_yolo(path)
        if det:
            self.yolo_det   = det
            self.model_name = name
            self.topbar.set_model(name, ok=True)
            det_panel = self._panels["detection"]
            if hasattr(det_panel, "lbl_model"):
                det_panel.lbl_model.config(text=name, fg=Theme.CYAN)
            self.toast.show(f"Model loaded: {name}", "success")
        else:
            self.topbar.set_model("Load failed", ok=False)
            self.toast.show("Failed to load model.", "error")

    # ── Screenshot ────────────────────────────────────────────────────────────
    def _take_screenshot(self):
        if self._last_frame is None:
            self.toast.show("No frame to capture.", "warning"); return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.ssdir, f"screenshot_{ts}.png")
        cv2.imwrite(path, self._last_frame)
        self.toast.show(f"Screenshot saved: {path}", "success")

    # ── Unified Excel export ───────────────────────────────────────────────────
    def _export_unified(self, auto: bool = False):
        if not TrafficReportExporter.available():
            messagebox.showerror("Error", "openpyxl not installed."); return
        if not self._exporter._rows:
            if not auto:
                self.toast.show("No crossing events to export.", "warning")
            return
        if auto:
            path = make_filename()
        else:
            path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel", "*.xlsx")],
                initialfile=make_filename())
            if not path:
                return
        try:
            self._exporter.export(path, self.var_limit.get())
            if not auto:
                self.toast.show(f"Exported: {os.path.basename(path)}", "success")
        except Exception as ex:
            if not auto:
                messagebox.showerror("Export Error", str(ex))

    def mainloop(self):
        self.root.mainloop()
