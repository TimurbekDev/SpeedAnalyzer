"""
LaneDistanceApp — single-screen, MULTI-lane vehicle-distance tool.

Flow:  Open video → add one or more lane ROIs → set real length/width (VERIFY
       GRID) → Start → ByteTrack tracks every vehicle in the frame ONCE → each
       ROI keeps the vehicles inside it → per-vehicle speed + gap to the vehicle
       ahead → one row per vehicle per lane → each lane auto-saves its own Excel.

Multi-ROI design
────────────────
Detection/tracking happens a single time per frame on the whole frame, so all
lanes share one stable set of ByteTrack ids (no ID churn / duplicate rows).  The
worker only *filters* those shared detections into each lane.  Lanes share the
real length/width calibration but each keeps its own homography (its corners
differ).

Threading mirrors a standard capture/worker/UI split:
  • capture thread  : reads frames, lossless push → in_q (every frame tracked)
  • Worker thread   : ByteTrack detect+track + per-lane metrics → out_q
  • Tk UI thread    : _poll drains out_q, renders, records rows
ROI editing is only allowed while stopped, so geometry never changes under the
worker — no locks required.
"""

import os
import queue
import threading
from collections import deque

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from config.defaults import (CAP_Q_SIZE, RES_Q_SIZE, ROI_HANDLE_PX, ROI_MIN_POINTS,
                             ROI_COLOR, LANE_LENGTH_M_DEFAULT, LANE_WIDTH_M_DEFAULT,
                             EXPORT_DIR, GRID_STEP_M, SAFE_GAP_M, WARN_GAP_M)
from vision.detection.loader        import load_yolo
from vision.rendering.frame_renderer import draw_tracks
from vision.pipeline.worker         import Worker
from vision.roi                     import RoiManager
from vision.export                  import DistanceExporter, make_filename
from ui.theme.dark                  import Theme
from ui.widgets.components          import (GlowButton, VideoPanel, ToastManager,
                                            section_label, divider, param_row)


class LaneDistanceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._init_state()
        self._load_backend()
        self._build_ui()

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
        self._last_tracks: list = []             # all vehicles tracked in stream
        self._inside_ids:  list = []             # union of in-lane ids (render)
        self._speeds:      dict = {}             # merged {track_id: km/h} (render)
        self._nexts:       dict = {}             # merged {id: (next_id, gap_m)}
        self._lanes_live:  dict = {}             # name -> per-lane live result
        self._ftimes = deque(maxlen=30)

        self.sx = self.sy = 1.0
        self.ox = self.oy = 0
        self._slider_busy = False

        # ROIs
        self.rois         = RoiManager()
        self.exporters:   dict = {}              # lane name -> DistanceExporter
        self._roi_mode:   str | None = None      # None | "draw" | "edit"
        self._roi_draft:  list       = []
        self._roi_tmp:    list | None = None
        self._drag_idx:   int | None = None

        self.var_lane_len = tk.DoubleVar(value=LANE_LENGTH_M_DEFAULT)
        self.var_lane_wid = tk.DoubleVar(value=LANE_WIDTH_M_DEFAULT)
        self.var_safe_gap = tk.DoubleVar(value=SAFE_GAP_M)
        self.var_show_grid = tk.BooleanVar(value=False)

    def _load_backend(self):
        self.yolo_det, self.model_name = load_yolo()

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("Lane Distance — Vehicle Spacing")
        self.root.geometry("1280x820")
        self.root.minsize(1040, 660)
        self.root.configure(bg=Theme.BG)

        # ── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=Theme.TOPBAR, height=50)
        header.pack(fill="x"); header.pack_propagate(False)
        tk.Label(header, text="LANE", bg=Theme.TOPBAR, fg=Theme.CYAN,
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=(16, 0))
        tk.Label(header, text="DISTANCE", bg=Theme.TOPBAR, fg=Theme.TEXT_2,
                 font=("Segoe UI", 14)).pack(side="left", padx=(4, 0))
        GlowButton(header, "⊕ OPEN VIDEO", self._open_video,
                   accent=Theme.CYAN, width=130, height=34).pack(
            side="left", padx=16, pady=8)
        self.lbl_status = tk.Label(header, text="Open a video to begin",
                                   bg=Theme.TOPBAR, fg=Theme.TEXT_3,
                                   font=Theme.F_MONO_S)
        self.lbl_status.pack(side="right", padx=16)
        self.lbl_tracked = tk.Label(header, text="● ByteTrack",
                                    bg=Theme.TOPBAR, fg=Theme.SUCCESS,
                                    font=Theme.F_MONO_S)
        self.lbl_tracked.pack(side="right", padx=8)
        tk.Frame(self.root, bg=Theme.DIVIDER, height=1).pack(fill="x")

        self.toast = ToastManager(self.root)

        body = tk.Frame(self.root, bg=Theme.BG)
        body.pack(fill="both", expand=True)

        # ── Video (left) ────────────────────────────────────────────────────
        self.video = VideoPanel(body)
        self.video.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.canvas = self.video.canvas
        self.canvas.bind("<ButtonPress-1>",   self._canvas_click)
        self.canvas.bind("<Motion>",          self._canvas_motion)
        self.canvas.bind("<B1-Motion>",       self._canvas_motion)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<ButtonPress-3>",   self._canvas_rclick)
        self.canvas.bind("<Double-Button-1>", self._canvas_dblclick)
        self.canvas.bind("<Configure>",
                         lambda _: self._render() if self._last_frame is not None else None)

        # ── Controls (right) ────────────────────────────────────────────────
        right = tk.Frame(body, bg=Theme.BG, width=300)
        right.pack(side="right", fill="y", pady=8, padx=(0, 8))
        right.pack_propagate(False)
        self._build_controls(right)

        # ── Playback bar (bottom) ───────────────────────────────────────────
        tk.Frame(self.root, bg=Theme.DIVIDER, height=1).pack(fill="x")
        bar = tk.Frame(self.root, bg=Theme.SURFACE, height=52)
        bar.pack(fill="x"); bar.pack_propagate(False)
        self.btn_play = GlowButton(bar, "▶  START", self._toggle_play,
                                   accent=Theme.SUCCESS, width=110, height=34)
        self.btn_play.pack(side="left", padx=10, pady=9)
        self.lbl_time = tk.Label(bar, text="0:00 / 0:00", bg=Theme.SURFACE,
                                 fg=Theme.TEXT_3, font=Theme.F_MONO_S)
        self.lbl_time.pack(side="left", padx=6)
        style = ttk.Style(); style.theme_use("clam")
        style.configure("Dark.Horizontal.TScale",
                        troughcolor=Theme.SURFACE2, background=Theme.SURFACE,
                        darkcolor=Theme.CYAN, lightcolor=Theme.CYAN,
                        sliderlength=14, sliderthickness=14)
        self.slider = ttk.Scale(bar, from_=0, to=100, orient="horizontal",
                                command=self._seek, style="Dark.Horizontal.TScale")
        self.slider.pack(side="left", fill="x", expand=True, padx=12)

        # ── Keyboard shortcuts ──────────────────────────────────────────────
        self.root.bind("<space>",  lambda _e: self._toggle_play())
        self.root.bind("<Escape>", lambda _e: self._cancel_roi_mode())
        self.root.bind("<Delete>", lambda _e: self._roi_delete())
        self.root.bind("<d>",      lambda _e: self._roi_add())
        self.root.bind("<g>",      lambda _e: self._toggle_grid())

    def _build_controls(self, parent):
        # Calibration
        cal = tk.Frame(parent, bg=Theme.SURFACE, padx=12, pady=10)
        cal.pack(fill="x", pady=(0, 8))
        section_label(cal, "LANE CALIBRATION (ALL ROIs)", bg=Theme.SURFACE)
        divider(cal, bg=Theme.SURFACE)
        fields = tk.Frame(cal, bg=Theme.SURFACE)
        fields.pack(fill="x", pady=8)
        param_row(fields, "Length (m)", self.var_lane_len, bg=Theme.SURFACE)
        param_row(fields, "Width (m)",  self.var_lane_wid, bg=Theme.SURFACE)
        fields2 = tk.Frame(cal, bg=Theme.SURFACE)
        fields2.pack(fill="x", pady=(0, 6))
        param_row(fields2, "Safe gap (m)", self.var_safe_gap, bg=Theme.SURFACE)
        self.btn_grid = GlowButton(cal, "▦ VERIFY GRID", self._toggle_grid,
                                   accent=Theme.CYAN, height=30)
        self.btn_grid.pack(fill="x", pady=(2, 4))
        tk.Label(cal, text="Length & width apply to every ROI. Trace each lane's\n"
                           "4 corners, then VERIFY GRID — the 5 m grid should\n"
                           "line up with road markings on the selected ROI.",
                 bg=Theme.SURFACE, fg=Theme.TEXT_3, font=Theme.F_CAP,
                 justify="left").pack(anchor="w")

        # ROI tools — multi-ROI list + add/edit/delete
        roi = tk.Frame(parent, bg=Theme.SURFACE, padx=12, pady=10)
        roi.pack(fill="x", pady=(0, 8))
        section_label(roi, "ROAD LANE ROIs", bg=Theme.SURFACE)
        divider(roi, bg=Theme.SURFACE)
        lb_wrap = tk.Frame(roi, bg=Theme.SURFACE2)
        lb_wrap.pack(fill="x", pady=(6, 4))
        self.roi_list = tk.Listbox(
            lb_wrap, height=4, bg=Theme.SURFACE2, fg=Theme.TEXT_2,
            font=Theme.F_MONO_S, bd=0, highlightthickness=0,
            selectbackground=Theme.SURFACE3, selectforeground=Theme.CYAN,
            activestyle="none", exportselection=False)
        self.roi_list.pack(side="left", fill="x", expand=True)
        self.roi_list.bind("<<ListboxSelect>>", self._on_roi_select)
        self.btn_add = GlowButton(roi, "✚ ADD ROI", self._roi_add,
                                  accent=Theme.PURPLE, height=32)
        self.btn_add.pack(fill="x", pady=2)
        self.btn_edit = GlowButton(roi, "✎ EDIT POINTS", self._roi_toggle_edit,
                                   accent=Theme.WARNING, height=32)
        self.btn_edit.pack(fill="x", pady=2)
        GlowButton(roi, "✕ DELETE ROI", self._roi_delete,
                   accent=Theme.DANGER, height=32).pack(fill="x", pady=2)

        # Lane traffic: count + avg speed (for the selected ROI)
        traf = tk.Frame(parent, bg=Theme.SURFACE, padx=12, pady=10)
        traf.pack(fill="x", pady=(0, 8))
        section_label(traf, "LANE TRAFFIC", bg=Theme.SURFACE)
        divider(traf, bg=Theme.SURFACE)
        self.lbl_traffic_lane = tk.Label(traf, text="— no ROI selected —",
                                         bg=Theme.SURFACE, fg=Theme.TEXT_3,
                                         font=Theme.F_MONO_S)
        self.lbl_traffic_lane.pack(anchor="w", pady=(4, 0))
        grid = tk.Frame(traf, bg=Theme.SURFACE)
        grid.pack(fill="x", pady=(6, 0))
        c1 = tk.Frame(grid, bg=Theme.SURFACE)
        c1.pack(side="left", expand=True, fill="x")
        tk.Label(c1, text="VEHICLES", bg=Theme.SURFACE, fg=Theme.TEXT_3,
                 font=Theme.F_CAP).pack(anchor="w")
        self.lbl_count = tk.Label(c1, text="0", bg=Theme.SURFACE, fg=Theme.CYAN,
                                  font=("Segoe UI", 22, "bold"))
        self.lbl_count.pack(anchor="w")
        c2 = tk.Frame(grid, bg=Theme.SURFACE)
        c2.pack(side="left", expand=True, fill="x")
        tk.Label(c2, text="AVG SPEED", bg=Theme.SURFACE, fg=Theme.TEXT_3,
                 font=Theme.F_CAP).pack(anchor="w")
        self.lbl_avg_speed = tk.Label(c2, text="— km/h", bg=Theme.SURFACE,
                                      fg=Theme.SUCCESS, font=("Segoe UI", 22, "bold"))
        self.lbl_avg_speed.pack(anchor="w")

        # Latest record
        res = tk.Frame(parent, bg=Theme.SURFACE, padx=12, pady=10)
        res.pack(fill="x", pady=(0, 8))
        section_label(res, "LATEST RECORD (GAP TO NEXT)", bg=Theme.SURFACE)
        divider(res, bg=Theme.SURFACE)
        self.lbl_dist = tk.Label(res, text="—", bg=Theme.SURFACE, fg=Theme.CYAN,
                                 font=("Segoe UI", 28, "bold"))
        self.lbl_dist.pack(anchor="w", pady=(6, 0))
        self.lbl_pair = tk.Label(res, text="no record yet", bg=Theme.SURFACE,
                                 fg=Theme.TEXT_3, font=Theme.F_MONO_S)
        self.lbl_pair.pack(anchor="w")

        # History
        hist = tk.Frame(parent, bg=Theme.SURFACE, padx=12, pady=10)
        hist.pack(fill="both", expand=True)
        section_label(hist, "MEASUREMENT HISTORY", bg=Theme.SURFACE)
        divider(hist, bg=Theme.SURFACE)
        inner = tk.Frame(hist, bg=Theme.SURFACE2)
        inner.pack(fill="both", expand=True, pady=6)
        self.hist = tk.Text(inner, bg=Theme.SURFACE2, fg=Theme.TEXT_2,
                            font=Theme.F_MONO_S, bd=0, state="disabled",
                            wrap="none", height=6)
        sb = tk.Scrollbar(inner, command=self.hist.yview,
                          bg=Theme.SURFACE2, troughcolor=Theme.SURFACE, width=8)
        self.hist.config(yscrollcommand=sb.set)
        self.hist.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        GlowButton(parent, "⊞  EXPORT EXCEL", self._export_excel,
                   accent=Theme.SUCCESS, height=36).pack(fill="x", pady=(8, 0))

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
        self.slider.configure(to=max(1, total - 1))
        # Reset session
        self.yolo_det.reset()                     # forget ByteTrack IDs
        self.exporters = {}
        self._clear_hist()
        self._last_tracks = []; self._inside_ids = []
        self._speeds = {}; self._nexts = {}; self._lanes_live = {}
        self.rois.clear(); self._roi_draft = []; self._roi_mode = None
        self._roi_tmp = None; self._drag_idx = None
        self.lbl_dist.config(text="—"); self.lbl_pair.config(text="no record yet")
        self.lbl_count.config(text="0"); self.lbl_avg_speed.config(text="— km/h")
        self.lbl_traffic_lane.config(text="— no ROI selected —")
        self._refresh_roi_list()
        ok, f = self.cap.read()
        if ok:
            self._last_frame = f
            self.video.set_has_frame()
            self._render()
        self._set_time(0, total / self.fps)
        self._status(f"{self.video_name} — add a lane ROI to begin")
        self.toast.show("Add a road-lane ROI (you can add several), then Start.", "info")
        self._roi_add()

    # ── Playback ──────────────────────────────────────────────────────────────
    def _toggle_play(self):
        if not self.cap:
            self.toast.show("Open a video file first.", "warning"); return
        if self.playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        # Mandatory ROI gate — need at least one finished lane polygon.
        lanes = self.rois.valid_lanes()
        if not lanes:
            self.toast.show("Add at least one lane ROI before processing.", "warning")
            self._roi_add()
            return
        # Calibrate every lane from its corners + the shared lane dimensions.
        ready = [l for l in lanes
                 if l.calib.calibrate(l.roi.points,
                                      self.var_lane_wid.get(),
                                      self.var_lane_len.get())]
        if not ready:
            self.toast.show("Calibration failed — need valid lane polygons.", "error")
            return

        self.playing = True
        self._roi_mode = None
        self.btn_edit.set_active(False)
        self.btn_play.update_text("⏸  STOP", accent=Theme.WARNING)
        self._status(f"PROCESSING… {len(ready)} ROI(s)")
        for q in (self.in_q, self.out_q):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break
        self.yolo_det.reset()                     # restart ByteTrack IDs from 1
        self._speeds = {}; self._nexts = {}; self._inside_ids = []; self._lanes_live = {}
        self.lbl_count.config(text="0"); self.lbl_avg_speed.config(text="— km/h")

        # Fresh auto-save file PER lane for this run.
        self.exporters = {}
        for lane in ready:
            ex = DistanceExporter()
            ex.set_auto_path(os.path.join(EXPORT_DIR, make_filename(lane.name)))
            self.exporters[lane.name] = ex

        self.worker = Worker(self.yolo_det, self.in_q, self.out_q, ready, self.fps)
        self.worker.start()
        self.cap_thread = threading.Thread(target=self._cap_loop, daemon=True)
        self.cap_thread.start()
        self._schedule_poll()

    def _stop_play(self):
        if not self.playing:
            return
        self.playing = False
        self.btn_play.update_text("▶  START", accent=Theme.SUCCESS)
        self._status("IDLE")
        try:
            self.in_q.put_nowait(None)
        except queue.Full:
            pass
        if self.worker:
            self.worker.stop()
            self.worker = None

    def _cap_loop(self):
        # Lossless: block until the worker has room so EVERY frame is tracked.
        # ByteTrack's motion model needs consecutive frames for stable IDs, so we
        # never drop frames — playback runs as fast as the worker can keep up.
        while self.playing:
            ok, frame = self.cap.read()
            if not ok:
                try: self.in_q.put(None, timeout=1.0)
                except queue.Full: pass
                break
            fid = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            while self.playing:
                try:
                    self.in_q.put((fid, frame), timeout=0.2)
                    break
                except queue.Full:
                    continue

    def _schedule_poll(self):
        delay = max(8, int(1000.0 / self.fps))
        self.root.after(delay, self._poll)

    def _poll(self):
        if not self.playing:
            return
        latest_frame  = None
        latest_tracks = None
        latest_lanes  = None
        records: list = []
        try:
            while True:
                item = self.out_q.get_nowait()
                if item is None:
                    self._stop_play()
                    self.toast.show("Video finished.", "info")
                    return
                frame_d, tracks_d, lanes_d = item
                latest_frame  = frame_d
                latest_tracks = tracks_d
                latest_lanes  = lanes_d
                for lane in lanes_d:
                    records.extend(lane["records"])
        except queue.Empty:
            pass

        if latest_frame is not None:
            self._last_frame  = latest_frame
            self._last_tracks = latest_tracks
            self._lanes_live  = {l["name"]: l for l in (latest_lanes or [])}
            # Merge per-lane live data for rendering (union of inside ids etc.).
            merged_inside: list = []
            merged_speeds: dict = {}
            merged_nexts:  dict = {}
            for l in (latest_lanes or []):
                merged_inside.extend(l["inside_ids"])
                merged_speeds.update(l["speeds"])
                merged_nexts.update(l["nexts"])
            self._inside_ids = merged_inside
            self._speeds     = merged_speeds
            self._nexts      = merged_nexts
            self._refresh_traffic()
            for r in records:
                self._record_vehicle(r)
            self.video.set_has_frame()
            self._render()
            if self.cap:
                pos   = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._slider_busy = True
                try:
                    self.slider.set(pos)
                    self._set_time(pos / self.fps, total / self.fps)
                finally:
                    self._slider_busy = False
        self._schedule_poll()

    def _refresh_traffic(self):
        lane = self.rois.current()
        live = self._lanes_live.get(lane.name) if lane else None
        if live is not None:
            self.lbl_traffic_lane.config(text=f"{lane.name}")
            self.lbl_count.config(text=str(live["count"]))
            moving = [s for s in live["speeds"].values() if s > 1]
            self.lbl_avg_speed.config(
                text=f"{sum(moving) / len(moving):.0f} km/h" if moving else "— km/h")
        else:
            total = sum(l["count"] for l in self._lanes_live.values())
            self.lbl_traffic_lane.config(
                text=lane.name if lane else "— no ROI selected —")
            self.lbl_count.config(text=str(total))
            self.lbl_avg_speed.config(text="— km/h")
        self.lbl_tracked.config(
            text=f"● tracked {len(self._last_tracks)} / ROIs {len(self.rois)}")

    def _record_vehicle(self, r: dict):
        ex = self.exporters.get(r["roi"])
        if ex is not None:
            ex.add(r["id"], r["speed"], r["dist_m"], r["next_id"], r["vtime"])
        gap_txt = f"{r['dist_m']:.1f} m" if r["dist_m"] is not None else "—"
        nxt_txt = f"→ #{r['next_id']}" if r["next_id"] is not None else "(lead)"
        self.lbl_dist.config(text=gap_txt)
        self.lbl_pair.config(
            text=f"{r['roi']}  #{r['id']}  {r['speed']:.0f} km/h   {nxt_txt}")
        self._append_hist(f"[{r['vtime']}] {r['roi']:>6} #{r['id']:>3} "
                          f"{r['speed']:>3.0f} km/h {nxt_txt:>7} {gap_txt:>7}\n")

    # ── ROI list / selection ────────────────────────────────────────────────────
    def _refresh_roi_list(self):
        self.roi_list.delete(0, "end")
        for lane in self.rois.lanes:
            n = len(lane.roi.points)
            tag = f"{n} pts" if lane.valid else "drawing…"
            self.roi_list.insert("end", f"{lane.name}  ({tag})")
        sel = self.rois.selected
        if 0 <= sel < len(self.rois.lanes):
            self.roi_list.selection_clear(0, "end")
            self.roi_list.selection_set(sel)
            self.roi_list.see(sel)

    def _on_roi_select(self, _e):
        sel = self.roi_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx == self.rois.selected:
            return
        # Don't switch away from a polygon still being drawn — keep the draft.
        if self._roi_mode == "draw":
            self._refresh_roi_list()
            return
        if self._roi_mode == "edit":
            self._roi_mode = None
            self.btn_edit.set_active(False)
        self.rois.select(idx)
        # Show the newly-selected lane's grid preview / stats.
        self._refresh_traffic()
        self._render()

    # ── ROI drawing / editing ──────────────────────────────────────────────────
    def _roi_add(self):
        if self._last_frame is None:
            self.toast.show("Open a video file first.", "warning"); return
        if self.playing:
            self.toast.show("Stop processing before editing ROIs.", "warning"); return
        # If a fresh, still-empty draft lane is selected, reuse it instead of
        # stacking empty lanes.
        cur = self.rois.current()
        if not (self._roi_mode == "draw" and cur is not None and not cur.valid):
            self.rois.add()
        self._roi_mode  = "draw"
        self._roi_draft = []
        self._roi_tmp   = None
        self._drag_idx  = None
        self.btn_edit.set_active(False)
        self._refresh_roi_list()
        self.toast.show("Click the lane corners. Right-click / double-click to close.",
                        "info")
        self._render()

    def _roi_toggle_edit(self):
        if self.playing:
            self.toast.show("Stop processing before editing ROIs.", "warning"); return
        lane = self.rois.current()
        if lane is None or not lane.valid:
            self.toast.show("Select a finished ROI first.", "warning"); return
        if self._roi_mode == "edit":
            self._roi_mode = None
            self.btn_edit.set_active(False)
        else:
            self._roi_mode = "edit"
            self.btn_edit.set_active(True)
            self.toast.show("Drag a vertex to move; right-click a vertex to delete.",
                            "info")
        self._render()

    def _finalize_draft(self):
        lane = self.rois.current()
        if lane is None:
            return
        if (len(self._roi_draft) >= 2
                and abs(self._roi_draft[-1][0] - self._roi_draft[-2][0]) < 4
                and abs(self._roi_draft[-1][1] - self._roi_draft[-2][1]) < 4):
            self._roi_draft.pop()
        if len(self._roi_draft) < ROI_MIN_POINTS:
            return
        lane.roi.set_points(self._roi_draft)
        self._roi_draft = []
        self._roi_tmp   = None
        self._roi_mode  = None
        self._refresh_roi_list()
        self._render()
        self.toast.show(f"{lane.name} set. Add another ROI or press Start.", "success")
        self._status(f"{self.video_name} — {len(self.rois.valid_lanes())} ROI(s) ready")

    def _roi_delete(self):
        if self.playing:
            self.toast.show("Stop processing before editing ROIs.", "warning"); return
        lane = self.rois.current()
        if lane is None:
            self.toast.show("No ROI selected to delete.", "warning"); return
        name = lane.name
        self.rois.remove_current()
        self._roi_mode  = None
        self._roi_draft = []
        self._roi_tmp   = None
        self.btn_edit.set_active(False)
        self._refresh_roi_list()
        self._render()
        self.toast.show(f"{name} deleted.", "warning")

    def _cancel_roi_mode(self):
        if self._roi_mode is None:
            return
        # Abandoning a half-drawn polygon — drop the empty lane it created.
        if self._roi_mode == "draw":
            cur = self.rois.current()
            if cur is not None and not cur.valid:
                self.rois.remove_current()
        self._roi_mode = None
        self._roi_draft = []
        self._roi_tmp = None
        self.btn_edit.set_active(False)
        self._refresh_roi_list()
        if self._last_frame is not None:
            self._render()

    def _toggle_grid(self):
        new = not self.var_show_grid.get()
        self.var_show_grid.set(new)
        self.btn_grid.set_active(new)
        lane = self.rois.current()
        if new:
            if lane is not None and lane.valid:
                lane.calib.calibrate(lane.roi.points,
                                     self.var_lane_wid.get(),
                                     self.var_lane_len.get())
            else:
                self.toast.show("Select or draw a lane ROI first.", "warning")
        if self._last_frame is not None:
            self._render()

    def _grab_radius(self) -> float:
        return max(ROI_HANDLE_PX, 10.0 * self.sx)

    def _canvas_click(self, e):
        if self._last_frame is None:
            return
        vx, vy = self._c2v(e.x, e.y)
        if self._roi_mode == "draw":
            self._roi_draft.append([vx, vy]); self._render()
        elif self._roi_mode == "edit":
            lane = self.rois.current()
            if lane is not None:
                self._drag_idx = lane.roi.hit_point(vx, vy, self._grab_radius())

    def _canvas_motion(self, e):
        if self._last_frame is None:
            return
        vx, vy = self._c2v(e.x, e.y)
        if self._roi_mode == "draw":
            self._roi_tmp = [vx, vy]; self._render()
        elif self._roi_mode == "edit" and self._drag_idx is not None:
            lane = self.rois.current()
            if lane is not None:
                lane.roi.move_point(self._drag_idx, vx, vy); self._render()

    def _canvas_release(self, _e):
        self._drag_idx = None

    def _canvas_rclick(self, e):
        if self._roi_mode == "draw":
            if len(self._roi_draft) >= ROI_MIN_POINTS:
                self._finalize_draft()
            else:
                self._roi_draft = []; self._roi_tmp = None; self._render()
        elif self._roi_mode == "edit":
            lane = self.rois.current()
            if lane is None:
                return
            vx, vy = self._c2v(e.x, e.y)
            idx = lane.roi.hit_point(vx, vy, self._grab_radius())
            if idx is not None:
                if lane.roi.delete_point(idx):
                    self._render(); self.toast.show("Vertex deleted.", "info")
                else:
                    self.toast.show("A polygon needs at least 3 points.", "warning")

    def _canvas_dblclick(self, _e):
        if self._roi_mode == "draw" and len(self._roi_draft) >= ROI_MIN_POINTS:
            self._finalize_draft()

    def _c2v(self, ex: int, ey: int):
        return (max(0, min(int((ex - self.ox) * self.sx), 99999)),
                max(0, min(int((ey - self.oy) * self.sy), 99999)))

    # ── Seek ───────────────────────────────────────────────────────────────────
    def _seek(self, v):
        if not self.cap or self._slider_busy or self.playing:
            return
        pos = int(float(v))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, f = self.cap.read()
        if ok:
            self._last_frame = f
            self.video.set_has_frame()
            self._render()
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._set_time(pos / self.fps, total / self.fps)

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

        self._draw_rois(d)
        if self.var_show_grid.get():
            lane = self.rois.current()
            if not self.playing and lane is not None and lane.valid:   # live preview
                lane.calib.calibrate(lane.roi.points, self.var_lane_wid.get(),
                                     self.var_lane_len.get())
            if lane is not None and lane.calib.ready:
                self._draw_grid(d, lane.calib)
        draw_tracks(d, self._last_tracks, self._inside_ids,
                    speeds=self._speeds, show_outside=True)
        self._draw_following(d)

        if not self.playing and not self.rois.valid_lanes():
            self._banner(d, "ADD A ROAD-LANE ROI TO BEGIN")

        if nw > 0 and nh > 0:
            disp = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_LINEAR)
            img  = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
            self.canvas.delete("frame")
            self.canvas.create_image(self.ox, self.oy, anchor="nw",
                                     image=img, tags="frame")
            self.canvas._img_ref = img

    def _draw_rois(self, d):
        editing = self._roi_mode in ("draw", "edit")
        cur = self.rois.current()
        for lane in self.rois.lanes:
            if not lane.valid:
                continue
            is_cur = lane is cur
            color  = lane.color
            arr    = np.array(lane.roi.points, np.int32)
            ov     = d.copy()
            cv2.fillPoly(ov, [arr], color)
            cv2.addWeighted(ov, 0.16 if is_cur else 0.10, d,
                            0.84 if is_cur else 0.90, 0, d)
            cv2.polylines(d, [arr], True, color, 3 if is_cur else 2, cv2.LINE_AA)
            ax, ay = min(lane.roi.points, key=lambda p: p[1])
            cv2.putText(d, lane.name, (ax + 6, max(16, ay - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(d, lane.name, (ax + 6, max(16, ay - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if editing and is_cur:
                for (px, py) in lane.roi.points:
                    cv2.circle(d, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(d, (px, py), 5, color, 1, cv2.LINE_AA)
        # Draft polygon being drawn for the current lane.
        if self._roi_mode == "draw" and self._roi_draft:
            color = cur.color if cur is not None else ROI_COLOR
            for i, pt in enumerate(self._roi_draft):
                cv2.circle(d, tuple(pt), 5, color, -1, cv2.LINE_AA)
                if i > 0:
                    cv2.line(d, tuple(self._roi_draft[i-1]), tuple(pt),
                             color, 2, cv2.LINE_AA)
            if self._roi_tmp:
                cv2.line(d, tuple(self._roi_draft[-1]), tuple(self._roi_tmp),
                         color, 1, cv2.LINE_AA)

    def _gap_color(self, gap_m: float) -> tuple:
        """Green/amber/red by following gap vs the safe threshold (BGR)."""
        safe = self.var_safe_gap.get()
        warn = max(WARN_GAP_M, safe * 0.5)
        if gap_m >= safe:
            return (0, 220, 80)
        if gap_m >= warn:
            return (0, 200, 240)
        return (0, 60, 255)

    def _draw_following(self, d):
        """Live line from each in-lane vehicle to the vehicle ahead + gap label,
        colour-coded by the safe-following-distance threshold."""
        if not self._nexts or not self._last_tracks:
            return
        gp = {}
        for t in self._last_tracks:
            x, y, w, h = t.bbox
            gp[t.id] = (int(x + w / 2), int(y + h))
        for tid, (nid, gap) in self._nexts.items():
            if nid is None or gap is None or tid not in gp or nid not in gp:
                continue
            pa, pb = gp[tid], gp[nid]
            color = self._gap_color(gap)
            cv2.line(d, pa, pb, color, 2, cv2.LINE_AA)
            cv2.circle(d, pa, 4, color, -1, cv2.LINE_AA)
            mx, my = (pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2
            txt = f"{gap:.1f} m"
            cv2.putText(d, txt, (mx + 4, my - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(d, txt, (mx + 4, my - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)

    def _draw_grid(self, d, calib):
        """Metric grid over the selected lane to verify calibration vs markings."""
        for (p1, p2, major) in calib.grid_segments(GRID_STEP_M):
            cv2.line(d, p1, p2, (0, 210, 255) if major else (90, 160, 90),
                     2 if major else 1, cv2.LINE_AA)

    @staticmethod
    def _banner(d, text):
        fh, fw = d.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = (fw - tw) // 2
        y = int(fh * 0.12)
        cv2.rectangle(d, (x - 16, y - th - 12), (x + tw + 16, y + 12), (0, 0, 0), -1)
        cv2.rectangle(d, (x - 16, y - th - 12), (x + tw + 16, y + 12),
                      (0, 200, 255), 1)
        cv2.putText(d, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 200, 255), 2, cv2.LINE_AA)

    # ── Excel export ────────────────────────────────────────────────────────────
    def _export_excel(self):
        if not DistanceExporter.available():
            messagebox.showerror("Error", "openpyxl not installed."); return
        nonempty = {name: ex for name, ex in self.exporters.items() if ex.rows}
        if not nonempty:
            self.toast.show("No measurements to export yet.", "warning"); return
        folder = filedialog.askdirectory(title="Choose a folder for the ROI files")
        if not folder:
            return
        saved = 0
        try:
            for name, ex in nonempty.items():
                ex.save(os.path.join(folder, make_filename(name)))
                saved += 1
            self.toast.show(f"Exported {saved} file(s) to {os.path.basename(folder)}.",
                            "success")
        except Exception as ex:
            messagebox.showerror("Export Error", str(ex))

    # ── Small helpers ──────────────────────────────────────────────────────────
    def _status(self, text):
        self.lbl_status.config(text=text)

    def _set_time(self, cur_s, total_s):
        def fmt(s):
            m, sec = divmod(int(s), 60)
            return f"{m}:{sec:02d}"
        self.lbl_time.config(text=f"{fmt(cur_s)} / {fmt(total_s)}")

    def _append_hist(self, line):
        self.hist.config(state="normal")
        self.hist.insert("end", line)
        self.hist.see("end")
        self.hist.config(state="disabled")

    def _clear_hist(self):
        self.hist.config(state="normal")
        self.hist.delete("1.0", "end")
        self.hist.config(state="disabled")

    def mainloop(self):
        self.root.mainloop()
