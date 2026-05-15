"""
Bird's-eye perspective calibration dialog.

User clicks 4 corners (TL → TR → BR → BL) on a still frame, then enters
the real-world width and height of that rectangle in metres.
The callback receives (src_pts_4x2, width_m, height_m).
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from ui.theme.dark import Theme
from ui.widgets.components import GlowButton


_CORNER_COLORS = ["#00e5ff", "#00ff88", "#ffb800", "#ff3366"]
_CORNER_LABELS = ["TL", "TR", "BR", "BL"]


class BirdseyeDialog(tk.Toplevel):
    def __init__(self, parent, frame_bgr: np.ndarray, callback):
        super().__init__(parent)
        self.title("Bird's-Eye Calibration")
        self.resizable(False, False)
        self.configure(bg=Theme.BG)
        self.transient(parent)
        self.grab_set()

        self._frame_bgr = frame_bgr
        self._callback  = callback
        self._pts:       list = []     # up to 4 [img_x, img_y]
        self._scale      = 1.0         # canvas → image coordinate multiplier

        self._build()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        tk.Label(self, text="Click 4 corners:  TL → TR → BR → BL",
                 bg=Theme.BG, fg=Theme.TEXT_1, font=Theme.F_BODY).pack(pady=(10, 4))

        # Canvas
        h, w = self._frame_bgr.shape[:2]
        max_w, max_h = 640, 400
        scale_canvas = min(max_w / w, max_h / h)
        cw, ch = int(w * scale_canvas), int(h * scale_canvas)
        self._scale = 1.0 / scale_canvas   # canvas px → image px

        frame_rgb = cv2.cvtColor(
            cv2.resize(self._frame_bgr, (cw, ch)), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

        self._canvas = tk.Canvas(self, width=cw, height=ch,
                                 bg="#000", highlightthickness=1,
                                 highlightbackground=Theme.BORDER,
                                 cursor="crosshair")
        self._canvas.pack(padx=12)
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._canvas.bind("<Button-1>", self._on_click)

        # Known dimensions
        dim_row = tk.Frame(self, bg=Theme.BG)
        dim_row.pack(pady=8)

        for label, attr, default in [
            ("Width (m)",  "_var_w", 20.0),
            ("Height (m)", "_var_h", 30.0),
        ]:
            var = tk.DoubleVar(value=default)
            setattr(self, attr, var)
            tk.Label(dim_row, text=label, bg=Theme.BG,
                     fg=Theme.TEXT_2, font=Theme.F_BODY).pack(side="left", padx=(10, 2))
            tk.Entry(dim_row, textvariable=var, width=8,
                     bg=Theme.SURFACE2, fg=Theme.TEXT_1, font=Theme.F_MONO_S,
                     bd=0, relief="flat", insertbackground=Theme.CYAN,
                     ).pack(side="left")

        self._status = tk.Label(self, text="0 / 4 corners selected",
                                bg=Theme.BG, fg=Theme.TEXT_3, font=Theme.F_CAP)
        self._status.pack(pady=(2, 6))

        btn_row = tk.Frame(self, bg=Theme.BG)
        btn_row.pack(pady=(0, 12))
        GlowButton(btn_row, "CALIBRATE",
                   self._calibrate, accent=Theme.CYAN,  width=120, height=34
                   ).pack(side="left", padx=4)
        GlowButton(btn_row, "RESET",
                   self._reset,     accent=Theme.WARNING, width=100, height=34
                   ).pack(side="left", padx=4)
        GlowButton(btn_row, "CANCEL",
                   self.destroy,    accent=Theme.DANGER,  width=100, height=34
                   ).pack(side="left", padx=4)

    # ── Canvas interaction ────────────────────────────────────────────────────

    def _on_click(self, e):
        if len(self._pts) >= 4:
            return
        img_x = int(e.x * self._scale)
        img_y = int(e.y * self._scale)
        self._pts.append([img_x, img_y])

        idx   = len(self._pts) - 1
        color = _CORNER_COLORS[idx]
        r = 6
        self._canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r,
                                  fill=color, outline="white", width=1)
        self._canvas.create_text(e.x + 10, e.y - 10,
                                  text=_CORNER_LABELS[idx],
                                  fill=color, font=("Consolas", 9, "bold"))

        self._status.config(text=f"{len(self._pts)} / 4 corners selected")

        if len(self._pts) == 4:
            # Draw outline polygon
            flat = [coord for pt in self._pts
                    for coord in (pt[0] / self._scale, pt[1] / self._scale)]
            self._canvas.create_polygon(flat,
                                         outline=Theme.CYAN, fill="",
                                         dash=(4, 4), width=1)

    def _reset(self):
        self._pts = []
        self._status.config(text="0 / 4 corners selected")
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

    # ── Calibrate ─────────────────────────────────────────────────────────────

    def _calibrate(self):
        if len(self._pts) != 4:
            messagebox.showwarning("Incomplete",
                                   "Select all 4 corners first.", parent=self)
            return
        w_m = self._var_w.get()
        h_m = self._var_h.get()
        if w_m <= 0 or h_m <= 0:
            messagebox.showwarning("Invalid dimensions",
                                   "Width and height must be > 0.", parent=self)
            return
        src = np.array(self._pts, dtype=np.float32)
        self._callback(src, w_m, h_m)
        self.destroy()
