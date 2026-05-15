"""Camera calibration popup window."""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json

BG_C  = "#0a0c12"
SB_C  = "#0d1018"
ACC_C = "#f59e0b"


class CalibratorWindow:
    """
    Popup camera calibration window.
    Methods: auto (Hough vanishing point), manual parallel lines, 2-point+distance.
    Calls callback(params) on apply.
    """
    def __init__(self, parent, callback=None, frame_ref=None):
        self.callback  = callback
        self.frame     = frame_ref.copy() if frame_ref is not None else None
        self.vw = self.vh = 0
        if self.frame is not None:
            self.vh, self.vw = self.frame.shape[:2]

        self.lines        = []
        self.drawing_line = False
        self.line_start   = None
        self._temp        = None
        self.ref_pts      = []
        self.setting_ref  = False
        self.vp           = None
        self.pitch        = None
        self.focal        = None
        self.px_per_m     = None
        self.sx = self.sy = 1.0
        self.ox = self.oy = 0

        self.win = tk.Toplevel(parent)
        self.win.title("Kamera Kalibrlash")
        self.win.geometry("1050x650")
        self.win.configure(bg=BG_C)
        self._build()

    def _build(self):
        st = ttk.Style(); st.theme_use("clam")
        st.configure("TFrame",  background=BG_C)
        st.configure("TLabel",  background=BG_C, foreground="#b0b8c8",
                     font=("Consolas", 10))
        st.configure("TButton", font=("Consolas", 9, "bold"), padding=5)

        left = ttk.Frame(self.win)
        left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self.canvas = tk.Canvas(left, bg="#000", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._click)
        self.canvas.bind("<Motion>",        self._motion)
        self.canvas.bind("<Configure>",     lambda _: self._render())

        ctrl = ttk.Frame(left); ctrl.pack(fill="x", pady=3)
        ttk.Button(ctrl, text="📂 Rasm/Video kadr",
                   command=self._open).pack(side="left", padx=2)
        ttk.Button(ctrl, text="🔍 Avto-aniqlash",
                   command=self._auto_detect).pack(side="left", padx=2)
        ttk.Button(ctrl, text="📏 Qo'lda chiziq",
                   command=self._start_line).pack(side="left", padx=2)
        ttk.Button(ctrl, text="📐 2 nuqta+masofa",
                   command=self._start_ref).pack(side="left", padx=2)
        ttk.Button(ctrl, text="🗑 Tozalash",
                   command=self._clear).pack(side="left", padx=2)

        right = tk.Frame(self.win, bg=SB_C, width=300)
        right.pack(side="right", fill="y"); right.pack_propagate(False)
        tk.Label(right, text="KAMERA PARAMETRLARI", bg=SB_C, fg=ACC_C,
                 font=("Consolas", 11, "bold")).pack(pady=10)

        self.lbl_vp    = tk.Label(right, text="Vanishing Point: —",
                                  bg=SB_C, fg="#94a3b8", font=("Consolas", 9))
        self.lbl_pitch = tk.Label(right, text="Egilish burchagi: —",
                                  bg=SB_C, fg="#94a3b8", font=("Consolas", 9))
        self.lbl_focal = tk.Label(right, text="Fokus masofasi: —",
                                  bg=SB_C, fg="#94a3b8", font=("Consolas", 9))
        for lbl in (self.lbl_vp, self.lbl_pitch, self.lbl_focal):
            lbl.pack(anchor="w", padx=14, pady=2)

        tk.Frame(right, bg="#1e293b", height=1).pack(fill="x", padx=12, pady=8)
        tk.Label(right, text="QO'LDA KIRITISH", bg=SB_C, fg="#475569",
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=12)

        fields = [
            ("Kamera balandligi (m):", "e_cam_h",    "6.0"),
            ("Fokus taxmini (px):",    "e_cam_f",    "800"),
            ("Sensor kengligi (mm):",  "e_sensor_w", "6.17"),
        ]
        for lbl_text, attr, default in fields:
            fr = tk.Frame(right, bg=SB_C); fr.pack(fill="x", padx=14, pady=2)
            tk.Label(fr, text=lbl_text, bg=SB_C, fg="#94a3b8",
                     font=("Consolas", 9)).pack(side="left")
            e = ttk.Entry(fr, width=8); e.pack(side="right")
            e.insert(0, default)
            setattr(self, attr, e)

        ttk.Button(right, text="Burchakni hisoblash",
                   command=self._calc_pitch).pack(fill="x", padx=14, pady=6)

        tk.Frame(right, bg="#1e293b", height=1).pack(fill="x", padx=12, pady=6)
        tk.Label(right, text="REFERENS MASOFA", bg=SB_C, fg="#475569",
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=12)
        fr2 = tk.Frame(right, bg=SB_C); fr2.pack(fill="x", padx=14, pady=2)
        tk.Label(fr2, text="Haqiqiy masofa (m):", bg=SB_C, fg="#94a3b8",
                 font=("Consolas", 9)).pack(side="left")
        self.e_ref_dist = ttk.Entry(fr2, width=8); self.e_ref_dist.pack(side="right")
        self.e_ref_dist.insert(0, "5.0")

        tk.Frame(right, bg="#1e293b", height=1).pack(fill="x", padx=12, pady=6)
        ttk.Button(right, text="💾 JSON saqlash",
                   command=self._save_json).pack(fill="x", padx=14, pady=2)
        ttk.Button(right, text="✅ Asosiy dasturga qo'llash",
                   command=self._apply_to_main).pack(fill="x", padx=14, pady=4)

        self.lbl_result = tk.Label(right, text="", bg=SB_C, fg="#34d399",
                                   font=("Consolas", 9), wraplength=260, justify="left")
        self.lbl_result.pack(anchor="w", padx=14, pady=6)

        if self.frame is not None:
            self._render()

    # ── I/O ──────────────────────────────────────────────────────────────────
    def _open(self):
        p = filedialog.askopenfilename(
            filetypes=[("Rasm/Video", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov"),
                       ("All", "*.*")],
            parent=self.win)
        if not p:
            return
        if p.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(p)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
            ok, frame = cap.read(); cap.release()
            if ok: self.frame = frame
        else:
            self.frame = cv2.imread(p)
        if self.frame is not None:
            self.vh, self.vw = self.frame.shape[:2]
            self.lines = []; self.ref_pts = []; self.vp = None
            self._render()

    def _render(self):
        if self.frame is None: return
        disp = self.frame.copy()
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        r  = min(cw / max(self.vw, 1), ch / max(self.vh, 1))
        nw, nh = int(self.vw * r), int(self.vh * r)
        self.sx, self.sy = self.vw / max(nw, 1), self.vh / max(nh, 1)
        self.ox, self.oy = (cw - nw) // 2, (ch - nh) // 2

        for i, (x1, y1, x2, y2) in enumerate(self.lines):
            cv2.line(disp, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp, str(i+1), (x1+5, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if self.drawing_line and self.line_start and self._temp:
            cv2.line(disp, tuple(self.line_start), tuple(self._temp), (0, 200, 200), 1)

        for i, pt in enumerate(self.ref_pts):
            cv2.circle(disp, tuple(pt), 8, (255, 100, 100), -1)
            cv2.putText(disp, f"R{i+1}", (pt[0]+10, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        if len(self.ref_pts) == 2:
            cv2.line(disp, tuple(self.ref_pts[0]), tuple(self.ref_pts[1]),
                     (255, 100, 100), 2)

        if self.vp:
            vx, vy = int(self.vp[0]), int(self.vp[1])
            if 0 <= vx < self.vw and 0 <= vy < self.vh:
                cv2.circle(disp, (vx, vy), 12, (0, 0, 255), 3)
                cv2.putText(disp, "VP", (vx+15, vy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.line(disp, (0, vy), (self.vw, vy), (0, 0, 200), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._tk = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (nw, nh))))
        self.canvas.delete("all")
        self.canvas.create_image(self.ox, self.oy, anchor="nw", image=self._tk)

    def _c2v(self, ex, ey):
        return (max(0, min(self.vw-1, int((ex - self.ox) * self.sx))),
                max(0, min(self.vh-1, int((ey - self.oy) * self.sy))))

    def _click(self, e):
        if self.frame is None: return
        vx, vy = self._c2v(e.x, e.y)
        if self.drawing_line:
            if self.line_start is None:
                self.line_start = [vx, vy]
            else:
                self.lines.append((self.line_start[0], self.line_start[1], vx, vy))
                self.line_start = None; self.drawing_line = False
                self._render()
                if len(self.lines) >= 2:
                    self._calc_vp()
        elif self.setting_ref:
            self.ref_pts.append([vx, vy])
            if len(self.ref_pts) >= 2:
                self.setting_ref = False
                self._calc_from_ref()
            self._render()

    def _motion(self, e):
        if self.drawing_line and self.line_start and self.frame is not None:
            self._temp = list(self._c2v(e.x, e.y))
            self._render()

    def _start_line(self):
        self.drawing_line = True; self.line_start = None

    def _start_ref(self):
        self.setting_ref = True; self.ref_pts = []

    def _clear(self):
        self.lines = []; self.ref_pts = []; self.vp = None
        self.pitch = self.focal = None
        self.lbl_vp.configure(text="VP: —")
        self.lbl_pitch.configure(text="Egilish burchagi: —")
        self.lbl_focal.configure(text="Fokus masofasi: —")
        self.lbl_result.configure(text="")
        if self.frame is not None: self._render()

    # ── Detection ─────────────────────────────────────────────────────────────
    def _auto_detect(self):
        if self.frame is None: return
        gray  = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        mask  = np.zeros_like(edges)
        mask[self.vh // 3:, :] = 255
        edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=80, maxLineGap=30)
        if lines is None:
            messagebox.showinfo("Info", "Chiziqlar topilmadi. Qo'lda chizing.",
                                parent=self.win); return
        good = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if 15 < angle < 85:
                good.append((x1, y1, x2, y2))
        if len(good) < 2:
            messagebox.showinfo("Info", f"Faqat {len(good)} chiziq topildi.",
                                parent=self.win); return
        good.sort(key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)
        self.lines = good[:6]
        self._render(); self._calc_vp()

    def _calc_vp(self):
        if len(self.lines) < 2: return
        pts = []
        for i in range(len(self.lines)):
            for j in range(i+1, len(self.lines)):
                pt = self._intersect(self.lines[i], self.lines[j])
                if pt and (-self.vw < pt[0] < 2*self.vw) and (-self.vh < pt[1] < 2*self.vh):
                    pts.append(pt)
        if not pts:
            messagebox.showinfo("Info", "Kesishish topilmadi.", parent=self.win); return
        self.vp = (np.median([p[0] for p in pts]), np.median([p[1] for p in pts]))
        self._calc_pitch(); self._render()

    @staticmethod
    def _intersect(l1, l2):
        x1, y1, x2, y2 = l1; x3, y3, x4, y4 = l2
        d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(d) < 1e-6: return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / d
        return (x1 + t*(x2-x1), y1 + t*(y2-y1))

    def _calc_pitch(self):
        if self.vp is None: return
        try:
            f = float(self.e_cam_f.get())
        except ValueError:
            f = 800.0
        cy          = self.vh / 2.0
        self.pitch  = np.degrees(np.arctan((cy - self.vp[1]) / f))
        self.focal  = f
        self.lbl_vp.configure(text=f"VP: ({self.vp[0]:.0f}, {self.vp[1]:.0f})")
        self.lbl_pitch.configure(text=f"Egilish: {self.pitch:.1f}°", fg="#34d399")
        self.lbl_focal.configure(text=f"Fokus: {f:.0f} px")
        self.lbl_result.configure(
            text=f"Egilish: {self.pitch:.1f}°\nVP: ({self.vp[0]:.0f},{self.vp[1]:.0f})\n"
                 f"Fokus: {f:.0f} px")

    def _calc_from_ref(self):
        if len(self.ref_pts) < 2: return
        try:
            real_d = float(self.e_ref_dist.get())
            H      = float(self.e_cam_h.get())
            f      = float(self.e_cam_f.get())
        except ValueError:
            messagebox.showwarning("Xato", "Qiymatlarni to'g'ri kiriting!",
                                   parent=self.win); return
        p1, p2 = self.ref_pts
        px_d   = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        cy     = self.vh / 2.0
        avg_y  = (p1[1] + p2[1]) / 2.0
        self.pitch    = np.degrees(np.arctan(H/real_d) + np.arctan((cy-avg_y)/f))
        self.focal    = f
        self.px_per_m = px_d / real_d
        self.lbl_pitch.configure(text=f"Egilish: {self.pitch:.1f}° (taxminiy)", fg="#fbbf24")
        self.lbl_result.configure(
            text=f"Piksel/metr: {self.px_per_m:.1f}\nEgilish: {self.pitch:.1f}°\n"
                 f"H={H}m f={f}px\nBu px/m qo'llandi ✓")
        self._render()

    def _save_json(self):
        try:   H = float(self.e_cam_h.get())
        except: H = 6.0
        try:   f = float(self.e_cam_f.get())
        except: f = 800.0
        data = {
            "camera_height_m":  H,
            "camera_pitch_deg": round(self.pitch, 2) if self.pitch else 0,
            "focal_length_px":  round(f, 1),
            "frame_width":  self.vw,
            "frame_height": self.vh,
            "vanishing_point": [round(self.vp[0], 1), round(self.vp[1], 1)] if self.vp else None,
        }
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            initialfile="camera_params.json", parent=self.win)
        if path:
            with open(path, "w") as fp:
                json.dump(data, fp, indent=2)
            messagebox.showinfo("Saqlandi!", path, parent=self.win)

    def _apply_to_main(self):
        if self.pitch is None:
            messagebox.showwarning("Hisoblang", "Avval burchakni hisoblang!",
                                   parent=self.win); return
        if self.callback:
            try:   H = float(self.e_cam_h.get())
            except: H = 6.0
            try:   f = float(self.e_cam_f.get())
            except: f = 800.0
            params = {
                "camera_height_m":  H,
                "camera_pitch_deg": self.pitch,
                "focal_length_px":  f,
                "frame_width":  self.vw  if self.vw  else 1280,
                "frame_height": self.vh  if self.vh  else  720,
            }
            if self.px_per_m is not None:
                params["px_per_m"] = self.px_per_m
            self.callback(params)
        messagebox.showinfo("Qo'llandi",
                            "Kamera parametrlari asosiy dasturga qo'llandi!",
                            parent=self.win)
        self.win.destroy()
