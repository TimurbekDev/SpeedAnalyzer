"""
Reusable modern UI widgets — no external GUI deps beyond optional matplotlib.
"""

import tkinter as tk
from tkinter import ttk
import time
from collections import deque
from ui.theme.dark import Theme

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np
    _MPL = True
except ImportError:
    _MPL = False


class GlowButton(tk.Canvas):
    def __init__(self, parent, text, command=None,
                 accent=Theme.CYAN, width=140, height=36, icon="", **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=Theme.BG, highlightthickness=0, cursor="hand2", **kwargs)
        self.text    = text
        self.command = command
        self.accent  = accent
        self.icon    = icon
        self._hover  = False
        self._press  = False
        self._active = False   # persistent ON state: accent border+text even without hover
        self.bind("<Enter>",           self._enter)
        self.bind("<Leave>",           self._leave)
        self.bind("<ButtonPress-1>",   self._press_cb)
        self.bind("<ButtonRelease-1>", self._release_cb)
        self.bind("<Configure>",       lambda _: self._draw())
        self._draw()

    def _draw(self):
        self.delete("all")
        w = max(self.winfo_width(),  int(self.cget("width")))
        h = max(self.winfo_height(), int(self.cget("height")))
        r = Theme.RADIUS
        lit = self._hover or self._active
        bg  = Theme.SURFACE3 if self._hover else Theme.SURFACE2
        if self._press:
            bg = Theme.SURFACE
        self._rrect(0, 0, w, h, r, fill=bg,
                    outline=self.accent if lit else Theme.BORDER, width=1)
        if lit:
            self.create_rectangle(r, h - 2, w - r, h, fill=self.accent, outline="")
        label = f"{self.icon} {self.text}" if self.icon else self.text
        color = self.accent if lit else Theme.TEXT_3
        self.create_text(w // 2, h // 2, text=label, fill=color,
                         font=Theme.F_MONO_S, anchor="center")

    def _rrect(self, x0, y0, x1, y1, r, **kw):
        pts = [x0+r, y0, x1-r, y0, x1, y0, x1, y0+r,
               x1, y1-r, x1, y1, x1-r, y1, x0+r, y1,
               x0, y1, x0, y1-r, x0, y0+r, x0, y0]
        self.create_polygon(pts, smooth=True, **kw)

    def _enter(self, _):    self._hover = True;  self._draw()
    def _leave(self, _):    self._hover = False;  self._press = False; self._draw()
    def _press_cb(self, _): self._press = True;   self._draw()
    def _release_cb(self, _):
        self._press = False; self._draw()
        if self.command:
            self.command()

    def set_active(self, active: bool) -> None:
        """Persistent lit state: accent border + text shown even without hover."""
        self._active = active
        self._draw()

    def update_text(self, text, accent=None):
        self.text = text
        if accent:
            self.accent = accent
        self._draw()


class StatCard(tk.Canvas):
    def __init__(self, parent, title, unit="",
                 accent=Theme.CYAN, icon="", width=170, height=110, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=Theme.BG, highlightthickness=0, **kwargs)
        self.title    = title
        self.unit     = unit
        self.accent   = accent
        self.icon     = icon
        self._disp    = 0.0
        self._target  = 0.0
        self._raw     = "0"
        self._numeric = True
        self._anim    = None
        self.bind("<Configure>", lambda _: self._draw())

    def set(self, value):
        if isinstance(value, (int, float)):
            self._target  = float(value)
            self._raw     = str(value)
            self._numeric = True
            if self._anim:
                self.after_cancel(self._anim)
            self._tick()
        else:
            self._numeric = False
            self._raw = str(value)
            self._draw()

    def _tick(self):
        diff = self._target - self._disp
        if abs(diff) > 0.3:
            self._disp += diff * 0.2
            self._draw()
            self._anim = self.after(16, self._tick)
        else:
            self._disp = self._target
            self._draw()

    def _draw(self):
        self.delete("all")
        w = max(self.winfo_width(),  170)
        h = max(self.winfo_height(), 110)
        self.create_rectangle(1, 1, w-1, h-1,
                              fill=Theme.SURFACE, outline=Theme.BORDER, width=1)
        self.create_rectangle(1, 1, w-1, 3, fill=self.accent, outline="")
        self.create_rectangle(1, 3, w-1, 24, fill=Theme.SURFACE2, outline="")
        title_text = f"{self.icon}  {self.title}" if self.icon else self.title
        self.create_text(12, 14, text=title_text.upper(), anchor="w",
                         fill=Theme.TEXT_3, font=Theme.F_CAP)
        val = f"{self._disp:,.0f}" if self._numeric else self._raw
        self.create_text(14, h // 2 + 12, text=val, anchor="w",
                         fill=Theme.TEXT_1, font=("Segoe UI", 28, "bold"))
        if self.unit:
            self.create_text(w-10, h // 2 + 18, text=self.unit, anchor="e",
                             fill=self.accent, font=Theme.F_MONO_S)
        self.create_rectangle(1, h-18, w-1, h-1, fill=Theme.SURFACE2, outline="")
        self.create_oval(8, h-14, 15, h-7, fill=self.accent, outline="")
        self.create_text(20, h-10, text="LIVE", anchor="w",
                         fill=Theme.TEXT_3, font=Theme.F_CAP)


class TopBar(tk.Frame):
    def __init__(self, parent, model_name="", **kwargs):
        super().__init__(parent, bg=Theme.TOPBAR, height=52, **kwargs)
        self.pack_propagate(False)
        self._build(model_name)

    def _build(self, model_name):
        left = tk.Frame(self, bg=Theme.TOPBAR)
        left.pack(side="left", padx=16, fill="y")
        brand = tk.Frame(left, bg=Theme.TOPBAR)
        brand.pack(side="left", fill="y")
        tk.Label(brand, text="SPEED", bg=Theme.TOPBAR, fg=Theme.CYAN,
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        tk.Label(brand, text="ANALYZER", bg=Theme.TOPBAR, fg=Theme.TEXT_2,
                 font=("Segoe UI", 14)).pack(side="left")
        tk.Label(brand, text=" PRO", bg=Theme.TOPBAR, fg=Theme.TEXT_3,
                 font=("Segoe UI", 11)).pack(side="left")
        tk.Frame(self, bg=Theme.DIVIDER, width=1).pack(side="left", fill="y", pady=8)
        mb = tk.Frame(self, bg=Theme.SURFACE2, padx=10)
        mb.pack(side="left", padx=14, fill="y", pady=8)
        tk.Label(mb, text="MODEL", bg=Theme.SURFACE2, fg=Theme.TEXT_3,
                 font=Theme.F_CAP).pack(anchor="w")
        self.lbl_model = tk.Label(mb, text=model_name or "BG MODE",
                                  bg=Theme.SURFACE2, fg=Theme.CYAN, font=Theme.F_MONO_S)
        self.lbl_model.pack(anchor="w")
        right = tk.Frame(self, bg=Theme.TOPBAR)
        right.pack(side="right", padx=16, fill="y")
        self._indicators = {}
        for key, label, color in [
            ("status", "STATUS", Theme.TEXT_3),
            ("tracks", "TRACKS", Theme.CYAN),
            ("fps",    "FPS",    Theme.SUCCESS),
        ]:
            frame = tk.Frame(right, bg=Theme.SURFACE2, padx=12)
            frame.pack(side="right", padx=4, fill="y", pady=8)
            tk.Label(frame, text=label, bg=Theme.SURFACE2,
                     fg=Theme.TEXT_3, font=Theme.F_CAP).pack(anchor="center")
            val_lbl = tk.Label(frame, text="—", bg=Theme.SURFACE2,
                               fg=color, font=("Consolas", 13, "bold"))
            val_lbl.pack(anchor="center")
            self._indicators[key] = val_lbl

    def set_fps(self, fps: float):
        color = (Theme.SUCCESS if fps >= 20 else
                 Theme.WARNING if fps >= 10 else Theme.DANGER)
        self._indicators["fps"].config(text=f"{fps:.0f}", fg=color)

    def set_tracks(self, n: int):
        self._indicators["tracks"].config(text=str(n))

    def set_status(self, text: str, color: str = None):
        STATUS_COLORS = {
            "IDLE":    Theme.TEXT_3,
            "PLAYING": Theme.SUCCESS,
            "PAUSED":  Theme.WARNING,
            "LOADING": Theme.INFO,
            "ERROR":   Theme.DANGER,
        }
        c = color or STATUS_COLORS.get(text, Theme.TEXT_2)
        self._indicators["status"].config(text=text, fg=c)

    def update_model_name(self, model_name: str):
        """Update displayed model name in topbar."""
        self.lbl_model.config(text=model_name or "BG MODE")

    def set_model(self, name: str, ok: bool = True):
        self.lbl_model.config(text=name, fg=Theme.CYAN if ok else Theme.DANGER)


class SidebarNav(tk.Frame):
    ITEMS = [
        ("⬡", "DASH",      "dashboard"),
        ("◈", "DETECT",    "detection"),
        ("◉", "ANALYTICS", "analytics"),
        ("◎", "PROXIMITY", "proximity"),
        ("⚙", "SETTINGS",  "settings"),
        ("⊞", "EXPORT",    "export"),
    ]

    def __init__(self, parent, on_nav, **kwargs):
        super().__init__(parent, bg=Theme.SIDEBAR, width=64, **kwargs)
        self.pack_propagate(False)
        self.on_nav   = on_nav
        self._active  = "dashboard"
        self._widgets = {}
        self._build()

    def _build(self):
        logo = tk.Frame(self, bg=Theme.SIDEBAR, height=52)
        logo.pack(fill="x"); logo.pack_propagate(False)
        tk.Label(logo, text="⬡", bg=Theme.SIDEBAR, fg=Theme.CYAN,
                 font=("Segoe UI", 20, "bold")).pack(expand=True)
        tk.Frame(self, bg=Theme.DIVIDER, height=1).pack(fill="x")
        tk.Frame(self, bg=Theme.SIDEBAR, height=6).pack()
        for icon, label, key in self.ITEMS:
            self._make_btn(icon, label, key)
        tk.Frame(self, bg=Theme.SIDEBAR).pack(fill="both", expand=True)
        tk.Label(self, text="v3", bg=Theme.SIDEBAR, fg=Theme.TEXT_3,
                 font=Theme.F_CAP).pack(pady=8)
        self._highlight("dashboard")

    def _make_btn(self, icon, label, key):
        outer = tk.Frame(self, bg=Theme.SIDEBAR, height=54, cursor="hand2")
        outer.pack(fill="x"); outer.pack_propagate(False)
        indicator = tk.Frame(outer, bg=Theme.SIDEBAR, width=3)
        indicator.pack(side="left", fill="y")
        inner = tk.Frame(outer, bg=Theme.SIDEBAR)
        inner.pack(fill="both", expand=True)
        ic = tk.Label(inner, text=icon, bg=Theme.SIDEBAR,
                      fg=Theme.TEXT_3, font=("Segoe UI", 15))
        ic.pack(pady=(8, 0))
        tx = tk.Label(inner, text=label, bg=Theme.SIDEBAR,
                      fg=Theme.TEXT_3, font=Theme.F_CAP)
        tx.pack()
        self._widgets[key] = (outer, indicator, ic, tx)

        def click(_, k=key):
            self._highlight(k); self.on_nav(k)

        def enter(_, k=key):
            if k != self._active:
                _, _, ic2, _ = self._widgets[k]; ic2.config(fg=Theme.TEXT_2)

        def leave(_, k=key):
            if k != self._active:
                _, _, ic2, _ = self._widgets[k]; ic2.config(fg=Theme.TEXT_3)

        for w in (outer, inner, ic, tx):
            w.bind("<Button-1>", click)
            w.bind("<Enter>",    enter)
            w.bind("<Leave>",    leave)

    def _highlight(self, key):
        if self._active in self._widgets:
            _, ind, ic, tx = self._widgets[self._active]
            ind.config(bg=Theme.SIDEBAR); ic.config(fg=Theme.TEXT_3); tx.config(fg=Theme.TEXT_3)
        self._active = key
        if key in self._widgets:
            _, ind, ic, tx = self._widgets[key]
            ind.config(bg=Theme.CYAN); ic.config(fg=Theme.CYAN); tx.config(fg=Theme.CYAN)


class SpeedChart(tk.Frame):
    def __init__(self, parent, maxpts: int = 60, **kwargs):
        super().__init__(parent, bg=Theme.SURFACE, **kwargs)
        self.maxpts   = maxpts
        self._speeds: deque = deque(maxlen=maxpts)
        self._times:  deque = deque(maxlen=maxpts)
        self._t0      = time.time()
        if _MPL:
            self._build_mpl()
        else:
            tk.Label(self, text="pip install matplotlib\nfor live chart",
                     bg=Theme.SURFACE, fg=Theme.TEXT_3,
                     font=Theme.F_MONO_S, justify="center").pack(expand=True)

    def _build_mpl(self):
        self.fig, self.ax = plt.subplots(figsize=(4.8, 2.0), dpi=92)
        self.fig.patch.set_facecolor(Theme.SURFACE)
        self.ax.set_facecolor(Theme.SURFACE2)
        self.ax.tick_params(colors=Theme.TEXT_3, labelsize=7)
        for side in ("top", "right"):
            self.ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            self.ax.spines[side].set_color(Theme.BORDER)
        self.ax.set_xlabel("s", color=Theme.TEXT_3, fontsize=7)
        self.ax.set_ylabel("km/h", color=Theme.TEXT_3, fontsize=7)
        self.line, = self.ax.plot([], [], color=Theme.CYAN, lw=1.5, alpha=0.9)
        self._fill = None
        self._limit_line = None
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.mpl_canvas.draw()
        self.mpl_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.fig.tight_layout(pad=0.6)

    def push(self, speed_kmh: float, limit: float = 60.0):
        if not _MPL:
            return
        t = time.time() - self._t0
        self._speeds.append(speed_kmh)
        self._times.append(t)
        if len(self._speeds) < 2:
            return
        xs, ys = list(self._times), list(self._speeds)
        self.line.set_data(xs, ys)
        self.ax.set_xlim(max(0, xs[-1] - 30), xs[-1] + 1)
        self.ax.set_ylim(0, max(max(ys) * 1.2, 20))
        if self._fill:
            self._fill.remove()
        self._fill = self.ax.fill_between(xs, ys, alpha=0.12, color=Theme.CYAN)
        if self._limit_line:
            try:
                self._limit_line.remove()
            except Exception:
                pass
        self._limit_line = self.ax.axhline(limit, color=Theme.WARNING,
                                           lw=1, ls="--", alpha=0.6)
        self.mpl_canvas.draw_idle()

    def reset(self):
        if not _MPL:
            return
        self._speeds.clear(); self._times.clear()
        self._t0 = time.time()
        self.line.set_data([], [])
        if self._fill:
            self._fill.remove(); self._fill = None
        self.mpl_canvas.draw_idle()


class VideoPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#000", **kwargs)
        self.canvas = tk.Canvas(self, bg="#030912",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self._has_frame = False
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.canvas.delete("placeholder")
        w  = max(self.canvas.winfo_width(), 800)
        h  = max(self.canvas.winfo_height(), 480)
        cx, cy = w // 2, h // 2
        for x in range(0, w, 48):
            self.canvas.create_line(x, 0, x, h, fill="#0a1425",
                                    width=1, tags="placeholder")
        for y in range(0, h, 48):
            self.canvas.create_line(0, y, w, y, fill="#0a1425",
                                    width=1, tags="placeholder")
        self.canvas.create_oval(cx-44, cy-44, cx+44, cy+44,
                                outline=Theme.CYAN, width=1.5,
                                dash=(5, 4), tags="placeholder")
        self.canvas.create_oval(cx-28, cy-28, cx+28, cy+28,
                                outline=Theme.CYAN_DIM, width=1,
                                dash=(3, 5), tags="placeholder")
        self.canvas.create_text(cx, cy, text="▶", fill=Theme.CYAN,
                                font=("Segoe UI", 26), tags="placeholder")
        self.canvas.create_text(cx, cy+60,
                                text="Open a video file to begin",
                                fill=Theme.TEXT_3, font=Theme.F_BODY,
                                tags="placeholder")

    def _on_resize(self, _):
        if not self._has_frame:
            self._draw_placeholder()

    def set_has_frame(self):
        if not self._has_frame:
            self._has_frame = True
            self.canvas.delete("placeholder")


class PlaybackBar(tk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=Theme.SURFACE, height=58, **kwargs)
        self.pack_propagate(False)
        self.app = app
        self._build()

    def _build(self):
        tk.Frame(self, bg=Theme.BORDER, height=1).pack(fill="x")
        row = tk.Frame(self, bg=Theme.SURFACE)
        row.pack(fill="both", expand=True, padx=10)

        left = tk.Frame(row, bg=Theme.SURFACE)
        left.pack(side="left", fill="y", pady=6)
        GlowButton(left, "⊕ OPEN", self.app._open_video,
                   accent=Theme.CYAN, width=88, height=36).pack(side="left", padx=(0, 4))
        self.btn_play = GlowButton(left, "▶  PLAY", self.app._toggle_play,
                                   accent=Theme.SUCCESS, width=92, height=36)
        self.btn_play.pack(side="left", padx=4)
        for txt, n in [("◀◀", -10), ("◀", -1), ("▶", 1), ("▶▶", 10)]:
            GlowButton(left, txt, lambda n=n: self.app._step(n),
                       accent=Theme.TEXT_3, width=36, height=36).pack(side="left", padx=2)

        mid = tk.Frame(row, bg=Theme.SURFACE)
        mid.pack(side="left", fill="both", expand=True, padx=14)
        time_row = tk.Frame(mid, bg=Theme.SURFACE)
        time_row.pack(fill="x", pady=(6, 0))
        self.lbl_cur   = tk.Label(time_row, text="0:00", bg=Theme.SURFACE,
                                  fg=Theme.CYAN, font=Theme.F_MONO_S)
        self.lbl_cur.pack(side="left")
        self.lbl_total = tk.Label(time_row, text="/ 0:00", bg=Theme.SURFACE,
                                  fg=Theme.TEXT_3, font=Theme.F_MONO_S)
        self.lbl_total.pack(side="left", padx=4)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Horizontal.TScale",
                        troughcolor=Theme.SURFACE2, background=Theme.SURFACE,
                        darkcolor=Theme.CYAN, lightcolor=Theme.CYAN,
                        sliderlength=14, sliderthickness=14)
        self.slider = ttk.Scale(mid, from_=0, to=100, orient="horizontal",
                                command=self.app._seek, style="Dark.Horizontal.TScale")
        self.slider.pack(fill="x")

        right = tk.Frame(row, bg=Theme.SURFACE)
        right.pack(side="right", fill="y", pady=6)

        GlowButton(right, "📷 SHOT", self.app._take_screenshot,
                   accent=Theme.PURPLE, width=74, height=36).pack(side="right", padx=(2, 0))

        tk.Frame(right, bg=Theme.BORDER, width=1).pack(side="right", fill="y", padx=6)

        GlowButton(right, "✕ CLR", self.app._clear_line,
                   accent=Theme.DANGER, width=64, height=36).pack(side="right", padx=2)
        self.btn_line = GlowButton(right, "— LINE", self.app._line_start,
                                   accent=Theme.WARNING, width=74, height=36)
        self.btn_line.pack(side="right", padx=2)

        tk.Frame(right, bg=Theme.BORDER, width=1).pack(side="right", fill="y", padx=6)

        GlowButton(right, "✕ ZONE", self.app._clear_prox,
                   accent=Theme.DANGER, width=74, height=36).pack(side="right", padx=2)
        self.btn_zone = GlowButton(right, "◎ ZONE", self.app._prox_start,
                                   accent=Theme.PURPLE, width=74, height=36)
        self.btn_zone.pack(side="right", padx=(0, 2))

        tk.Frame(right, bg=Theme.BORDER, width=1).pack(side="right", fill="y", padx=6)

        # Analytics toggle buttons — active (lit) by default.
        self.btn_dist = GlowButton(right, "DST", self.app._toggle_dist,
                                   accent=Theme.CYAN, width=60, height=36)
        self.btn_dist.set_active(True)
        self.btn_dist.pack(side="right", padx=2)
        self.btn_speed = GlowButton(right, "SPD", self.app._toggle_speed,
                                    accent=Theme.SUCCESS, width=60, height=36)
        self.btn_speed.set_active(True)
        self.btn_speed.pack(side="right", padx=(0, 2))

    def set_playing(self, playing: bool):
        self.btn_play.update_text(
            "⏸  PAUSE" if playing else "▶  PLAY",
            accent=Theme.WARNING if playing else Theme.SUCCESS
        )

    def set_time(self, cur_s: float, total_s: float):
        def fmt(s):
            m, sec = divmod(int(s), 60)
            return f"{m}:{sec:02d}"
        self.lbl_cur.config(text=fmt(cur_s))
        self.lbl_total.config(text=f"/ {fmt(total_s)}")


class ToastManager:
    _COLORS = {
        "info":    (Theme.CYAN,    Theme.SURFACE2),
        "success": (Theme.SUCCESS, "#0a1f14"),
        "warning": (Theme.WARNING, "#1f1500"),
        "error":   (Theme.DANGER,  "#1f0010"),
    }
    _ICONS = {"info": "ℹ", "success": "✔", "warning": "⚠", "error": "✖"}

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self._stack: list = []

    def show(self, message: str, kind: str = "info", duration: int = 3200):
        fg, bg = self._COLORS.get(kind, self._COLORS["info"])
        icon   = self._ICONS.get(kind, "•")
        t = tk.Toplevel(self.parent)
        t.overrideredirect(True)
        t.attributes("-topmost", True)
        t.configure(bg=bg)
        pw = self.parent.winfo_width()  or 1200
        ph = self.parent.winfo_height() or 800
        px = self.parent.winfo_x()
        py = self.parent.winfo_y()
        y_off = 24 + len(self._stack) * 58
        t.geometry(f"360x48+{px + pw - 380}+{py + ph - 72 - y_off}")
        tk.Frame(t, bg=fg, width=3).place(x=0, y=0, relheight=1.0)
        body = tk.Frame(t, bg=bg, padx=14, pady=10)
        body.place(x=3, y=0, relwidth=1.0, relheight=1.0)
        tk.Label(body, text=icon, bg=bg, fg=fg,
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=(0, 8))
        tk.Label(body, text=message, bg=bg, fg=Theme.TEXT_1,
                 font=Theme.F_BODY, wraplength=270, justify="left").pack(side="left")
        self._stack.append(t)
        self.parent.after(duration, lambda: self._dismiss(t))

    def _dismiss(self, t):
        if t in self._stack:
            self._stack.remove(t)
        try:
            t.destroy()
        except Exception:
            pass


def section_label(parent, text: str, bg=Theme.BG):
    tk.Label(parent, text=text.upper(), bg=bg, fg=Theme.TEXT_3,
             font=Theme.F_CAP).pack(anchor="w", padx=16, pady=(14, 4))


def divider(parent, bg=Theme.BG):
    tk.Frame(parent, bg=Theme.DIVIDER, height=1).pack(fill="x", padx=0)


def param_row(parent, label: str, var, bg=Theme.SURFACE):
    row = tk.Frame(parent, bg=bg)
    row.pack(fill="x", padx=14, pady=2)
    tk.Label(row, text=label, bg=bg, fg=Theme.TEXT_3,
             font=Theme.F_MONO_S).pack(side="left")
    tk.Entry(row, textvariable=var, width=9,
             bg=Theme.SURFACE2, fg=Theme.TEXT_1, font=Theme.F_MONO_S,
             bd=0, relief="flat", insertbackground=Theme.CYAN).pack(side="right")


def toggle_row(parent, label: str, var, bg=Theme.SURFACE, command=None):
    row = tk.Frame(parent, bg=bg)
    row.pack(fill="x", padx=14, pady=2)
    tk.Label(row, text=label, bg=bg, fg=Theme.TEXT_2,
             font=Theme.F_BODY).pack(side="left")
    tk.Checkbutton(row, variable=var, bg=bg, fg=Theme.CYAN,
                   activebackground=bg, selectcolor=Theme.SURFACE2,
                   command=command).pack(side="right")
