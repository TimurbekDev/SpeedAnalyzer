"""
Minimal UI widgets for the lane-distance app — no matplotlib, no dashboards.

  GlowButton    — flat accent button
  VideoPanel    — video canvas with an idle placeholder
  ToastManager  — transient corner notifications
  helpers       — section_label, divider, param_row
"""

import tkinter as tk
from ui.theme.dark import Theme


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
        self._active = False
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
    def _leave(self, _):    self._hover = False; self._press = False; self._draw()
    def _press_cb(self, _): self._press = True;  self._draw()
    def _release_cb(self, _):
        self._press = False; self._draw()
        if self.command:
            self.command()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._draw()

    def update_text(self, text, accent=None):
        self.text = text
        if accent:
            self.accent = accent
        self._draw()


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
        self.canvas.create_text(cx, cy, text="▶", fill=Theme.CYAN,
                                font=("Segoe UI", 26), tags="placeholder")
        self.canvas.create_text(cx, cy + 40,
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
        pw = self.parent.winfo_width()  or 1100
        ph = self.parent.winfo_height() or 760
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
             font=Theme.F_CAP).pack(anchor="w", padx=12, pady=(10, 4))


def divider(parent, bg=Theme.BG):
    tk.Frame(parent, bg=Theme.DIVIDER, height=1).pack(fill="x")


def param_row(parent, label: str, var, bg=Theme.SURFACE, width=7):
    row = tk.Frame(parent, bg=bg)
    row.pack(side="left", padx=(0, 14))
    tk.Label(row, text=label, bg=bg, fg=Theme.TEXT_3,
             font=Theme.F_MONO_S).pack(side="left", padx=(0, 6))
    tk.Entry(row, textvariable=var, width=width,
             bg=Theme.SURFACE2, fg=Theme.TEXT_1, font=Theme.F_MONO_S,
             bd=0, relief="flat", insertbackground=Theme.CYAN).pack(side="left")
    return row
