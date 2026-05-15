"""Dashboard panel — stat cards, video feed, live chart, detection log."""

import tkinter as tk
from datetime import datetime
from ui.theme.dark import Theme
from ui.widgets.components import StatCard, VideoPanel, SpeedChart, section_label, divider


class DashboardPanel(tk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=Theme.BG, **kwargs)
        self.app = app
        self._build()

    def _build(self):
        cards = tk.Frame(self, bg=Theme.BG)
        cards.pack(fill="x", padx=10, pady=(10, 6))

        self.c_total   = StatCard(cards, "VEHICLES",   "total", Theme.CYAN,    "⬡")
        self.c_speed   = StatCard(cards, "AVG SPEED",  "km/h",  Theme.PURPLE,  "◉")
        self.c_violate = StatCard(cards, "VIOLATIONS", "over",  Theme.DANGER,  "⚠")
        self.c_fps     = StatCard(cards, "FRAME RATE", "fps",   Theme.SUCCESS, "◈")

        for c in (self.c_total, self.c_speed, self.c_violate, self.c_fps):
            c.pack(side="left", fill="both", expand=True, padx=4)

        body = tk.Frame(self, bg=Theme.BG)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.video = VideoPanel(body)
        self.video.pack(side="left", fill="both", expand=True, padx=(0, 6))

        right = tk.Frame(body, bg=Theme.BG, width=264)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        chart_card = tk.Frame(right, bg=Theme.SURFACE)
        chart_card.pack(fill="x", pady=(0, 6))
        section_label(chart_card, "SPEED GRAPH", bg=Theme.SURFACE)
        self.chart = SpeedChart(chart_card, maxpts=60)
        self.chart.pack(fill="x", padx=8, pady=(0, 8))

        cross_card = tk.Frame(right, bg=Theme.SURFACE)
        cross_card.pack(fill="x", pady=(0, 6))
        section_label(cross_card, "CROSSING LINE", bg=Theme.SURFACE)
        cross_row = tk.Frame(cross_card, bg=Theme.SURFACE)
        cross_row.pack(fill="x", padx=14, pady=(0, 10))
        self.lbl_in  = tk.Label(cross_row, text="IN  0", bg=Theme.SURFACE,
                                fg=Theme.SUCCESS, font=("Consolas", 14, "bold"))
        self.lbl_in.pack(side="left")
        self.lbl_out = tk.Label(cross_row, text="OUT  0", bg=Theme.SURFACE,
                                fg=Theme.DANGER, font=("Consolas", 14, "bold"))
        self.lbl_out.pack(side="right")

        log_card = tk.Frame(right, bg=Theme.SURFACE)
        log_card.pack(fill="both", expand=True)
        section_label(log_card, "DETECTION LOG", bg=Theme.SURFACE)
        divider(log_card, bg=Theme.SURFACE)

        log_inner = tk.Frame(log_card, bg=Theme.SURFACE2)
        log_inner.pack(fill="both", expand=True, padx=8, pady=8)

        self.log = tk.Text(log_inner, bg=Theme.SURFACE2, fg=Theme.TEXT_2,
                           font=Theme.F_MONO_S, bd=0, state="disabled",
                           wrap="none", height=1)
        sb = tk.Scrollbar(log_inner, command=self.log.yview,
                          bg=Theme.SURFACE2, troughcolor=Theme.SURFACE, width=8)
        self.log.config(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.log.tag_config("ok",   foreground=Theme.SUCCESS)
        self.log.tag_config("warn", foreground=Theme.WARNING)
        self.log.tag_config("bad",  foreground=Theme.DANGER)
        self.log.tag_config("meta", foreground=Theme.TEXT_3)

    def update_cards(self, total, avg_speed, violations, fps):
        self.c_total.set(total)
        self.c_speed.set(avg_speed)
        self.c_violate.set(violations)
        self.c_fps.set(fps)

    def update_crossing(self, count_in: int, count_out: int):
        self.lbl_in.config(text=f"IN  {count_in}")
        self.lbl_out.config(text=f"OUT  {count_out}")

    def log_detection(self, track_id: int, speed: float, direction: str, limit: float):
        tag = ("ok" if speed < limit else "warn" if speed < limit * 1.3 else "bad")
        ts  = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  #{track_id:03d}  {direction:3s}  {speed:5.1f} km/h\n"
        self.log.config(state="normal")
        self.log.insert("end", line, tag)
        self.log.see("end")
        self.log.config(state="disabled")
        content = self.log.get("1.0", "end-1c").splitlines()
        if len(content) > 200:
            self.log.config(state="normal")
            self.log.delete("1.0", f"{len(content)-200}.0")
            self.log.config(state="disabled")
