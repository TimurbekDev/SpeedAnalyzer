"""Analytics panel — speed histogram, session summary grid."""

import time
import tkinter as tk
import numpy as np
from ui.theme.dark import Theme
from ui.widgets.components import section_label, divider

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _MPL = True
except ImportError:
    _MPL = False


class AnalyticsPanel(tk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=Theme.BG, **kwargs)
        self.app  = app
        self._mpl = _MPL
        self._build()

    def _build(self):
        section_label(self, "ANALYTICS")

        body = tk.Frame(self, bg=Theme.BG)
        body.pack(fill="both", expand=True, padx=10)

        if self._mpl:
            hist_card = tk.Frame(body, bg=Theme.SURFACE, padx=12, pady=10)
            hist_card.pack(fill="x", pady=6)
            section_label(hist_card, "SPEED DISTRIBUTION", bg=Theme.SURFACE)
            self.fig, self.ax = plt.subplots(figsize=(7, 2.4), dpi=90)
            self.fig.patch.set_facecolor(Theme.SURFACE)
            self.ax.set_facecolor(Theme.SURFACE2)
            self.ax.tick_params(colors=Theme.TEXT_3, labelsize=7)
            for s in ("top", "right"):
                self.ax.spines[s].set_visible(False)
            for s in ("bottom", "left"):
                self.ax.spines[s].set_color(Theme.BORDER)
            self.hist_widget = FigureCanvasTkAgg(self.fig, master=hist_card)
            self.hist_widget.get_tk_widget().pack(fill="x")
            self.fig.tight_layout(pad=0.6)

        stats_card = tk.Frame(body, bg=Theme.SURFACE, padx=12, pady=12)
        stats_card.pack(fill="x", pady=6)
        section_label(stats_card, "SESSION SUMMARY", bg=Theme.SURFACE)
        divider(stats_card, bg=Theme.SURFACE)

        grid = tk.Frame(stats_card, bg=Theme.SURFACE)
        grid.pack(fill="x", padx=0, pady=8)

        self._stats = {}
        items = [
            ("Total Vehicles", "total"),
            ("Avg Speed",      "avg"),
            ("Max Speed",      "max"),
            ("Violations",     "violations"),
            ("Over Limit %",   "pct"),
            ("Session Time",   "time"),
        ]
        for i, (label, key) in enumerate(items):
            col   = i % 3
            row_n = i // 3
            cell  = tk.Frame(grid, bg=Theme.SURFACE2, padx=12, pady=8)
            cell.grid(row=row_n, column=col, padx=4, pady=4, sticky="nsew")
            grid.columnconfigure(col, weight=1)
            tk.Label(cell, text=label, bg=Theme.SURFACE2,
                     fg=Theme.TEXT_3, font=Theme.F_CAP).pack(anchor="w")
            val = tk.Label(cell, text="—", bg=Theme.SURFACE2,
                           fg=Theme.TEXT_1, font=("Segoe UI", 15, "bold"))
            val.pack(anchor="w")
            self._stats[key] = val

    def refresh(self, results, session_start: float, limit: float):
        if not results:
            return
        speeds = [r.get("speed_kmh", 0) for r in results if r.get("speed_kmh", 0) > 1]
        if not speeds:
            return

        violations = sum(1 for s in speeds if s > limit)
        self._stats["total"].config(text=str(len(results)))
        self._stats["avg"].config(text=f"{np.mean(speeds):.1f} km/h")
        self._stats["max"].config(
            text=f"{max(speeds):.1f} km/h",
            fg=Theme.DANGER if max(speeds) > limit else Theme.TEXT_1)
        self._stats["violations"].config(
            text=str(violations),
            fg=Theme.DANGER if violations else Theme.SUCCESS)
        self._stats["pct"].config(
            text=f"{100 * violations / max(len(speeds), 1):.1f}%")
        elapsed = time.time() - session_start
        m, s = divmod(int(elapsed), 60)
        self._stats["time"].config(text=f"{m:02d}:{s:02d}")

        if not self._mpl:
            return
        self.ax.clear()
        self.ax.set_facecolor(Theme.SURFACE2)
        self.ax.tick_params(colors=Theme.TEXT_3, labelsize=7)
        bins = np.arange(0, max(speeds) + 20, 10)
        n, bins_out, patches = self.ax.hist(
            speeds, bins=bins, color=Theme.CYAN, alpha=0.7, edgecolor=Theme.SURFACE)
        for patch, b in zip(patches, bins_out):
            if b > limit * 1.3:
                patch.set_facecolor(Theme.DANGER)
            elif b > limit:
                patch.set_facecolor(Theme.WARNING)
        self.ax.axvline(limit, color=Theme.WARNING, lw=1.5, ls="--", alpha=0.7)
        for s in ("top", "right"):
            self.ax.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            self.ax.spines[s].set_color(Theme.BORDER)
        self.fig.tight_layout(pad=0.6)
        self.hist_widget.draw_idle()
