"""Proximity analytics panel — inter-vehicle distance monitoring dashboard."""

import tkinter as tk
from ui.theme.dark import Theme
from ui.widgets.components import (
    GlowButton, section_label, divider, param_row, toggle_row,
)


class ProximityPanel(tk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=Theme.BG, **kwargs)
        self.app = app
        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        section_label(self, "PROXIMITY ANALYTICS")

        body = tk.Frame(self, bg=Theme.BG)
        body.pack(fill="both", expand=True, padx=10)

        left = tk.Frame(body, bg=Theme.BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        right = tk.Frame(body, bg=Theme.BG, width=246)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_stats(left)
        self._build_congestion(left)
        self._build_pairs(left)
        self._build_calibration(right)
        self._build_viz(right)

    # ── Stats card ─────────────────────────────────────────────────────────

    def _build_stats(self, parent):
        card = tk.Frame(parent, bg=Theme.SURFACE, padx=14, pady=12)
        card.pack(fill="x", pady=6)
        section_label(card, "ZONE STATISTICS", bg=Theme.SURFACE)
        divider(card, bg=Theme.SURFACE)

        grid = tk.Frame(card, bg=Theme.SURFACE)
        grid.pack(fill="x", pady=8)

        self._stat_lbl: dict = {}
        specs = [
            ("Vehicles",     "count",    Theme.CYAN,    "—"),
            ("Min Dist",     "min_dist", Theme.SUCCESS, "— m"),
            ("Avg Dist",     "avg_dist", Theme.WARNING, "— m"),
            ("Max Dist",     "max_dist", Theme.TEXT_1,  "— m"),
        ]
        for i, (title, key, color, default) in enumerate(specs):
            cell = tk.Frame(grid, bg=Theme.SURFACE2, padx=10, pady=8)
            cell.grid(row=0, column=i, padx=4, pady=2, sticky="nsew")
            grid.columnconfigure(i, weight=1)
            tk.Label(cell, text=title, bg=Theme.SURFACE2,
                     fg=Theme.TEXT_3, font=Theme.F_CAP).pack(anchor="w")
            lbl = tk.Label(cell, text=default, bg=Theme.SURFACE2,
                           fg=color, font=("Segoe UI", 14, "bold"))
            lbl.pack(anchor="w")
            self._stat_lbl[key] = (lbl, color)

    # ── Congestion card ───────────────────────────────────────────────────────

    def _build_congestion(self, parent):
        card = tk.Frame(parent, bg=Theme.SURFACE, padx=14, pady=12)
        card.pack(fill="x", pady=6)
        section_label(card, "CONGESTION & DENSITY", bg=Theme.SURFACE)
        divider(card, bg=Theme.SURFACE)

        bar_row = tk.Frame(card, bg=Theme.SURFACE)
        bar_row.pack(fill="x", pady=(8, 4))

        self._cong_bar = tk.Canvas(bar_row, bg=Theme.SURFACE2,
                                   height=22, highlightthickness=0)
        self._cong_bar.pack(side="left", fill="x", expand=True)

        self._cong_pct = tk.Label(bar_row, text="0%", bg=Theme.SURFACE,
                                  fg=Theme.TEXT_1, font=Theme.F_MONO_S, width=5)
        self._cong_pct.pack(side="right", padx=(6, 0))

        info = tk.Frame(card, bg=Theme.SURFACE)
        info.pack(fill="x")
        self._cong_level = tk.Label(info, text="Level: —",
                                    bg=Theme.SURFACE, fg=Theme.TEXT_2,
                                    font=Theme.F_MONO_S)
        self._cong_level.pack(side="left")
        self._density_lbl = tk.Label(info, text="Density: — veh/1000m²",
                                     bg=Theme.SURFACE, fg=Theme.TEXT_3,
                                     font=Theme.F_CAP)
        self._density_lbl.pack(side="right")

    # ── Pairs table card ──────────────────────────────────────────────────────

    def _build_pairs(self, parent):
        card = tk.Frame(parent, bg=Theme.SURFACE, padx=14, pady=12)
        card.pack(fill="both", expand=True, pady=6)
        section_label(card, "CLOSEST VEHICLE PAIRS", bg=Theme.SURFACE)
        divider(card, bg=Theme.SURFACE)

        hdr = tk.Frame(card, bg=Theme.SURFACE)
        hdr.pack(fill="x", pady=(6, 2))
        for txt in ("Vehicles", "Distance", "Status"):
            tk.Label(hdr, text=txt, bg=Theme.SURFACE,
                     fg=Theme.TEXT_3, font=Theme.F_CAP).pack(side="left", expand=True)

        self._pairs_frame = tk.Frame(card, bg=Theme.SURFACE)
        self._pairs_frame.pack(fill="both", expand=True)

    # ── Calibration card ──────────────────────────────────────────────────────

    def _build_calibration(self, parent):
        card = tk.Frame(parent, bg=Theme.SURFACE, padx=14, pady=12)
        card.pack(fill="x", pady=6)
        section_label(card, "CALIBRATION", bg=Theme.SURFACE)
        divider(card, bg=Theme.SURFACE)

        param_row(card, "px / metre",    self.app.var_prox_px_m)
        param_row(card, "Safe dist (m)", self.app.var_safe_dist)
        param_row(card, "Warn dist (m)", self.app.var_warn_dist)

        GlowButton(card, "APPLY SETTINGS", self.app._apply_prox_settings,
                   accent=Theme.CYAN, height=32).pack(fill="x", pady=(10, 3))
        GlowButton(card, "BIRD'S-EYE CALIB", self.app._open_birdseye_calib,
                   accent=Theme.PURPLE, height=32).pack(fill="x", pady=3)
        GlowButton(card, "EXPORT PROXIMITY", self.app._export_proximity_excel,
                   accent=Theme.SUCCESS, height=32).pack(fill="x", pady=3)

    # ── Visualisation toggles card ────────────────────────────────────────────

    def _build_viz(self, parent):
        card = tk.Frame(parent, bg=Theme.SURFACE, padx=14, pady=12)
        card.pack(fill="x", pady=6)
        section_label(card, "VISUALISATION", bg=Theme.SURFACE)
        divider(card, bg=Theme.SURFACE)

        toggle_row(card, "Distance Lines",  self.app.var_show_dist)
        toggle_row(card, "Distance Labels", self.app.var_show_labels)
        toggle_row(card, "Heatmap Overlay", self.app.var_show_heatmap)
        toggle_row(card, "Bird's-Eye Mode", self.app.var_use_birdseye)

    # ── Live refresh ──────────────────────────────────────────────────────────

    def refresh(self, snap: dict):
        count      = snap.get("count",      0)
        min_m      = snap.get("min_m",      0.0)
        avg_m      = snap.get("avg_m",      0.0)
        max_m      = snap.get("max_m",      0.0)
        congestion = snap.get("congestion", 0.0)
        density    = snap.get("density",    0.0)
        pairs      = snap.get("pairs",      [])

        safe_m = self.app.var_safe_dist.get()
        warn_m = self.app.var_warn_dist.get()

        # Vehicle count
        lbl, _ = self._stat_lbl["count"]
        lbl.config(text=str(count))

        # Distance cells
        def _color(m):
            if m >= safe_m: return Theme.SUCCESS
            if m >= warn_m: return Theme.WARNING
            return Theme.DANGER

        if count >= 2:
            for key, val in (("min_dist", min_m), ("avg_dist", avg_m), ("max_dist", max_m)):
                lbl, _ = self._stat_lbl[key]
                lbl.config(text=f"{val:.1f} m", fg=_color(val))
        else:
            for key in ("min_dist", "avg_dist", "max_dist"):
                lbl, orig_color = self._stat_lbl[key]
                lbl.config(text="— m", fg=Theme.TEXT_3)

        self._refresh_congestion(congestion, density)
        self._refresh_pairs(pairs, safe_m, warn_m)

    def _refresh_congestion(self, pct: float, density: float):
        self._cong_pct.config(text=f"{pct:.0f}%")

        self._cong_bar.update_idletasks()
        w = max(self._cong_bar.winfo_width(), 10)
        h = max(self._cong_bar.winfo_height(), 10)

        if pct < 25:
            bar_color = Theme.SUCCESS;  level = "LOW"
        elif pct < 50:
            bar_color = Theme.WARNING;  level = "MODERATE"
        elif pct < 75:
            bar_color = "#ff8c00";      level = "HIGH"
        else:
            bar_color = Theme.DANGER;   level = "CRITICAL"

        fill_w = max(0, int(w * pct / 100.0))
        self._cong_bar.delete("all")
        self._cong_bar.create_rectangle(0, 0, w, h,
                                        fill=Theme.SURFACE2, outline="")
        if fill_w > 2:
            self._cong_bar.create_rectangle(0, 0, fill_w, h,
                                            fill=bar_color, outline="")

        self._cong_level.config(
            text=f"Level: {level}",
            fg=bar_color if pct > 5 else Theme.TEXT_3)
        self._density_lbl.config(text=f"Density: {density:.1f} veh/1000m²")

    def _refresh_pairs(self, pairs, safe_m: float, warn_m: float):
        for w in self._pairs_frame.winfo_children():
            w.destroy()

        sorted_pairs = sorted(pairs, key=lambda p: p.m_dist)[:10]
        for pair in sorted_pairs:
            if pair.m_dist >= safe_m:
                dot_color = Theme.SUCCESS;  status = "SAFE"
            elif pair.m_dist >= warn_m:
                dot_color = Theme.WARNING;  status = "WARN"
            else:
                dot_color = Theme.DANGER;   status = "DANGER"

            row = tk.Frame(self._pairs_frame, bg=Theme.SURFACE2)
            row.pack(fill="x", pady=1)

            tk.Label(row, text=f"#{pair.id_a} ↔ #{pair.id_b}",
                     bg=Theme.SURFACE2, fg=Theme.TEXT_1,
                     font=Theme.F_MONO_S).pack(side="left", padx=(8, 0), expand=True, anchor="w")
            tk.Label(row, text=f"{pair.m_dist:.1f} m",
                     bg=Theme.SURFACE2, fg=dot_color,
                     font=Theme.F_MONO_S).pack(side="left", expand=True)
            tk.Label(row, text=f"● {status}",
                     bg=Theme.SURFACE2, fg=dot_color,
                     font=Theme.F_CAP).pack(side="right", padx=8, expand=True)
