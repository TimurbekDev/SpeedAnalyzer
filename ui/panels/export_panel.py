"""Export panel — Excel report, results preview table."""

import tkinter as tk
from tkinter import ttk
from ui.theme.dark import Theme
from ui.widgets.components import GlowButton, section_label, divider, toggle_row


class ExportPanel(tk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=Theme.BG, **kwargs)
        self.app = app
        self._build()

    def _build(self):
        section_label(self, "EXPORT & REPORTS")

        body = tk.Frame(self, bg=Theme.BG)
        body.pack(fill="both", expand=True, padx=10)

        xls_card = tk.Frame(body, bg=Theme.SURFACE, padx=14, pady=14)
        xls_card.pack(fill="x", pady=6)
        section_label(xls_card, "EXCEL REPORT", bg=Theme.SURFACE)
        divider(xls_card, bg=Theme.SURFACE)
        toggle_row(xls_card, "Auto-save when stopping", self.app.var_autosave)
        tk.Frame(xls_card, bg=Theme.SURFACE, height=6).pack()
        GlowButton(xls_card, "⊞  Export Unified Report",
                   self.app._export_unified, accent=Theme.SUCCESS,
                   height=40).pack(fill="x", padx=14, pady=4)

        tbl_card = tk.Frame(body, bg=Theme.SURFACE, padx=14, pady=12)
        tbl_card.pack(fill="both", expand=True, pady=6)
        section_label(tbl_card, "RESULTS PREVIEW", bg=Theme.SURFACE)
        divider(tbl_card, bg=Theme.SURFACE)

        cols = ("ID", "Speed", "Direction", "Time")
        self.tree = ttk.Treeview(tbl_card, columns=cols, show="headings", height=14)
        widths = {"ID": 60, "Speed": 90, "Direction": 80, "Time": 80}
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=widths[col], anchor="center")

        st = ttk.Style()
        st.configure("Modern.Treeview",
                     background=Theme.SURFACE2, foreground=Theme.TEXT_2,
                     fieldbackground=Theme.SURFACE2, rowheight=26,
                     font=Theme.F_MONO_S)
        st.configure("Modern.Treeview.Heading",
                     background=Theme.SURFACE3, foreground=Theme.TEXT_3,
                     font=Theme.F_CAP)
        st.map("Modern.Treeview",
               background=[("selected", Theme.SURFACE3)],
               foreground=[("selected", Theme.CYAN)])
        self.tree.configure(style="Modern.Treeview")
        self.tree.tag_configure("over", foreground=Theme.DANGER)
        self.tree.pack(fill="both", expand=True, padx=0, pady=8)

    def refresh(self, results, limit: float):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for r in results[-100:]:
            tid = r.get("id", "?")
            spd = r.get("speed_kmh", 0)
            dr  = r.get("cross_dir", "—")
            ts  = r.get("timestamp", "")[:8]
            tag = "over" if spd > limit else ""
            self.tree.insert("", "end",
                             values=(f"#{tid}", f"{spd:.1f}", dr, ts),
                             tags=(tag,))
