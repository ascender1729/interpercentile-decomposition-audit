# -*- coding: utf-8 -*-
"""Venue-standard figure style for ICML / ICLR / NeurIPS submissions.
Import and call apply() before plotting. Use save() to emit vector PDF + 300dpi PNG
with fonts embedded (Type42) so the PDF passes camera-ready font checks.

Standards enforced:
- serif family matching LaTeX body text; 8pt base, 7pt ticks (NeurIPS/ICML body size)
- Okabe-Ito colourblind-safe palette
- PDF/PS fonttype 42 (TrueType embedded) - required by most camera-ready checkers
- no top/right spines, light y-grid only, no chartjunk
- figure widths: COL single-column 3.25in, WIDE double-column 5.5in (NeurIPS text width)
- captions do NOT go in the image (they go in the LaTeX \\caption); save() writes a .txt caption instead
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE_ITO = {
    "blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "pink": "#CC79A7",
    "yellow": "#E69F00", "sky": "#56B4E9", "vermillion": "#E41A1C", "grey": "#999999",
}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
COL, WIDE = 3.25, 5.5   # inches


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42,      # embed TrueType, camera-ready safe
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.grid": True, "axes.grid.axis": "y", "grid.linewidth": 0.4,
        "grid.alpha": 0.35, "grid.color": "#B0B0B0",
        "legend.frameon": False, "figure.dpi": 120,
    })


def finish(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, stem, caption):
    """Emit stem.pdf (vector) + stem.png (300dpi) + stem.caption.txt (for the LaTeX \\caption)."""
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    with open(stem + ".caption.txt", "w", encoding="utf-8") as f:
        f.write(caption.strip() + "\n")


def legend_above(ax, ncol, y=1.02):
    """Place the legend ABOVE the axes, spanning full width - never overlaps bars.
    Standard for grouped bar charts in ICML/ICLR/NeurIPS figures."""
    return ax.legend(loc="lower left", bbox_to_anchor=(0, y, 1, 0.12),
                     mode="expand", ncol=ncol, borderaxespad=0, frameon=False,
                     handlelength=1.2, columnspacing=1.4)
