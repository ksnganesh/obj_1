"""Figure styling shared by the scripts (Matplotlib, non-interactive Agg backend).

Categorical slots (fixed order, validated for colour-vision deficiency on a light surface; slots 3-4 have low contrast
against the surface, so every figure also carries a legend, direct labels or line-style/marker redundancy, and the
numbers are tabulated in results/tables.md).  Figures carry no in-figure titles (captions are in the article) and are
written both as PNG and as vector PDF (``save``):
    1 blue #2a78d6   2 orange #eb6834   3 aqua #1baf7a   4 yellow #eda100
Text uses neutral ink, never the series colour.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0, "lines.markersize": 6.5,
        "figure.dpi": 110, "savefig.dpi": 200,
    })


def label(ax, x, y, text, color=INK_2, **kw):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=kw.pop("xytext", (5, 3)), fontsize=8, color=color, **kw)


def save(fig, stem, tight: bool = True) -> None:
    """Write ``stem``.png (200 dpi) and ``stem``.pdf (vector) and close the figure."""
    from pathlib import Path
    stem = Path(stem)
    if tight:
        fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
