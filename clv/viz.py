"""One visual style for every chart in the project.

Categorical colours follow a fixed, colourblind checked order (blue, orange,
aqua, yellow). Magnitude uses one blue ramp, light to dark. Text never takes a
series colour, grids stay faint, and every chart with two or more series gets
a legend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#b9b8b2"
HIGHLIGHT = SERIES[1]
RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("blues", RAMP)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "legend.frameon": False,
    "legend.labelcolor": TEXT_2,
    "lines.linewidth": 2,
})


def _pct(v, _=None):
    x = round(v * 100, 6)
    return f"{x:.0f}%" if float(x).is_integer() else f"{x:.1f}%"


pct = FuncFormatter(_pct)


def _gbp(v, _=None):
    a = abs(v)
    sign = "-" if v < 0 and round(a) > 0 else ""
    if a >= 1e6:
        return f"{sign}£{a / 1e6:,.1f}M"
    if a >= 1000:
        return f"{sign}£{a / 1000:,.0f}k"
    return f"{sign}£{a:,.0f}"


def money(v: float) -> str:
    """Full pounds for labels, minus sign in front of the symbol."""
    return f"-£{abs(v):,.0f}" if v < 0 and round(abs(v)) > 0 else f"£{abs(v):,.0f}"


gbp = FuncFormatter(_gbp)


def legend_top(ax, ncols: int, **kw) -> None:
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncols=ncols, borderaxespad=0,
              handletextpad=0.4, columnspacing=1.4, **kw)


def subtitle(ax, text: str, above_legend: bool = False) -> None:
    ax.text(0, 1.085 if above_legend else 1.015, text, transform=ax.transAxes, color=TEXT_2,
            fontsize=9.5, va="bottom")


def footnote(fig, text: str) -> None:
    fig.text(0.01, 0.005, text, color=TEXT_2, fontsize=8, va="bottom")


def save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
