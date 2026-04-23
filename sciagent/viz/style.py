"""
统一样式 / 配色，给论文风格绘图做兜底。
Agent 调用时可以 import apply_paper_style() 一键切换。
"""

from __future__ import annotations
from typing import Dict


PAPER_STYLE: Dict[str, object] = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

SLIDE_STYLE: Dict[str, object] = {
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "lines.linewidth": 1.8,
    "legend.frameon": True,
}

ENG_PALETTE = [
    "#0072B5",  # blue
    "#E18727",  # orange
    "#20854E",  # green
    "#BC3C29",  # red
    "#7876B1",  # purple
    "#6F99AD",  # steel
    "#FFDC91",  # yellow
    "#EE4C97",  # pink
]


def apply_paper_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_STYLE)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=ENG_PALETTE)


def apply_slide_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(SLIDE_STYLE)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=ENG_PALETTE)


def reset_style():
    import matplotlib.pyplot as plt
    plt.rcdefaults()
