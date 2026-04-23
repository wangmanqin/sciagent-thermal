"""
收敛曲线可视化：给单目标/多目标优化的历史数据画图。
"""

from __future__ import annotations
from typing import Sequence, Optional, List, Tuple
import os


def _ensure_backend():
    import matplotlib
    matplotlib.use("Agg")


def plot_convergence(
    history: Sequence[float],
    output_path: str,
    *,
    title: str = "Convergence",
    ylabel: str = "Objective",
    xlabel: str = "Iteration",
    log_y: bool = False,
    figsize: Tuple[float, float] = (6.0, 4.0),
    dpi: int = 150,
) -> str:
    _ensure_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    xs = list(range(len(history)))
    ax.plot(xs, list(history), "-", color="steelblue", lw=1.4)
    ax.scatter(xs, list(history), s=10, color="navy", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, ls=":", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_multi_convergence(
    histories: dict,
    output_path: str,
    *,
    title: str = "Algorithm Comparison",
    ylabel: str = "Objective",
    xlabel: str = "Iteration",
    figsize: Tuple[float, float] = (6.5, 4.2),
    dpi: int = 150,
) -> str:
    """比较多个优化器的收敛历史。histories: {"DE": [...], "PSO": [...], ...}"""
    _ensure_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for name, hist in histories.items():
        ax.plot(list(range(len(hist))), list(hist), "-", lw=1.3, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_hypervolume_history(
    hv_history: Sequence[float],
    output_path: str,
    *,
    title: str = "Hypervolume Over Generations",
    figsize: Tuple[float, float] = (6.0, 4.0),
    dpi: int = 150,
) -> str:
    _ensure_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    xs = list(range(len(hv_history)))
    ax.plot(xs, list(hv_history), "-", color="darkgreen", lw=1.5)
    ax.fill_between(xs, 0, list(hv_history), color="darkgreen", alpha=0.15)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
