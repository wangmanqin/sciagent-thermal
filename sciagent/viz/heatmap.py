"""
热力图 / 等值线：用于一维/二维温度场、压力场、速度场可视化。
"""

from __future__ import annotations
from typing import Sequence, Tuple, Optional, List
import os


def _ensure_backend():
    import matplotlib
    matplotlib.use("Agg")


def plot_1d_profile(
    x: Sequence[float],
    y: Sequence[float],
    output_path: str,
    *,
    xlabel: str = "x (m)",
    ylabel: str = "T (°C)",
    title: str = "1D Profile",
    color: str = "crimson",
    figsize: Tuple[float, float] = (6.5, 4.0),
    dpi: int = 150,
) -> str:
    _ensure_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(list(x), list(y), "-", color=color, lw=1.6)
    ax.scatter(list(x), list(y), s=14, color=color, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_2d_heatmap(
    matrix: Sequence[Sequence[float]],
    output_path: str,
    *,
    extent: Optional[Tuple[float, float, float, float]] = None,
    cmap: str = "inferno",
    title: str = "2D Field",
    cbar_label: str = "Value",
    figsize: Tuple[float, float] = (6.5, 5.0),
    dpi: int = 150,
) -> str:
    """matrix[i][j] 画成颜色图；extent = (x_min, x_max, y_min, y_max)。"""
    _ensure_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        list(matrix),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=extent,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_title(title)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_contour(
    x: Sequence[float],
    y: Sequence[float],
    Z: Sequence[Sequence[float]],
    output_path: str,
    *,
    levels: int = 15,
    cmap: str = "viridis",
    title: str = "Contour",
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: Tuple[float, float] = (6.5, 5.0),
    dpi: int = 150,
) -> str:
    _ensure_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    cs = ax.contourf(list(x), list(y), list(Z), levels=levels, cmap=cmap)
    fig.colorbar(cs, ax=ax)
    cl = ax.contour(list(x), list(y), list(Z), levels=levels, colors="black",
                    linewidths=0.4, alpha=0.4)
    ax.clabel(cl, inline=True, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
