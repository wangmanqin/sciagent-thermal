"""
Pareto 前沿可视化。
  - 2D 前沿散点 + 连线
  - 3D 前沿散点（可选）
  - knee point / 极端点标注

统一使用 matplotlib + Agg 后端，保证无窗口环境下也能保存图像。
"""

from __future__ import annotations
from typing import List, Sequence, Tuple, Optional
import os


def _ensure_backend():
    import matplotlib
    matplotlib.use("Agg")


def plot_pareto_2d(
    all_objs: Sequence[Sequence[float]],
    pareto_objs: Sequence[Sequence[float]],
    output_path: str,
    *,
    obj_labels: Tuple[str, str] = ("Objective 1", "Objective 2"),
    title: str = "Pareto Front",
    knee_index: Optional[int] = None,
    extreme_indices: Optional[Sequence[int]] = None,
    figsize: Tuple[float, float] = (6.5, 4.8),
    dpi: int = 150,
) -> str:
    """绘制 2D Pareto 前沿：灰色散点是所有评估点，红色是前沿，可选 knee/extremes。"""
    _ensure_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    if all_objs:
        ax.scatter(
            [p[0] for p in all_objs], [p[1] for p in all_objs],
            s=12, c="lightgray", alpha=0.5, label="Evaluated",
        )
    # 排序后连线
    sorted_front = sorted(pareto_objs, key=lambda p: p[0])
    ax.plot(
        [p[0] for p in sorted_front], [p[1] for p in sorted_front],
        "-", color="crimson", lw=1.2, alpha=0.7,
    )
    ax.scatter(
        [p[0] for p in pareto_objs], [p[1] for p in pareto_objs],
        s=28, c="crimson", label="Pareto front", zorder=3,
    )

    if knee_index is not None and 0 <= knee_index < len(pareto_objs):
        kp = pareto_objs[knee_index]
        ax.scatter([kp[0]], [kp[1]], s=90, marker="*",
                   c="gold", edgecolors="black", lw=0.8,
                   label="Knee", zorder=4)

    if extreme_indices:
        ex = [pareto_objs[i] for i in extreme_indices]
        ax.scatter(
            [p[0] for p in ex], [p[1] for p in ex],
            s=60, marker="s", facecolors="none", edgecolors="navy",
            lw=1.2, label="Extremes", zorder=4,
        )

    ax.set_xlabel(obj_labels[0])
    ax.set_ylabel(obj_labels[1])
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_pareto_3d(
    pareto_objs: Sequence[Sequence[float]],
    output_path: str,
    *,
    obj_labels: Tuple[str, str, str] = ("Obj1", "Obj2", "Obj3"),
    title: str = "3D Pareto Front",
    dpi: int = 150,
) -> str:
    _ensure_backend()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    xs = [p[0] for p in pareto_objs]
    ys = [p[1] for p in pareto_objs]
    zs = [p[2] for p in pareto_objs]
    ax.scatter(xs, ys, zs, s=30, c="crimson", alpha=0.85)
    ax.set_xlabel(obj_labels[0])
    ax.set_ylabel(obj_labels[1])
    ax.set_zlabel(obj_labels[2])
    ax.set_title(title)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_parallel_coordinates(
    pareto_vars: Sequence[Sequence[float]],
    output_path: str,
    *,
    var_labels: Optional[Sequence[str]] = None,
    title: str = "Design Space (Parallel Coordinates)",
    dpi: int = 150,
) -> str:
    """用平行坐标图展示 Pareto 前沿的设计变量分布。"""
    _ensure_backend()
    import matplotlib.pyplot as plt

    if not pareto_vars:
        raise ValueError("pareto_vars 不能为空")
    n_dim = len(pareto_vars[0])
    if var_labels is None:
        var_labels = [f"x{i+1}" for i in range(n_dim)]

    # 归一化到 [0,1]
    mins = [min(p[j] for p in pareto_vars) for j in range(n_dim)]
    maxs = [max(p[j] for p in pareto_vars) for j in range(n_dim)]
    normed = [
        [(p[j] - mins[j]) / (maxs[j] - mins[j]) if maxs[j] > mins[j] else 0.5
         for j in range(n_dim)]
        for p in pareto_vars
    ]

    fig, ax = plt.subplots(figsize=(max(6, n_dim * 1.2), 4.5))
    xs = list(range(n_dim))
    for row in normed:
        ax.plot(xs, row, color="steelblue", alpha=0.4, lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(var_labels)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Normalized value")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
