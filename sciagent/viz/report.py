"""
多面板报告图：把 Pareto 前沿 + 收敛曲线 + 参数分布组合成一张"优化结果一览"。
"""

from __future__ import annotations
from typing import Sequence, Tuple, Optional, List
import os


def _ensure_backend():
    import matplotlib
    matplotlib.use("Agg")


def build_optimization_report(
    output_path: str,
    *,
    all_objs: Sequence[Sequence[float]],
    pareto_objs: Sequence[Sequence[float]],
    pareto_vars: Sequence[Sequence[float]],
    obj_labels: Tuple[str, str] = ("Objective 1", "Objective 2"),
    var_labels: Optional[Sequence[str]] = None,
    hv_history: Optional[Sequence[float]] = None,
    knee_index: Optional[int] = None,
    title: str = "NSGA-II Optimization Report",
    dpi: int = 150,
) -> str:
    """2x2 面板：左上 Pareto、右上 HV 收敛、左下 平行坐标、右下 变量分布箱线"""
    _ensure_backend()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle(title, fontsize=13, y=0.995)

    # 1) Pareto 前沿
    ax = axes[0, 0]
    if all_objs:
        ax.scatter(
            [p[0] for p in all_objs], [p[1] for p in all_objs],
            s=10, c="lightgray", alpha=0.5, label="Evaluated",
        )
    sorted_front = sorted(pareto_objs, key=lambda p: p[0])
    ax.plot([p[0] for p in sorted_front],
            [p[1] for p in sorted_front], "-", color="crimson", lw=1.0, alpha=0.7)
    ax.scatter([p[0] for p in pareto_objs], [p[1] for p in pareto_objs],
               s=22, c="crimson", label="Pareto")
    if knee_index is not None and 0 <= knee_index < len(pareto_objs):
        kp = pareto_objs[knee_index]
        ax.scatter([kp[0]], [kp[1]], s=90, marker="*",
                   c="gold", edgecolors="black", label="Knee", zorder=4)
    ax.set_xlabel(obj_labels[0])
    ax.set_ylabel(obj_labels[1])
    ax.set_title("Pareto Front")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    # 2) HV 收敛
    ax = axes[0, 1]
    if hv_history:
        xs = list(range(len(hv_history)))
        ax.plot(xs, list(hv_history), "-", color="darkgreen", lw=1.5)
        ax.fill_between(xs, 0, list(hv_history), color="darkgreen", alpha=0.15)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Hypervolume")
    else:
        ax.text(0.5, 0.5, "No HV history", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_title("Hypervolume History")
    ax.grid(True, ls=":", alpha=0.5)

    # 3) 平行坐标（归一化后）
    ax = axes[1, 0]
    if pareto_vars:
        n_dim = len(pareto_vars[0])
        labels = list(var_labels) if var_labels else [f"x{i+1}" for i in range(n_dim)]
        mins = [min(p[j] for p in pareto_vars) for j in range(n_dim)]
        maxs = [max(p[j] for p in pareto_vars) for j in range(n_dim)]
        normed = [
            [(p[j] - mins[j]) / (maxs[j] - mins[j]) if maxs[j] > mins[j] else 0.5
             for j in range(n_dim)]
            for p in pareto_vars
        ]
        xs = list(range(n_dim))
        for row in normed:
            ax.plot(xs, row, color="steelblue", alpha=0.35, lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Normalized")
    ax.set_title("Design Space")
    ax.grid(True, ls=":", alpha=0.5)

    # 4) 变量箱线
    ax = axes[1, 1]
    if pareto_vars:
        n_dim = len(pareto_vars[0])
        labels = list(var_labels) if var_labels else [f"x{i+1}" for i in range(n_dim)]
        by_dim = [[p[j] for p in pareto_vars] for j in range(n_dim)]
        ax.boxplot(by_dim, labels=labels, showmeans=True)
        ax.tick_params(axis="x", rotation=20)
    ax.set_title("Variable Distribution (Pareto)")
    ax.grid(True, ls=":", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
