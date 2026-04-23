"""viz 子包：优化 / 场 / 收敛可视化。

所有绘图函数都会：
  - 强制 Agg 后端（保证无窗口环境可用）
  - 自动 mkdir -p 输出目录
  - 返回 output_path 作为凭证
"""

from sciagent.viz.pareto_plot import (
    plot_pareto_2d,
    plot_pareto_3d,
    plot_parallel_coordinates,
)
from sciagent.viz.convergence import (
    plot_convergence,
    plot_multi_convergence,
    plot_hypervolume_history,
)
from sciagent.viz.heatmap import (
    plot_1d_profile,
    plot_2d_heatmap,
    plot_contour,
)
from sciagent.viz.report import build_optimization_report
from sciagent.viz.style import (
    apply_paper_style,
    apply_slide_style,
    reset_style,
    PAPER_STYLE,
    SLIDE_STYLE,
    ENG_PALETTE,
)

__all__ = [
    "plot_pareto_2d", "plot_pareto_3d", "plot_parallel_coordinates",
    "plot_convergence", "plot_multi_convergence", "plot_hypervolume_history",
    "plot_1d_profile", "plot_2d_heatmap", "plot_contour",
    "build_optimization_report",
    "apply_paper_style", "apply_slide_style", "reset_style",
    "PAPER_STYLE", "SLIDE_STYLE", "ENG_PALETTE",
]
