"""optim 子包：多目标 / 单目标优化 + Pareto 前沿分析。

设计思路：
  - NSGA-II 主打多目标（Pareto 前沿，适合"换热 vs 压降"这种博弈）
  - DE / PSO / SA 做单目标兜底
  - pareto.py 提供前沿分析的公共工具（非支配排序、超体积、knee）
"""

from sciagent.optim.nsga2 import run_nsga2, NSGA2Result
from sciagent.optim.single_objective import (
    differential_evolution,
    particle_swarm,
    simulated_annealing,
    OptimResult,
)
from sciagent.optim.pareto import (
    dominates,
    non_dominated_sort,
    pareto_front_indices,
    hypervolume_2d,
    hypervolume_monte_carlo,
    spacing_metric,
    pick_knee_point,
    pick_extremes,
    representative_solutions,
)

__all__ = [
    # NSGA-II
    "run_nsga2", "NSGA2Result",
    # 单目标
    "differential_evolution", "particle_swarm", "simulated_annealing",
    "OptimResult",
    # Pareto 工具
    "dominates", "non_dominated_sort", "pareto_front_indices",
    "hypervolume_2d", "hypervolume_monte_carlo", "spacing_metric",
    "pick_knee_point", "pick_extremes", "representative_solutions",
]
