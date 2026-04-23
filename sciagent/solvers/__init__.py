"""solvers 子包：热阻网络 / ODE / 线性代数。

这些求解器暴露两种接口：
  - Python 层：直接 import 使用
  - 工具层：通过 TOOL_DEFINITIONS / TOOL_EXECUTORS 给 Agent 调用
"""

from sciagent.solvers.thermal_network import (
    Resistance, ResistanceNetwork,
    conduction_resistance_plane, conduction_resistance_cylinder,
    convection_resistance, caloric_resistance,
    TOOL_DEFINITIONS as _TN_TOOLS,
    TOOL_EXECUTORS as _TN_EXEC,
)

from sciagent.solvers.ode import (
    rk4_step, solve_ode_rk4, solve_ode_rk45,
    fin_temperature_distribution,
)

from sciagent.solvers.linalg import (
    thomas, lu_decompose, lu_solve, solve_linear_system,
    norm_2, norm_inf, vec_sub, mat_vec_mul,
    solve_1d_conduction_dirichlet,
)

TOOL_DEFINITIONS = list(_TN_TOOLS)
TOOL_EXECUTORS = dict(_TN_EXEC)

__all__ = [
    "Resistance", "ResistanceNetwork",
    "conduction_resistance_plane", "conduction_resistance_cylinder",
    "convection_resistance", "caloric_resistance",
    "rk4_step", "solve_ode_rk4", "solve_ode_rk45",
    "fin_temperature_distribution",
    "thomas", "lu_decompose", "lu_solve", "solve_linear_system",
    "norm_2", "norm_inf", "vec_sub", "mat_vec_mul",
    "solve_1d_conduction_dirichlet",
    "TOOL_DEFINITIONS", "TOOL_EXECUTORS",
]
