"""10 道微通道散热 benchmark 题目。

每题是一个 Task 对象，包含：
  - id
  - category
  - question (喂给 Agent 的自然语言)
  - reference_answer (数值或键值答案，用于自动评分)
  - tolerance (允许误差百分比)
  - required_tools (ReAct log 里应当出现的工具名)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


@dataclass
class Task:
    id: int
    category: str
    question: str
    reference_answer: Dict[str, float]
    tolerance: float = 0.05  # 默认 5%
    required_tools: List[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# 参考答案的来源都是独立手算 + 交叉验证（不是 Agent 自己生成再回放）
# -----------------------------------------------------------------------------

TASKS: List[Task] = [
    Task(
        id=1,
        category="basic_heat_transfer",
        question=(
            "一个 1 cm² 的 CPU 芯片发热 80 W，用 20 条矩形水冷通道散热，"
            "通道截面 1mm × 2mm，长度 2cm，水以 1 L/min 的总流量进入，"
            "入口水温 25°C。请估算进出口水温差 ΔT。"
        ),
        reference_answer={"delta_T_C": 1.15},
        tolerance=0.10,
        required_tools=["water_properties", "caloric_resistance"],
        notes="能量守恒：Q = m·cp·ΔT，水的 cp≈4180，m=1/60 kg/s",
    ),
    Task(
        id=2,
        category="thermal_network_full",
        question=(
            "沿用题 1 的几何：硅衬底厚 0.5mm，通道底面全部被硅覆盖。"
            "水 25°C 进、流量 2 L/min，算对流换热系数 h 并估算最高结温。"
        ),
        reference_answer={"h_W_per_m2K": 12000.0, "T_max_C": 45.0},
        tolerance=0.20,
        required_tools=["water_properties", "shah_london",
                        "conduction_resistance_plane"],
        notes="矩形通道 aspect_ratio=0.5，Shah-London 给 Nu ≈ 3.39",
    ),
    Task(
        id=3,
        category="non_circular_channel",
        question=(
            "换成梯形通道：上宽 1mm、下宽 0.5mm、高 2mm。请算水力直径 "
            "Dh，说明用什么关联式算 Nu 比较合适。"
        ),
        reference_answer={"hydraulic_diameter_m": 1.2e-3},
        tolerance=0.10,
        required_tools=["trapezoidal_cross_section"],
        notes="Dh = 4A/P，A=(1+0.5)/2 * 2 = 1.5 mm², P 要用斜边",
    ),
    Task(
        id=4,
        category="parameter_sweep_plot",
        question=(
            "沿用题 2 的几何。请把总流量从 0.5 L/min 扫到 5 L/min（10 点），"
            "算每个流量下的 Re、Nu、h，并画出 h-flow 曲线。"
        ),
        reference_answer={"h_at_5_Lpm_W_per_m2K": 30000.0},
        tolerance=0.30,
        required_tools=["water_properties", "shah_london", "save_xy_plot"],
        notes="流量大时进入湍流区需要 Dittus-Boelter 或 Gnielinski",
    ),
    Task(
        id=5,
        category="pressure_drop_full",
        question=(
            "沿用题 2 的几何，流量 3 L/min。硅壁粗糙度约 1 μm。"
            "请计算通道沿程、入口突缩、出口突扩的总压降，并估算所需泵功。"
        ),
        reference_answer={"total_dp_Pa": 4500.0, "pump_power_W": 0.225},
        tolerance=0.30,
        required_tools=["colebrook", "darcy_weisbach", "minor_loss",
                        "pump_power"],
    ),
    Task(
        id=6,
        category="multi_objective_optim",
        question=(
            "以题 5 为基础，做换热（最小化结温）与压降（最小化泵功）的"
            "Pareto 优化。变量：n_channels ∈ [10,40]、通道宽 [0.3,1.5] mm、"
            "通道高 [1,3] mm、流量 [0.5,4] L/min。用 NSGA-II。"
        ),
        reference_answer={"pareto_points_min": 8},
        tolerance=0.0,  # 用 min 值比较，不走百分比
        required_tools=["run_nsga2"],
        notes="代表解的 knee 点应在 T_max 35-40°C、泵功 0.1-0.5 W 之间",
    ),
    Task(
        id=7,
        category="alternative_fluid",
        question=(
            "把题 1 的水换成 40% 乙二醇水溶液。其它条件不变，"
            "重新算进出口温差。"
        ),
        reference_answer={"delta_T_C": 1.55},
        tolerance=0.15,
        required_tools=["ethylene_glycol_properties"],
        notes="40% EG 的 cp 约 3500 J/(kg·K)",
    ),
    Task(
        id=8,
        category="nanofluid",
        question=(
            "把题 1 的水换成 2% 体积分数的 Al₂O₃ 纳米流体（k 用 Maxwell，"
            "μ 用 Einstein）。估算 h 的提升比例。"
        ),
        reference_answer={"k_ratio": 1.06, "mu_ratio": 1.05},
        tolerance=0.15,
        required_tools=["water_properties", "nanofluid_properties"],
    ),
    Task(
        id=9,
        category="fd_1d_conduction",
        question=(
            "一块 1 cm 厚的硅衬底，底部均匀加热 80 W/cm²，顶部对流 "
            "h=5e4 W/m²K、T_∞=25°C，求稳态 1D 温度分布（20 个单元）。"
        ),
        reference_answer={"T_bottom_C": 80.0, "T_top_C": 60.0},
        tolerance=0.10,
        required_tools=["run_python_code"],
        notes="调 solve_1d_conduction_dirichlet",
    ),
    Task(
        id=10,
        category="design_report",
        question=(
            "综合题 1-6，写一份 Markdown 格式的"
            "微通道散热器设计报告，包含参数、性能、权衡点（knee）、"
            "推荐设计方案。"
        ),
        reference_answer={"has_pareto_table": 1.0, "has_recommended_design": 1.0},
        tolerance=0.0,
        required_tools=["run_nsga2", "run_python_code"],
        notes="最后一题本质是"组合能力"考察",
    ),
]


def get_task(task_id: int) -> Task:
    for t in TASKS:
        if t.id == task_id:
            return t
    raise KeyError(f"task id={task_id} 不存在")


def all_tasks() -> List[Task]:
    return list(TASKS)


def dump_tasks_json(path: str) -> str:
    import json, os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([t.as_dict() for t in TASKS], f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    print(f"{len(TASKS)} tasks defined.")
    for t in TASKS:
        print(f"  {t.id:2d} | {t.category:30s} | tools={len(t.required_tools)}")
