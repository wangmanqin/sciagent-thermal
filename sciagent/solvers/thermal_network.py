"""
一维热阻网络求解器。

散热器的总热阻 = 导热热阻 + 对流热阻 + 焦耳热阻（流体升温段）
这里把串联/并联热阻封装成一个小 "RC 网络"，Agent 只要把每段热阻
传进来就行，不用自己手算 sum。

用法：
    net = ResistanceNetwork([
        Resistance("base_conduction", 0.01, kind="series"),
        Resistance("wall_convection", 0.04, kind="series"),
        Resistance("caloric_rise", 0.02, kind="series"),
    ])
    net.total()  # 0.07 K/W
    net.temperature_rise(heat_flux_W=100)  # 7.0 K
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class Resistance:
    name: str
    value_K_per_W: float
    kind: Literal["series", "parallel_group"] = "series"
    group_id: Optional[str] = None

    def __post_init__(self):
        if self.value_K_per_W < 0:
            raise ValueError(f"热阻 {self.name} 必须非负")


@dataclass
class ResistanceNetwork:
    resistances: List[Resistance] = field(default_factory=list)

    def add(self, *resistances: Resistance) -> "ResistanceNetwork":
        self.resistances.extend(resistances)
        return self

    def total(self) -> float:
        # 分组：串联 vs 并联组
        series_sum = 0.0
        groups: dict[str, list[float]] = {}
        for r in self.resistances:
            if r.kind == "series":
                series_sum += r.value_K_per_W
            elif r.kind == "parallel_group":
                if r.group_id is None:
                    raise ValueError(f"并联热阻 {r.name} 缺少 group_id")
                groups.setdefault(r.group_id, []).append(r.value_K_per_W)
            else:
                raise ValueError(f"未知 kind '{r.kind}'")

        # 每组并联求 1/sum(1/Ri)
        parallel_sum = 0.0
        for gid, vals in groups.items():
            inv = sum(1.0 / v for v in vals if v > 0)
            if inv == 0:
                raise ValueError(f"并联组 '{gid}' 所有热阻为 0")
            parallel_sum += 1.0 / inv

        return series_sum + parallel_sum

    def temperature_rise(self, heat_flux_W: float) -> float:
        return self.total() * heat_flux_W

    def summary(self) -> dict:
        return {
            "total_K_per_W": self.total(),
            "components": [
                {"name": r.name,
                 "R_K_per_W": r.value_K_per_W,
                 "kind": r.kind,
                 "group_id": r.group_id}
                for r in self.resistances
            ],
        }


# ---------------------------------------------------------------------------
# 常见热阻计算小工具
# ---------------------------------------------------------------------------

def conduction_resistance_plane(
    thickness_m: float, area_m2: float, k_W_per_mK: float,
) -> dict:
    if min(thickness_m, area_m2, k_W_per_mK) <= 0:
        raise ValueError("参数必须为正")
    R = thickness_m / (k_W_per_mK * area_m2)
    return {"R_K_per_W": R, "thickness_m": thickness_m,
            "area_m2": area_m2, "k_W_per_mK": k_W_per_mK}


def conduction_resistance_cylinder(
    r_inner: float, r_outer: float, length_m: float, k_W_per_mK: float,
) -> dict:
    import math
    if r_outer <= r_inner or length_m <= 0 or k_W_per_mK <= 0:
        raise ValueError("参数非法")
    R = math.log(r_outer / r_inner) / (2 * math.pi * k_W_per_mK * length_m)
    return {"R_K_per_W": R, "r_inner": r_inner, "r_outer": r_outer,
            "length_m": length_m, "k_W_per_mK": k_W_per_mK}


def convection_resistance(
    h_W_per_m2K: float, area_m2: float,
) -> dict:
    if h_W_per_m2K <= 0 or area_m2 <= 0:
        raise ValueError("h 和 area 必须为正")
    R = 1.0 / (h_W_per_m2K * area_m2)
    return {"R_K_per_W": R, "h_W_per_m2K": h_W_per_m2K, "area_m2": area_m2}


def caloric_resistance(
    mass_flow_kg_per_s: float, cp_J_per_kgK: float,
) -> dict:
    """热容流量热阻：fluid 从入口到出口的温升热阻 = 1/(m·cp)"""
    if mass_flow_kg_per_s <= 0 or cp_J_per_kgK <= 0:
        raise ValueError("参数必须为正")
    R = 1.0 / (mass_flow_kg_per_s * cp_J_per_kgK)
    return {"R_K_per_W": R,
            "mass_flow_kg_per_s": mass_flow_kg_per_s,
            "cp_J_per_kgK": cp_J_per_kgK}


# ---------------------------------------------------------------------------
# 工具注册（可选，给 Agent 用）
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "conduction_resistance_plane",
        "description": "平板导热热阻 R = t / (k·A)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "thickness_m": {"type": "number"},
                "area_m2": {"type": "number"},
                "k_W_per_mK": {"type": "number"},
            },
            "required": ["thickness_m", "area_m2", "k_W_per_mK"],
        },
    },
    {
        "name": "conduction_resistance_cylinder",
        "description": "圆柱壁面导热热阻 R = ln(r_o/r_i) / (2πkL)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "r_inner": {"type": "number"},
                "r_outer": {"type": "number"},
                "length_m": {"type": "number"},
                "k_W_per_mK": {"type": "number"},
            },
            "required": ["r_inner", "r_outer", "length_m", "k_W_per_mK"],
        },
    },
    {
        "name": "convection_resistance",
        "description": "对流热阻 R = 1 / (h·A)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "h_W_per_m2K": {"type": "number"},
                "area_m2": {"type": "number"},
            },
            "required": ["h_W_per_m2K", "area_m2"],
        },
    },
    {
        "name": "caloric_resistance",
        "description": "焦耳/热容热阻 R = 1 / (m·cp)，用于流体温升。",
        "input_schema": {
            "type": "object",
            "properties": {
                "mass_flow_kg_per_s": {"type": "number"},
                "cp_J_per_kgK": {"type": "number"},
            },
            "required": ["mass_flow_kg_per_s", "cp_J_per_kgK"],
        },
    },
]


def _wrap(fn):
    def _exec(args):
        return json.dumps(fn(**args), ensure_ascii=False, indent=2)
    return _exec


TOOL_EXECUTORS = {
    "conduction_resistance_plane": _wrap(conduction_resistance_plane),
    "conduction_resistance_cylinder": _wrap(conduction_resistance_cylinder),
    "convection_resistance": _wrap(convection_resistance),
    "caloric_resistance": _wrap(caloric_resistance),
}
