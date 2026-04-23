"""
压降计算工具：摩擦损失 + 入口/出口/弯头局部损失 + 两相乘子。

覆盖工程里最常用的几条路径：
  - Darcy-Weisbach：dP = f * (L/Dh) * 0.5 * rho * v^2
  - 摩擦因子：层流 f=64/Re，过渡/湍流用 Colebrook（迭代）或 Swamee-Jain 显式解
  - 局部损失：K 值表（渐缩、渐扩、弯头、阀门、T 型……）
  - 两相：Chisholm B 系数，Martinelli 参数
  - 泵功率：P = dP * Q / eta
所有函数返回 dict，便于工具链接入。
"""

from __future__ import annotations
import math
import json
from typing import Optional


# ---------------------------------------------------------------------------
# 1) 层流摩擦因子 Darcy friction factor
# ---------------------------------------------------------------------------

def laminar_friction_factor(Re: float) -> dict:
    if Re <= 0:
        raise ValueError("Re 必须为正")
    f = 64.0 / Re
    ok = Re < 2300
    return {"friction_factor_darcy": f,
            "Re": Re,
            "applicability_ok": ok,
            "notes": "" if ok else "Re > 2300，不再是层流。"}


# ---------------------------------------------------------------------------
# 2) Colebrook-White（隐式）— 粗糙管湍流摩擦因子
#     1/sqrt(f) = -2 log10(eps/(3.7*D) + 2.51/(Re sqrt(f)))
#    用不动点迭代求解。
# ---------------------------------------------------------------------------

def colebrook(
    Re: float, roughness_m: float, diameter_m: float,
    tol: float = 1e-8, max_iter: int = 200,
) -> dict:
    if Re <= 0 or diameter_m <= 0 or roughness_m < 0:
        raise ValueError("Re, diameter 必须为正，roughness 必须非负")
    if Re < 4000:
        raise ValueError("Colebrook 仅适用湍流 Re > 4000")

    eps_rel = roughness_m / diameter_m
    # 以 Swamee-Jain 作初值
    f = (-2.0 * math.log10(eps_rel / 3.7 + 5.74 / Re ** 0.9)) ** -2
    for i in range(max_iter):
        lhs = 1.0 / math.sqrt(f)
        rhs = -2.0 * math.log10(eps_rel / 3.7 + 2.51 / (Re * math.sqrt(f)))
        if abs(lhs - rhs) < tol:
            break
        f_new = 1.0 / rhs ** 2
        if abs(f_new - f) < tol:
            f = f_new
            break
        f = 0.5 * (f + f_new)
    return {
        "friction_factor_darcy": f,
        "Re": Re,
        "relative_roughness": eps_rel,
        "iterations": i + 1,
    }


def swamee_jain(
    Re: float, roughness_m: float, diameter_m: float,
) -> dict:
    """显式拟合 Colebrook，精度 ±1%，覆盖 5e3<Re<1e8, 1e-6<eps/D<1e-2。"""
    if Re <= 0 or diameter_m <= 0 or roughness_m < 0:
        raise ValueError("参数非法")
    eps_rel = roughness_m / diameter_m
    f = 0.25 / math.log10(eps_rel / 3.7 + 5.74 / Re ** 0.9) ** 2
    return {"friction_factor_darcy": f,
            "Re": Re,
            "relative_roughness": eps_rel}


# ---------------------------------------------------------------------------
# 3) Darcy-Weisbach 压降
# ---------------------------------------------------------------------------

def darcy_weisbach(
    friction_factor: float,
    length_m: float,
    hydraulic_diameter_m: float,
    density: float,
    velocity: float,
) -> dict:
    if min(length_m, hydraulic_diameter_m, density, velocity) <= 0:
        raise ValueError("参数必须为正")
    dP = friction_factor * (length_m / hydraulic_diameter_m) * 0.5 * density * velocity ** 2
    return {
        "pressure_drop_Pa": dP,
        "dynamic_head_Pa": 0.5 * density * velocity ** 2,
        "L_over_D": length_m / hydraulic_diameter_m,
        "friction_factor": friction_factor,
    }


# ---------------------------------------------------------------------------
# 4) 局部损失 K 值表 + 合成
# ---------------------------------------------------------------------------

MINOR_LOSS_K = {
    "entrance_sharp": 0.5,
    "entrance_rounded": 0.05,
    "entrance_reentrant": 1.0,
    "exit_sharp": 1.0,
    "elbow_90_sharp": 1.5,
    "elbow_90_smooth": 0.3,
    "elbow_45": 0.35,
    "tee_flow_through": 0.4,
    "tee_branch": 1.0,
    "gate_valve_open": 0.15,
    "ball_valve_open": 0.05,
    "globe_valve_open": 10.0,
    "check_valve_swing": 2.0,
    "sudden_contraction_half": 0.3,   # 面积比 0.5
    "sudden_expansion_half": 0.25,    # 面积比 0.5，由 Borda-Carnot
}


def minor_loss(
    components: list,
    density: float,
    velocity: float,
) -> dict:
    """
    components: List[str] 或 List[Tuple[str, int]] — 部件名与数量
    """
    if density <= 0 or velocity <= 0:
        raise ValueError("density 和 velocity 必须为正")

    total_K = 0.0
    detail = {}
    for item in components:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            name, count = item
        else:
            name, count = item, 1
        if name not in MINOR_LOSS_K:
            raise ValueError(
                f"未知部件 '{name}'，已知项：{sorted(MINOR_LOSS_K.keys())}"
            )
        K = MINOR_LOSS_K[name] * count
        total_K += K
        detail[name] = {"K": MINOR_LOSS_K[name], "count": count}

    dP = total_K * 0.5 * density * velocity ** 2
    return {
        "total_K": total_K,
        "pressure_drop_Pa": dP,
        "dynamic_head_Pa": 0.5 * density * velocity ** 2,
        "components": detail,
    }


# ---------------------------------------------------------------------------
# 5) Borda-Carnot 突然扩张/收缩
# ---------------------------------------------------------------------------

def borda_carnot_expansion(
    A1_m2: float, A2_m2: float, density: float, velocity1: float,
) -> dict:
    if min(A1_m2, A2_m2, density, velocity1) <= 0:
        raise ValueError("参数必须为正")
    if A2_m2 <= A1_m2:
        raise ValueError("突然扩张要求 A2 > A1")
    v2 = velocity1 * A1_m2 / A2_m2
    K = (1.0 - A1_m2 / A2_m2) ** 2
    dP = K * 0.5 * density * velocity1 ** 2
    return {
        "K": K,
        "pressure_drop_Pa": dP,
        "velocity_upstream": velocity1,
        "velocity_downstream": v2,
    }


def sudden_contraction(
    A1_m2: float, A2_m2: float, density: float, velocity2: float,
) -> dict:
    if A2_m2 >= A1_m2:
        raise ValueError("突然收缩要求 A1 > A2")
    beta = A2_m2 / A1_m2
    # Weisbach 拟合：K_c ≈ 0.5 (1 - beta)
    K = 0.5 * (1.0 - beta)
    dP = K * 0.5 * density * velocity2 ** 2
    return {"K": K, "pressure_drop_Pa": dP, "area_ratio": beta}


# ---------------------------------------------------------------------------
# 6) 两相 — Lockhart-Martinelli + Chisholm
#    dP_tp = phi_L^2 * dP_L（液相单独流时的压降）
#    phi_L^2 = 1 + C/X + 1/X^2,  X = sqrt(dP_L / dP_G)
#    C 取决于两相流型（turb-turb: 20; lam-lam: 5; lam-turb: 10; turb-lam: 12）
# ---------------------------------------------------------------------------

def lockhart_martinelli(
    dP_liquid_alone_Pa: float,
    dP_gas_alone_Pa: float,
    flow_regime: str = "turb-turb",
) -> dict:
    if dP_liquid_alone_Pa <= 0 or dP_gas_alone_Pa <= 0:
        raise ValueError("单相压降必须为正")
    C_map = {
        "turb-turb": 20,
        "lam-lam": 5,
        "lam-turb": 10,
        "turb-lam": 12,
    }
    if flow_regime not in C_map:
        raise ValueError(f"flow_regime 必须是 {list(C_map)}")
    C = C_map[flow_regime]

    X = math.sqrt(dP_liquid_alone_Pa / dP_gas_alone_Pa)
    phi_L_sq = 1.0 + C / X + 1.0 / X ** 2
    phi_G_sq = 1.0 + C * X + X ** 2
    dP_tp = phi_L_sq * dP_liquid_alone_Pa
    return {
        "Martinelli_X": X,
        "phi_L_squared": phi_L_sq,
        "phi_G_squared": phi_G_sq,
        "C_coefficient": C,
        "two_phase_pressure_drop_Pa": dP_tp,
    }


# ---------------------------------------------------------------------------
# 7) 泵功率 P = dP * Q / eta
# ---------------------------------------------------------------------------

def pump_power(
    pressure_drop_Pa: float,
    volume_flow_m3_per_s: float,
    pump_efficiency: float = 1.0,
) -> dict:
    if pressure_drop_Pa <= 0 or volume_flow_m3_per_s <= 0:
        raise ValueError("压降与流量必须为正")
    if not (0 < pump_efficiency <= 1):
        raise ValueError("泵效率 ∈ (0, 1]")
    P_hyd = pressure_drop_Pa * volume_flow_m3_per_s
    P_shaft = P_hyd / pump_efficiency
    return {
        "hydraulic_power_W": P_hyd,
        "shaft_power_W": P_shaft,
        "pump_efficiency": pump_efficiency,
    }


# ---------------------------------------------------------------------------
# 8) 矩形通道摩擦因子（基于 fRe 查表）
# ---------------------------------------------------------------------------

def rectangular_channel_friction(
    Re: float, aspect_ratio_alpha: float,
) -> dict:
    """用 correlations.rectangular_nusselt_fRe 里的 fRe 除以 Re 即为 f。"""
    from sciagent.tools.correlations import rectangular_nusselt_fRe
    fRe = rectangular_nusselt_fRe(aspect_ratio_alpha)["fRe"]
    f = fRe / Re
    return {"friction_factor_darcy": f, "fRe": fRe, "Re": Re,
            "aspect_ratio": aspect_ratio_alpha}


# ---------------------------------------------------------------------------
# 工具注册
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "laminar_friction_factor",
        "description": "层流 Darcy 摩擦因子 f=64/Re。Re 必须 < 2300。",
        "input_schema": {
            "type": "object",
            "properties": {"Re": {"type": "number"}},
            "required": ["Re"],
        },
    },
    {
        "name": "colebrook",
        "description": "Colebrook-White 粗糙管湍流摩擦因子（迭代隐式解）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number"},
                "roughness_m": {"type": "number"},
                "diameter_m": {"type": "number"},
            },
            "required": ["Re", "roughness_m", "diameter_m"],
        },
    },
    {
        "name": "swamee_jain",
        "description": "Swamee-Jain 显式拟合 Colebrook，精度约±1%。",
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number"},
                "roughness_m": {"type": "number"},
                "diameter_m": {"type": "number"},
            },
            "required": ["Re", "roughness_m", "diameter_m"],
        },
    },
    {
        "name": "darcy_weisbach",
        "description": "Darcy-Weisbach 方程计算摩擦压降。",
        "input_schema": {
            "type": "object",
            "properties": {
                "friction_factor": {"type": "number"},
                "length_m": {"type": "number"},
                "hydraulic_diameter_m": {"type": "number"},
                "density": {"type": "number"},
                "velocity": {"type": "number"},
            },
            "required": ["friction_factor", "length_m",
                         "hydraulic_diameter_m", "density", "velocity"],
        },
    },
    {
        "name": "minor_loss",
        "description": (
            "按部件清单合成局部损失。部件可选："
            + ", ".join(sorted(MINOR_LOSS_K.keys()))
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "description": (
                        "部件列表，每项可以是字符串或 [name, count]。"
                    ),
                    "items": {},
                },
                "density": {"type": "number"},
                "velocity": {"type": "number"},
            },
            "required": ["components", "density", "velocity"],
        },
    },
    {
        "name": "borda_carnot_expansion",
        "description": "突然扩张压降（Borda-Carnot）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "A1_m2": {"type": "number"},
                "A2_m2": {"type": "number"},
                "density": {"type": "number"},
                "velocity1": {"type": "number"},
            },
            "required": ["A1_m2", "A2_m2", "density", "velocity1"],
        },
    },
    {
        "name": "sudden_contraction",
        "description": "突然收缩压降（Weisbach 拟合）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "A1_m2": {"type": "number"},
                "A2_m2": {"type": "number"},
                "density": {"type": "number"},
                "velocity2": {"type": "number"},
            },
            "required": ["A1_m2", "A2_m2", "density", "velocity2"],
        },
    },
    {
        "name": "lockhart_martinelli",
        "description": "Lockhart-Martinelli 两相压降乘子（Chisholm C）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "dP_liquid_alone_Pa": {"type": "number"},
                "dP_gas_alone_Pa": {"type": "number"},
                "flow_regime": {
                    "type": "string",
                    "enum": ["turb-turb", "lam-lam", "lam-turb", "turb-lam"],
                },
            },
            "required": ["dP_liquid_alone_Pa", "dP_gas_alone_Pa"],
        },
    },
    {
        "name": "pump_power",
        "description": "泵轴功率 P=dP*Q/eta。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pressure_drop_Pa": {"type": "number"},
                "volume_flow_m3_per_s": {"type": "number"},
                "pump_efficiency": {"type": "number"},
            },
            "required": ["pressure_drop_Pa", "volume_flow_m3_per_s"],
        },
    },
    {
        "name": "rectangular_channel_friction",
        "description": "矩形通道层流摩擦因子，由 fRe/Re 得到。",
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number"},
                "aspect_ratio_alpha": {"type": "number"},
            },
            "required": ["Re", "aspect_ratio_alpha"],
        },
    },
]


def _wrap(fn):
    def _exec(args):
        return json.dumps(fn(**args), ensure_ascii=False, indent=2)
    return _exec


TOOL_EXECUTORS = {
    "laminar_friction_factor": _wrap(laminar_friction_factor),
    "colebrook": _wrap(colebrook),
    "swamee_jain": _wrap(swamee_jain),
    "darcy_weisbach": _wrap(darcy_weisbach),
    "minor_loss": _wrap(minor_loss),
    "borda_carnot_expansion": _wrap(borda_carnot_expansion),
    "sudden_contraction": _wrap(sudden_contraction),
    "lockhart_martinelli": _wrap(lockhart_martinelli),
    "pump_power": _wrap(pump_power),
    "rectangular_channel_friction": _wrap(rectangular_channel_friction),
}
