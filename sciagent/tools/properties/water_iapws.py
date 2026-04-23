"""
扩展版水物性：IAPWS-IF97 区域 1 （压缩液态水，T=0~200°C, P=0.1~100 bar）

注意：本模块使用 **简化拟合** 而非完整 IAPWS 方程组。
目标是把 fluid_properties.py 的 20~80°C 范围扩展到 0~100°C，并
提供密度随压力的弱修正，用于散热器里 1~5 bar 的工况。完整 IAPWS
计算请接入 iapws / CoolProp 这类专业库。

所有返回值单位 SI：kg/m^3, Pa·s, W/(m·K), J/(kg·K)
"""

from __future__ import annotations
import math
import json
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 饱和液态水拟合表（0~100°C，每 5°C 一个节点）
# ---------------------------------------------------------------------------

_SAT_WATER = [
    # T,    rho,      mu,         k,       cp,       beta,      sigma
    (  0.0, 999.87,  1.7920e-3,  0.5610,  4217.0, -6.80e-5,  75.64e-3),
    (  5.0, 999.97,  1.5185e-3,  0.5710,  4202.0,  1.60e-5,  74.90e-3),
    ( 10.0, 999.70,  1.3060e-3,  0.5800,  4192.0,  8.80e-5,  74.20e-3),
    ( 15.0, 999.10,  1.1390e-3,  0.5890,  4186.0,  1.50e-4,  73.48e-3),
    ( 20.0, 998.20,  1.0020e-3,  0.5984,  4182.0,  2.07e-4,  72.75e-3),
    ( 25.0, 997.04,  0.8903e-3,  0.6070,  4180.0,  2.57e-4,  71.97e-3),
    ( 30.0, 995.70,  0.7980e-3,  0.6154,  4178.0,  3.03e-4,  71.18e-3),
    ( 35.0, 994.03,  0.7200e-3,  0.6231,  4178.0,  3.45e-4,  70.38e-3),
    ( 40.0, 992.20,  0.6530e-3,  0.6305,  4179.0,  3.85e-4,  69.55e-3),
    ( 45.0, 990.22,  0.5960e-3,  0.6373,  4180.0,  4.22e-4,  68.74e-3),
    ( 50.0, 988.10,  0.5470e-3,  0.6435,  4181.0,  4.58e-4,  67.92e-3),
    ( 55.0, 985.68,  0.5040e-3,  0.6493,  4183.0,  4.93e-4,  67.05e-3),
    ( 60.0, 983.20,  0.4670e-3,  0.6543,  4185.0,  5.26e-4,  66.19e-3),
    ( 65.0, 980.55,  0.4340e-3,  0.6590,  4187.0,  5.59e-4,  65.33e-3),
    ( 70.0, 977.80,  0.4040e-3,  0.6631,  4190.0,  5.90e-4,  64.42e-3),
    ( 75.0, 974.85,  0.3780e-3,  0.6668,  4193.0,  6.22e-4,  63.53e-3),
    ( 80.0, 971.80,  0.3550e-3,  0.6700,  4196.0,  6.52e-4,  62.60e-3),
    ( 85.0, 968.63,  0.3340e-3,  0.6729,  4200.0,  6.81e-4,  61.68e-3),
    ( 90.0, 965.30,  0.3150e-3,  0.6753,  4205.0,  7.11e-4,  60.73e-3),
    ( 95.0, 961.92,  0.2970e-3,  0.6775,  4211.0,  7.39e-4,  59.78e-3),
    (100.0, 958.40,  0.2820e-3,  0.6794,  4217.0,  7.67e-4,  58.91e-3),
]

T_MIN = 0.0
T_MAX = 100.0


@dataclass
class WaterPropsExtended:
    temperature_C: float
    pressure_bar: float
    density: float
    viscosity: float
    conductivity: float
    specific_heat: float
    prandtl: float
    thermal_diffusivity: float
    kinematic_viscosity: float
    thermal_expansion: float      # beta, 1/K
    surface_tension: float        # N/m
    saturation_pressure_Pa: float


def _bisect_segment(T: float):
    for i in range(len(_SAT_WATER) - 1):
        if _SAT_WATER[i][0] <= T <= _SAT_WATER[i + 1][0]:
            return i
    return None


def _interp(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def saturation_pressure(T_C: float) -> float:
    """Antoine 方程（water, 1~100°C 简化式，结果 Pa）。"""
    if T_C < 1 or T_C > 100:
        raise ValueError("饱和压力查询仅支持 1~100°C")
    # Antoine: log10(P_mmHg) = A - B/(C+T)
    A, B, C = 8.07131, 1730.63, 233.426
    P_mmHg = 10 ** (A - B / (C + T_C))
    return P_mmHg * 133.322


def _pressure_density_correction(rho_sat: float, P_bar: float) -> float:
    """
    简化的等温可压缩性修正：rho(P) ≈ rho_sat * (1 + kappa * (P - 1bar))
    水的等温可压缩性 ~5e-10 1/Pa，对 1-10 bar 影响 <0.1%，几乎可忽略。
    """
    kappa = 5e-10
    dP = (P_bar - 1.013) * 1e5
    return rho_sat * (1.0 + kappa * dP)


def water_properties_extended(
    temperature_C: float, pressure_bar: float = 1.013,
) -> WaterPropsExtended:
    if not (T_MIN <= temperature_C <= T_MAX):
        raise ValueError(
            f"温度 {temperature_C}°C 超出支持范围 [{T_MIN}, {T_MAX}]"
        )
    if pressure_bar <= 0:
        raise ValueError("压力必须为正 (bar)")

    seg = _bisect_segment(temperature_C)
    if seg is None:
        raise RuntimeError("温度区间未命中")

    T0, *row0 = _SAT_WATER[seg]
    T1, *row1 = _SAT_WATER[seg + 1]
    rho0, mu0, k0, cp0, beta0, sigma0 = row0
    rho1, mu1, k1, cp1, beta1, sigma1 = row1

    rho = _interp(temperature_C, T0, T1, rho0, rho1)
    mu = _interp(temperature_C, T0, T1, mu0, mu1)
    k = _interp(temperature_C, T0, T1, k0, k1)
    cp = _interp(temperature_C, T0, T1, cp0, cp1)
    beta = _interp(temperature_C, T0, T1, beta0, beta1)
    sigma = _interp(temperature_C, T0, T1, sigma0, sigma1)

    rho = _pressure_density_correction(rho, pressure_bar)
    Pr = mu * cp / k
    alpha = k / (rho * cp)
    nu = mu / rho

    try:
        P_sat = saturation_pressure(max(temperature_C, 1.0))
    except ValueError:
        P_sat = float("nan")

    return WaterPropsExtended(
        temperature_C=temperature_C,
        pressure_bar=pressure_bar,
        density=rho,
        viscosity=mu,
        conductivity=k,
        specific_heat=cp,
        prandtl=Pr,
        thermal_diffusivity=alpha,
        kinematic_viscosity=nu,
        thermal_expansion=beta,
        surface_tension=sigma,
        saturation_pressure_Pa=P_sat,
    )


# ---------------------------------------------------------------------------
# 工具 schema
# ---------------------------------------------------------------------------

TOOL_DEFINITION = {
    "name": "water_properties_extended",
    "description": (
        "水的扩展物性（0~100°C，含 beta、alpha、nu、sigma、饱和压力）。"
        "比 water_properties 覆盖范围更宽，用于自然对流、相变、沸腾等场景。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "temperature_C": {"type": "number"},
            "pressure_bar": {"type": "number"},
        },
        "required": ["temperature_C"],
    },
}


def execute(args: dict) -> str:
    props = water_properties_extended(
        float(args["temperature_C"]),
        float(args.get("pressure_bar", 1.013)),
    )
    return json.dumps(props.__dict__, ensure_ascii=False, indent=2)
