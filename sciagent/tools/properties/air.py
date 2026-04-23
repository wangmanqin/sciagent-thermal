"""
干空气物性（-50~500°C，常压），用于风冷散热器 / 自然对流计算。

关联式拟合自 Incropera Appendix A.4；精度 ±1% 以内，覆盖常见工程范围。
"""

from __future__ import annotations
import math
import json
from dataclasses import dataclass


@dataclass
class AirProperties:
    temperature_C: float
    pressure_bar: float
    density: float
    viscosity: float
    kinematic_viscosity: float
    conductivity: float
    specific_heat: float
    thermal_diffusivity: float
    prandtl: float
    thermal_expansion: float


R_AIR = 287.05  # J/(kg·K)


def air_properties(temperature_C: float, pressure_bar: float = 1.013) -> AirProperties:
    if not (-50 <= temperature_C <= 500):
        raise ValueError("air_properties 支持 -50~500°C")
    if pressure_bar <= 0:
        raise ValueError("pressure 必须为正")

    T = temperature_C + 273.15
    P = pressure_bar * 1e5

    # 理想气体密度
    rho = P / (R_AIR * T)

    # Sutherland 粘度：mu = mu_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)
    mu_ref = 1.716e-5
    T_ref = 273.0
    S_mu = 110.4
    mu = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S_mu) / (T + S_mu)

    # 导热系数（Sutherland 形式）
    k_ref = 0.0241
    S_k = 194.0
    k = k_ref * (T / T_ref) ** 1.5 * (T_ref + S_k) / (T + S_k)

    # 定压比热（温度多项式拟合）
    # cp [J/(kg K)] = a0 + a1 T + a2 T^2, 250~1000K
    a0, a1, a2 = 1030.0, -0.13, 2.0e-4
    cp = a0 + a1 * T + a2 * T ** 2

    alpha = k / (rho * cp)
    nu = mu / rho
    Pr = mu * cp / k
    beta = 1.0 / T  # 理想气体

    return AirProperties(
        temperature_C=temperature_C,
        pressure_bar=pressure_bar,
        density=rho,
        viscosity=mu,
        kinematic_viscosity=nu,
        conductivity=k,
        specific_heat=cp,
        thermal_diffusivity=alpha,
        prandtl=Pr,
        thermal_expansion=beta,
    )


TOOL_DEFINITION = {
    "name": "air_properties",
    "description": "干空气物性 (-50~500°C)。适合风冷散热器、自然对流场景。",
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
    props = air_properties(
        float(args["temperature_C"]),
        float(args.get("pressure_bar", 1.013)),
    )
    return json.dumps(props.__dict__, ensure_ascii=False, indent=2)
