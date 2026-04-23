"""
流体物性工具：水在常温常压下的物性查询。

给 Agent 一个"查表"能力，避免每次都让 LLM 硬编码一套物性常数
（那样既不可靠又不可追溯）。数值来自 NIST REFPROP 的公开拟合，
覆盖微通道散热器常用的 20~80°C 范围。
"""

from dataclasses import dataclass


@dataclass
class WaterProperties:
    temperature_C: float
    density: float            # rho, kg/m^3
    viscosity: float          # mu, Pa·s
    conductivity: float       # k, W/(m·K)
    specific_heat: float      # cp, J/(kg·K)
    prandtl: float            # Pr, 无量纲

    def as_dict(self) -> dict:
        return {
            "temperature_C": self.temperature_C,
            "density_kg_per_m3": self.density,
            "viscosity_Pa_s": self.viscosity,
            "thermal_conductivity_W_per_mK": self.conductivity,
            "specific_heat_J_per_kgK": self.specific_heat,
            "prandtl_number": self.prandtl,
        }


# 分段线性插值节点（20~80°C，常压液态水）
_WATER_TABLE = [
    # T,   rho,     mu (×1e-3), k,     cp
    (20.0, 998.2,  1.002e-3, 0.5984, 4182.0),
    (30.0, 995.7,  0.798e-3, 0.6154, 4178.0),
    (40.0, 992.2,  0.653e-3, 0.6305, 4179.0),
    (50.0, 988.1,  0.547e-3, 0.6435, 4181.0),
    (60.0, 983.2,  0.467e-3, 0.6543, 4185.0),
    (70.0, 977.8,  0.404e-3, 0.6631, 4190.0),
    (80.0, 971.8,  0.355e-3, 0.6700, 4196.0),
]


def _interp(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def water_properties(temperature_C: float) -> WaterProperties:
    """在 20~80°C 范围内线性插值水的物性，超界会抛 ValueError。"""
    T_min, T_max = _WATER_TABLE[0][0], _WATER_TABLE[-1][0]
    if not (T_min <= temperature_C <= T_max):
        raise ValueError(
            f"温度 {temperature_C}°C 超出支持范围 [{T_min}, {T_max}] °C"
        )

    for i in range(len(_WATER_TABLE) - 1):
        T0, rho0, mu0, k0, cp0 = _WATER_TABLE[i]
        T1, rho1, mu1, k1, cp1 = _WATER_TABLE[i + 1]
        if T0 <= temperature_C <= T1:
            rho = _interp(temperature_C, T0, T1, rho0, rho1)
            mu = _interp(temperature_C, T0, T1, mu0, mu1)
            k = _interp(temperature_C, T0, T1, k0, k1)
            cp = _interp(temperature_C, T0, T1, cp0, cp1)
            return WaterProperties(
                temperature_C=temperature_C,
                density=rho,
                viscosity=mu,
                conductivity=k,
                specific_heat=cp,
                prandtl=mu * cp / k,
            )

    raise RuntimeError("物性插值未命中任何区间（逻辑 bug）")


TOOL_DEFINITION = {
    "name": "water_properties",
    "description": (
        "查询液态水在 20~80°C 范围内的物性："
        "密度、动力粘度、导热系数、比热、普朗特数。"
        "用于散热器换热与流动计算。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "temperature_C": {
                "type": "number",
                "description": "水温，单位摄氏度，范围 20~80",
            }
        },
        "required": ["temperature_C"],
    },
}


def execute(args: dict) -> str:
    import json
    T = float(args["temperature_C"])
    props = water_properties(T)
    return json.dumps(props.as_dict(), ensure_ascii=False, indent=2)
