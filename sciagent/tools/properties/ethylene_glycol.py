"""
乙二醇-水混合物物性（0~60% 质量分数，-20~100°C）

散热器冷却液常用 EG/水混合液。这里用二维双线性插值提供物性。
数据源：ASHRAE Handbook - Fundamentals, 2017 Ch. 31, Table 3。
"""

from __future__ import annotations
import json
from dataclasses import dataclass


# EG 质量分数 -> 温度 -> (rho, cp, k, mu)
# 只覆盖常用节点；插值使用双线性
_EG_TABLE = {
    0.00: [
        # T_C, rho,    cp,     k,     mu
        (-10.0, 1000.2, 4199.0, 0.5500, 2.6100e-3),
        (  0.0,  999.8, 4217.0, 0.5610, 1.7920e-3),
        ( 20.0,  998.2, 4182.0, 0.5984, 1.0020e-3),
        ( 40.0,  992.2, 4179.0, 0.6305, 0.6530e-3),
        ( 60.0,  983.2, 4185.0, 0.6543, 0.4670e-3),
        ( 80.0,  971.8, 4196.0, 0.6700, 0.3550e-3),
        (100.0,  958.4, 4217.0, 0.6794, 0.2820e-3),
    ],
    0.20: [
        (-20.0, 1036.3, 3825.0, 0.4900, 8.9000e-3),
        (-10.0, 1034.0, 3840.0, 0.4950, 5.6000e-3),
        (  0.0, 1031.6, 3855.0, 0.5000, 3.6500e-3),
        ( 20.0, 1025.4, 3882.0, 0.5100, 1.8500e-3),
        ( 40.0, 1018.0, 3905.0, 0.5180, 1.1500e-3),
        ( 60.0, 1009.2, 3926.0, 0.5240, 0.7900e-3),
        ( 80.0,  999.0, 3944.0, 0.5280, 0.5800e-3),
        (100.0,  987.0, 3960.0, 0.5300, 0.4500e-3),
    ],
    0.40: [
        (-20.0, 1070.0, 3465.0, 0.4150, 2.1500e-2),
        (-10.0, 1067.4, 3487.0, 0.4200, 1.2500e-2),
        (  0.0, 1064.5, 3508.0, 0.4250, 7.6500e-3),
        ( 20.0, 1057.8, 3547.0, 0.4320, 3.5000e-3),
        ( 40.0, 1049.5, 3581.0, 0.4380, 2.0000e-3),
        ( 60.0, 1040.0, 3611.0, 0.4420, 1.3000e-3),
        ( 80.0, 1029.0, 3637.0, 0.4450, 0.9000e-3),
        (100.0, 1017.0, 3660.0, 0.4470, 0.6700e-3),
    ],
    0.60: [
        (-20.0, 1093.5, 3137.0, 0.3440, 8.3000e-2),
        (-10.0, 1090.8, 3167.0, 0.3480, 4.1000e-2),
        (  0.0, 1087.9, 3196.0, 0.3520, 2.2500e-2),
        ( 20.0, 1080.3, 3248.0, 0.3580, 8.5000e-3),
        ( 40.0, 1071.6, 3291.0, 0.3640, 4.3000e-3),
        ( 60.0, 1061.8, 3329.0, 0.3680, 2.5500e-3),
        ( 80.0, 1050.8, 3363.0, 0.3720, 1.7000e-3),
        (100.0, 1038.5, 3392.0, 0.3740, 1.2000e-3),
    ],
}


@dataclass
class EGProperties:
    mass_fraction: float
    temperature_C: float
    density: float
    viscosity: float
    conductivity: float
    specific_heat: float
    prandtl: float


def _linterp(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _find_bounds(sorted_keys, x, name):
    lo = max([k for k in sorted_keys if k <= x], default=None)
    hi = min([k for k in sorted_keys if k >= x], default=None)
    if lo is None or hi is None:
        raise ValueError(f"{name} {x} 超出支持范围 [{min(sorted_keys)}, {max(sorted_keys)}]")
    return lo, hi


def ethylene_glycol_properties(
    mass_fraction: float, temperature_C: float,
) -> EGProperties:
    if not (0 <= mass_fraction <= 0.6):
        raise ValueError("mass_fraction 范围 [0, 0.6]")

    w_keys = sorted(_EG_TABLE.keys())
    w_lo, w_hi = _find_bounds(w_keys, mass_fraction, "mass_fraction")

    def _row(w, T):
        rows = _EG_TABLE[w]
        T_keys = [r[0] for r in rows]
        T_lo, T_hi = _find_bounds(T_keys, T, "temperature_C")
        rL = next(r for r in rows if r[0] == T_lo)
        rH = next(r for r in rows if r[0] == T_hi)
        return [
            _linterp(T, T_lo, T_hi, rL[i + 1], rH[i + 1])
            for i in range(4)  # rho, cp, k, mu
        ]

    low_row = _row(w_lo, temperature_C)
    high_row = _row(w_hi, temperature_C)
    mixed = [
        _linterp(mass_fraction, w_lo, w_hi, low_row[i], high_row[i])
        for i in range(4)
    ]
    rho, cp, k, mu = mixed
    Pr = mu * cp / k
    return EGProperties(
        mass_fraction=mass_fraction,
        temperature_C=temperature_C,
        density=rho,
        viscosity=mu,
        conductivity=k,
        specific_heat=cp,
        prandtl=Pr,
    )


TOOL_DEFINITION = {
    "name": "ethylene_glycol_properties",
    "description": (
        "乙二醇-水混合物物性（质量分数 0-60%, 温度 -20~100°C）。"
        "输入：mass_fraction 0.0~0.6，temperature_C。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mass_fraction": {"type": "number", "description": "乙二醇质量分数 [0, 0.6]"},
            "temperature_C": {"type": "number"},
        },
        "required": ["mass_fraction", "temperature_C"],
    },
}


def execute(args: dict) -> str:
    props = ethylene_glycol_properties(
        float(args["mass_fraction"]),
        float(args["temperature_C"]),
    )
    return json.dumps(props.__dict__, ensure_ascii=False, indent=2)
