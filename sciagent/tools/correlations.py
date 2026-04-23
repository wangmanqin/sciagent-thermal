"""
微通道换热与流动关联式。

把领域里最常用、但 LLM 常常记错系数的几个公式固化为工具：
  - 矩形通道充分发展层流的 Nu 和 fRe（长宽比修正）
  - Dittus-Boelter 湍流换热关联
  - 矩形通道水力直径
  - 翅片效率
这样 Agent 不用再去"回忆"系数，而是直接调用确定的实现。
"""

from __future__ import annotations
import math
import json


# ---------------------------------------------------------------------------
# 矩形通道充分发展层流 — Shah & London 拟合
# alpha = b/a （短边/长边），范围 0 < alpha <= 1
# ---------------------------------------------------------------------------

def rectangular_nusselt_fRe(alpha: float, boundary: str = "constant_heat_flux") -> dict:
    if not (0 < alpha <= 1):
        raise ValueError(f"aspect_ratio alpha 必须在 (0, 1]，输入 {alpha}")
    if boundary not in ("constant_heat_flux", "constant_wall_temperature"):
        raise ValueError(
            "boundary 必须是 'constant_heat_flux' 或 'constant_wall_temperature'"
        )

    # Shah & London (1978) 多项式拟合
    # 以 alpha=b/a 为自变量，给出 Nu_H, Nu_T, f·Re
    a = alpha
    nu_H = (
        8.235 * (1 - 2.0421 * a + 3.0853 * a**2 - 2.4765 * a**3
                 + 1.0578 * a**4 - 0.1861 * a**5)
    )
    nu_T = (
        7.541 * (1 - 2.610 * a + 4.970 * a**2 - 5.119 * a**3
                 + 2.702 * a**4 - 0.548 * a**5)
    )
    f_Re = (
        24.0 * (1 - 1.3553 * a + 1.9467 * a**2 - 1.7012 * a**3
                + 0.9564 * a**4 - 0.2537 * a**5)
    )

    nu = nu_H if boundary == "constant_heat_flux" else nu_T
    return {
        "aspect_ratio": alpha,
        "boundary_condition": boundary,
        "nusselt": nu,
        "fRe": f_Re,
    }


# ---------------------------------------------------------------------------
# Dittus-Boelter: Nu = 0.023 Re^0.8 Pr^n
# n = 0.4 加热（流体被加热），n = 0.3 冷却
# 适用：Re > 1e4，0.6 < Pr < 160
# ---------------------------------------------------------------------------

def dittus_boelter(Re: float, Pr: float, mode: str = "heating") -> dict:
    if Re <= 0 or Pr <= 0:
        raise ValueError("Re 和 Pr 必须为正")
    if mode == "heating":
        n = 0.4
    elif mode == "cooling":
        n = 0.3
    else:
        raise ValueError("mode 必须是 'heating' 或 'cooling'")

    Nu = 0.023 * Re**0.8 * Pr**n
    valid = (Re > 1e4) and (0.6 < Pr < 160)
    return {
        "reynolds": Re,
        "prandtl": Pr,
        "mode": mode,
        "nusselt": Nu,
        "applicability_ok": valid,
    }


# ---------------------------------------------------------------------------
# 矩形通道水力直径 Dh = 4A/P
# ---------------------------------------------------------------------------

def hydraulic_diameter(width_m: float, height_m: float) -> dict:
    if width_m <= 0 or height_m <= 0:
        raise ValueError("width 和 height 必须为正")
    A = width_m * height_m
    P = 2 * (width_m + height_m)
    Dh = 4 * A / P
    return {
        "width_m": width_m,
        "height_m": height_m,
        "cross_section_area_m2": A,
        "wetted_perimeter_m": P,
        "hydraulic_diameter_m": Dh,
        "aspect_ratio_short_over_long": min(width_m, height_m) / max(width_m, height_m),
    }


# ---------------------------------------------------------------------------
# 直翅片效率 eta_f = tanh(mL) / (mL),  m = sqrt(2h / (k*t))
# 适用于等截面直翅片的一维绝热尖端近似
# ---------------------------------------------------------------------------

def fin_efficiency(
    h_W_per_m2K: float,
    k_W_per_mK: float,
    thickness_m: float,
    length_m: float,
) -> dict:
    for label, val in [
        ("h", h_W_per_m2K), ("k", k_W_per_mK),
        ("thickness", thickness_m), ("length", length_m),
    ]:
        if val <= 0:
            raise ValueError(f"{label} 必须为正")

    m = math.sqrt(2 * h_W_per_m2K / (k_W_per_mK * thickness_m))
    mL = m * length_m
    eta = math.tanh(mL) / mL if mL > 0 else 1.0
    return {
        "h_W_per_m2K": h_W_per_m2K,
        "k_W_per_mK": k_W_per_mK,
        "thickness_m": thickness_m,
        "length_m": length_m,
        "m_per_m": m,
        "mL": mL,
        "fin_efficiency": eta,
    }


# ---------------------------------------------------------------------------
# Tool 注册
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "rectangular_nusselt_fRe",
        "description": (
            "矩形通道充分发展层流的 Nu 数和 f·Re 乘积（Shah-London 拟合）。"
            "输入长宽比 alpha=b/a ∈ (0,1] 和边界条件类型。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "alpha": {
                    "type": "number",
                    "description": "长宽比 b/a（短边/长边），取值 (0, 1]",
                },
                "boundary": {
                    "type": "string",
                    "enum": ["constant_heat_flux", "constant_wall_temperature"],
                    "description": "边界条件",
                },
            },
            "required": ["alpha"],
        },
    },
    {
        "name": "dittus_boelter",
        "description": (
            "Dittus-Boelter 湍流换热关联式 Nu=0.023 Re^0.8 Pr^n。"
            "适用 Re>1e4，0.6<Pr<160。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number", "description": "雷诺数"},
                "Pr": {"type": "number", "description": "普朗特数"},
                "mode": {
                    "type": "string",
                    "enum": ["heating", "cooling"],
                    "description": "加热或冷却模式，决定 Pr 的指数",
                },
            },
            "required": ["Re", "Pr"],
        },
    },
    {
        "name": "hydraulic_diameter",
        "description": "矩形通道水力直径 Dh=4A/P。输入长、宽（米）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "width_m": {"type": "number"},
                "height_m": {"type": "number"},
            },
            "required": ["width_m", "height_m"],
        },
    },
    {
        "name": "fin_efficiency",
        "description": (
            "直翅片效率 eta=tanh(mL)/(mL)，绝热尖端近似。"
            "输入对流换热系数 h、翅片材料导热系数 k、翅片厚度、长度。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "h_W_per_m2K": {"type": "number"},
                "k_W_per_mK": {"type": "number"},
                "thickness_m": {"type": "number"},
                "length_m": {"type": "number"},
            },
            "required": ["h_W_per_m2K", "k_W_per_mK", "thickness_m", "length_m"],
        },
    },
]


def _exec_rect(args):
    res = rectangular_nusselt_fRe(
        float(args["alpha"]),
        args.get("boundary", "constant_heat_flux"),
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


def _exec_db(args):
    res = dittus_boelter(
        float(args["Re"]), float(args["Pr"]), args.get("mode", "heating"),
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


def _exec_dh(args):
    res = hydraulic_diameter(float(args["width_m"]), float(args["height_m"]))
    return json.dumps(res, ensure_ascii=False, indent=2)


def _exec_fin(args):
    res = fin_efficiency(
        float(args["h_W_per_m2K"]),
        float(args["k_W_per_mK"]),
        float(args["thickness_m"]),
        float(args["length_m"]),
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


TOOL_EXECUTORS = {
    "rectangular_nusselt_fRe": _exec_rect,
    "dittus_boelter": _exec_db,
    "hydraulic_diameter": _exec_dh,
    "fin_efficiency": _exec_fin,
}
