"""
强制对流 / 自然对流 / 辐射换热关联式库。

补足 correlations.py 之外的常用公式，覆盖：
  - 湍流管内换热：Gnielinski, Petukhov, Sieder-Tate, Colburn
  - 层流入口段：Hausen, Sieder-Tate 入口修正
  - 外掠：Churchill-Bernstein（圆柱），Zukauskas（管束）
  - 自然对流：Churchill-Chu（竖板/竖圆柱），McAdams（水平板）
  - 辐射：两面灰体空间辐射，净辐射换热
  - 组合：NTU-epsilon 换热器模型，LMTD 方法

每个函数：
  - 明确输入范围与适用条件
  - 返回 dict（Nu, h, 以及判定是否落在适用区间）
  - 暴露为 Agent 可调用的工具（TOOL_DEFINITIONS/TOOL_EXECUTORS）

References:
  - Incropera, DeWitt, Bergman, Lavine, "Fundamentals of Heat and Mass
    Transfer", 7th ed., 2011.
  - Kays & Crawford, "Convective Heat and Mass Transfer".
  - Shah & London, "Laminar Flow Forced Convection in Ducts" (1978).
"""

from __future__ import annotations
import math
import json
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 共用数据结构
# ---------------------------------------------------------------------------

@dataclass
class CorrelationResult:
    """关联式输出的统一结构。"""
    nusselt: float
    heat_transfer_coefficient_W_per_m2K: Optional[float]
    applicability_ok: bool
    notes: str
    inputs: dict

    def as_dict(self) -> dict:
        return {
            "nusselt": self.nusselt,
            "h_W_per_m2K": self.heat_transfer_coefficient_W_per_m2K,
            "applicability_ok": self.applicability_ok,
            "notes": self.notes,
            "inputs": self.inputs,
        }


def _convert_nu_to_h(Nu: float, k: float, Dh: float) -> Optional[float]:
    """Nu = h*L/k → h，若缺少 k 或 Dh 则返回 None。"""
    if k is None or Dh is None or Dh <= 0:
        return None
    return Nu * k / Dh


# ---------------------------------------------------------------------------
# 1) Gnielinski — 湍流管内换热（更准确，Re > 3000）
#    Nu = (f/8)(Re-1000)Pr / (1 + 12.7*sqrt(f/8)*(Pr^(2/3)-1))
#    f 用 Petukhov 摩擦因子拟合：f = (0.79 ln Re - 1.64)^(-2)
# ---------------------------------------------------------------------------

def gnielinski(
    Re: float,
    Pr: float,
    k_fluid: Optional[float] = None,
    hydraulic_diameter_m: Optional[float] = None,
) -> dict:
    if Re <= 0 or Pr <= 0:
        raise ValueError("Re 和 Pr 必须为正")

    f = (0.79 * math.log(Re) - 1.64) ** -2
    numer = (f / 8.0) * (Re - 1000.0) * Pr
    denom = 1.0 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0)
    Nu = numer / denom

    ok = (3e3 < Re < 5e6) and (0.5 < Pr < 2000)
    notes = ""
    if not ok:
        notes = "输入超出 Gnielinski 适用范围 (3e3<Re<5e6, 0.5<Pr<2000)。"
    h = _convert_nu_to_h(Nu, k_fluid, hydraulic_diameter_m)
    res = CorrelationResult(
        nusselt=Nu,
        heat_transfer_coefficient_W_per_m2K=h,
        applicability_ok=ok,
        notes=notes,
        inputs={"Re": Re, "Pr": Pr, "friction_factor_Petukhov": f},
    )
    return res.as_dict()


# ---------------------------------------------------------------------------
# 2) Petukhov — 更高精度的湍流换热（适用 0.5 < Pr < 2000, 1e4 < Re < 5e6）
#    Nu = (f/8)*Re*Pr / (1.07 + 12.7*sqrt(f/8)*(Pr^(2/3) - 1))
# ---------------------------------------------------------------------------

def petukhov(
    Re: float,
    Pr: float,
    k_fluid: Optional[float] = None,
    hydraulic_diameter_m: Optional[float] = None,
) -> dict:
    if Re <= 0 or Pr <= 0:
        raise ValueError("Re 和 Pr 必须为正")
    f = (0.79 * math.log(Re) - 1.64) ** -2
    Nu = (f / 8.0) * Re * Pr / (1.07 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2 / 3) - 1))
    ok = (1e4 < Re < 5e6) and (0.5 < Pr < 2000)
    notes = "" if ok else "输入超出 Petukhov 适用范围 (1e4<Re<5e6, 0.5<Pr<2000)。"
    h = _convert_nu_to_h(Nu, k_fluid, hydraulic_diameter_m)
    return CorrelationResult(Nu, h, ok, notes,
                             {"Re": Re, "Pr": Pr, "friction_factor": f}).as_dict()


# ---------------------------------------------------------------------------
# 3) Sieder-Tate — 带壁面粘度修正，适合 mu 随温度变化较大的情形
#    Nu = 0.027 Re^0.8 Pr^(1/3) (mu/mu_wall)^0.14
# ---------------------------------------------------------------------------

def sieder_tate(
    Re: float,
    Pr: float,
    mu_bulk: float,
    mu_wall: float,
    k_fluid: Optional[float] = None,
    hydraulic_diameter_m: Optional[float] = None,
) -> dict:
    if Re <= 0 or Pr <= 0 or mu_bulk <= 0 or mu_wall <= 0:
        raise ValueError("Re, Pr, mu_bulk, mu_wall 必须为正")
    Nu = 0.027 * Re ** 0.8 * Pr ** (1.0 / 3.0) * (mu_bulk / mu_wall) ** 0.14
    ok = (Re > 1e4) and (0.7 < Pr < 1.7e4)
    notes = ("" if ok else
             "输入超出 Sieder-Tate 适用范围 (Re>1e4, 0.7<Pr<1.7e4)。")
    h = _convert_nu_to_h(Nu, k_fluid, hydraulic_diameter_m)
    return CorrelationResult(
        Nu, h, ok, notes,
        {"Re": Re, "Pr": Pr, "mu_bulk": mu_bulk, "mu_wall": mu_wall,
         "viscosity_correction": (mu_bulk / mu_wall) ** 0.14},
    ).as_dict()


# ---------------------------------------------------------------------------
# 4) Colburn analogy — j-factor 与 Reynolds 类比
#    j = St Pr^(2/3) = f/8
#    St = Nu / (Re Pr)
# ---------------------------------------------------------------------------

def colburn_j_factor(
    Re: float,
    Pr: float,
    friction_factor: Optional[float] = None,
    k_fluid: Optional[float] = None,
    hydraulic_diameter_m: Optional[float] = None,
) -> dict:
    if Re <= 0 or Pr <= 0:
        raise ValueError("Re 和 Pr 必须为正")
    if friction_factor is None:
        friction_factor = (0.79 * math.log(Re) - 1.64) ** -2
    j = friction_factor / 8.0
    St = j / Pr ** (2.0 / 3.0)
    Nu = St * Re * Pr
    ok = Re > 1e4
    notes = ("" if ok else
             "Colburn 类比建立在充分发展湍流，Re 较低时仅作参考。")
    h = _convert_nu_to_h(Nu, k_fluid, hydraulic_diameter_m)
    return CorrelationResult(
        Nu, h, ok, notes,
        {"Re": Re, "Pr": Pr, "friction_factor": friction_factor,
         "stanton": St, "j_factor": j},
    ).as_dict()


# ---------------------------------------------------------------------------
# 5) Hausen 层流入口修正（等壁温，圆管）
#    Nu = 3.66 + 0.0668*(D/L)*Re*Pr / (1 + 0.04*((D/L)*Re*Pr)^(2/3))
# ---------------------------------------------------------------------------

def hausen_entry(
    Re: float,
    Pr: float,
    diameter_m: float,
    length_m: float,
    k_fluid: Optional[float] = None,
) -> dict:
    if diameter_m <= 0 or length_m <= 0:
        raise ValueError("diameter 和 length 必须为正")
    gz = (diameter_m / length_m) * Re * Pr  # Graetz 数
    Nu = 3.66 + 0.0668 * gz / (1.0 + 0.04 * gz ** (2.0 / 3.0))
    ok = (Re < 2300)
    notes = ("" if ok else
             "Hausen 入口关联式假设层流（Re<2300），当前 Re 偏大。")
    h = _convert_nu_to_h(Nu, k_fluid, diameter_m)
    return CorrelationResult(
        Nu, h, ok, notes,
        {"Re": Re, "Pr": Pr, "diameter_m": diameter_m,
         "length_m": length_m, "graetz_number": gz},
    ).as_dict()


# ---------------------------------------------------------------------------
# 6) Sieder-Tate 层流入口修正（带 mu 修正）
#    Nu = 1.86 * (Re*Pr*D/L)^(1/3) * (mu/mu_wall)^0.14
# ---------------------------------------------------------------------------

def sieder_tate_entry(
    Re: float,
    Pr: float,
    diameter_m: float,
    length_m: float,
    mu_bulk: float,
    mu_wall: float,
    k_fluid: Optional[float] = None,
) -> dict:
    if min(Re, Pr, diameter_m, length_m, mu_bulk, mu_wall) <= 0:
        raise ValueError("输入参数必须为正")
    term = Re * Pr * diameter_m / length_m
    Nu = 1.86 * term ** (1.0 / 3.0) * (mu_bulk / mu_wall) ** 0.14
    ok = (Re < 2300) and (term > 10)
    notes = ("" if ok else
             "Sieder-Tate 入口关联式要求 Re<2300 且 RePrD/L>10。")
    h = _convert_nu_to_h(Nu, k_fluid, diameter_m)
    return CorrelationResult(
        Nu, h, ok, notes,
        {"Re": Re, "Pr": Pr, "diameter_m": diameter_m,
         "length_m": length_m, "RePrD_over_L": term,
         "mu_bulk": mu_bulk, "mu_wall": mu_wall},
    ).as_dict()


# ---------------------------------------------------------------------------
# 7) Churchill-Bernstein — 外掠圆柱（所有 Re Pr，Re*Pr > 0.2）
# ---------------------------------------------------------------------------

def churchill_bernstein(
    Re: float,
    Pr: float,
    k_fluid: Optional[float] = None,
    diameter_m: Optional[float] = None,
) -> dict:
    if Re <= 0 or Pr <= 0:
        raise ValueError("Re 和 Pr 必须为正")
    base = (
        0.3
        + (0.62 * Re ** 0.5 * Pr ** (1.0 / 3.0))
        / (1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25
    )
    correction = (1.0 + (Re / 2.82e5) ** (5.0 / 8.0)) ** (4.0 / 5.0)
    Nu = base * correction
    ok = Re * Pr > 0.2
    notes = ("" if ok else
             "Churchill-Bernstein 要求 Re*Pr > 0.2，当前乘积偏小。")
    h = _convert_nu_to_h(Nu, k_fluid, diameter_m)
    return CorrelationResult(
        Nu, h, ok, notes,
        {"Re": Re, "Pr": Pr},
    ).as_dict()


# ---------------------------------------------------------------------------
# 8) Zukauskas — 管束外掠换热
#    Nu = C * Re^m * Pr^0.36 * (Pr/Pr_w)^0.25
#    C, m 查表（对齐 / 错列，Re 分段）
# ---------------------------------------------------------------------------

_ZUKAUSKAS_COEFFS = {
    # arrangement, Re_range: (C, m)
    ("aligned", (10, 100)): (0.80, 0.40),
    ("aligned", (100, 1000)): (0.80, 0.40),
    ("aligned", (1000, 2e5)): (0.27, 0.63),
    ("aligned", (2e5, 2e6)): (0.021, 0.84),
    ("staggered", (10, 100)): (0.90, 0.40),
    ("staggered", (100, 1000)): (0.90, 0.40),
    ("staggered", (1000, 2e5)): (0.35, 0.60),  # 与 SL/ST 有关，取常用值
    ("staggered", (2e5, 2e6)): (0.022, 0.84),
}


def zukauskas_tube_bank(
    Re: float,
    Pr: float,
    Pr_wall: float,
    arrangement: str = "staggered",
    k_fluid: Optional[float] = None,
    diameter_m: Optional[float] = None,
) -> dict:
    if arrangement not in ("aligned", "staggered"):
        raise ValueError("arrangement 必须是 'aligned' 或 'staggered'")
    if min(Re, Pr, Pr_wall) <= 0:
        raise ValueError("Re, Pr, Pr_wall 必须为正")

    # 查找 Re 区间
    chosen = None
    for (arr, (lo, hi)), coeffs in _ZUKAUSKAS_COEFFS.items():
        if arr == arrangement and lo <= Re < hi:
            chosen = coeffs
            break
    if chosen is None:
        raise ValueError(f"Re={Re} 超出 Zukauskas 覆盖范围 [10, 2e6]")

    C, m = chosen
    Nu = C * Re ** m * Pr ** 0.36 * (Pr / Pr_wall) ** 0.25
    ok = (10 <= Re <= 2e6) and (0.7 <= Pr <= 500)
    notes = ("" if ok else
             "Zukauskas 适用 10 ≤ Re ≤ 2e6，0.7 ≤ Pr ≤ 500。")
    h = _convert_nu_to_h(Nu, k_fluid, diameter_m)
    return CorrelationResult(
        Nu, h, ok, notes,
        {"Re": Re, "Pr": Pr, "Pr_wall": Pr_wall,
         "arrangement": arrangement, "C": C, "m": m},
    ).as_dict()


# ---------------------------------------------------------------------------
# 9) Churchill-Chu — 竖板 / 竖圆柱自然对流
# ---------------------------------------------------------------------------

def churchill_chu_vertical(
    Ra: float,
    Pr: float,
    k_fluid: Optional[float] = None,
    characteristic_length_m: Optional[float] = None,
) -> dict:
    if Ra <= 0 or Pr <= 0:
        raise ValueError("Ra 和 Pr 必须为正")
    # 全 Ra 适用公式（含 Ra<1e9 层流部分）
    numer = 0.387 * Ra ** (1.0 / 6.0)
    denom = (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    Nu = (0.825 + numer / denom) ** 2
    ok = Ra < 1e12
    notes = ("" if ok else
             "Ra > 1e12 已超经验范围，仅供粗估。")
    h = _convert_nu_to_h(Nu, k_fluid, characteristic_length_m)
    return CorrelationResult(Nu, h, ok, notes, {"Ra": Ra, "Pr": Pr}).as_dict()


# ---------------------------------------------------------------------------
# 10) McAdams — 水平板自然对流，上热面 / 下热面
# ---------------------------------------------------------------------------

def mcadams_horizontal(
    Ra: float,
    surface: str = "hot_upward",
    k_fluid: Optional[float] = None,
    characteristic_length_m: Optional[float] = None,
) -> dict:
    if Ra <= 0:
        raise ValueError("Ra 必须为正")

    # 有效 Rayleigh 分段
    if surface == "hot_upward":
        if 1e4 <= Ra <= 1e7:
            Nu = 0.54 * Ra ** 0.25
            notes = "层流段 (1e4 ≤ Ra ≤ 1e7)"
            ok = True
        elif 1e7 < Ra <= 1e11:
            Nu = 0.15 * Ra ** (1.0 / 3.0)
            notes = "湍流段 (1e7 < Ra ≤ 1e11)"
            ok = True
        else:
            Nu = 0.54 * Ra ** 0.25
            notes = "Ra 超经验范围 (1e4-1e11)，按层流公式外推，仅供粗估。"
            ok = False
    elif surface == "hot_downward":
        if 1e5 <= Ra <= 1e10:
            Nu = 0.27 * Ra ** 0.25
            notes = "下热面 (1e5 ≤ Ra ≤ 1e10)"
            ok = True
        else:
            Nu = 0.27 * Ra ** 0.25
            notes = "Ra 超经验范围 (1e5-1e10)，外推结果仅供粗估。"
            ok = False
    else:
        raise ValueError("surface 必须是 'hot_upward' 或 'hot_downward'")

    h = _convert_nu_to_h(Nu, k_fluid, characteristic_length_m)
    return CorrelationResult(
        Nu, h, ok, notes, {"Ra": Ra, "surface": surface},
    ).as_dict()


# ---------------------------------------------------------------------------
# 11) 两灰面空间辐射换热：q = sigma*(T1^4-T2^4) / (1/eps1 + 1/eps2 - 1)
# ---------------------------------------------------------------------------

SIGMA = 5.670374419e-8  # Stefan-Boltzmann, W/(m^2 K^4)


def gray_body_radiation(
    T1_K: float,
    T2_K: float,
    emissivity_1: float,
    emissivity_2: float,
) -> dict:
    if T1_K <= 0 or T2_K <= 0:
        raise ValueError("温度必须为正（Kelvin）")
    if not (0 < emissivity_1 <= 1) or not (0 < emissivity_2 <= 1):
        raise ValueError("发射率取值 (0, 1]")

    denom = 1.0 / emissivity_1 + 1.0 / emissivity_2 - 1.0
    q = SIGMA * (T1_K ** 4 - T2_K ** 4) / denom
    # 等效辐射换热系数
    h_rad = q / (T1_K - T2_K) if T1_K != T2_K else float("inf")
    return {
        "q_net_W_per_m2": q,
        "h_rad_W_per_m2K": h_rad,
        "T1_K": T1_K,
        "T2_K": T2_K,
        "emissivity_1": emissivity_1,
        "emissivity_2": emissivity_2,
    }


# ---------------------------------------------------------------------------
# 12) NTU-epsilon 换热器模型（按流动排列）
#    C = Cmin/Cmax ∈ [0, 1]
#    effectiveness:
#      - parallel flow:    eps = (1 - exp(-NTU(1+C))) / (1 + C)
#      - counterflow:      eps = (1 - exp(-NTU(1-C))) / (1 - C*exp(-NTU(1-C)))
#      - shell-and-tube:   ... (single shell)
# ---------------------------------------------------------------------------

def ntu_effectiveness(
    NTU: float,
    C_ratio: float,
    arrangement: str = "counterflow",
) -> dict:
    if NTU < 0:
        raise ValueError("NTU 必须 ≥ 0")
    if not (0 <= C_ratio <= 1):
        raise ValueError("C_ratio ∈ [0, 1]")

    if arrangement == "counterflow":
        if abs(C_ratio - 1.0) < 1e-9:
            eps = NTU / (1.0 + NTU)
        else:
            num = 1.0 - math.exp(-NTU * (1.0 - C_ratio))
            den = 1.0 - C_ratio * math.exp(-NTU * (1.0 - C_ratio))
            eps = num / den
    elif arrangement == "parallel":
        eps = (1.0 - math.exp(-NTU * (1.0 + C_ratio))) / (1.0 + C_ratio)
    elif arrangement == "crossflow_unmixed":
        # 两侧均未混合，Kays & Crawford 近似
        term = NTU ** 0.22
        eps = 1.0 - math.exp((1.0 / C_ratio) * term * (math.exp(-C_ratio * NTU ** 0.78) - 1.0))
    elif arrangement == "shell_and_tube_1_shell":
        if abs(C_ratio) < 1e-9:
            eps = 1.0 - math.exp(-NTU)
        else:
            root = math.sqrt(1.0 + C_ratio ** 2)
            exp_term = math.exp(-NTU * root)
            eps = 2.0 / (
                1.0 + C_ratio + root * (1.0 + exp_term) / (1.0 - exp_term)
            )
    else:
        raise ValueError(
            "arrangement 必须是 "
            "counterflow / parallel / crossflow_unmixed / shell_and_tube_1_shell"
        )

    return {
        "effectiveness": eps,
        "NTU": NTU,
        "C_ratio": C_ratio,
        "arrangement": arrangement,
    }


# ---------------------------------------------------------------------------
# 13) LMTD — 对数平均温差
# ---------------------------------------------------------------------------

def lmtd(dT_1: float, dT_2: float) -> dict:
    if dT_1 <= 0 or dT_2 <= 0:
        raise ValueError("dT_1 和 dT_2 必须为正温差")
    if abs(dT_1 - dT_2) < 1e-9:
        lm = dT_1
    else:
        lm = (dT_1 - dT_2) / math.log(dT_1 / dT_2)
    return {"LMTD_K": lm, "dT_1": dT_1, "dT_2": dT_2}


# ---------------------------------------------------------------------------
# 14) Grashof & Rayleigh numbers — 自然对流常用
# ---------------------------------------------------------------------------

def grashof_number(
    beta: float, g: float, dT: float, L: float, nu: float,
) -> dict:
    if nu <= 0 or L <= 0:
        raise ValueError("nu 和 L 必须为正")
    Gr = g * beta * abs(dT) * L ** 3 / nu ** 2
    return {"Grashof": Gr, "beta_1_per_K": beta, "g_m_per_s2": g,
            "dT_K": dT, "L_m": L, "nu_m2_per_s": nu}


def rayleigh_number(
    beta: float, g: float, dT: float, L: float, nu: float, alpha: float,
) -> dict:
    if nu <= 0 or alpha <= 0 or L <= 0:
        raise ValueError("nu, alpha, L 必须为正")
    Ra = g * beta * abs(dT) * L ** 3 / (nu * alpha)
    return {"Rayleigh": Ra, "Grashof": g * beta * abs(dT) * L ** 3 / nu ** 2,
            "Prandtl": nu / alpha}


# ---------------------------------------------------------------------------
# 工具注册
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "gnielinski",
        "description": (
            "Gnielinski 湍流管内换热关联式，适用 3000<Re<5e6, 0.5<Pr<2000。"
            "比 Dittus-Boelter 更准确。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number"},
                "Pr": {"type": "number"},
                "k_fluid": {"type": "number", "description": "流体导热系数 W/(m·K)"},
                "hydraulic_diameter_m": {"type": "number"},
            },
            "required": ["Re", "Pr"],
        },
    },
    {
        "name": "petukhov",
        "description": "Petukhov 湍流换热，1e4<Re<5e6, 0.5<Pr<2000。",
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number"},
                "Pr": {"type": "number"},
                "k_fluid": {"type": "number"},
                "hydraulic_diameter_m": {"type": "number"},
            },
            "required": ["Re", "Pr"],
        },
    },
    {
        "name": "sieder_tate",
        "description": (
            "Sieder-Tate 湍流换热，带 (mu_bulk/mu_wall)^0.14 粘度修正。"
            "适合大温差下粘度变化显著的液体。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number"},
                "Pr": {"type": "number"},
                "mu_bulk": {"type": "number"},
                "mu_wall": {"type": "number"},
                "k_fluid": {"type": "number"},
                "hydraulic_diameter_m": {"type": "number"},
            },
            "required": ["Re", "Pr", "mu_bulk", "mu_wall"],
        },
    },
    {
        "name": "colburn_j_factor",
        "description": "Colburn j-factor 类比：j=St*Pr^(2/3)=f/8。",
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number"},
                "Pr": {"type": "number"},
                "friction_factor": {"type": "number"},
                "k_fluid": {"type": "number"},
                "hydraulic_diameter_m": {"type": "number"},
            },
            "required": ["Re", "Pr"],
        },
    },
    {
        "name": "hausen_entry",
        "description": "Hausen 层流圆管入口换热关联式（等壁温）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number"},
                "Pr": {"type": "number"},
                "diameter_m": {"type": "number"},
                "length_m": {"type": "number"},
                "k_fluid": {"type": "number"},
            },
            "required": ["Re", "Pr", "diameter_m", "length_m"],
        },
    },
    {
        "name": "sieder_tate_entry",
        "description": "Sieder-Tate 层流入口关联式，带粘度修正。",
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number"},
                "Pr": {"type": "number"},
                "diameter_m": {"type": "number"},
                "length_m": {"type": "number"},
                "mu_bulk": {"type": "number"},
                "mu_wall": {"type": "number"},
                "k_fluid": {"type": "number"},
            },
            "required": ["Re", "Pr", "diameter_m", "length_m", "mu_bulk", "mu_wall"],
        },
    },
    {
        "name": "churchill_bernstein",
        "description": "Churchill-Bernstein 外掠圆柱换热，适用 Re*Pr>0.2。",
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number"},
                "Pr": {"type": "number"},
                "k_fluid": {"type": "number"},
                "diameter_m": {"type": "number"},
            },
            "required": ["Re", "Pr"],
        },
    },
    {
        "name": "zukauskas_tube_bank",
        "description": "Zukauskas 管束外掠换热（对齐 / 错列）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "Re": {"type": "number"},
                "Pr": {"type": "number"},
                "Pr_wall": {"type": "number"},
                "arrangement": {"type": "string",
                                "enum": ["aligned", "staggered"]},
                "k_fluid": {"type": "number"},
                "diameter_m": {"type": "number"},
            },
            "required": ["Re", "Pr", "Pr_wall"],
        },
    },
    {
        "name": "churchill_chu_vertical",
        "description": "Churchill-Chu 竖板/竖圆柱自然对流，全 Ra 适用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "Ra": {"type": "number", "description": "瑞利数"},
                "Pr": {"type": "number"},
                "k_fluid": {"type": "number"},
                "characteristic_length_m": {"type": "number"},
            },
            "required": ["Ra", "Pr"],
        },
    },
    {
        "name": "mcadams_horizontal",
        "description": "McAdams 水平板自然对流（上热面 / 下热面）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "Ra": {"type": "number"},
                "surface": {"type": "string",
                            "enum": ["hot_upward", "hot_downward"]},
                "k_fluid": {"type": "number"},
                "characteristic_length_m": {"type": "number"},
            },
            "required": ["Ra"],
        },
    },
    {
        "name": "gray_body_radiation",
        "description": "两灰体空间辐射净换热。输入两面温度 (K) 和发射率。",
        "input_schema": {
            "type": "object",
            "properties": {
                "T1_K": {"type": "number"},
                "T2_K": {"type": "number"},
                "emissivity_1": {"type": "number"},
                "emissivity_2": {"type": "number"},
            },
            "required": ["T1_K", "T2_K", "emissivity_1", "emissivity_2"],
        },
    },
    {
        "name": "ntu_effectiveness",
        "description": (
            "NTU-epsilon 换热器有效度。arrangement 支持 counterflow / "
            "parallel / crossflow_unmixed / shell_and_tube_1_shell。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "NTU": {"type": "number"},
                "C_ratio": {"type": "number"},
                "arrangement": {
                    "type": "string",
                    "enum": [
                        "counterflow", "parallel",
                        "crossflow_unmixed", "shell_and_tube_1_shell",
                    ],
                },
            },
            "required": ["NTU", "C_ratio"],
        },
    },
    {
        "name": "lmtd",
        "description": "对数平均温差 LMTD=(dT1-dT2)/ln(dT1/dT2)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "dT_1": {"type": "number"},
                "dT_2": {"type": "number"},
            },
            "required": ["dT_1", "dT_2"],
        },
    },
    {
        "name": "grashof_number",
        "description": "Grashof 数 Gr=g*beta*dT*L^3/nu^2。",
        "input_schema": {
            "type": "object",
            "properties": {
                "beta": {"type": "number"},
                "g": {"type": "number"},
                "dT": {"type": "number"},
                "L": {"type": "number"},
                "nu": {"type": "number"},
            },
            "required": ["beta", "g", "dT", "L", "nu"],
        },
    },
    {
        "name": "rayleigh_number",
        "description": "Rayleigh 数 Ra=Gr*Pr。",
        "input_schema": {
            "type": "object",
            "properties": {
                "beta": {"type": "number"},
                "g": {"type": "number"},
                "dT": {"type": "number"},
                "L": {"type": "number"},
                "nu": {"type": "number"},
                "alpha": {"type": "number"},
            },
            "required": ["beta", "g", "dT", "L", "nu", "alpha"],
        },
    },
]


def _mk(fn):
    def _exec(args):
        return json.dumps(fn(**args), ensure_ascii=False, indent=2)
    return _exec


TOOL_EXECUTORS = {
    "gnielinski": _mk(gnielinski),
    "petukhov": _mk(petukhov),
    "sieder_tate": _mk(sieder_tate),
    "colburn_j_factor": _mk(colburn_j_factor),
    "hausen_entry": _mk(hausen_entry),
    "sieder_tate_entry": _mk(sieder_tate_entry),
    "churchill_bernstein": _mk(churchill_bernstein),
    "zukauskas_tube_bank": _mk(zukauskas_tube_bank),
    "churchill_chu_vertical": _mk(churchill_chu_vertical),
    "mcadams_horizontal": _mk(mcadams_horizontal),
    "gray_body_radiation": _mk(gray_body_radiation),
    "ntu_effectiveness": _mk(ntu_effectiveness),
    "lmtd": _mk(lmtd),
    "grashof_number": _mk(grashof_number),
    "rayleigh_number": _mk(rayleigh_number),
}
