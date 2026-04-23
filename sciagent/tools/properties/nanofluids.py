"""
纳米流体有效物性模型。

微通道散热器里纳米流体（Al2O3-水、CuO-水）是热点方向，但它的
物性不是查表可得 —— 需要用"颗粒 + 基液"的有效介质模型组合。
本模块实现几组经典模型：
  - 密度：基液-颗粒加权（线性组合）
  - 比热：质量加权平均
  - 粘度：Einstein, Brinkman, Batchelor, Maiga 拟合
  - 导热：Maxwell, Hamilton-Crosser, Yu-Choi
输入体积分数 phi（一般 0~5%），返回与基液同格式的物性。
"""

from __future__ import annotations
import math
import json
from dataclasses import dataclass


# 颗粒库：密度 / 比热 / 导热
PARTICLES = {
    "Al2O3":   {"rho": 3970.0, "cp":  765.0, "k": 40.0},
    "CuO":     {"rho": 6500.0, "cp":  540.0, "k": 18.0},
    "TiO2":    {"rho": 4157.0, "cp":  710.0, "k":  8.4},
    "SiO2":    {"rho": 2220.0, "cp":  745.0, "k":  1.4},
    "Cu":      {"rho": 8933.0, "cp":  385.0, "k": 401.0},
    "Ag":      {"rho": 10500.0,"cp":  235.0, "k": 429.0},
    "CNT":     {"rho": 2100.0, "cp":  425.0, "k": 3000.0},
    "graphene":{"rho": 2200.0, "cp":  700.0, "k": 5000.0},
}


@dataclass
class NanofluidProperties:
    particle: str
    volume_fraction: float
    base_fluid_density: float
    base_fluid_cp: float
    base_fluid_k: float
    base_fluid_mu: float
    density: float
    specific_heat: float
    conductivity: float
    viscosity: float
    prandtl: float
    model_viscosity: str
    model_conductivity: str


def _density(rho_bf, rho_p, phi):
    return (1 - phi) * rho_bf + phi * rho_p


def _cp(rho_bf, cp_bf, rho_p, cp_p, phi):
    rho_mix = _density(rho_bf, rho_p, phi)
    return ((1 - phi) * rho_bf * cp_bf + phi * rho_p * cp_p) / rho_mix


def _mu_einstein(mu_bf, phi):
    # 稀溶液极限，phi << 1
    return mu_bf * (1 + 2.5 * phi)


def _mu_brinkman(mu_bf, phi):
    # Brinkman (1952)：对稀到中等浓度更适用
    return mu_bf / (1 - phi) ** 2.5


def _mu_batchelor(mu_bf, phi):
    # Batchelor (1977)：含二阶项
    return mu_bf * (1 + 2.5 * phi + 6.5 * phi ** 2)


def _mu_maiga(mu_bf, phi):
    # Maiga et al. (2005)：Al2O3-water 拟合
    return mu_bf * (1 + 7.3 * phi + 123 * phi ** 2)


_MU_MODELS = {
    "einstein": _mu_einstein,
    "brinkman": _mu_brinkman,
    "batchelor": _mu_batchelor,
    "maiga": _mu_maiga,
}


def _k_maxwell(k_bf, k_p, phi):
    # Maxwell-Garnett，球形颗粒
    num = k_p + 2 * k_bf + 2 * (k_p - k_bf) * phi
    den = k_p + 2 * k_bf - (k_p - k_bf) * phi
    return k_bf * num / den


def _k_hamilton_crosser(k_bf, k_p, phi, shape_factor_n: float = 3.0):
    # n = 3/psi，球形 psi=1 → n=3
    num = k_p + (shape_factor_n - 1) * k_bf + (shape_factor_n - 1) * (k_p - k_bf) * phi
    den = k_p + (shape_factor_n - 1) * k_bf - (k_p - k_bf) * phi
    return k_bf * num / den


def _k_yu_choi(k_bf, k_p, phi, beta: float = 0.1):
    # Yu & Choi (2003) 带表面纳米层修正
    k_pe = k_p * ((1 + beta) ** 3)
    num = k_pe + 2 * k_bf + 2 * (k_pe - k_bf) * (1 + beta) ** 3 * phi
    den = k_pe + 2 * k_bf - (k_pe - k_bf) * (1 + beta) ** 3 * phi
    return k_bf * num / den


_K_MODELS = {
    "maxwell": lambda kbf, kp, phi: _k_maxwell(kbf, kp, phi),
    "hamilton_crosser": lambda kbf, kp, phi: _k_hamilton_crosser(kbf, kp, phi),
    "yu_choi": lambda kbf, kp, phi: _k_yu_choi(kbf, kp, phi),
}


def nanofluid_properties(
    particle: str,
    volume_fraction: float,
    base_fluid_density: float,
    base_fluid_cp: float,
    base_fluid_k: float,
    base_fluid_mu: float,
    viscosity_model: str = "brinkman",
    conductivity_model: str = "maxwell",
) -> NanofluidProperties:
    if particle not in PARTICLES:
        raise ValueError(f"未知颗粒 '{particle}'，已知：{sorted(PARTICLES)}")
    if not (0.0 <= volume_fraction <= 0.1):
        raise ValueError("体积分数建议 ∈ [0, 0.1]（10%）")
    if viscosity_model not in _MU_MODELS:
        raise ValueError(f"viscosity_model 必须是 {list(_MU_MODELS)}")
    if conductivity_model not in _K_MODELS:
        raise ValueError(f"conductivity_model 必须是 {list(_K_MODELS)}")

    p = PARTICLES[particle]
    phi = volume_fraction

    rho = _density(base_fluid_density, p["rho"], phi)
    cp = _cp(base_fluid_density, base_fluid_cp, p["rho"], p["cp"], phi)
    mu = _MU_MODELS[viscosity_model](base_fluid_mu, phi)
    k = _K_MODELS[conductivity_model](base_fluid_k, p["k"], phi)
    Pr = mu * cp / k

    return NanofluidProperties(
        particle=particle,
        volume_fraction=phi,
        base_fluid_density=base_fluid_density,
        base_fluid_cp=base_fluid_cp,
        base_fluid_k=base_fluid_k,
        base_fluid_mu=base_fluid_mu,
        density=rho,
        specific_heat=cp,
        conductivity=k,
        viscosity=mu,
        prandtl=Pr,
        model_viscosity=viscosity_model,
        model_conductivity=conductivity_model,
    )


TOOL_DEFINITION = {
    "name": "nanofluid_properties",
    "description": (
        "纳米流体（Al2O3/CuO/TiO2/SiO2/Cu/Ag/CNT/graphene + 基液）"
        "有效物性。可选多种粘度 / 导热模型。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "particle": {
                "type": "string",
                "enum": list(PARTICLES.keys()),
            },
            "volume_fraction": {"type": "number"},
            "base_fluid_density": {"type": "number"},
            "base_fluid_cp": {"type": "number"},
            "base_fluid_k": {"type": "number"},
            "base_fluid_mu": {"type": "number"},
            "viscosity_model": {
                "type": "string",
                "enum": list(_MU_MODELS.keys()),
            },
            "conductivity_model": {
                "type": "string",
                "enum": list(_K_MODELS.keys()),
            },
        },
        "required": [
            "particle", "volume_fraction",
            "base_fluid_density", "base_fluid_cp",
            "base_fluid_k", "base_fluid_mu",
        ],
    },
}


def execute(args: dict) -> str:
    props = nanofluid_properties(**args)
    return json.dumps(props.__dict__, ensure_ascii=False, indent=2)
