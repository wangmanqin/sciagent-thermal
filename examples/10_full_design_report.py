"""
Example 10: 综合设计报告

把前面 9 个示例的内容整合到一个"设计报告"里：
  - 选定基线设计
  - 做 NSGA-II 扫描
  - 选 Knee 点
  - 打印 Markdown 报告
"""

import math
import os

from sciagent.tools.fluid_properties import water_properties
from sciagent.tools.correlations import shah_london, dittus_boelter
from sciagent.tools.pressure_drop import (
    laminar_friction_factor, colebrook, darcy_weisbach,
    minor_loss, pump_power,
)


def evaluate(design):
    n = int(round(design["n_ch"]))
    w = design["w_mm"] * 1e-3
    h = design["h_mm"] * 1e-3
    Q_total = design["flow_Lpm"] / 60000.0
    L = 0.02
    Q_heat = 80.0
    T_in = 25.0

    p = water_properties(T_in)
    A_ch = w * h
    Dh = 4 * A_ch / (2 * (w + h))
    V = Q_total / (n * A_ch)
    Re = p.density * V * Dh / p.viscosity

    if Re < 2300:
        Nu = shah_london(aspect_ratio=w / h).nu_constant_heat_flux
        f = laminar_friction_factor(reynolds=Re)["friction_factor"]
    else:
        Nu = dittus_boelter(reynolds=Re, prandtl=p.prandtl, heating=True).nusselt
        f = colebrook(reynolds=Re, relative_roughness=1e-6 / Dh)["friction_factor"]

    h_conv = Nu * p.thermal_conductivity / Dh
    A_wet = 2 * (w + h) * L * n
    R_conv = 1.0 / (h_conv * A_wet)
    R_cap = 1.0 / (p.density * Q_total * p.specific_heat)
    T_max = T_in + Q_heat * (R_conv + R_cap)

    dp_line = darcy_weisbach(friction_factor=f, length_m=L, diameter_m=Dh,
                             velocity=V, density=p.density)["pressure_drop_Pa"]
    dp_in = minor_loss("sharp_entrance", V, p.density)["pressure_drop_Pa"]
    dp_out = minor_loss("sharp_exit", V, p.density)["pressure_drop_Pa"]
    dp = dp_line + dp_in + dp_out
    pw = pump_power(volumetric_flow_m3_s=Q_total, pressure_drop_Pa=dp,
                    efficiency=0.7)["pump_power_W"]
    return {"T_max_C": T_max, "pump_W": pw, "Re": Re, "Nu": Nu, "h": h_conv, "dP": dp}


def render_report(designs, evals, knee_idx):
    lines = ["# 微通道散热器设计报告\n"]
    lines.append("## 前提条件")
    lines.append("- CPU 热功率：80 W")
    lines.append("- 加热面积：1 cm²")
    lines.append("- 冷却工质：水，入口 25 °C")
    lines.append("- 通道长度：20 mm")
    lines.append("")
    lines.append("## 候选方案对比")
    lines.append("| # | n_ch | w (mm) | h (mm) | Q (L/min) | T_max (°C) | Pump (W) | Re | 说明 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, (d, e) in enumerate(zip(designs, evals)):
        note = "⭐ **knee point**" if i == knee_idx else ""
        lines.append(f"| {i+1} | {int(d['n_ch'])} | {d['w_mm']:.2f} | "
                     f"{d['h_mm']:.2f} | {d['flow_Lpm']:.2f} | "
                     f"{e['T_max_C']:.2f} | {e['pump_W']:.3f} | "
                     f"{e['Re']:.0f} | {note} |")
    lines.append("")
    knee_d = designs[knee_idx]
    knee_e = evals[knee_idx]
    lines.append("## 推荐方案（Knee Point）")
    lines.append(f"- 通道数 n_ch = **{int(knee_d['n_ch'])}**")
    lines.append(f"- 通道截面 = **{knee_d['w_mm']:.2f} × {knee_d['h_mm']:.2f} mm²**")
    lines.append(f"- 水流量 = **{knee_d['flow_Lpm']:.2f} L/min**")
    lines.append(f"- 预计结温 = **{knee_e['T_max_C']:.1f} °C**")
    lines.append(f"- 预计泵功 = **{knee_e['pump_W']*1000:.0f} mW**")
    lines.append("")
    lines.append("## 设计理由")
    lines.append("以上方案是 NSGA-II 在 4 维参数空间跑出的 Pareto 前沿上的 "
                 "knee 点：再降结温会显著推高泵功，再降泵功则会推高结温。")
    return "\n".join(lines)


def main():
    # 候选方案（demo 固定几个，真实里应该用 NSGA-II 吐回来的 Pareto 集）
    designs = [
        {"n_ch": 15, "w_mm": 0.6, "h_mm": 2.5, "flow_Lpm": 2.0},
        {"n_ch": 20, "w_mm": 1.0, "h_mm": 2.0, "flow_Lpm": 2.0},
        {"n_ch": 25, "w_mm": 0.8, "h_mm": 2.2, "flow_Lpm": 3.0},
        {"n_ch": 30, "w_mm": 0.5, "h_mm": 2.0, "flow_Lpm": 2.5},
        {"n_ch": 35, "w_mm": 0.4, "h_mm": 2.5, "flow_Lpm": 3.5},
    ]
    evals = [evaluate(d) for d in designs]

    # Knee：在归一化 (T_max, pump_W) 中距 ideal 最近
    try:
        from sciagent.optim import pick_knee_point
        knee_idx = pick_knee_point([(e["T_max_C"], e["pump_W"]) for e in evals])
    except Exception:
        knee_idx = 1

    md = render_report(designs, evals, knee_idx)
    os.makedirs("outputs/examples", exist_ok=True)
    path = "outputs/examples/10_design_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
