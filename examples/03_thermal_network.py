"""
Example 03: 多层热阻网络

简化模型：
  芯片 (Q=80W)
    ↓ 导热 (硅 0.5mm × 1cm²)
    ↓ 并联 (20 × 水通道对流)
    ↓ 能量守恒温升 (caloric)
    → 入口水 25°C
"""

from sciagent.solvers import (
    ResistanceNetwork,
    conduction_resistance_plane,
    convection_resistance,
    caloric_resistance,
)
from sciagent.tools.fluid_properties import water_properties
from sciagent.tools.correlations import shah_london


def main():
    # 导热层
    R_cond = conduction_resistance_plane(
        thickness_m=0.5e-3, area_m2=1e-4, thermal_conductivity=130.0,
    )["resistance_K_per_W"]

    # 每个通道的对流热阻
    p = water_properties(25.0)
    w, h, L = 1e-3, 2e-3, 0.02
    Dh = 4 * (w * h) / (2 * (w + h))
    aspect = w / h
    Nu = shah_london(aspect_ratio=aspect).nu_constant_heat_flux
    h_conv = Nu * p.thermal_conductivity / Dh
    A_wet_per_ch = 2 * (w + h) * L  # 简化：所有内壁面
    R_conv_one = convection_resistance(
        h_W_per_m2K=h_conv, area_m2=A_wet_per_ch,
    )["resistance_K_per_W"]

    # 热容温升（单通道流量 = 总流量 / 20）
    m_dot_total = 2.0 / 60 / 1000 * 1000  # 2 L/min = 0.0333 kg/s（水）
    m_dot_per_ch = m_dot_total / 20
    R_cap_one = caloric_resistance(
        mass_flow_kg_per_s=m_dot_per_ch,
        specific_heat_J_per_kgK=p.specific_heat,
    )["resistance_K_per_W"]

    # 组装：R_cond 串联 (R_conv_parallel 串 R_cap_parallel)
    net = ResistanceNetwork()
    net.add("conduction", R_cond)
    net.add_parallel_group("convection", [R_conv_one] * 20)
    net.add_parallel_group("caloric", [R_cap_one] * 20)

    R_total = net.total()
    Q = 80.0
    dT = Q * R_total
    T_junction = 25.0 + dT

    print(f"R_cond       = {R_cond*1e3:.2f} mK/W")
    print(f"R_conv (1ch) = {R_conv_one*1e3:.2f} mK/W  → parallel × 20 = {R_conv_one/20*1e3:.2f}")
    print(f"R_cap  (1ch) = {R_cap_one*1e3:.2f} mK/W  → parallel × 20 = {R_cap_one/20*1e3:.2f}")
    print(f"R_total      = {R_total*1e3:.2f} mK/W")
    print(f"Q = {Q} W → ΔT = {dT:.2f} K, T_junction ≈ {T_junction:.2f} °C")

    # 详细网络
    summary = net.summary()
    print("\n--- network ---")
    for row in summary:
        print(f"  {row}")


if __name__ == "__main__":
    main()
