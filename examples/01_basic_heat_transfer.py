"""
Example 01: 基本内流换热

场景：
  矩形微通道，水流入口 25°C、速度 1 m/s、通道 1mm × 2mm × 20mm。
  算 Re, Nu（Shah-London）, h, 并看看层流/湍流分界。

这个示例直接在 Python 里用工具函数，不走 LLM，用来演示"工具层可以
独立使用"。
"""

from sciagent.tools.fluid_properties import water_properties
from sciagent.tools.correlations import shah_london, hydraulic_diameter
from sciagent.tools.geometry import rectangular_cross_section


def main():
    # 几何
    w = 1e-3     # channel width
    h = 2e-3     # channel height
    L = 20e-3    # channel length
    geom = rectangular_cross_section(width_m=w, height_m=h)
    Dh = geom["hydraulic_diameter_m"]
    A_c = geom["area_m2"]
    print(f"Dh = {Dh*1e3:.3f} mm, A = {A_c*1e6:.3f} mm²")

    # 物性（25°C）
    p = water_properties(temperature_C=25.0)
    print(f"ρ={p.density:.1f} kg/m³, cp={p.specific_heat:.0f} J/kg·K, "
          f"k={p.thermal_conductivity:.3f} W/m·K, μ={p.viscosity*1e3:.3f} mPa·s, "
          f"Pr={p.prandtl:.2f}")

    # Re
    V = 1.0
    Re = p.density * V * Dh / p.viscosity
    print(f"Re = {Re:.0f}")

    # 选 Nu
    if Re < 2300:
        aspect_ratio = w / h  # 0.5
        r = shah_london(aspect_ratio=aspect_ratio)
        Nu = r.nu_constant_heat_flux
        print(f"[laminar] Shah-London, AR={aspect_ratio}: Nu = {Nu:.2f}")
    else:
        from sciagent.tools.heat_transfer import gnielinski
        r = gnielinski(reynolds=Re, prandtl=p.prandtl)
        Nu = r.nusselt
        print(f"[turbulent] Gnielinski: Nu = {Nu:.2f}")

    # h
    h_conv = Nu * p.thermal_conductivity / Dh
    print(f"h = {h_conv:.1f} W/m²K")

    # ΔT
    m_dot = p.density * V * A_c
    Q = 4.0  # 4 W per channel as a sanity number
    dT = Q / (m_dot * p.specific_heat)
    print(f"m_dot = {m_dot*1e3:.3f} g/s, ΔT (per channel) = {dT:.3f} °C")


if __name__ == "__main__":
    main()
