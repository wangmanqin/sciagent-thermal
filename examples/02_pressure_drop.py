"""
Example 02: 压降全链路

计算：层流 f = 64/Re；湍流走 Colebrook / Swamee-Jain；加入入口突缩和
出口突扩的 K 值；最后算泵功。
"""

from sciagent.tools.fluid_properties import water_properties
from sciagent.tools.pressure_drop import (
    laminar_friction_factor, colebrook, swamee_jain,
    darcy_weisbach, minor_loss, pump_power,
)


def main():
    # 水 25°C
    p = water_properties(25.0)
    D = 1.2e-3
    L = 0.02
    V = 2.0
    roughness = 1e-6
    eps_D = roughness / D
    Re = p.density * V * D / p.viscosity
    print(f"Re = {Re:.0f}, ε/D = {eps_D:.2e}")

    # 沿程
    if Re < 2300:
        f = laminar_friction_factor(reynolds=Re)["friction_factor"]
        source = "laminar"
    else:
        f_cb = colebrook(reynolds=Re, relative_roughness=eps_D)["friction_factor"]
        f_sj = swamee_jain(reynolds=Re, relative_roughness=eps_D)["friction_factor"]
        f = f_cb
        source = f"Colebrook (SJ approx={f_sj:.4f})"
    print(f"f = {f:.4f} ({source})")

    dp_line = darcy_weisbach(
        friction_factor=f, length_m=L, diameter_m=D,
        velocity=V, density=p.density,
    )["pressure_drop_Pa"]
    print(f"ΔP_沿程 = {dp_line:.1f} Pa")

    # 入口突缩 + 出口突扩
    dp_in = minor_loss(
        fitting_type="sharp_entrance", velocity=V, density=p.density,
    )["pressure_drop_Pa"]
    dp_out = minor_loss(
        fitting_type="sharp_exit", velocity=V, density=p.density,
    )["pressure_drop_Pa"]
    print(f"ΔP_入口 = {dp_in:.1f} Pa, ΔP_出口 = {dp_out:.1f} Pa")

    total = dp_line + dp_in + dp_out
    print(f"ΔP_total = {total:.1f} Pa")

    # 泵功 (假设 Q 对应 V * A_c)
    A_c = 3.14159 * (D / 2) ** 2
    Q = V * A_c
    pw = pump_power(
        volumetric_flow_m3_s=Q, pressure_drop_Pa=total, efficiency=0.7,
    )["pump_power_W"]
    print(f"Q = {Q*1e6:.2f} mL/s, pump power = {pw*1e3:.2f} mW")


if __name__ == "__main__":
    main()
