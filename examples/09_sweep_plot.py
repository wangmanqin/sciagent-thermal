"""
Example 09: 参数扫描 + 可视化

扫描流量从 0.5 到 5 L/min（20 点），输出 Re、Nu、h、ΔP 四合一图。
"""

from sciagent.tools.fluid_properties import water_properties
from sciagent.tools.correlations import shah_london, dittus_boelter
from sciagent.tools.pressure_drop import (
    laminar_friction_factor, colebrook, darcy_weisbach,
)


def main():
    p = water_properties(25.0)
    n_ch = 20
    w, h = 1e-3, 2e-3
    L = 0.02
    A_ch = w * h
    Dh = 4 * A_ch / (2 * (w + h))
    aspect = w / h

    flows_Lpm = [0.5 + i * (5 - 0.5) / 19 for i in range(20)]
    Res, Nus, hs, dps = [], [], [], []

    for Q in flows_Lpm:
        V = (Q / 60 / 1000) / (n_ch * A_ch)
        Re = p.density * V * Dh / p.viscosity
        if Re < 2300:
            Nu = shah_london(aspect_ratio=aspect).nu_constant_heat_flux
            f = laminar_friction_factor(reynolds=Re)["friction_factor"]
        else:
            Nu = dittus_boelter(reynolds=Re, prandtl=p.prandtl, heating=True).nusselt
            f = colebrook(reynolds=Re, relative_roughness=1e-6 / Dh)["friction_factor"]
        h_conv = Nu * p.thermal_conductivity / Dh
        dp = darcy_weisbach(friction_factor=f, length_m=L, diameter_m=Dh,
                            velocity=V, density=p.density)["pressure_drop_Pa"]
        Res.append(Re)
        Nus.append(Nu)
        hs.append(h_conv)
        dps.append(dp)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os

        os.makedirs("outputs/examples", exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, ys, ylabel in zip(
            axes.flat,
            [Res, Nus, hs, dps],
            ["Re", "Nu", "h (W/m²K)", "ΔP (Pa)"],
        ):
            ax.plot(flows_Lpm, ys, "o-", color="crimson")
            ax.set_xlabel("Flow (L/min)")
            ax.set_ylabel(ylabel)
            ax.grid(True, ls=":", alpha=0.5)
        fig.suptitle("Flow Sweep: Re, Nu, h, ΔP")
        fig.tight_layout()
        fig.savefig("outputs/examples/09_sweep.png", dpi=150)
        plt.close(fig)
        print("→ outputs/examples/09_sweep.png")
    except Exception as e:
        print(f"[plot skipped] {e}")

    # 顺带打印 transition 点
    for i, Re in enumerate(Res):
        if Re >= 2300:
            print(f"层湍过渡发生在 Q ≈ {flows_Lpm[i]:.2f} L/min (Re={Re:.0f})")
            break


if __name__ == "__main__":
    main()
