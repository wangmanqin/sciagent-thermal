"""
Example 06: 一维稳态导热 + Thomas 算法

物理：-k d²T/dx² = S, 两端 Dirichlet。
"""

import math

from sciagent.solvers import solve_1d_conduction_dirichlet


def main():
    # 1cm 硅棒，体积热源 1e8 W/m³, 两端 T=25°C / T=100°C
    sol = solve_1d_conduction_dirichlet(
        length_m=0.01,
        n_cells=40,
        k_W_per_mK=130.0,
        source_W_per_m3=1e8,
        T_left=25.0,
        T_right=100.0,
    )
    print(f"n_cells={sol['n_cells']}, dx={sol['dx_m']*1e3:.3f} mm")
    for i in range(0, len(sol["x_m"]), 4):
        print(f"  x={sol['x_m'][i]*1e3:.3f} mm, T={sol['T_C'][i]:.2f} °C")

    # 画图
    try:
        from sciagent.viz import plot_1d_profile
        plot_1d_profile(
            x=[x * 1e3 for x in sol["x_m"]],
            y=sol["T_C"],
            output_path="outputs/examples/06_conduction.png",
            xlabel="x (mm)",
            ylabel="T (°C)",
            title="1D Conduction with Volumetric Source",
        )
        print("→ outputs/examples/06_conduction.png")
    except Exception as e:
        print(f"[viz skipped] {e}")


if __name__ == "__main__":
    main()
