"""
Example 04: 微通道的换热-压降 Pareto 优化（NSGA-II）

变量：
  n_channels   ∈ [10, 40]  (整数化)
  w_channel_mm ∈ [0.3, 1.5]
  h_channel_mm ∈ [1.0, 3.0]
  flow_Lpm     ∈ [0.5, 4.0]

目标（都要最小化）：
  T_max_C       结温
  pump_power_W  泵功
"""

import math

from sciagent.tools.fluid_properties import water_properties
from sciagent.tools.correlations import shah_london, dittus_boelter
from sciagent.tools.pressure_drop import (
    laminar_friction_factor, colebrook, darcy_weisbach,
    minor_loss, pump_power,
)


def eval_design(x):
    n = max(5, int(round(x[0])))
    w = x[1] * 1e-3
    h = x[2] * 1e-3
    Q_total = x[3] / 60000.0  # L/min → m³/s
    L = 0.02
    Q_heat = 80.0
    T_in = 25.0

    Dh = 4 * (w * h) / (2 * (w + h))
    p = water_properties(T_in)

    # 单通道速度
    A_ch = w * h
    V = Q_total / (n * A_ch)

    Re = p.density * V * Dh / p.viscosity

    # Nu / h
    if Re < 2300:
        aspect = w / h
        Nu = shah_london(aspect_ratio=aspect).nu_constant_heat_flux
    else:
        Nu = dittus_boelter(reynolds=Re, prandtl=p.prandtl, heating=True).nusselt
    h_conv = Nu * p.thermal_conductivity / Dh

    # 简化热阻
    A_wet = 2 * (w + h) * L * n
    R_conv = 1.0 / (h_conv * A_wet)
    m_dot = p.density * Q_total
    R_cap = 1.0 / (m_dot * p.specific_heat)
    R_total = R_conv + R_cap
    T_max = T_in + Q_heat * R_total

    # 压降
    if Re < 2300:
        f = laminar_friction_factor(reynolds=Re)["friction_factor"]
    else:
        f = colebrook(reynolds=Re, relative_roughness=1e-6 / Dh)["friction_factor"]
    dp_line = darcy_weisbach(friction_factor=f, length_m=L, diameter_m=Dh,
                             velocity=V, density=p.density)["pressure_drop_Pa"]
    dp_in = minor_loss("sharp_entrance", V, p.density)["pressure_drop_Pa"]
    dp_out = minor_loss("sharp_exit", V, p.density)["pressure_drop_Pa"]
    dp_total = dp_line + dp_in + dp_out

    pw = pump_power(volumetric_flow_m3_s=Q_total, pressure_drop_Pa=dp_total,
                    efficiency=0.7)["pump_power_W"]

    return [T_max, pw]


def main():
    from sciagent.optim import run_nsga2, pick_knee_point

    bounds = [
        (10.0, 40.0),   # n
        (0.3, 1.5),     # w mm
        (1.0, 3.0),     # h mm
        (0.5, 4.0),     # Q L/min
    ]

    result = run_nsga2(
        objective_function=eval_design,
        bounds=bounds,
        n_objectives=2,
        population_size=60,
        n_generations=40,
        seed=0,
    )

    print(f"\nPareto 点数: {len(result.pareto_variables)}")
    print("前 5 个解（T_max, pump_W）:")
    for v, o in list(zip(result.pareto_variables, result.pareto_objectives))[:5]:
        print(f"  vars={[round(x, 3) for x in v]}  objs={[round(x, 3) for x in o]}")

    knee = pick_knee_point(result.pareto_objectives)
    kv = result.pareto_variables[knee]
    ko = result.pareto_objectives[knee]
    print(f"\nKnee: vars={[round(x, 3) for x in kv]}, objs={[round(x, 3) for x in ko]}")

    # 可视化
    try:
        from sciagent.viz import plot_pareto_2d, build_optimization_report
        plot_pareto_2d(
            all_objs=result.all_objectives,
            pareto_objs=result.pareto_objectives,
            output_path="outputs/examples/04_pareto.png",
            obj_labels=("T_max (°C)", "Pump power (W)"),
            knee_index=knee,
        )
        build_optimization_report(
            output_path="outputs/examples/04_report.png",
            all_objs=result.all_objectives,
            pareto_objs=result.pareto_objectives,
            pareto_vars=result.pareto_variables,
            obj_labels=("T_max (°C)", "Pump power (W)"),
            var_labels=["n_ch", "w_mm", "h_mm", "Q_Lpm"],
            knee_index=knee,
            title="Microchannel Heat Sink: NSGA-II",
        )
        print("\n图已保存到 outputs/examples/")
    except Exception as e:
        print(f"[viz skipped] {e}")


if __name__ == "__main__":
    main()
