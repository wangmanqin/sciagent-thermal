"""
Example 07: 纳米流体物性增强

对比水 / Al₂O₃-水 / CNT-水（体积分数 2%）的 k, μ, Nu 提升比。
"""

from sciagent.tools.fluid_properties import water_properties
from sciagent.tools.properties.nanofluids import nanofluid_properties


def main():
    base = water_properties(30.0)
    base_dict = {
        "density": base.density,
        "specific_heat": base.specific_heat,
        "thermal_conductivity": base.thermal_conductivity,
        "viscosity": base.viscosity,
    }

    cases = [
        ("Al2O3-water (Maxwell, Einstein)", {
            "particle": "Al2O3", "volume_fraction": 0.02,
            "k_model": "maxwell", "mu_model": "einstein",
        }),
        ("Al2O3-water (Maxwell, Brinkman)", {
            "particle": "Al2O3", "volume_fraction": 0.02,
            "k_model": "maxwell", "mu_model": "brinkman",
        }),
        ("CuO-water", {
            "particle": "CuO", "volume_fraction": 0.02,
            "k_model": "maxwell", "mu_model": "einstein",
        }),
        ("CNT-water (Hamilton-Crosser)", {
            "particle": "CNT", "volume_fraction": 0.01,
            "k_model": "hamilton_crosser", "mu_model": "einstein",
            "particle_shape_factor": 6.0,
        }),
    ]

    print(f"Base water 30°C: ρ={base.density:.1f}, cp={base.specific_heat:.0f}, "
          f"k={base.thermal_conductivity:.3f}, μ={base.viscosity*1e3:.3f} mPa·s")
    print()
    print(f"{'Case':45s}  k_ratio  mu_ratio  k_nf    μ_nf")
    for name, kw in cases:
        r = nanofluid_properties(base_fluid=base_dict, **kw)
        k_ratio = r["thermal_conductivity"] / base.thermal_conductivity
        mu_ratio = r["viscosity"] / base.viscosity
        print(f"{name:45s}  {k_ratio:.3f}   {mu_ratio:.3f}    "
              f"{r['thermal_conductivity']:.3f}  "
              f"{r['viscosity']*1e3:.3f} mPa·s")


if __name__ == "__main__":
    main()
