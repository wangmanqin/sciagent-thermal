"""流体物性测试：水、乙二醇、空气、纳米流体。"""

from __future__ import annotations


def test_water_basic_range():
    from sciagent.tools.fluid_properties import water_properties
    p = water_properties(25.0)
    # 25°C 的水：ρ≈997, cp≈4182, k≈0.607, mu≈0.00089, Pr≈6.1
    assert 990 < p.density < 1005
    assert 4170 < p.specific_heat < 4200
    assert 0.59 < p.thermal_conductivity < 0.63
    assert 0.0007 < p.viscosity < 0.001
    assert 5 < p.prandtl < 7


def test_water_iapws_extended():
    from sciagent.tools.properties.water_iapws import water_properties_extended
    p = water_properties_extended(50.0)
    # 50°C: ρ≈988, cp≈4181, Pr≈3.55
    assert 985 < p.density < 992
    assert 3.0 < p.prandtl < 4.2
    # beta 在正值
    assert p.thermal_expansion > 0
    # 饱和压力应 < 1 atm
    assert p.saturation_pressure_Pa < 1.2e4


def test_water_iapws_monotonic_density():
    from sciagent.tools.properties.water_iapws import water_properties_extended
    # 密度随温度降低
    p10 = water_properties_extended(10.0)
    p90 = water_properties_extended(90.0)
    assert p10.density > p90.density


def test_ethylene_glycol_viscosity_increases_with_concentration():
    from sciagent.tools.properties.ethylene_glycol import ethylene_glycol_properties
    p0 = ethylene_glycol_properties(temperature_C=30, mass_fraction=0.0)
    p40 = ethylene_glycol_properties(temperature_C=30, mass_fraction=0.4)
    # 40% EG 的粘度远大于纯水
    assert p40.viscosity > 1.5 * p0.viscosity


def test_air_sutherland():
    from sciagent.tools.properties.air import air_properties
    p25 = air_properties(25.0, pressure_Pa=101325.0)
    # 25°C 空气：ρ≈1.18, mu≈1.85e-5, k≈0.026
    assert 1.1 < p25.density < 1.25
    assert 1.7e-5 < p25.viscosity < 2.0e-5
    assert 0.024 < p25.thermal_conductivity < 0.028


def test_air_pressure_scaling():
    from sciagent.tools.properties.air import air_properties
    p1 = air_properties(25.0, pressure_Pa=101325.0)
    p2 = air_properties(25.0, pressure_Pa=202650.0)
    # 密度随压力线性
    assert abs(p2.density - 2 * p1.density) / p1.density < 1e-6


def test_nanofluid_k_maxwell_increases():
    from sciagent.tools.properties.nanofluids import nanofluid_properties
    r = nanofluid_properties(
        base_fluid={"density": 997, "specific_heat": 4180,
                    "thermal_conductivity": 0.607, "viscosity": 8.9e-4},
        particle="Al2O3",
        volume_fraction=0.02,
        k_model="maxwell",
        mu_model="einstein",
    )
    # Al2O3 纳米流体 k 应该大于纯水
    assert r["thermal_conductivity"] > 0.607
    # 2% 体积分数，Einstein 下 μ = μ0 * (1+2.5φ) = μ0 * 1.05
    assert abs(r["viscosity"] / 8.9e-4 - 1.05) < 0.01


def test_nanofluid_cnt_hamilton_crosser_stronger():
    from sciagent.tools.properties.nanofluids import nanofluid_properties
    base = {"density": 997, "specific_heat": 4180,
            "thermal_conductivity": 0.607, "viscosity": 8.9e-4}
    # CNT 非球形颗粒，Hamilton-Crosser 会比 Maxwell 给出更大的 k 增益
    r_max = nanofluid_properties(
        base_fluid=base, particle="CNT", volume_fraction=0.01,
        k_model="maxwell", mu_model="einstein",
    )
    r_hc = nanofluid_properties(
        base_fluid=base, particle="CNT", volume_fraction=0.01,
        k_model="hamilton_crosser", mu_model="einstein",
        particle_shape_factor=6.0,
    )
    assert r_hc["thermal_conductivity"] >= r_max["thermal_conductivity"]


def test_water_tool_definition_registered():
    from sciagent.tools import TOOL_DEFINITIONS
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert "water_properties" in names


def test_all_properties_registered():
    from sciagent.tools import TOOL_DEFINITIONS
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    # 四种介质都挂在工具总线上
    for n in ["water_properties_extended", "ethylene_glycol_properties",
              "air_properties", "nanofluid_properties"]:
        assert n in names, f"缺少工具 {n}"


if __name__ == "__main__":
    import sys
    fns = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed.")
    sys.exit(0 if passed == len(fns) else 1)
