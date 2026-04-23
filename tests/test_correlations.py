"""对流换热 / 压降经验关联式的单元测试。"""

from __future__ import annotations
import math


def test_dittus_boelter_typical_water():
    from sciagent.tools.correlations import dittus_boelter
    # 典型水流：Re=10000, Pr=7，heating
    r = dittus_boelter(reynolds=10000, prandtl=7.0, heating=True)
    # Nu = 0.023 * 10000^0.8 * 7^0.4 ≈ 0.023 * 1584.89 * 2.178 ≈ 79.4
    assert 70 < r.nusselt < 90


def test_shah_london_square_channel():
    from sciagent.tools.correlations import shah_london
    # 方形通道 aspect_ratio=1, fRe ≈ 56.92, Nu_const_q ≈ 3.61
    r = shah_london(aspect_ratio=1.0)
    assert abs(r.f_reynolds - 56.92) < 0.5
    assert abs(r.nu_constant_heat_flux - 3.61) < 0.1


def test_gnielinski_vs_dittus_boelter_order():
    from sciagent.tools.correlations import dittus_boelter
    from sciagent.tools.heat_transfer import gnielinski
    # Gnielinski 在湍流区通常给出比 Dittus-Boelter 低一些的 Nu
    r_db = dittus_boelter(reynolds=50000, prandtl=4.0, heating=True)
    r_g = gnielinski(reynolds=50000, prandtl=4.0)
    # 至少是同一个量级
    assert 0.3 * r_db.nusselt < r_g.nusselt < 1.5 * r_db.nusselt


def test_petukhov_high_re():
    from sciagent.tools.heat_transfer import petukhov
    # Re=1e5, Pr=4.0，Nu 应该在几百量级
    r = petukhov(reynolds=1e5, prandtl=4.0)
    assert 200 < r.nusselt < 2000


def test_colburn_j_factor_symmetric():
    from sciagent.tools.heat_transfer import colburn_j_factor
    # j = St * Pr^(2/3)，和 Nu/(Re Pr) 等价换算
    r = colburn_j_factor(reynolds=20000, prandtl=4.0)
    # j 因子的量级在 1e-3
    assert 1e-4 < r.j_factor < 1e-1


def test_hausen_entry_regions():
    from sciagent.tools.heat_transfer import hausen_entry
    # 短管（L/D 小）Nu 比 fully developed 大
    r_short = hausen_entry(reynolds=500, prandtl=4.0, length_m=0.1, diameter_m=0.01)
    r_long = hausen_entry(reynolds=500, prandtl=4.0, length_m=5.0, diameter_m=0.01)
    assert r_short.nusselt > r_long.nusselt


def test_churchill_bernstein_cross_flow():
    from sciagent.tools.heat_transfer import churchill_bernstein
    r = churchill_bernstein(reynolds=1000, prandtl=0.7)
    # 圆柱横掠，Pr=0.7 的 Nu 量级在 10 左右
    assert 5 < r.nusselt < 50


def test_churchill_chu_vertical_plate():
    from sciagent.tools.heat_transfer import churchill_chu_vertical
    # Ra=1e9, Pr=0.7，典型自然对流
    r = churchill_chu_vertical(rayleigh=1e9, prandtl=0.7)
    assert 50 < r.nusselt < 200


def test_zukauskas_tube_bank():
    from sciagent.tools.heat_transfer import zukauskas_tube_bank
    r = zukauskas_tube_bank(
        reynolds=5000, prandtl=4.0,
        arrangement="aligned",
        n_rows=10,
    )
    assert r.nusselt > 0


def test_lmtd_counterflow():
    from sciagent.tools.heat_transfer import lmtd
    # 热侧 80→50，冷侧 20→40，逆流
    r = lmtd(T_hot_in=80, T_hot_out=50, T_cold_in=20, T_cold_out=40,
             flow="counterflow")
    # 端差 = 80-40 = 40, 50-20 = 30，LMTD = (40-30)/ln(40/30) ≈ 34.76
    assert 34 < r["lmtd_C"] < 36


def test_ntu_effectiveness_counterflow():
    from sciagent.tools.heat_transfer import ntu_effectiveness
    r = ntu_effectiveness(ntu=2.0, C_r=0.5, flow="counterflow")
    # 逆流, NTU=2, Cr=0.5 → ε ≈ 0.82
    assert 0.75 < r["effectiveness"] < 0.9


def test_grashof_rayleigh_consistency():
    from sciagent.tools.heat_transfer import grashof_number, rayleigh_number
    gr = grashof_number(g=9.81, beta=3e-4, delta_T=20, length_m=0.1, nu=1e-6)
    ra = rayleigh_number(g=9.81, beta=3e-4, delta_T=20,
                         length_m=0.1, nu=1e-6, alpha=1.4e-7)
    # Ra = Gr * Pr，Pr = nu/alpha ≈ 7
    pr = 1e-6 / 1.4e-7
    assert abs(ra["rayleigh"] - gr["grashof"] * pr) / ra["rayleigh"] < 1e-6


def test_laminar_friction_factor_round():
    from sciagent.tools.pressure_drop import laminar_friction_factor
    r = laminar_friction_factor(reynolds=1000)
    # f = 64/Re
    assert abs(r["friction_factor"] - 0.064) < 1e-6


def test_colebrook_vs_swamee_jain():
    from sciagent.tools.pressure_drop import colebrook, swamee_jain
    kwargs = dict(reynolds=1e5, relative_roughness=1e-4)
    r1 = colebrook(**kwargs)
    r2 = swamee_jain(**kwargs)
    # 两者差异应该 <5%
    diff = abs(r1["friction_factor"] - r2["friction_factor"])
    rel = diff / r1["friction_factor"]
    assert rel < 0.05


def test_darcy_weisbach_sanity():
    from sciagent.tools.pressure_drop import darcy_weisbach
    r = darcy_weisbach(friction_factor=0.02, length_m=1.0,
                       diameter_m=0.01, velocity=1.0, density=1000)
    # ΔP = f * L/D * ρv²/2 = 0.02 * 100 * 500 = 1000 Pa
    assert abs(r["pressure_drop_Pa"] - 1000) < 1


def test_minor_loss_entrance():
    from sciagent.tools.pressure_drop import minor_loss
    r = minor_loss(fitting_type="sharp_entrance", velocity=1.0, density=1000)
    # K ≈ 0.5, ΔP = 0.5 * ρv²/2 = 250
    assert abs(r["pressure_drop_Pa"] - 250) < 1


def test_pump_power_order():
    from sciagent.tools.pressure_drop import pump_power
    r = pump_power(volumetric_flow_m3_s=1e-4, pressure_drop_Pa=1e5,
                   efficiency=0.7)
    # P = Q*ΔP/η ≈ 1e-4 * 1e5 / 0.7 ≈ 14.3 W
    assert 13 < r["pump_power_W"] < 15


if __name__ == "__main__":
    import inspect, sys
    fns = [(name, obj) for name, obj in globals().items()
           if name.startswith("test_") and callable(obj)]
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
    print(f"\n{passed}/{len(fns)} tests passed.")
    sys.exit(0 if passed == len(fns) else 1)
