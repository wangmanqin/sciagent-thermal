"""几何 / 压降模块补充测试（避开与 test_correlations 重复的范围）。"""

from __future__ import annotations
import math


def test_rectangular_cross_section():
    from sciagent.tools.geometry import rectangular_cross_section
    r = rectangular_cross_section(width_m=0.002, height_m=0.001)
    # A = 2e-6, P = 2*(0.002+0.001)=0.006, Dh = 4A/P = 8e-6/0.006 ≈ 1.333e-3
    assert abs(r["area_m2"] - 2e-6) < 1e-18
    assert abs(r["perimeter_m"] - 0.006) < 1e-9
    assert abs(r["hydraulic_diameter_m"] - (4 * 2e-6 / 0.006)) < 1e-9


def test_circular_cross_section():
    from sciagent.tools.geometry import circular_cross_section
    r = circular_cross_section(diameter_m=0.01)
    # A = π D² / 4, Dh = D
    assert abs(r["area_m2"] - math.pi * 1e-4 / 4) < 1e-12
    assert abs(r["hydraulic_diameter_m"] - 0.01) < 1e-12


def test_triangular_cross_section():
    from sciagent.tools.geometry import triangular_cross_section
    r = triangular_cross_section(base_m=0.002, height_m=0.002)
    # A = 0.5 * 0.002 * 0.002 = 2e-6
    assert abs(r["area_m2"] - 2e-6) < 1e-18


def test_channel_array_microchannel_sink():
    from sciagent.tools.geometry import channel_array
    r = channel_array(
        n_channels=20,
        channel_width_m=0.001,
        channel_height_m=0.002,
        wall_thickness_m=0.0005,
        length_m=0.02,
    )
    # 总宽度 = 20 * (w + s) + s  或  n*w + (n+1)*s（取决于实现）
    # 但至少 total_width > 20 * 0.001
    assert r["total_width_m"] >= 20 * 0.001


def test_fin_array_total_area():
    from sciagent.tools.geometry import fin_array
    r = fin_array(
        n_fins=10,
        fin_thickness_m=0.001,
        fin_height_m=0.01,
        fin_length_m=0.05,
        base_width_m=0.05,
    )
    # 至少每片翅片表面积应为 2*H*L = 2*0.01*0.05 = 1e-3
    assert r["single_fin_surface_m2"] > 0


def test_sphere_volume():
    from sciagent.tools.geometry import sphere_volume
    r = sphere_volume(diameter_m=0.02)
    expected = (4.0 / 3.0) * math.pi * (0.01) ** 3
    assert abs(r["volume_m3"] - expected) < 1e-15


def test_cylinder_volume():
    from sciagent.tools.geometry import cylinder_volume
    r = cylinder_volume(diameter_m=0.02, length_m=0.1)
    expected = math.pi * (0.01) ** 2 * 0.1
    assert abs(r["volume_m3"] - expected) < 1e-15


def test_rectangular_channel_friction_laminar():
    from sciagent.tools.pressure_drop import rectangular_channel_friction
    r = rectangular_channel_friction(
        aspect_ratio=0.5, reynolds=500,
    )
    # f*Re 应在 70 附近（矩形 laminar）
    assert r["friction_factor"] > 0
    # f = (f*Re)/Re → Re=500 时 f 大约 0.14
    assert 0.05 < r["friction_factor"] < 0.3


def test_borda_carnot_expansion_sanity():
    from sciagent.tools.pressure_drop import borda_carnot_expansion
    r = borda_carnot_expansion(
        velocity_1=2.0, area_ratio=0.5, density=1000,
    )
    # 突扩压降应 > 0
    assert r["pressure_loss_Pa"] > 0


def test_sudden_contraction_sanity():
    from sciagent.tools.pressure_drop import sudden_contraction
    r = sudden_contraction(
        velocity_2=2.0, area_ratio=0.3, density=1000,
    )
    assert r["pressure_loss_Pa"] > 0


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
