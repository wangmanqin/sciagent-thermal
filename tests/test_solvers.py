"""求解器测试：热阻网络 / ODE / 线性代数。"""

from __future__ import annotations
import math


def test_resistance_series():
    from sciagent.solvers import ResistanceNetwork
    net = ResistanceNetwork()
    net.add("r1", 0.1)
    net.add("r2", 0.2)
    net.add("r3", 0.3)
    assert abs(net.total() - 0.6) < 1e-12


def test_resistance_parallel_group():
    from sciagent.solvers import ResistanceNetwork
    net = ResistanceNetwork()
    net.add("ra", 0.1)
    net.add_parallel_group("conv", [0.2, 0.2])  # 并联 → 0.1
    # series: 0.1 + 0.1 = 0.2
    assert abs(net.total() - 0.2) < 1e-12


def test_conduction_resistance_plane():
    from sciagent.solvers import conduction_resistance_plane
    r = conduction_resistance_plane(
        thickness_m=0.01, area_m2=0.1, thermal_conductivity=10.0,
    )
    # R = L/(kA) = 0.01 / (10 * 0.1) = 0.01 K/W
    assert abs(r["resistance_K_per_W"] - 0.01) < 1e-12


def test_conduction_resistance_cylinder():
    from sciagent.solvers import conduction_resistance_cylinder
    r = conduction_resistance_cylinder(
        r_inner_m=0.01, r_outer_m=0.02, length_m=1.0,
        thermal_conductivity=10.0,
    )
    # R = ln(ro/ri) / (2πkL)
    expected = math.log(2.0) / (2 * math.pi * 10.0 * 1.0)
    assert abs(r["resistance_K_per_W"] - expected) < 1e-12


def test_convection_resistance():
    from sciagent.solvers import convection_resistance
    r = convection_resistance(h_W_per_m2K=1000.0, area_m2=0.01)
    # R = 1/(hA) = 1/10 = 0.1
    assert abs(r["resistance_K_per_W"] - 0.1) < 1e-12


def test_rk4_exponential_decay():
    from sciagent.solvers import solve_ode_rk4
    # dy/dt = -y, y(0)=1 → y(t) = e^(-t)
    def f(t, y):
        return [-y[0]]
    sol = solve_ode_rk4(f, t_span=(0, 1), y0=[1.0], n_steps=200)
    err = abs(sol["y"][-1][0] - math.exp(-1.0))
    assert err < 1e-5


def test_rk45_adaptive_tolerance():
    from sciagent.solvers import solve_ode_rk45
    # dy/dt = y, y(0)=1 → y(t) = e^t
    def f(t, y):
        return [y[0]]
    sol = solve_ode_rk45(f, t_span=(0, 1), y0=[1.0], rtol=1e-6, atol=1e-8)
    err = abs(sol["y"][-1][0] - math.e)
    assert err < 1e-4


def test_fin_temperature_distribution_tip_colder():
    from sciagent.solvers import fin_temperature_distribution
    sol = fin_temperature_distribution(
        length_m=0.05, perimeter_m=0.02, area_m2=1e-4,
        thermal_conductivity=200.0, h_W_per_m2K=25.0,
        T_base_C=80.0, T_inf_C=25.0, n_points=50,
    )
    # 翅片基部应 = T_base，尖端应 < T_base
    assert abs(sol["T_C"][0] - 80.0) < 1e-9
    assert sol["T_C"][-1] < 80.0
    # 尖端仍应 > T_inf
    assert sol["T_C"][-1] > 25.0


def test_thomas_tridiagonal():
    from sciagent.solvers import thomas
    # 解 [[2,-1,0],[-1,2,-1],[0,-1,2]] x = [1,0,1]
    a = [0.0, -1.0, -1.0]
    b = [2.0, 2.0, 2.0]
    c = [-1.0, -1.0, 0.0]
    d = [1.0, 0.0, 1.0]
    x = thomas(a, b, c, d)
    # 真解：x = [1, 1, 1]
    for xi in x:
        assert abs(xi - 1.0) < 1e-12


def test_lu_decompose_and_solve():
    from sciagent.solvers import solve_linear_system
    A = [[4, 3], [6, 3]]
    b = [10, 12]
    x = solve_linear_system(A, b)
    # 解：x = [1, 2]
    assert abs(x[0] - 1.0) < 1e-12
    assert abs(x[1] - 2.0) < 1e-12


def test_1d_conduction_no_source_linear():
    from sciagent.solvers import solve_1d_conduction_dirichlet
    sol = solve_1d_conduction_dirichlet(
        length_m=1.0, n_cells=20, k_W_per_mK=1.0,
        source_W_per_m3=0.0, T_left=0.0, T_right=100.0,
    )
    # 无源，Dirichlet 两端 → 温度应线性分布
    xs = sol["x_m"]
    Ts = sol["T_C"]
    for x, T in zip(xs, Ts):
        assert abs(T - 100.0 * x) < 1e-4


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
