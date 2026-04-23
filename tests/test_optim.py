"""优化器测试：DE / PSO / SA / NSGA-II + Pareto 工具。"""

from __future__ import annotations
import math


def _sphere(x):
    return sum(xi ** 2 for xi in x)


def _rosenbrock(x):
    return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
               for i in range(len(x) - 1))


def test_de_sphere_converges():
    from sciagent.optim import differential_evolution
    r = differential_evolution(
        _sphere,
        bounds=[(-5, 5)] * 5,
        population_size=30,
        n_iter=150,
        seed=0,
    )
    assert r.best_f < 1e-3


def test_pso_sphere_converges():
    from sciagent.optim import particle_swarm
    r = particle_swarm(
        _sphere,
        bounds=[(-5, 5)] * 5,
        n_particles=30,
        n_iter=200,
        seed=0,
    )
    assert r.best_f < 1e-2


def test_sa_rosenbrock_improvement():
    from sciagent.optim import simulated_annealing
    r = simulated_annealing(
        _rosenbrock,
        bounds=[(-2, 2)] * 2,
        n_iter=3000,
        seed=0,
    )
    # SA 不保证接近 0，但应远好于一个随机初值的典型 rosenbrock 值（几十）
    assert r.best_f < 50.0


def test_de_maximize_flag():
    from sciagent.optim import differential_evolution
    # 最大化 -x^2 ≡ 最大值 0 在 x=0
    r = differential_evolution(
        lambda x: -(x[0] ** 2),
        bounds=[(-5, 5)],
        population_size=20,
        n_iter=80,
        maximize=True,
        seed=1,
    )
    assert r.best_f > -0.01


def test_dominates():
    from sciagent.optim import dominates
    assert dominates([1, 1], [2, 2])
    assert not dominates([1, 3], [2, 2])
    assert not dominates([1, 1], [1, 1])


def test_non_dominated_sort():
    from sciagent.optim import non_dominated_sort
    # 4 个点：A(1,4) B(2,3) C(3,2) D(4,1) 全在一个前沿上
    objs = [(1, 4), (2, 3), (3, 2), (4, 1)]
    fronts = non_dominated_sort(objs)
    assert len(fronts[0]) == 4


def test_hypervolume_2d_known():
    from sciagent.optim import hypervolume_2d
    # 单点 (1,1)，ref (2,2)，HV = 1
    hv = hypervolume_2d([(1, 1)], (2, 2))
    assert abs(hv - 1.0) < 1e-12


def test_knee_point_returns_valid_index():
    from sciagent.optim import pick_knee_point
    front = [(1, 5), (2, 3), (3, 2), (5, 1)]
    i = pick_knee_point(front)
    assert 0 <= i < len(front)


def test_spacing_metric_uniform():
    from sciagent.optim import spacing_metric
    # 等间距前沿
    front = [(i, 10 - i) for i in range(5)]
    s = spacing_metric(front)
    # 等间距 → spacing std 很小
    assert s < 0.5


def test_nsga2_bi_objective_front_nontrivial():
    """NSGA-II 在一个简单双目标问题上应该至少产出一个前沿。"""
    try:
        from sciagent.optim import run_nsga2
    except ImportError:
        return  # 没装 DEAP 就跳过

    def obj(x):
        # ZDT1-like: f1 = x1, f2 = 1 - sqrt(x1)
        f1 = x[0]
        g = 1.0
        f2 = g * (1 - math.sqrt(x[0] / g))
        return [f1, f2]

    r = run_nsga2(
        objective_function=obj,
        bounds=[(0.01, 1.0)],
        n_objectives=2,
        population_size=30,
        n_generations=15,
        seed=0,
    )
    assert len(r.pareto_variables) >= 1


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
