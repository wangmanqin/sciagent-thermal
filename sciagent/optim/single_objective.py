"""
单目标全局优化：差分进化 (DE)，粒子群 (PSO)。

两者都是无梯度元启发式，常用于散热器单目标优化（比如"最小化热阻"
或"最大化换热系数"）。接口风格和 NSGA-II 保持一致：给函数和 bounds，
拿 best_x / best_f / history。
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple


Bounds = Sequence[Tuple[float, float]]
ScalarFn = Callable[[Sequence[float]], float]


@dataclass
class OptimResult:
    best_x: List[float]
    best_f: float
    history_f: List[float]
    n_evaluations: int
    method: str
    converged: bool


# ---------------------------------------------------------------------------
# Differential Evolution (classic rand/1/bin)
# ---------------------------------------------------------------------------

def differential_evolution(
    fn: ScalarFn,
    bounds: Bounds,
    population_size: int = 30,
    F: float = 0.5,
    CR: float = 0.9,
    n_iter: int = 200,
    tol: float = 1e-8,
    seed: int = 42,
    maximize: bool = False,
) -> OptimResult:
    random.seed(seed)
    n = len(bounds)
    sign = -1 if maximize else 1

    def clip(x):
        return [max(lo, min(hi, xi)) for xi, (lo, hi) in zip(x, bounds)]

    pop = [
        [random.uniform(lo, hi) for lo, hi in bounds]
        for _ in range(population_size)
    ]
    fitness = [sign * fn(x) for x in pop]
    n_eval = population_size

    best_idx = min(range(population_size), key=lambda i: fitness[i])
    best_x, best_f = list(pop[best_idx]), fitness[best_idx]
    history = [best_f]
    converged = False

    for _ in range(n_iter):
        new_pop = []
        new_fit = []
        for i in range(population_size):
            idxs = [k for k in range(population_size) if k != i]
            a, b, c = random.sample(idxs, 3)
            mutant = [pop[a][j] + F * (pop[b][j] - pop[c][j]) for j in range(n)]
            mutant = clip(mutant)
            # binomial crossover
            R = random.randrange(n)
            trial = [
                mutant[j] if random.random() < CR or j == R else pop[i][j]
                for j in range(n)
            ]
            f_trial = sign * fn(trial)
            n_eval += 1
            if f_trial <= fitness[i]:
                new_pop.append(trial)
                new_fit.append(f_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = list(trial)
            else:
                new_pop.append(pop[i])
                new_fit.append(fitness[i])
        pop = new_pop
        fitness = new_fit
        history.append(best_f)
        # 收敛判据：种群适应度方差
        mean = sum(fitness) / len(fitness)
        var = sum((f - mean) ** 2 for f in fitness) / len(fitness)
        if var < tol:
            converged = True
            break

    return OptimResult(
        best_x=best_x,
        best_f=sign * best_f,
        history_f=[sign * h for h in history],
        n_evaluations=n_eval,
        method="differential_evolution",
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Particle Swarm Optimization (classic inertia weight)
# ---------------------------------------------------------------------------

def particle_swarm(
    fn: ScalarFn,
    bounds: Bounds,
    n_particles: int = 40,
    n_iter: int = 200,
    w: float = 0.72,
    c1: float = 1.49,
    c2: float = 1.49,
    v_max_ratio: float = 0.2,
    seed: int = 42,
    maximize: bool = False,
) -> OptimResult:
    random.seed(seed)
    n = len(bounds)
    sign = -1 if maximize else 1

    span = [hi - lo for lo, hi in bounds]
    v_max = [v_max_ratio * s for s in span]

    x = [[random.uniform(lo, hi) for lo, hi in bounds] for _ in range(n_particles)]
    v = [[random.uniform(-vm, vm) for vm in v_max] for _ in range(n_particles)]
    p_best = [list(xi) for xi in x]
    p_best_f = [sign * fn(xi) for xi in x]
    g_idx = min(range(n_particles), key=lambda i: p_best_f[i])
    g_best = list(x[g_idx])
    g_best_f = p_best_f[g_idx]

    n_eval = n_particles
    history = [g_best_f]

    for _ in range(n_iter):
        for i in range(n_particles):
            for j in range(n):
                r1 = random.random()
                r2 = random.random()
                v[i][j] = (
                    w * v[i][j]
                    + c1 * r1 * (p_best[i][j] - x[i][j])
                    + c2 * r2 * (g_best[j] - x[i][j])
                )
                v[i][j] = max(-v_max[j], min(v_max[j], v[i][j]))
                x[i][j] += v[i][j]
                lo, hi = bounds[j]
                if x[i][j] < lo:
                    x[i][j] = lo
                    v[i][j] *= -0.5
                elif x[i][j] > hi:
                    x[i][j] = hi
                    v[i][j] *= -0.5

            f = sign * fn(x[i])
            n_eval += 1
            if f < p_best_f[i]:
                p_best_f[i] = f
                p_best[i] = list(x[i])
                if f < g_best_f:
                    g_best_f = f
                    g_best = list(x[i])
        history.append(g_best_f)

    return OptimResult(
        best_x=g_best,
        best_f=sign * g_best_f,
        history_f=[sign * h for h in history],
        n_evaluations=n_eval,
        method="particle_swarm",
        converged=False,
    )


# ---------------------------------------------------------------------------
# Simulated annealing（标量问题小玩具）
# ---------------------------------------------------------------------------

def simulated_annealing(
    fn: ScalarFn,
    bounds: Bounds,
    n_iter: int = 5000,
    T0: float = 1.0,
    alpha: float = 0.995,
    seed: int = 42,
    maximize: bool = False,
) -> OptimResult:
    import math

    random.seed(seed)
    n = len(bounds)
    sign = -1 if maximize else 1

    x = [random.uniform(lo, hi) for lo, hi in bounds]
    f = sign * fn(x)
    best_x, best_f = list(x), f
    T = T0
    history = [f]

    for _ in range(n_iter):
        step = [random.gauss(0, 1) * (hi - lo) * 0.1
                for lo, hi in bounds]
        cand = [max(lo, min(hi, xi + si))
                for xi, si, (lo, hi) in zip(x, step, bounds)]
        f_cand = sign * fn(cand)
        if f_cand < f or random.random() < math.exp(-(f_cand - f) / max(T, 1e-12)):
            x = cand
            f = f_cand
            if f < best_f:
                best_f = f
                best_x = list(x)
        T *= alpha
        history.append(best_f)

    return OptimResult(
        best_x=best_x,
        best_f=sign * best_f,
        history_f=[sign * h for h in history],
        n_evaluations=n_iter + 1,
        method="simulated_annealing",
        converged=False,
    )
