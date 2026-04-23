"""
NSGA-II 包装层：把 DEAP 的 NSGA-II 封装成一个对 Agent 友好的 API。

Agent 直接调用 run_nsga2(problem)，不需要理解 DEAP 的 toolbox 细节。
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple


Bounds = Sequence[Tuple[float, float]]
ObjFn = Callable[[Sequence[float]], Sequence[float]]


@dataclass
class NSGA2Result:
    pareto_variables: List[List[float]]
    pareto_objectives: List[List[float]]
    all_variables: List[List[float]]
    all_objectives: List[List[float]]
    n_generations: int
    population_size: int
    seed: int

    def summary(self) -> dict:
        return {
            "n_pareto": len(self.pareto_variables),
            "n_evaluated": len(self.all_variables),
            "n_generations": self.n_generations,
            "population_size": self.population_size,
            "seed": self.seed,
        }


def run_nsga2(
    objective_function: ObjFn,
    bounds: Bounds,
    n_objectives: int,
    minimize: Sequence[bool] = None,
    population_size: int = 80,
    n_generations: int = 60,
    mutation_eta: float = 20.0,
    crossover_eta: float = 15.0,
    seed: int = 42,
) -> NSGA2Result:
    try:
        from deap import base, creator, tools, algorithms
    except ImportError as e:
        raise ImportError(
            "运行 NSGA-II 需要安装 deap：pip install deap"
        ) from e

    random.seed(seed)
    n_vars = len(bounds)
    if n_vars == 0:
        raise ValueError("bounds 不能为空")
    if minimize is None:
        minimize = [True] * n_objectives
    weights = tuple(-1.0 if m else 1.0 for m in minimize)

    # DEAP 用模块级 creator 注册 class，多次调用会报警告
    fit_name = f"_Fit_{n_objectives}_{hash(weights) & 0xffff}"
    ind_name = f"_Ind_{n_objectives}_{hash(weights) & 0xffff}"
    if not hasattr(creator, fit_name):
        creator.create(fit_name, base.Fitness, weights=weights)
    if not hasattr(creator, ind_name):
        creator.create(ind_name, list, fitness=getattr(creator, fit_name))
    FitCls = getattr(creator, fit_name)
    IndCls = getattr(creator, ind_name)

    def _random_individual():
        return IndCls([random.uniform(lo, hi) for lo, hi in bounds])

    def _evaluate(ind):
        objs = objective_function(ind)
        if len(objs) != n_objectives:
            raise ValueError(
                f"目标函数应返回 {n_objectives} 个值，实际返回 {len(objs)}"
            )
        return tuple(objs)

    toolbox = base.Toolbox()
    toolbox.register("individual", _random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", _evaluate)
    toolbox.register(
        "mate", tools.cxSimulatedBinaryBounded,
        low=[b[0] for b in bounds],
        up=[b[1] for b in bounds],
        eta=crossover_eta,
    )
    toolbox.register(
        "mutate", tools.mutPolynomialBounded,
        low=[b[0] for b in bounds],
        up=[b[1] for b in bounds],
        eta=mutation_eta,
        indpb=1.0 / n_vars,
    )
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=population_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    all_vars = [list(ind) for ind in pop]
    all_objs = [list(ind.fitness.values) for ind in pop]

    for gen in range(n_generations):
        offspring = algorithms.varAnd(
            pop, toolbox, cxpb=0.9, mutpb=1.0 / n_vars,
        )
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
                all_vars.append(list(ind))
                all_objs.append(list(ind.fitness.values))
        pop = toolbox.select(pop + offspring, population_size)

    # 提取最终 Pareto 前沿
    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    pareto_vars = [list(ind) for ind in front]
    pareto_objs = [list(ind.fitness.values) for ind in front]

    return NSGA2Result(
        pareto_variables=pareto_vars,
        pareto_objectives=pareto_objs,
        all_variables=all_vars,
        all_objectives=all_objs,
        n_generations=n_generations,
        population_size=population_size,
        seed=seed,
    )
