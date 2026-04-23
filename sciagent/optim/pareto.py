"""
Pareto 前沿分析工具：非支配排序、超体积、代表解挑选、前沿指标。
"""

from __future__ import annotations
from typing import List, Sequence, Tuple
import math


Objs = Sequence[Sequence[float]]


# ---------------------------------------------------------------------------
# 非支配排序（朴素 O(n^2)，足够用于 n<1000 的微通道题）
# ---------------------------------------------------------------------------

def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """a 支配 b：所有维度 a <= b 且至少一维 a < b。"""
    strictly_better = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            strictly_better = True
    return strictly_better


def non_dominated_sort(objs: Objs) -> List[List[int]]:
    """返回前沿分层，每层是 objs 的下标列表。"""
    n = len(objs)
    S = [[] for _ in range(n)]
    n_p = [0] * n
    rank = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n_p[p] += 1
        if n_p[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_p[q] -= 1
                if n_p[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # 最后一层空
    return fronts


def pareto_front_indices(objs: Objs) -> List[int]:
    return non_dominated_sort(objs)[0]


# ---------------------------------------------------------------------------
# 超体积（hypervolume）指标 — 2D 快速版本，N 维用包含法
# ---------------------------------------------------------------------------

def hypervolume_2d(front: Objs, reference_point: Sequence[float]) -> float:
    pts = sorted(front, key=lambda p: p[0])
    rx, ry = reference_point
    hv = 0.0
    prev_y = ry
    for x, y in pts:
        if x >= rx or y >= ry:
            continue
        if y < prev_y:
            hv += (rx - x) * (prev_y - y)
            prev_y = y
    return hv


def hypervolume_monte_carlo(
    front: Objs, reference_point: Sequence[float],
    n_samples: int = 20000, seed: int = 42,
) -> float:
    """N 维近似：向参考矩形盒子里撒点，统计被前沿支配的比例。"""
    import random
    random.seed(seed)
    m = len(reference_point)
    # 前沿的 min 作为盒子下界
    box_min = [min(p[i] for p in front) for i in range(m)]
    box_max = list(reference_point)
    box_vol = 1.0
    for lo, hi in zip(box_min, box_max):
        if hi <= lo:
            return 0.0
        box_vol *= (hi - lo)

    count = 0
    for _ in range(n_samples):
        pt = [random.uniform(lo, hi) for lo, hi in zip(box_min, box_max)]
        # 若有任一前沿点支配 pt，则 pt 在被覆盖区域内
        for p in front:
            if all(pi <= xi for pi, xi in zip(p, pt)):
                count += 1
                break
    return box_vol * count / n_samples


# ---------------------------------------------------------------------------
# 前沿分布均匀度：最近邻间距的标准差
# ---------------------------------------------------------------------------

def spacing_metric(front: Objs) -> float:
    n = len(front)
    if n < 2:
        return 0.0
    d = []
    for i in range(n):
        min_d = float("inf")
        for j in range(n):
            if i == j:
                continue
            dist = sum(abs(a - b) for a, b in zip(front[i], front[j]))
            if dist < min_d:
                min_d = dist
        d.append(min_d)
    d_bar = sum(d) / n
    s = math.sqrt(sum((di - d_bar) ** 2 for di in d) / n)
    return s


# ---------------------------------------------------------------------------
# 代表解挑选：knee point / 两端 / 妥协解
# ---------------------------------------------------------------------------

def pick_knee_point(front: Objs) -> int:
    """找到距离"理想极小点"最近的前沿点，作为 knee。"""
    if not front:
        raise ValueError("front 不能为空")
    m = len(front[0])
    ideal = [min(p[i] for p in front) for i in range(m)]
    nadir = [max(p[i] for p in front) for i in range(m)]
    best_i, best_d = 0, float("inf")
    for i, p in enumerate(front):
        normed = [(pi - id_) / (na - id_) if na > id_ else 0.0
                  for pi, id_, na in zip(p, ideal, nadir)]
        d = sum(x * x for x in normed) ** 0.5
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def pick_extremes(front: Objs) -> List[int]:
    m = len(front[0])
    indices = []
    for j in range(m):
        ext = min(range(len(front)), key=lambda i: front[i][j])
        if ext not in indices:
            indices.append(ext)
    return indices


def representative_solutions(front: Objs, variables: Objs = None) -> dict:
    """一次性返回 knee + 两端点。"""
    knee = pick_knee_point(front)
    extremes = pick_extremes(front)
    return {
        "knee_index": knee,
        "knee_objective": list(front[knee]),
        "knee_variables": list(variables[knee]) if variables else None,
        "extreme_indices": extremes,
        "extreme_objectives": [list(front[i]) for i in extremes],
        "extreme_variables": [list(variables[i]) for i in extremes] if variables else None,
    }
