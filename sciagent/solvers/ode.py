"""
常微分方程求解器：RK4（定步长）、RK45（自适应步长）

用途：一维稳态/瞬态导热方程、翅片温度分布、热电耦合方程等。
"""

from __future__ import annotations
from typing import Callable, List, Sequence, Tuple
import math


State = Sequence[float]


# ---------------------------------------------------------------------------
# RK4 定步长
# ---------------------------------------------------------------------------

def rk4_step(
    f: Callable[[float, State], State],
    t: float, y: State, h: float,
) -> list:
    k1 = f(t, y)
    y2 = [yi + 0.5 * h * ki for yi, ki in zip(y, k1)]
    k2 = f(t + 0.5 * h, y2)
    y3 = [yi + 0.5 * h * ki for yi, ki in zip(y, k2)]
    k3 = f(t + 0.5 * h, y3)
    y4 = [yi + h * ki for yi, ki in zip(y, k3)]
    k4 = f(t + h, y4)
    return [
        yi + h / 6 * (k1i + 2 * k2i + 2 * k3i + k4i)
        for yi, k1i, k2i, k3i, k4i in zip(y, k1, k2, k3, k4)
    ]


def solve_ode_rk4(
    f: Callable[[float, State], State],
    t_span: Tuple[float, float],
    y0: State,
    n_steps: int,
) -> Tuple[list, list]:
    t0, t1 = t_span
    if n_steps <= 0:
        raise ValueError("n_steps 必须为正")
    h = (t1 - t0) / n_steps
    ts = [t0]
    ys = [list(y0)]
    t = t0
    y = list(y0)
    for _ in range(n_steps):
        y = rk4_step(f, t, y, h)
        t += h
        ts.append(t)
        ys.append(list(y))
    return ts, ys


# ---------------------------------------------------------------------------
# RK45（Dormand-Prince 变体 — 简化实现）
# ---------------------------------------------------------------------------

_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
_B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640,
       -92097 / 339200, 187 / 2100, 1 / 40]
_C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]


def _vec_add(y, scales, ks):
    out = list(y)
    for s, k in zip(scales, ks):
        for i, ki in enumerate(k):
            out[i] += s * ki
    return out


def solve_ode_rk45(
    f: Callable[[float, State], State],
    t_span: Tuple[float, float],
    y0: State,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h_init: float = None,
    max_steps: int = 10000,
) -> Tuple[list, list]:
    t0, t1 = t_span
    if h_init is None:
        h = (t1 - t0) / 100.0
    else:
        h = h_init

    t = t0
    y = list(y0)
    ts = [t]
    ys = [y[:]]

    for _ in range(max_steps):
        if t + h > t1:
            h = t1 - t
            if h <= 0:
                break

        ks = []
        for i in range(7):
            yi = list(y)
            for j in range(i):
                for idx in range(len(y)):
                    yi[idx] += h * _A[i][j] * ks[j][idx]
            ks.append(f(t + _C[i] * h, yi))

        y5 = _vec_add(y, [h * b for b in _B5], ks)
        y4 = _vec_add(y, [h * b for b in _B4], ks)
        err = max(
            abs(a - b) / (atol + rtol * max(abs(ai), abs(bi)))
            for a, b, ai, bi in zip(y5, y4, y5, y4)
        ) if y else 0.0

        if err <= 1.0:
            t += h
            y = y5
            ts.append(t)
            ys.append(y[:])
            if t >= t1:
                break
            h = h * min(5.0, 0.9 * err ** (-1 / 5) if err > 0 else 5.0)
        else:
            h = h * max(0.1, 0.9 * err ** (-1 / 5))

    return ts, ys


# ---------------------------------------------------------------------------
# 一维稳态翅片方程：d2T/dx2 - m^2 (T - T_inf) = 0
# 用 shooting 法转成初值问题
# ---------------------------------------------------------------------------

def fin_temperature_distribution(
    m_per_m: float, length_m: float, T_base: float, T_inf: float,
    boundary_tip: str = "adiabatic", n_points: int = 100,
) -> dict:
    """
    求沿翅片方向的温度分布。
    m = sqrt(2h / (k*t))
    """
    if m_per_m <= 0 or length_m <= 0:
        raise ValueError("m 和 length 必须为正")
    if boundary_tip == "adiabatic":
        theta_over_theta_b = lambda x: (
            math.cosh(m_per_m * (length_m - x)) / math.cosh(m_per_m * length_m)
        )
    elif boundary_tip == "infinite":
        theta_over_theta_b = lambda x: math.exp(-m_per_m * x)
    else:
        raise ValueError("boundary_tip 必须是 adiabatic / infinite")

    theta_b = T_base - T_inf
    xs = [i * length_m / (n_points - 1) for i in range(n_points)]
    Ts = [T_inf + theta_b * theta_over_theta_b(x) for x in xs]
    return {
        "x_m": xs,
        "T_C": Ts,
        "boundary_tip": boundary_tip,
        "T_base": T_base,
        "T_inf": T_inf,
        "m_per_m": m_per_m,
    }
