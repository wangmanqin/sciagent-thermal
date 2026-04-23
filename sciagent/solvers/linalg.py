"""
轻量线性代数工具：不依赖 numpy，在 sandbox 外侧（工具层）使用。

主要用途：
  - 解三对角线性方程组（TDMA / Thomas 算法）：1D 稳态导热有限差分的常客
  - 小矩阵 LU 分解 + 回代
  - 向量 2 范数 / 无穷范数
"""

from __future__ import annotations
from typing import List, Sequence


Vec = Sequence[float]
Mat = Sequence[Sequence[float]]


# ---------------------------------------------------------------------------
# 三对角矩阵算法（Thomas）
#   a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i
# ---------------------------------------------------------------------------

def thomas(a: Vec, b: Vec, c: Vec, d: Vec) -> List[float]:
    n = len(d)
    if not (len(a) == len(b) == len(c) == n):
        raise ValueError("a, b, c, d 长度必须相同")

    cp = [0.0] * n
    dp = [0.0] * n

    if b[0] == 0:
        raise ZeroDivisionError("b[0] == 0，Thomas 算法失败")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if denom == 0:
            raise ZeroDivisionError(f"第 {i} 步除以 0")
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x = [0.0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ---------------------------------------------------------------------------
# LU 分解（无主元选）
# ---------------------------------------------------------------------------

def lu_decompose(A: Mat) -> tuple:
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0

    for j in range(n):
        for i in range(j + 1):
            s = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - s
        for i in range(j, n):
            s = sum(L[i][k] * U[k][j] for k in range(j))
            if U[j][j] == 0:
                raise ZeroDivisionError(
                    "LU 分解失败（无主元选，矩阵可能奇异）"
                )
            L[i][j] = (A[i][j] - s) / U[j][j]
    return L, U


def lu_solve(L: Mat, U: Mat, b: Vec) -> list:
    n = len(b)
    # 前代 L y = b
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][k] * y[k] for k in range(i))
    # 回代 U x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][k] * x[k] for k in range(i + 1, n))
        if U[i][i] == 0:
            raise ZeroDivisionError("U 对角出现 0")
        x[i] = (y[i] - s) / U[i][i]
    return x


def solve_linear_system(A: Mat, b: Vec) -> list:
    L, U = lu_decompose(A)
    return lu_solve(L, U, b)


# ---------------------------------------------------------------------------
# 范数
# ---------------------------------------------------------------------------

def norm_2(v: Vec) -> float:
    return sum(x * x for x in v) ** 0.5


def norm_inf(v: Vec) -> float:
    return max(abs(x) for x in v)


def vec_sub(a: Vec, b: Vec) -> list:
    return [ai - bi for ai, bi in zip(a, b)]


def mat_vec_mul(A: Mat, x: Vec) -> list:
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


# ---------------------------------------------------------------------------
# 一维稳态导热 — 组装 + Thomas 求解
#   -k d2T/dx2 = S(x)
# ---------------------------------------------------------------------------

def solve_1d_conduction_dirichlet(
    length_m: float, n_cells: int, k_W_per_mK: float,
    source_W_per_m3: float, T_left: float, T_right: float,
) -> dict:
    if n_cells < 3:
        raise ValueError("至少 3 个单元")
    dx = length_m / n_cells
    # 在 n-1 个内部节点上离散
    n_int = n_cells - 1
    a = [0.0] * n_int
    b = [0.0] * n_int
    c = [0.0] * n_int
    d = [0.0] * n_int
    coeff = k_W_per_mK / dx ** 2
    for i in range(n_int):
        a[i] = -coeff if i > 0 else 0.0
        c[i] = -coeff if i < n_int - 1 else 0.0
        b[i] = 2 * coeff
        d[i] = source_W_per_m3

    # 边界修正
    d[0] -= -coeff * T_left if False else 0.0  # no-op sentinel
    d[0] += coeff * T_left
    d[-1] += coeff * T_right

    T_int = thomas(a, b, c, d)
    x = [i * dx for i in range(n_cells + 1)]
    T = [T_left] + T_int + [T_right]
    return {"x_m": x, "T_C": T, "dx_m": dx, "n_cells": n_cells}
