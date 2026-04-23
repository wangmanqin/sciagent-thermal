"""
命名式绘图工具：Agent 传数据 + 图形类型，拿到一张落盘的 PNG。

相比让 Agent 自己写 matplotlib 样板代码，这条路径的好处是：
  - 数据格式固定，便于评测系统直接比对数值
  - 样式统一，不依赖 LLM 的"绘图审美"
  - 不走 run_python_code，省一次 subprocess 开销
适合评测里纯绘图的子任务；复杂可视化仍走 run_python_code。
"""

from __future__ import annotations
import os
import json
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs"
)


def save_xy_plot(
    x: List[float],
    y: List[float],
    filename: str,
    *,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    kind: str = "line",
) -> str:
    if len(x) != len(y):
        raise ValueError(f"x 和 y 长度必须相等，收到 {len(x)} vs {len(y)}")
    if kind not in ("line", "scatter"):
        raise ValueError("kind 必须是 'line' 或 'scatter'")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    if not filename.lower().endswith(".png"):
        filename = filename + ".png"
    path = os.path.join(OUTPUTS_DIR, filename)

    fig, ax = plt.subplots(figsize=(6, 4))
    if kind == "line":
        ax.plot(x, y, linewidth=1.8)
    else:
        ax.scatter(x, y, s=18)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


TOOL_DEFINITION = {
    "name": "save_xy_plot",
    "description": (
        "把一组 (x, y) 数据保存为 PNG 折线图或散点图。"
        "适合简单曲线。复杂子图 / Pareto 前沿等仍建议用 run_python_code。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "x": {"type": "array", "items": {"type": "number"}},
            "y": {"type": "array", "items": {"type": "number"}},
            "filename": {"type": "string", "description": "输出文件名（不带目录）"},
            "title": {"type": "string"},
            "xlabel": {"type": "string"},
            "ylabel": {"type": "string"},
            "kind": {"type": "string", "enum": ["line", "scatter"]},
        },
        "required": ["x", "y", "filename"],
    },
}


def execute(args: dict) -> str:
    path = save_xy_plot(
        x=list(args["x"]),
        y=list(args["y"]),
        filename=args["filename"],
        title=args.get("title", ""),
        xlabel=args.get("xlabel", "x"),
        ylabel=args.get("ylabel", "y"),
        kind=args.get("kind", "line"),
    )
    return json.dumps(
        {"saved": os.path.basename(path), "full_path": path},
        ensure_ascii=False,
    )
