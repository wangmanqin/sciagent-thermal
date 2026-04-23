"""对比两份 benchmark 运行结果（比如换 LLM 前后，或改 prompt 前后）。"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json
import os


def load_run_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scores(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delta_report(
    old_summary: dict, new_summary: dict,
    old_scores: List[dict], new_scores: List[dict],
    label_old: str = "old", label_new: str = "new",
) -> str:
    lines = [f"# Benchmark Comparison: {label_old} → {label_new}\n"]
    lines.append("## 汇总变化")
    lines.append("| 指标 | {o} | {n} | Δ |".format(o=label_old, n=label_new))
    lines.append("|---|---|---|---|")
    for key in ["mean_total", "mean_correctness", "mean_tool_usage",
                "mean_explainability", "mean_artifacts", "mean_conciseness",
                "pass_rate"]:
        if key in old_summary and key in new_summary:
            old = old_summary[key]
            new = new_summary[key]
            d = new - old
            sign = "+" if d >= 0 else ""
            lines.append(f"| {key} | {old:.3f} | {new:.3f} | {sign}{d:.3f} |")
    lines.append("")

    lines.append("## 每题变化")
    lines.append("| Task | {o} | {n} | Δ |".format(o=label_old, n=label_new))
    lines.append("|---|---|---|---|")
    old_by = {s["task_id"]: s["total"] for s in old_scores}
    new_by = {s["task_id"]: s["total"] for s in new_scores}
    all_ids = sorted(set(old_by) | set(new_by))
    for tid in all_ids:
        o = old_by.get(tid, "-")
        n = new_by.get(tid, "-")
        if isinstance(o, (int, float)) and isinstance(n, (int, float)):
            d = n - o
            sign = "+" if d >= 0 else ""
            lines.append(f"| {tid} | {o:.1f} | {n:.1f} | {sign}{d:.1f} |")
        else:
            lines.append(f"| {tid} | {o} | {n} | - |")
    return "\n".join(lines)


def save_compare_report(md: str, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path


if __name__ == "__main__":
    # 自测
    old = {"mean_total": 70.0, "mean_correctness": 35.0, "mean_tool_usage": 15.0,
           "mean_explainability": 10.0, "mean_artifacts": 6.0,
           "mean_conciseness": 4.0, "pass_rate": 0.7}
    new = {"mean_total": 88.8, "mean_correctness": 44.5, "mean_tool_usage": 18.5,
           "mean_explainability": 13.2, "mean_artifacts": 8.0,
           "mean_conciseness": 4.6, "pass_rate": 1.0}
    old_scores = [{"task_id": i, "total": 70 + i} for i in range(1, 11)]
    new_scores = [{"task_id": i, "total": 85 + i * 0.3} for i in range(1, 11)]
    print(delta_report(old, new, old_scores, new_scores,
                       label_old="DeepSeek", label_new="Claude Opus 4.7")[:800])
