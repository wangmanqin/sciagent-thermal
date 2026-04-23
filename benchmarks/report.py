"""把 runner + scorer 的结果组合成一份 Markdown 报告。"""

from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import asdict


def render_markdown_report(
    run_results: List[Any],
    task_scores: List[Any],
    summary: Dict[str, Any],
    title: str = "SciAgent-Thermal Benchmark Report",
) -> str:
    lines: List[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"**Total tasks:** {summary['n']}  ")
    lines.append(f"**Mean score:** {summary['mean_total']:.2f} / 100  ")
    lines.append(f"**Pass rate (≥60):** {summary['pass_rate']*100:.1f}% "
                 f"({summary['pass_count']}/{summary['n']})  \n")

    lines.append("## 分项平均")
    lines.append("| 维度 | 权重 | 平均分 |")
    lines.append("|---|---|---|")
    lines.append(f"| Correctness | 50 | {summary['mean_correctness']:.2f} |")
    lines.append(f"| Tool usage | 20 | {summary['mean_tool_usage']:.2f} |")
    lines.append(f"| Explainability | 15 | {summary['mean_explainability']:.2f} |")
    lines.append(f"| Artifacts | 10 | {summary['mean_artifacts']:.2f} |")
    lines.append(f"| Conciseness | 5 | {summary['mean_conciseness']:.2f} |")
    lines.append("")

    lines.append("## 每题详情")
    lines.append("| # | 得分 | 耗时 (s) | 工具调用 | Success |")
    lines.append("|---|---|---|---|---|")
    by_id = {r.task_id: r for r in run_results}
    for ts in task_scores:
        r = by_id.get(ts.task_id)
        if r:
            lines.append(
                f"| {ts.task_id} | {ts.total:.1f} | {r.duration_s:.1f} | "
                f"{r.n_tool_calls} | {'✓' if r.success else '✗'} |"
            )
        else:
            lines.append(f"| {ts.task_id} | {ts.total:.1f} | - | - | ? |")
    lines.append("")

    lines.append("## 低分题诊断")
    for ts in task_scores:
        if ts.total < 80:
            lines.append(f"\n### Task {ts.task_id} — score {ts.total}")
            r = by_id.get(ts.task_id)
            if r:
                lines.append(f"- Tools used: `{r.tools_used}`")
                lines.append(f"- Final answer: {r.final_answer[:300]}")
            for k, v in ts.details.items():
                lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def save_report(md: str, path: str) -> str:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path


if __name__ == "__main__":
    # 空跑自测
    from dataclasses import dataclass

    @dataclass
    class Dummy:
        task_id: int
        total: float = 90.0
        details: dict = None
        duration_s: float = 10.0
        n_tool_calls: int = 5
        success: bool = True
        tools_used: list = None
        final_answer: str = ""
        correctness: float = 45.0
        tool_usage: float = 18.0
        explainability: float = 13.0
        artifacts: float = 9.0
        conciseness: float = 5.0

    scores = [Dummy(task_id=i) for i in range(1, 4)]
    results = [Dummy(task_id=i) for i in range(1, 4)]
    summary = {
        "n": 3, "mean_total": 90.0, "pass_rate": 1.0, "pass_count": 3,
        "mean_correctness": 45.0, "mean_tool_usage": 18.0,
        "mean_explainability": 13.0, "mean_artifacts": 9.0,
        "mean_conciseness": 5.0,
    }
    print(render_markdown_report(results, scores, summary)[:500])
