"""Benchmark runner：跑 Agent，记录每题的 ReAct log 和产物。"""

from __future__ import annotations
import json
import os
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class RunResult:
    task_id: int
    question: str
    final_answer: str = ""
    n_iterations: int = 0
    n_tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    duration_s: float = 0.0
    success: bool = False
    error: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)


def run_single_task(
    agent: Any,
    task: Any,
    *,
    verbose: bool = False,
    timeout_s: Optional[float] = None,
) -> RunResult:
    result = RunResult(task_id=task.id, question=task.question)
    t0 = time.time()

    events_buf = []
    tools_used = []
    artifacts = []

    def on_event(ev):
        events_buf.append({
            "type": ev.event_type,
            "content": ev.content[:500] if isinstance(ev.content, str) else str(ev.content)[:500],
            "metadata": getattr(ev, "metadata", {}),
        })
        if ev.event_type == "tool_call":
            tool_name = getattr(ev, "metadata", {}).get("name") or ev.content
            tools_used.append(tool_name)
        elif ev.event_type == "tool_result":
            # artifact 检测：工具结果里含 output_path 字段
            meta = getattr(ev, "metadata", {})
            if isinstance(meta, dict) and "output_path" in meta:
                artifacts.append(meta["output_path"])
            if isinstance(ev.content, str) and ".png" in ev.content:
                artifacts.append(ev.content)
        if verbose:
            print(f"  [{ev.event_type}] {str(ev.content)[:80]}")

    try:
        agent.run(task.question, on_event=on_event)
        # 找最终 answer
        for e in reversed(events_buf):
            if e["type"] == "answer":
                result.final_answer = e["content"]
                break
        result.success = True
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        result.success = False

    result.duration_s = round(time.time() - t0, 2)
    result.n_tool_calls = sum(1 for e in events_buf if e["type"] == "tool_call")
    result.n_iterations = max(
        [e.get("metadata", {}).get("iteration", 0) for e in events_buf] or [0]
    )
    result.tools_used = tools_used
    result.artifacts = list(dict.fromkeys(artifacts))
    result.events = events_buf
    return result


def run_all(
    agent: Any,
    tasks: List[Any],
    output_dir: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for task in tasks:
        if verbose:
            print(f"\n=== Task {task.id} [{task.category}] ===")
        r = run_single_task(agent, task, verbose=verbose)
        results.append(r)
        # 单题 log
        with open(os.path.join(output_dir, f"task_{task.id:02d}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(asdict(r), f, ensure_ascii=False, indent=2)

    # summary
    summary = {
        "n_tasks": len(results),
        "n_success": sum(1 for r in results if r.success),
        "mean_duration_s": round(
            sum(r.duration_s for r in results) / max(1, len(results)), 2),
        "mean_tool_calls": round(
            sum(r.n_tool_calls for r in results) / max(1, len(results)), 2),
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    # 快速自测：用 MockLLM 跑第一道题，证明管线能跑通
    from sciagent.agent import Agent
    from sciagent.llm import MockLLM
    from benchmarks.tasks import all_tasks

    llm = MockLLM(scripted_responses=[
        {"content": "水温差约 1.15°C", "tool_calls": []},
    ])
    agent = Agent(llm=llm)
    tasks = all_tasks()[:1]
    summary = run_all(agent, tasks, output_dir="runs/selftest", verbose=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
