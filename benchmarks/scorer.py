"""对比 Agent 产出与参考答案，按 docs/BENCHMARK.md 打分。"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import math


@dataclass
class TaskScore:
    task_id: int
    correctness: float = 0.0   # /50
    tool_usage: float = 0.0    # /20
    explainability: float = 0.0  # /15
    artifacts: float = 0.0     # /10
    conciseness: float = 0.0   # /5
    total: float = 0.0         # /100
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 单项评分
# ---------------------------------------------------------------------------

def score_correctness(
    ref: Dict[str, float],
    agent_answers: Dict[str, float],
    tolerance: float,
) -> (float, Dict[str, Any]):
    """数值项按 tolerance 内给分；布尔项 1/0。"""
    if not ref:
        return 50.0, {"note": "no reference values"}

    n = len(ref)
    got = 0
    detail = {}
    for key, ref_val in ref.items():
        if key not in agent_answers:
            detail[key] = "missing"
            continue
        val = agent_answers[key]
        if ref_val == 0.0:
            ok = abs(val) < 1e-9
            err = abs(val)
        else:
            err = abs(val - ref_val) / abs(ref_val)
            ok = err <= tolerance
        detail[key] = {"ref": ref_val, "got": val,
                       "err_pct": round(err * 100, 2), "pass": ok}
        if ok:
            got += 1
        elif err < 2 * tolerance:
            got += 0.5  # 接近但未达标给半分

    score = 50.0 * got / n
    return round(score, 2), detail


def score_tool_usage(required: List[str], used: List[str]) -> (float, dict):
    if not required:
        return 20.0, {"note": "no required tools"}
    used_set = set(used)
    hit = sum(1 for r in required if r in used_set)
    score = 20.0 * hit / len(required)
    return round(score, 2), {
        "required": required, "used": used,
        "hit": hit, "total": len(required),
    }


def score_explainability(text: str, must_mention: List[str] = None) -> (float, dict):
    """粗糙版：看最终答案里是否提到关键概念 / 关联式名。"""
    must_mention = must_mention or []
    if not text:
        return 0.0, {"note": "empty answer"}
    hit = sum(1 for kw in must_mention if kw.lower() in text.lower())
    # 至少应解释用了什么关联式
    mention_score = 10.0 if hit >= max(1, len(must_mention) // 2) else 5.0
    # 长度过短判为没解释
    length_score = 5.0 if len(text) > 120 else 2.0
    return round(mention_score + length_score, 2), {
        "mentions_hit": hit, "mentions_required": must_mention,
        "answer_length": len(text),
    }


def score_artifacts(expected_any: List[str], produced: List[str]) -> (float, dict):
    """expected_any：'plot' / 'table' / 'code' / 'report'；任一命中即加分。"""
    if not expected_any:
        return 10.0, {"note": "no artifacts expected"}
    produced_set = set(produced)
    hit = any(a in produced_set for a in expected_any)
    return (10.0 if hit else 4.0), {
        "expected": expected_any, "produced": produced,
    }


def score_conciseness(n_messages: int) -> (float, dict):
    if n_messages <= 10:
        s = 5.0
    elif n_messages <= 20:
        s = 4.0
    elif n_messages <= 30:
        s = 3.0
    else:
        s = 1.0
    return s, {"n_messages": n_messages}


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def score_task(
    task_id: int,
    ref: Dict[str, float],
    agent_answers: Dict[str, float],
    tolerance: float,
    required_tools: List[str],
    used_tools: List[str],
    final_text: str,
    must_mention: List[str],
    expected_artifacts: List[str],
    produced_artifacts: List[str],
    n_messages: int,
) -> TaskScore:
    c, c_det = score_correctness(ref, agent_answers, tolerance)
    t, t_det = score_tool_usage(required_tools, used_tools)
    e, e_det = score_explainability(final_text, must_mention)
    a, a_det = score_artifacts(expected_artifacts, produced_artifacts)
    k, k_det = score_conciseness(n_messages)
    total = c + t + e + a + k
    return TaskScore(
        task_id=task_id,
        correctness=c,
        tool_usage=t,
        explainability=e,
        artifacts=a,
        conciseness=k,
        total=round(total, 2),
        details={
            "correctness": c_det,
            "tool_usage": t_det,
            "explainability": e_det,
            "artifacts": a_det,
            "conciseness": k_det,
        },
    )


def summarize_scores(scores: List[TaskScore]) -> dict:
    if not scores:
        return {"n": 0, "mean": 0.0, "pass_rate": 0.0}
    n = len(scores)
    total_mean = sum(s.total for s in scores) / n
    pass_n = sum(1 for s in scores if s.total >= 60.0)
    return {
        "n": n,
        "mean_total": round(total_mean, 2),
        "mean_correctness": round(sum(s.correctness for s in scores) / n, 2),
        "mean_tool_usage": round(sum(s.tool_usage for s in scores) / n, 2),
        "mean_explainability": round(sum(s.explainability for s in scores) / n, 2),
        "mean_artifacts": round(sum(s.artifacts for s in scores) / n, 2),
        "mean_conciseness": round(sum(s.conciseness for s in scores) / n, 2),
        "pass_rate": round(pass_n / n, 3),
        "pass_count": pass_n,
    }


if __name__ == "__main__":
    # 快速自测
    s = score_task(
        task_id=1,
        ref={"delta_T_C": 1.15},
        agent_answers={"delta_T_C": 1.18},
        tolerance=0.10,
        required_tools=["water_properties", "caloric_resistance"],
        used_tools=["water_properties", "caloric_resistance"],
        final_text="水的 cp=4180，Q=m·cp·ΔT，所以 ΔT ≈ 1.18°C。",
        must_mention=["cp"],
        expected_artifacts=[],
        produced_artifacts=[],
        n_messages=8,
    )
    print(s)
