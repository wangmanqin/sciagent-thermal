"""
从 Agent 的最终答案中抽取数值答案。

因为 Agent 回答是自然语言，我们需要把 `ΔT ≈ 1.18°C` 这类表达解析成
`{"delta_T_C": 1.18}` 用于打分。
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple


_NUM_RE = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"

# 预设提取规则：key 和候选 regex（按优先级）
EXTRACTION_RULES: Dict[str, List[str]] = {
    "delta_T_C": [
        rf"(?:ΔT|Δ\s*T|温差|DELTA[_ ]?T)\s*[≈=:~]?\s*{_NUM_RE}\s*°?C",
        rf"ΔT\s*[≈=:~]?\s*{_NUM_RE}",
    ],
    "h_W_per_m2K": [
        rf"h\s*[≈=:~]?\s*{_NUM_RE}\s*W/\(?m[²2]\s*·?\s*K\)?",
        rf"对流换热系数\s*[≈=:~]?\s*{_NUM_RE}",
    ],
    "T_max_C": [
        rf"T[_\s]?max\s*[≈=:~]?\s*{_NUM_RE}\s*°?C",
        rf"最高(?:结)?温\s*[≈=:~]?\s*{_NUM_RE}\s*°?C",
    ],
    "hydraulic_diameter_m": [
        rf"Dh\s*[≈=:~]?\s*{_NUM_RE}\s*(?:m|mm)",
        rf"水力直径\s*[≈=:~]?\s*{_NUM_RE}\s*(?:m|mm)",
    ],
    "h_at_5_Lpm_W_per_m2K": [
        rf"5\s*L/min.*?h\s*[≈=:~]?\s*{_NUM_RE}",
    ],
    "total_dp_Pa": [
        rf"(?:总压降|total.*pressure.*drop|ΔP[_\s]?total)\s*[≈=:~]?\s*{_NUM_RE}\s*Pa",
        rf"压降\s*[≈=:~]?\s*{_NUM_RE}\s*Pa",
    ],
    "pump_power_W": [
        rf"泵功(?:率)?\s*[≈=:~]?\s*{_NUM_RE}\s*W",
        rf"pump\s+power\s*[≈=:~]?\s*{_NUM_RE}\s*W",
    ],
    "k_ratio": [
        rf"k\s*比(?:例|值)?\s*[≈=:~]?\s*{_NUM_RE}",
        rf"k[_/]\s*ratio\s*[≈=:~]?\s*{_NUM_RE}",
    ],
    "mu_ratio": [
        rf"(?:μ|mu|粘度)\s*比(?:例|值)?\s*[≈=:~]?\s*{_NUM_RE}",
        rf"mu[_/]\s*ratio\s*[≈=:~]?\s*{_NUM_RE}",
    ],
    "T_bottom_C": [
        rf"底部(?:温度)?\s*[≈=:~]?\s*{_NUM_RE}\s*°?C",
        rf"T\s*bottom\s*[≈=:~]?\s*{_NUM_RE}",
    ],
    "T_top_C": [
        rf"顶部(?:温度)?\s*[≈=:~]?\s*{_NUM_RE}\s*°?C",
        rf"T\s*top\s*[≈=:~]?\s*{_NUM_RE}",
    ],
    "pareto_points_min": [
        rf"Pareto\s*(?:前沿)?[^0-9]*(\d+)\s*个点",
        rf"(\d+)\s*个\s*(?:非支配|Pareto)\s*解",
    ],
    "has_pareto_table": [
        r"(\|.*Pareto.*\|)",
    ],
    "has_recommended_design": [
        r"(推荐(?:设计|方案)|recommended\s+design)",
    ],
}


def _normalize_dh(text: str, val: float) -> float:
    """Dh 可能以 mm 为单位。如果数值 > 0.01 认为是 mm → 转 m。"""
    if val > 0.01:
        return val * 1e-3
    return val


def extract_answers(text: str, keys: List[str]) -> Dict[str, float]:
    """从 text 中提取 keys 对应的数值。找不到返回 inf 作为哨兵。"""
    result: Dict[str, float] = {}
    for key in keys:
        patterns = EXTRACTION_RULES.get(key, [])
        found = None
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                # 对"是否存在"类型的键：返回 1.0
                if key.startswith("has_"):
                    found = 1.0
                    break
                try:
                    found = float(m.group(1))
                except (ValueError, IndexError):
                    continue
                if key == "hydraulic_diameter_m":
                    found = _normalize_dh(text, found)
                break
        if found is None and key.startswith("has_"):
            found = 0.0
        result[key] = found if found is not None else float("nan")
    return result


def extract_answers_for_task(text: str, reference_answer: Dict[str, float]) -> Dict[str, float]:
    return extract_answers(text, list(reference_answer.keys()))


if __name__ == "__main__":
    # 自测
    sample = "水的 cp=4180 J/(kg·K)。根据能量守恒，ΔT ≈ 1.18 °C。"
    out = extract_answers_for_task(sample, {"delta_T_C": 1.15})
    print(out)
