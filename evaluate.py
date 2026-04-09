"""
SciAgent 评测脚本（微通道散热器方向）。
自动运行 benchmark.json 中的测试用例，对比 Agent 输出与标注答案，自动评分并生成评测报告。

用法：
  python evaluate.py                  # 运行全部测试
  python evaluate.py --ids 1 3 5      # 只运行指定编号的测试
  python evaluate.py --show-answers   # 只查看标注答案（不运行Agent）
  python evaluate.py --dry-run        # 预览测试用例信息
  python evaluate.py --verify-only    # 只运行验证代码，检查标注答案本身正确性
"""

import json
import argparse
import time
import re
import os
import sys
import io
import traceback
from datetime import datetime
from pathlib import Path

# 修复 Windows 终端中文+特殊字符编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 把项目根目录加入 path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_FILE = PROJECT_ROOT / "benchmark.json"
REPORT_DIR = PROJECT_ROOT / "eval_reports"


def load_benchmark(ids=None):
    """加载测试基准"""
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    return cases


def run_verification_code(code_str):
    """运行验证代码，返回 (passed: bool, error_msg: str)"""
    try:
        exec(code_str, {"__builtins__": __builtins__})
        return True, ""
    except AssertionError as e:
        return False, f"断言失败: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def extract_numbers_from_text(text):
    """从 Agent 回答文本中提取所有数值"""
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    return [float(x) for x in re.findall(pattern, text)]


def auto_score(case, agent_answer, agent_code_output):
    """
    自动评分。返回 dict:
    {
        "verification_passed": bool,    # 验证代码是否通过
        "verification_error": str,      # 验证错误信息
        "value_matches": list,          # 数值匹配结果
        "plot_generated": bool,         # 是否生成了图片
        "score": float,                 # 综合得分 0~1
        "details": str                  # 评分详情
    }
    """
    gt = case["ground_truth"]
    result = {
        "verification_passed": False,
        "verification_error": "",
        "value_matches": [],
        "plot_generated": False,
        "score": 0.0,
        "details": "",
    }

    total_points = 0
    earned_points = 0
    detail_lines = []

    # === 1. 运行验证代码 (40分) ===
    if "verification_code" in gt:
        total_points += 40
        passed, err = run_verification_code(gt["verification_code"])
        result["verification_passed"] = passed
        result["verification_error"] = err
        if passed:
            earned_points += 40
            detail_lines.append("[PASS] 验证代码通过 (+40)")
        else:
            detail_lines.append(f"[FAIL] 验证代码失败: {err} (+0/40)")

    # === 2. 检查期望数值是否出现在回答中 (40分) ===
    if "expected_values" in gt:
        expected = gt["expected_values"]
        tol_pct = gt.get("tolerance_percent", 10)
        tol_abs = gt.get("tolerance", None)

        # 合并 agent_answer 和 agent_code_output 作为搜索文本
        full_text = (agent_answer or "") + "\n" + (agent_code_output or "")
        found_numbers = extract_numbers_from_text(full_text)

        if isinstance(expected, dict):
            check_values = {k: v for k, v in expected.items() if isinstance(v, (int, float))}
        elif isinstance(expected, list):
            check_values = {f"value_{i}": v for i, v in enumerate(expected) if isinstance(v, (int, float))}
        else:
            check_values = {}

        if check_values:
            points_per_value = 40 / len(check_values)
            for name, expected_val in check_values.items():
                total_points += points_per_value
                matched = False
                for num in found_numbers:
                    if tol_abs is not None:
                        if abs(num - expected_val) <= tol_abs:
                            matched = True
                            break
                    else:
                        if expected_val != 0:
                            if abs(num - expected_val) / abs(expected_val) * 100 <= tol_pct:
                                matched = True
                                break
                        else:
                            if abs(num) < 0.01:
                                matched = True
                                break

                result["value_matches"].append({
                    "name": name, "expected": expected_val, "matched": matched
                })
                if matched:
                    earned_points += points_per_value
                    detail_lines.append(f"[PASS] {name}={expected_val} 在输出中找到匹配 (+{points_per_value:.0f})")
                else:
                    detail_lines.append(f"[FAIL] {name}={expected_val} 未在输出中找到匹配 (+0/{points_per_value:.0f})")
        else:
            total_points += 40
            # 如果没有可检查的数值，只要有回答就给20分
            if agent_answer and len(agent_answer) > 50:
                earned_points += 20
                detail_lines.append("[PARTIAL] 有回答但无法自动验证数值 (+20/40)")

    # === 3. 检查图片生成 (20分) ===
    has_plot_requirement = any("PNG" in c or "图" in c for c in gt["key_checks"])
    if has_plot_requirement:
        total_points += 20
        outputs_dir = PROJECT_ROOT / "outputs"
        if outputs_dir.exists():
            pngs = list(outputs_dir.glob("*.png"))
            if pngs:
                result["plot_generated"] = True
                earned_points += 20
                detail_lines.append(f"[PASS] 生成了图片: {[p.name for p in pngs[-3:]]} (+20)")
            else:
                detail_lines.append("[FAIL] 未生成图片文件 (+0/20)")
        else:
            detail_lines.append("[FAIL] outputs目录不存在 (+0/20)")

    # 计算总分
    result["score"] = earned_points / total_points if total_points > 0 else 0
    result["details"] = "\n".join(detail_lines)

    return result


def show_ground_truth(case):
    """显示单个测试用例的标注答案"""
    gt = case["ground_truth"]
    print(f"\n{'─' * 60}")
    print(f"  测试 #{case['id']} | {case['category']} | 难度: {case['difficulty']}")
    print(f"  问题: {case['query'][:80]}...")
    print(f"{'─' * 60}")
    print(f"  [标注] {gt['description']}")
    if "expected_values" in gt:
        print(f"  [期望] {json.dumps(gt['expected_values'], ensure_ascii=False)}")
    if "tolerance" in gt:
        print(f"  [容差] +/-{gt['tolerance']}")
    if "tolerance_percent" in gt:
        print(f"  [容差] +/-{gt['tolerance_percent']}%")
    print(f"  [检查项]")
    for i, check in enumerate(gt["key_checks"], 1):
        print(f"     {i}. {check}")
    if "verification_code" in gt:
        print(f"  [验证代码]")
        for line in gt["verification_code"].split("\n"):
            print(f"     {line}")
    print()


def show_all_answers(cases):
    """显示所有测试用例的标注答案"""
    print("\n" + "=" * 60)
    print("  SciAgent 微通道散热器评测基准 -- 标注答案一览")
    print(f"  共 {len(cases)} 道测试题")
    print("=" * 60)
    for case in cases:
        show_ground_truth(case)


def verify_all_answers(cases):
    """验证所有标注答案的正确性"""
    print("\n" + "=" * 60)
    print("  验证标注答案的正确性")
    print("=" * 60)
    all_passed = True
    for case in cases:
        gt = case["ground_truth"]
        code = gt.get("verification_code", "")
        if not code:
            print(f"  #{case['id']} -- 无验证代码，跳过")
            continue
        passed, err = run_verification_code(code)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  #{case['id']} [{status}] {gt['description']}")
        if err:
            print(f"         {err}")

    print(f"\n{'=' * 60}")
    if all_passed:
        print("  所有标注答案验证通过!")
    else:
        print("  部分标注答案验证失败，请检查 benchmark.json")
    print("=" * 60)


def run_single_test(case, llm_mode=None):
    """运行单个测试用例"""
    from sciagent.agent import Agent

    print(f"\n{'=' * 60}")
    print(f"  >> 运行测试 #{case['id']}: {case['query'][:40]}...")
    print(f"{'=' * 60}")

    # 先显示标注答案
    gt = case["ground_truth"]
    print(f"\n  [标注答案] {gt['description']}")
    if "expected_values" in gt:
        print(f"  [期望值]   {json.dumps(gt['expected_values'], ensure_ascii=False)}")
    for check in gt["key_checks"]:
        print(f"    - {check}")

    print(f"\n  [Agent 运行中...]")
    print(f"  {'─' * 40}")

    # 运行 Agent
    agent = Agent(llm_mode)
    start_time = time.time()
    iterations = 0
    errors = []
    code_outputs = []  # 收集代码执行输出

    def on_event(event):
        nonlocal iterations, errors
        if event.event_type == "tool_call":
            iterations += 1
            code_preview = event.metadata.get("code", "")[:80]
            print(f"     [iter {iterations}] {code_preview}...")
        elif event.event_type == "error":
            errors.append(event.content)
            print(f"     [ERROR] {event.content[:100]}")
        elif event.event_type == "tool_result":
            code_outputs.append(event.content)
            result_preview = event.content[:200].replace("\n", " ")
            print(f"     [OK] {result_preview}")

    try:
        answer = agent.run(case["query"], on_event=on_event)
    except Exception as e:
        answer = f"Agent异常: {type(e).__name__}: {e}"
        errors.append(answer)

    elapsed = time.time() - start_time

    # 显示 Agent 回答
    print(f"\n  [Agent 回答]")
    print(f"  {'─' * 40}")
    if answer:
        for line in answer.split("\n")[:30]:  # 最多显示30行
            print(f"     {line}")
        if len(answer.split("\n")) > 30:
            print(f"     ... (共{len(answer.split(chr(10)))}行，已截断)")
    else:
        print("     (无回答)")

    # 自动评分
    all_code_output = "\n".join(code_outputs)
    score_result = auto_score(case, answer, all_code_output)

    # 显示评分
    score_pct = score_result["score"] * 100
    print(f"\n  [自动评分] {score_pct:.0f}分/100分")
    print(f"  {'─' * 40}")
    for line in score_result["details"].split("\n"):
        print(f"     {line}")

    print(f"\n  [统计] 迭代: {iterations} | 错误: {len(errors)} | 耗时: {elapsed:.1f}s")

    return {
        "id": case["id"],
        "query": case["query"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "ground_truth": gt["description"],
        "agent_answer": answer,
        "iterations": iterations,
        "errors": len(errors),
        "elapsed_seconds": round(elapsed, 1),
        "error_details": errors,
        "score": round(score_result["score"] * 100, 1),
        "score_details": score_result["details"],
        "verification_passed": score_result["verification_passed"],
        "plot_generated": score_result["plot_generated"],
    }


def generate_report(results):
    """生成评测报告"""
    REPORT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"eval_report_{timestamp}.json"
    md_path = REPORT_DIR / f"eval_report_{timestamp}.md"

    # 统计
    total = len(results)
    avg_score = sum(r["score"] for r in results) / total if total > 0 else 0
    pass_count = sum(1 for r in results if r["score"] >= 60)
    perfect_count = sum(1 for r in results if r["score"] >= 90)
    avg_iterations = sum(r["iterations"] for r in results) / total if total > 0 else 0
    avg_time = sum(r["elapsed_seconds"] for r in results) / total if total > 0 else 0

    # 终端输出总结
    print("\n" + "=" * 60)
    print("  评测总结")
    print("=" * 60)
    print(f"  测试数量:     {total}")
    print(f"  平均得分:     {avg_score:.1f}/100")
    print(f"  通过(>=60):   {pass_count}/{total}")
    print(f"  优秀(>=90):   {perfect_count}/{total}")
    print(f"  平均迭代:     {avg_iterations:.1f}")
    print(f"  平均耗时:     {avg_time:.1f}s")
    print(f"  {'─' * 40}")
    for r in results:
        status = "PASS" if r["score"] >= 60 else "FAIL"
        print(f"  #{r['id']:2d} [{status}] {r['score']:5.1f}分 | {r['category']} | {r['difficulty']}")
    print("=" * 60)

    # JSON 报告
    report_data = {
        "timestamp": timestamp,
        "total_tests": total,
        "summary": {
            "avg_score": round(avg_score, 1),
            "pass_count": pass_count,
            "perfect_count": perfect_count,
            "pass_rate": f"{pass_count/total*100:.1f}%" if total > 0 else "0%",
            "avg_iterations": round(avg_iterations, 1),
            "avg_time_seconds": round(avg_time, 1),
        },
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # Markdown 报告
    md_lines = [
        f"# SciAgent 微通道散热器评测报告",
        f"",
        f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**测试数量**: {total}",
        f"**平均得分**: {avg_score:.1f}/100",
        f"**通过率**: {pass_count}/{total} ({pass_count/total*100:.0f}%)" if total > 0 else "",
        f"",
        f"## 总览",
        f"",
        f"| # | 类别 | 难度 | 得分 | 验证 | 画图 | 迭代 | 耗时(s) |",
        f"|---|------|------|------|------|------|------|---------|",
    ]
    for r in results:
        v = "PASS" if r["verification_passed"] else "FAIL"
        p = "YES" if r["plot_generated"] else "NO"
        md_lines.append(
            f"| {r['id']} | {r['category']} | {r['difficulty']} | "
            f"{r['score']}分 | {v} | {p} | {r['iterations']} | {r['elapsed_seconds']} |"
        )

    md_lines.append("")
    md_lines.append("## 详细结果")
    md_lines.append("")

    for r in results:
        md_lines.append(f"### 测试 #{r['id']}: {r['query'][:50]}")
        md_lines.append(f"")
        md_lines.append(f"**标注答案**: {r['ground_truth']}")
        md_lines.append(f"**得分**: {r['score']}/100")
        md_lines.append(f"")
        md_lines.append(f"**评分详情**:")
        md_lines.append(f"```")
        md_lines.append(r["score_details"])
        md_lines.append(f"```")
        md_lines.append(f"")
        md_lines.append(f"**Agent回答**:")
        md_lines.append(f"```")
        md_lines.append(r["agent_answer"][:1500] if r["agent_answer"] else "(无)")
        md_lines.append(f"```")
        md_lines.append(f"")
        if r["error_details"]:
            md_lines.append(f"**错误详情**: {'; '.join(r['error_details'][:3])}")
            md_lines.append(f"")
        md_lines.append(f"---")
        md_lines.append(f"")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n  报告已保存:")
    print(f"     JSON: {report_path}")
    print(f"     MD:   {md_path}")

    return report_path, md_path


def main():
    parser = argparse.ArgumentParser(description="SciAgent 微通道散热器评测脚本")
    parser.add_argument("--ids", nargs="+", type=int, help="只运行指定编号的测试（如 --ids 1 3 5）")
    parser.add_argument("--show-answers", action="store_true", help="只显示标注答案，不运行Agent")
    parser.add_argument("--dry-run", action="store_true", help="预览测试用例信息")
    parser.add_argument("--verify-only", action="store_true", help="只运行验证代码，检查标注答案正确性")
    parser.add_argument("--llm", type=str, default=None, choices=["mock", "deepseek", "claude"],
                        help="指定LLM模式")
    args = parser.parse_args()

    cases = load_benchmark(args.ids)

    if not cases:
        print("未找到匹配的测试用例。")
        return

    # 只看答案模式
    if args.show_answers:
        show_all_answers(cases)
        return

    # 验证标注答案模式
    if args.verify_only:
        verify_all_answers(cases)
        return

    # 预览模式
    if args.dry_run:
        print(f"\n共 {len(cases)} 个测试用例:")
        for c in cases:
            print(f"  #{c['id']} [{c['difficulty']}] {c['category']}: {c['query'][:50]}...")
        return

    # 运行评测
    print("\n" + "=" * 60)
    print(f"  SciAgent 微通道散热器评测 -- 共 {len(cases)} 道测试题")
    print("=" * 60)

    results = []
    for case in cases:
        result = run_single_test(case, args.llm)
        results.append(result)
        print()

    # 生成报告
    generate_report(results)


if __name__ == "__main__":
    main()
