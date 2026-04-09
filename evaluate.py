"""
SciAgent 评测脚本。
自动运行 benchmark.json 中的测试用例，对比 Agent 输出与标注答案，生成评测报告。

用法：
  python evaluate.py                  # 运行全部测试
  python evaluate.py --ids 1 3 5      # 只运行指定编号的测试
  python evaluate.py --show-answers   # 只查看标注答案（不运行Agent）
  python evaluate.py --dry-run        # 预览测试用例信息
"""

import json
import argparse
import time
import os
import sys
import io
from datetime import datetime
from pathlib import Path

# 修复 Windows 终端中文+特殊字符编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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


def show_ground_truth(case):
    """显示单个测试用例的标注答案"""
    gt = case["ground_truth"]
    print(f"\n{'─' * 60}")
    print(f"  测试 #{case['id']} | {case['category']} | 难度: {case['difficulty']}")
    print(f"  问题: {case['query']}")
    print(f"{'─' * 60}")
    print(f"  📋 标注答案: {gt['description']}")
    if "expected_values" in gt:
        print(f"  📊 期望值: {json.dumps(gt['expected_values'], ensure_ascii=False)}")
    if "tolerance" in gt:
        print(f"  📏 容差: ±{gt['tolerance']}")
    print(f"  ✅ 检查项:")
    for i, check in enumerate(gt["key_checks"], 1):
        print(f"     {i}. {check}")
    if "verification_code" in gt:
        print(f"  🔧 验证代码:")
        for line in gt["verification_code"].split("\n"):
            print(f"     {line}")
    print()


def show_all_answers(cases):
    """显示所有测试用例的标注答案"""
    print("\n" + "═" * 60)
    print("  SciAgent 测试基准 — 标注答案一览")
    print(f"  共 {len(cases)} 道测试题")
    print("═" * 60)
    for case in cases:
        show_ground_truth(case)


def run_single_test(case, llm_mode=None):
    """运行单个测试用例"""
    from sciagent.agent import Agent

    print(f"\n{'━' * 60}")
    print(f"  🚀 运行测试 #{case['id']}: {case['query'][:40]}...")
    print(f"{'━' * 60}")

    # 先显示标注答案
    print("\n  📌 【标注答案】")
    gt = case["ground_truth"]
    print(f"     {gt['description']}")
    if "expected_values" in gt:
        print(f"     期望值: {json.dumps(gt['expected_values'], ensure_ascii=False)}")
    for check in gt["key_checks"]:
        print(f"     • {check}")

    print(f"\n  🤖 【Agent 运行中...】")
    print(f"  {'─' * 40}")

    # 运行 Agent
    agent = Agent(llm_mode)
    start_time = time.time()
    iterations = 0
    errors = []

    def on_event(event):
        nonlocal iterations, errors
        if event.event_type == "tool_call":
            iterations += 1
            print(f"     [迭代 {iterations}] 调用工具: {event.metadata.get('code', '')[:80]}...")
        elif event.event_type == "error":
            errors.append(event.content)
            print(f"     ❌ 错误: {event.content[:100]}")
        elif event.event_type == "tool_result":
            result_preview = event.content[:200].replace("\n", " ")
            print(f"     ✅ 结果: {result_preview}")

    try:
        answer = agent.run(case["query"], on_event=on_event)
    except Exception as e:
        answer = f"Agent异常: {type(e).__name__}: {e}"
        errors.append(answer)

    elapsed = time.time() - start_time

    # 显示 Agent 回答
    print(f"\n  📝 【Agent 回答】")
    print(f"  {'─' * 40}")
    if answer:
        for line in answer.split("\n"):
            print(f"     {line}")
    else:
        print("     (无回答)")

    # 显示对比
    print(f"\n  📊 【对比结果】")
    print(f"  {'─' * 40}")
    print(f"     迭代次数: {iterations}")
    print(f"     错误次数: {len(errors)}")
    print(f"     耗时: {elapsed:.1f}秒")
    print(f"\n     🔍 请对照以下检查项判断正确性:")
    for i, check in enumerate(gt["key_checks"], 1):
        print(f"        {i}. {check}")

    # 检查是否生成了图片
    outputs_dir = PROJECT_ROOT / "outputs"
    if outputs_dir.exists():
        pngs = list(outputs_dir.glob("*.png"))
        has_plot = "应生成" in " ".join(gt["key_checks"]) and "PNG" in " ".join(gt["key_checks"])
        if has_plot:
            if pngs:
                print(f"     📈 生成了 {len(pngs)} 个图片文件: {[p.name for p in pngs[-3:]]}")
            else:
                print(f"     ⚠️  期望生成图片但未找到PNG文件")

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
    }


def generate_report(results):
    """生成评测报告"""
    REPORT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"eval_report_{timestamp}.json"
    md_path = REPORT_DIR / f"eval_report_{timestamp}.md"

    # JSON 报告
    report_data = {
        "timestamp": timestamp,
        "total_tests": len(results),
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # Markdown 报告
    md_lines = [
        f"# SciAgent 评测报告",
        f"",
        f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**测试数量**: {len(results)}",
        f"",
        f"## 总览",
        f"",
        f"| # | 类别 | 难度 | 迭代 | 错误 | 耗时(s) |",
        f"|---|------|------|------|------|---------|",
    ]
    for r in results:
        md_lines.append(
            f"| {r['id']} | {r['category']} | {r['difficulty']} | "
            f"{r['iterations']} | {r['errors']} | {r['elapsed_seconds']} |"
        )

    md_lines.append("")
    md_lines.append("## 详细结果")
    md_lines.append("")

    for r in results:
        md_lines.append(f"### 测试 #{r['id']}: {r['query'][:50]}")
        md_lines.append(f"")
        md_lines.append(f"**标注答案**: {r['ground_truth']}")
        md_lines.append(f"")
        md_lines.append(f"**Agent回答**:")
        md_lines.append(f"```")
        md_lines.append(r["agent_answer"][:1000] if r["agent_answer"] else "(无)")
        md_lines.append(f"```")
        md_lines.append(f"")
        if r["error_details"]:
            md_lines.append(f"**错误详情**: {'; '.join(r['error_details'][:3])}")
            md_lines.append(f"")
        md_lines.append(f"---")
        md_lines.append(f"")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'═' * 60}")
    print(f"  📄 报告已保存:")
    print(f"     JSON: {report_path}")
    print(f"     MD:   {md_path}")
    print(f"{'═' * 60}")

    return report_path, md_path


def main():
    parser = argparse.ArgumentParser(description="SciAgent 评测脚本")
    parser.add_argument("--ids", nargs="+", type=int, help="只运行指定编号的测试（如 --ids 1 3 5）")
    parser.add_argument("--show-answers", action="store_true", help="只显示标注答案，不运行Agent")
    parser.add_argument("--dry-run", action="store_true", help="预览测试用例信息")
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

    # 预览模式
    if args.dry_run:
        print(f"\n共 {len(cases)} 个测试用例:")
        for c in cases:
            print(f"  #{c['id']} [{c['difficulty']}] {c['category']}: {c['query'][:50]}...")
        return

    # 运行评测
    print("\n" + "═" * 60)
    print(f"  SciAgent 评测 — 共 {len(cases)} 道测试题")
    print("═" * 60)

    results = []
    for case in cases:
        result = run_single_test(case, args.llm)
        results.append(result)
        print()

    # 生成报告
    generate_report(results)


if __name__ == "__main__":
    main()
