"""
Step 7: SciAgent 评测集成 —— 三模型对比评测
=============================================
目标：
  1. 将 SciTune 微调后的 Qwen2.5 接入 SciAgent 评测框架
  2. 三模型对比: 原始 Qwen2.5 vs SciTune SFT vs SciTune SFT+DPO
  3. 在科学计算任务上生成定量评测报告

与前几步的关系：
  - Step 5 (SFT) → 训练出科学计算领域的 LoRA adapter
  - Step 6 (DPO) → 在 SFT 基础上做偏好对齐
  - Step 7 (本步) → 在标准化 benchmark 上评测三个模型，形成完整的实验闭环

评测思路：
  不走 SciAgent 的 Agent 循环（因为本地小模型不支持 tool_use），
  而是直接给模型科学计算问题，评估其代码生成质量。
  这是 SFT 模型最直接的能力检验：给一个问题，看它能不能写出正确的代码。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import re
import time
import argparse
import traceback
from datetime import datetime

# 国内网络访问 HuggingFace 不稳定，自动设置 hf-mirror.com 镜像
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Part 1: 模型加载工具
# ============================================================

class QwenModel:
    """
    封装 Qwen2.5 模型的加载和推理。
    支持三种模式：raw（原始模型）、sft（SFT adapter）、dpo（SFT + DPO adapter）
    """

    def __init__(self, model_name, adapter_path=None, mode_name="raw"):
        self.mode_name = mode_name
        self.model_name = model_name

        # 4bit 量化配置（与 Step 5/6 一致）
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"  加载 {mode_name} 模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if adapter_path and os.path.exists(adapter_path):
            print(f"  加载 adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"  adapter 已加载")

        self.model.eval()
        print(f"  {mode_name} 模型就绪")

    def generate(self, prompt, max_new_tokens=512):
        """生成回答（ChatML 格式）"""
        text = (
            f"<|im_start|>system\n"
            f"You are a scientific computing expert. "
            f"Solve problems with complete, runnable Python code. "
            f"Include all imports. Use print() to output results.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{prompt}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def unload(self):
        """释放显存（评测时需要轮流加载不同模型）"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        print(f"  {self.mode_name} 模型已释放")


# ============================================================
# Part 2: 评测核心逻辑
# ============================================================

def extract_python_code(text):
    """
    从模型回答中提取 Python 代码块。
    优先提取 ```python ... ``` 包裹的代码，
    如果没有则尝试提取 import/def/print 开头的连续行。
    """
    # 方式1: 匹配 markdown 代码块
    code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        # 合并所有代码块
        return "\n\n".join(code_blocks)

    # 方式2: 提取看起来像 Python 代码的连续行
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ', 'def ', 'class ',
                                'for ', 'while ', 'if ', 'with ',
                                'print(', '#', 'np.', 'plt.',
                                'x =', 'y =', 'result')):
            in_code = True
        if in_code:
            if stripped == '' and len(code_lines) > 3:
                # 连续空行可能意味着代码结束
                pass
            else:
                code_lines.append(line)

    return "\n".join(code_lines) if code_lines else ""


def execute_code_safely(code, timeout=30):
    """
    安全执行 Python 代码，捕获 stdout 输出。
    用于检验模型生成的代码是否可以运行。

    返回: (success: bool, output: str, error: str)
    """
    if not code.strip():
        return False, "", "没有可执行的代码"

    # 移除 plt.show()（无头环境不支持）
    code = code.replace("plt.show()", "# plt.show()  # disabled for evaluation")

    # 捕获 stdout
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        # 准备执行环境
        exec_globals = {"__builtins__": __builtins__}
        exec(code, exec_globals)
        output = redirected_output.getvalue()
        return True, output, ""
    except Exception as e:
        output = redirected_output.getvalue()
        error_msg = f"{type(e).__name__}: {e}"
        return False, output, error_msg
    finally:
        sys.stdout = old_stdout


def extract_numbers(text):
    """从文本中提取所有浮点数"""
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    return [float(x) for x in re.findall(pattern, text)]


def score_response(question, response, code_output):
    """
    对模型回答评分 (0-100)。

    评分维度（简化版，适配本地小模型）：
      - 代码可执行性 (30分): 代码能否成功运行
      - 代码完整性 (30分): 是否包含 import、是否有 print 输出
      - 输出相关性 (20分): 输出中是否包含数值结果
      - 回答结构 (20分): 是否有解释文字、代码结构是否清晰
    """
    score = 0
    details = []

    # 1. 提取代码
    code = extract_python_code(response)
    has_code = len(code.strip()) > 20

    if not has_code:
        details.append("[FAIL] 未检测到有效代码 (+0/30)")
        details.append("[FAIL] 无代码可执行 (+0/30)")
        details.append("[FAIL] 无输出 (+0/20)")
    else:
        # 2. 代码完整性 (30分)
        has_import = 'import' in code
        has_print = 'print' in code
        completeness = 0
        if has_import:
            completeness += 15
            details.append("[PASS] 包含 import 语句 (+15)")
        else:
            details.append("[FAIL] 缺少 import 语句 (+0/15)")
        if has_print:
            completeness += 15
            details.append("[PASS] 包含 print 输出 (+15)")
        else:
            details.append("[FAIL] 缺少 print 输出 (+0/15)")
        score += completeness

        # 3. 代码可执行性 (30分)
        success, output, error = execute_code_safely(code)
        if success:
            score += 30
            details.append(f"[PASS] 代码执行成功 (+30)")
        elif output:
            # 部分执行成功
            score += 15
            details.append(f"[PARTIAL] 代码部分执行: {error} (+15/30)")
        else:
            details.append(f"[FAIL] 代码执行失败: {error} (+0/30)")

        # 4. 输出相关性 (20分)
        combined_output = (output or "") + "\n" + (code_output or "")
        numbers = extract_numbers(combined_output)
        if len(numbers) >= 2:
            score += 20
            details.append(f"[PASS] 输出包含 {len(numbers)} 个数值结果 (+20)")
        elif len(numbers) >= 1:
            score += 10
            details.append(f"[PARTIAL] 输出仅包含 {len(numbers)} 个数值 (+10/20)")
        else:
            details.append("[FAIL] 输出中无数值结果 (+0/20)")

    # 5. 回答结构 (20分)
    non_code_text = response.replace(code, "") if has_code else response
    has_explanation = len(non_code_text.strip()) > 30
    has_structure = any(kw in response for kw in ['```', '步骤', 'Step', '原理', '解释', '分析', '结果'])

    structure_score = 0
    if has_explanation:
        structure_score += 10
        details.append("[PASS] 包含解释文字 (+10)")
    else:
        details.append("[FAIL] 缺少解释文字 (+0/10)")
    if has_structure:
        structure_score += 10
        details.append("[PASS] 回答结构清晰 (+10)")
    else:
        details.append("[FAIL] 回答结构不够清晰 (+0/10)")
    score += structure_score

    return score, "\n".join(details)


# ============================================================
# Part 3: 评测问题集
# ============================================================

def get_evaluation_questions():
    """
    评测问题集。
    优先加载 SciAgent benchmark.json，如果没有则使用内置的通用科学计算题。

    设计原则：覆盖 SFT 训练的 8 个领域，每个领域 1 道代表性问题。
    """
    # 尝试加载 SciAgent benchmark
    sciagent_path = os.path.join(
        os.path.dirname(__file__), "..", "sciagent_2", "benchmark.json"
    )
    if os.path.exists(sciagent_path):
        print(f"  加载 SciAgent benchmark: {sciagent_path}")
        with open(sciagent_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
        # 从 benchmark 中提取 query 作为评测题
        questions = []
        for case in benchmark:
            questions.append({
                "id": case["id"],
                "category": case.get("category", "unknown"),
                "question": case["query"],
                "source": "sciagent_benchmark",
                "ground_truth": case.get("ground_truth", {}),
            })
        return questions

    # 内置评测题（覆盖 SFT 训练的 8 个领域 + 2 个新增领域）
    print("  使用内置评测题（未找到 SciAgent benchmark）")
    return [
        {
            "id": 1,
            "category": "equation_solving",
            "question": "用Python求解方程 x^3 - 6x^2 + 11x - 6 = 0 的所有实数根，并验证每个根代入方程结果接近0",
            "source": "builtin",
        },
        {
            "id": 2,
            "category": "ode_solving",
            "question": "用scipy.integrate.solve_ivp求解常微分方程 dy/dx = -2y + sin(x), y(0)=1, 在x=[0,10]上求解，用print输出y(10)的值",
            "source": "builtin",
        },
        {
            "id": 3,
            "category": "optimization",
            "question": "用scipy.optimize.minimize最小化 Rosenbrock 函数 f(x,y)=(1-x)^2+100(y-x^2)^2，初始点(0,0)，输出最优解和最优值",
            "source": "builtin",
        },
        {
            "id": 4,
            "category": "statistics",
            "question": "生成1000个正态分布N(50,10)随机数，计算均值、标准差、偏度、峰度，用print输出所有统计量",
            "source": "builtin",
        },
        {
            "id": 5,
            "category": "signal_processing",
            "question": "生成包含30Hz和80Hz成分的信号(采样率500Hz, 持续1秒)，用FFT分析频谱，输出两个峰值频率",
            "source": "builtin",
        },
        {
            "id": 6,
            "category": "curve_fitting",
            "question": "对数据 x=[0,1,2,3,4,5], y=[2.1,7.7,13.6,27.2,40.9,61.1] 做指数拟合 y=a*exp(b*x)+c，输出拟合参数a,b,c",
            "source": "builtin",
        },
        {
            "id": 7,
            "category": "linear_algebra",
            "question": "计算矩阵 [[4,2,1],[1,3,1],[2,1,5]] 的特征值和特征向量，验证 Av = λv 对每个特征对成立",
            "source": "builtin",
        },
        {
            "id": 8,
            "category": "numerical_methods",
            "question": "用Simpson法则数值计算积分 ∫_0^π sin(x)dx (n=100)，与解析解2.0对比误差",
            "source": "builtin",
        },
        {
            "id": 9,
            "category": "pde_solving",
            "question": "用有限差分法求解一维热传导方程 ∂u/∂t = 0.01*∂²u/∂x²，x∈[0,1]，初始条件u(x,0)=sin(πx)，边界u(0,t)=u(1,t)=0，输出t=0.5时x=0.5处的温度",
            "source": "builtin",
        },
        {
            "id": 10,
            "category": "data_analysis",
            "question": "用numpy生成50个数据点(含3个异常值)，用Z-score方法(阈值=2)检测异常值，输出异常值的数量和位置",
            "source": "builtin",
        },
    ]


# ============================================================
# Part 4: 主评测流程
# ============================================================

def evaluate_model(model, questions, model_name):
    """
    用指定模型回答所有评测问题，返回评测结果列表。
    """
    results = []

    for q in questions:
        print(f"\n  [{model_name}] 问题 #{q['id']}: {q['question'][:60]}...")

        start_time = time.time()
        try:
            response = model.generate(q["question"], max_new_tokens=512)
        except Exception as e:
            response = f"生成失败: {e}"
        elapsed = time.time() - start_time

        # 提取代码并尝试执行
        code = extract_python_code(response)
        success, output, error = execute_code_safely(code) if code else (False, "", "无代码")

        # 评分
        score, details = score_response(q["question"], response, output)

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"][:80],
            "model": model_name,
            "response": response[:500],  # 截断保存
            "code_extracted": code[:300] if code else "",
            "code_success": success,
            "code_output": output[:300] if output else "",
            "code_error": error,
            "score": score,
            "score_details": details,
            "elapsed": round(elapsed, 1),
        })

        bar = "█" * (score // 5) + "░" * (20 - score // 5)
        status = "PASS" if score >= 60 else "FAIL"
        print(f"    [{status}] {score:3d}/100  {bar}  ({elapsed:.1f}s)")

    return results


def print_comparison_table(all_results, model_names):
    """打印三模型对比表"""
    print("\n" + "=" * 80)
    print("  三模型对比评测结果")
    print("=" * 80)

    # 表头
    header = f"  {'问题':^6} | {'类别':^20}"
    for name in model_names:
        header += f" | {name:^12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # 按问题 ID 汇总
    question_ids = sorted(set(r["id"] for r in all_results[model_names[0]]))
    for qid in question_ids:
        row = f"  #{qid:3d}   |"
        category = ""
        for name in model_names:
            r = next((r for r in all_results[name] if r["id"] == qid), None)
            if r:
                category = r["category"]
                score = r["score"]
                status = "✓" if score >= 60 else "✗"
                row += f" {status} {score:3d}分      |"
            else:
                row += f"   ---        |"
        # 插入类别
        row = row.replace("|", f" {category:^20} |", 1)
        print(row)

    # 汇总统计
    print("  " + "-" * (len(header) - 2))
    summary_row = f"  {'平均':^6} | {'':^20}"
    for name in model_names:
        scores = [r["score"] for r in all_results[name]]
        avg = sum(scores) / len(scores) if scores else 0
        summary_row += f" | {avg:5.1f}分      "
    print(summary_row)

    pass_row = f"  {'通过率':^5} | {'(>=60分)':^20}"
    for name in model_names:
        scores = [r["score"] for r in all_results[name]]
        total = len(scores)
        passed = sum(1 for s in scores if s >= 60)
        pass_row += f" | {passed}/{total}          "
    print(pass_row)
    print("=" * 80)


def save_report(all_results, model_names):
    """保存评测报告为 JSON"""
    report_dir = os.path.join(os.path.dirname(__file__), "eval_reports")
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"scitune_eval_{timestamp}.json")

    # 计算各模型统计
    summaries = {}
    for name in model_names:
        scores = [r["score"] for r in all_results[name]]
        summaries[name] = {
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "pass_count": sum(1 for s in scores if s >= 60),
            "total": len(scores),
            "pass_rate": f"{sum(1 for s in scores if s >= 60)/len(scores)*100:.1f}%" if scores else "0%",
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
        }

    report = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "summaries": summaries,
        "detailed_results": {name: all_results[name] for name in model_names},
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  评测报告已保存: {report_path}")
    return report_path


# ============================================================
# Part 5: 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SciTune 三模型对比评测")
    parser.add_argument("--models", nargs="+", default=["raw", "sft", "dpo"],
                        choices=["raw", "sft", "dpo"],
                        help="要评测的模型列表 (默认: raw sft dpo)")
    parser.add_argument("--ids", nargs="+", type=int, default=None,
                        help="只评测指定编号的题目")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="最大生成token数 (默认: 512)")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式: 只评测3道题 (题1,5,8)")
    args = parser.parse_args()

    print("=" * 60)
    print("  SciTune 三模型对比评测")
    print("=" * 60)
    print(f"  设备: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 确定模型路径
    LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "Qwen", "Qwen2.5-1.5B-Instruct")
    LOCAL_BASE_PATH = os.path.join(os.path.dirname(__file__), "models", "Qwen", "Qwen2.5-1.5B")

    if os.path.exists(LOCAL_MODEL_PATH):
        model_name = LOCAL_MODEL_PATH
    elif os.path.exists(LOCAL_BASE_PATH):
        model_name = LOCAL_BASE_PATH
    else:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    sft_adapter = os.path.join(os.path.dirname(__file__), "qwen-sft-lora", "final_adapter")
    dpo_adapter = os.path.join(os.path.dirname(__file__), "qwen-dpo-lora")

    print(f"\n  基座模型: {model_name}")
    print(f"  SFT adapter: {sft_adapter} {'(存在)' if os.path.exists(sft_adapter) else '(不存在)'}")
    print(f"  DPO adapter: {dpo_adapter} {'(存在)' if os.path.exists(dpo_adapter) else '(不存在)'}")

    # 加载评测题
    questions = get_evaluation_questions()
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]
    elif args.quick:
        questions = [q for q in questions if q["id"] in [1, 5, 8]]

    print(f"\n  评测题目: {len(questions)} 道")
    print(f"  评测模型: {', '.join(args.models)}")

    # 逐个模型评测（为了省显存，一次只加载一个模型）
    all_results = {}

    for mode in args.models:
        print(f"\n{'=' * 60}")
        print(f"  评测模型: {mode.upper()}")
        print(f"{'=' * 60}")

        if mode == "raw":
            model = QwenModel(model_name, adapter_path=None, mode_name="raw")
        elif mode == "sft":
            if not os.path.exists(sft_adapter):
                print(f"  跳过: SFT adapter 不存在 ({sft_adapter})")
                print(f"  请先运行 step5_sft_qwen.py 训练 SFT 模型")
                continue
            model = QwenModel(model_name, adapter_path=sft_adapter, mode_name="sft")
        elif mode == "dpo":
            if not os.path.exists(dpo_adapter):
                print(f"  跳过: DPO adapter 不存在 ({dpo_adapter})")
                print(f"  请先运行 step6_dpo_qwen.py 训练 DPO 模型")
                continue
            # DPO 模型 = (基座 merge SFT) + DPO LoRA
            # 先加载基座 + SFT merge，再加载 DPO adapter
            if os.path.exists(sft_adapter):
                model = QwenModel(model_name, adapter_path=sft_adapter, mode_name="dpo_base")
                # merge SFT adapter 到基座（与 step6 训练时一致）
                model.model = model.model.merge_and_unload()
                # 再加载 DPO adapter
                model.model = PeftModel.from_pretrained(model.model, dpo_adapter)
                model.mode_name = "dpo"
                model.model.eval()
                print(f"  SFT merged + DPO adapter 加载完成")
            else:
                model = QwenModel(model_name, adapter_path=dpo_adapter, mode_name="dpo")
        else:
            continue

        results = evaluate_model(model, questions, mode)
        all_results[mode] = results

        # 释放显存给下一个模型
        model.unload()

    # 打印对比表
    evaluated_models = list(all_results.keys())
    if len(evaluated_models) > 0:
        print_comparison_table(all_results, evaluated_models)

        # 保存报告
        report_path = save_report(all_results, evaluated_models)

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"""
Step 7 完成：SciTune 三模型对比评测

评测方法：
  1. 给模型科学计算问题，让它生成 Python 代码
  2. 提取代码并尝试执行，检查是否可运行
  3. 从四个维度评分：代码可执行性、代码完整性、输出相关性、回答结构
  4. 对比 raw / SFT / DPO 三个版本的得分差异

评分维度 (共100分)：
  - 代码可执行性 (30分): 代码能否成功运行
  - 代码完整性 (30分): 是否包含 import、print
  - 输出相关性 (20分): 输出是否包含数值结果
  - 回答结构 (20分): 是否有解释文字和清晰结构

完整训练链路回顾：
  Step 1: 理解Transformer
  Step 2: GPT-2 全量SFT
  Step 3: GPT-2 LoRA SFT
  Step 4: GPT-2 DPO (原理验证)
  Step 5: Qwen2.5 LoRA SFT (科学计算领域)
  Step 6: Qwen2.5 DPO 对齐
  Step 7: 三模型对比评测 (本步, 形成实验闭环)
""")


if __name__ == "__main__":
    main()
