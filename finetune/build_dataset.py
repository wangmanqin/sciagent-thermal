"""
build_dataset.py — 用 DeepSeek API 批量生成科学计算训练数据
============================================================
为 SciTune 项目构建两类数据：
  1. SFT 数据（Supervised Fine-Tuning，监督微调）：
     - 格式：科学计算问题 → 高质量 Python 代码解答
     - 目标约 200 条
     - 用途：教会模型"怎样正确回答科学计算问题"

  2. DPO 偏好数据（Direct Preference Optimization，直接偏好优化）：
     - 格式：同一问题 → (好回答 chosen, 差回答 rejected) 配对
     - 目标约 100 对
     - 用途：教会模型"区分好回答和差回答"，在 SFT 基础上进一步对齐

数据构建思路：
  - 人工定义 10 道"种子题目"，覆盖 8 个科学计算子领域
  - 用 DeepSeek API 将每道种子题扩展为多道变体题（增加多样性）
  - 再用 API 为每道题生成解答（SFT）和低质量对比回答（DPO）
  - 这种"用强模型生成数据训练弱模型"的做法属于知识蒸馏的思路
"""

# --- 标准库导入 ---
import sys
import io
# Windows 终端默认编码可能不是 UTF-8，这行强制设置标准输出为 UTF-8，
# 防止打印中文时出现乱码（UnicodeEncodeError）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json          # 用于 JSON 序列化/反序列化（保存和解析数据）
import time          # 用于 time.sleep() 控制 API 调用频率，防止触发限流
import os            # 用于路径拼接（os.path.join）和文件操作
from openai import OpenAI  # DeepSeek 的 API 兼容 OpenAI SDK，所以直接复用这个库
from dotenv import load_dotenv  # 自动加载项目根目录的 .env 文件

# --- 初始化 DeepSeek API 客户端 ---
# DeepSeek 提供了与 OpenAI 完全兼容的 API 接口，
# 所以可以直接用 OpenAI 的 Python SDK，只需要改 base_url 指向 DeepSeek
load_dotenv()
_api_key = os.environ.get("DEEPSEEK_API_KEY")
if not _api_key:
    raise RuntimeError(
        "未检测到 DEEPSEEK_API_KEY。请在项目根目录的 .env 中设置,或通过环境变量注入。"
    )
client = OpenAI(
    api_key=_api_key,                                 # 从环境变量读取,不硬编码
    base_url="https://api.deepseek.com",              # 指向 DeepSeek 而非 OpenAI
    timeout=60,  # 单次请求最多等 60 秒，超时自动报错，避免程序卡死
)

def call_deepseek(prompt, system_prompt="You are a helpful assistant.", max_tokens=1024):
    """
    封装 DeepSeek API 调用，带自动重试机制。

    参数：
      prompt       — 用户提问内容（每次调用不同，比如"生成变体题目"或"写解答代码"）
      system_prompt — 系统提示词，定义 AI 的角色（一般不改，用默认值）
      max_tokens   — 回复最大 token 数，1024 ≈ 大约 500-700 个中文字

    返回：
      成功 → API 返回的文本内容（str）
      失败 → None（3 次都失败才返回 None）
    """
    # 重试机制：网络请求可能因为网络波动、服务器繁忙等原因偶尔失败，
    # 所以最多尝试 3 次，这是 API 调用的工程标准做法
    for attempt in range(3):
        try:
            # 调用 Chat Completions API（聊天补全接口）
            # 这是目前主流大模型 API 的标准格式：messages 列表包含对话历史
            response = client.chat.completions.create(
                model="deepseek-chat",       # 使用 DeepSeek 的聊天模型
                messages=[
                    # system 消息：设定 AI 的行为角色（在对话开头，只出现一次）
                    {"role": "system", "content": system_prompt},
                    # user 消息：用户的具体请求
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,       # 限制回复长度，防止回答太长浪费 token
                temperature=0.7,             # 控制随机性：0=确定性最高，1=最随机
                                             # 0.7 是比较好的平衡点，既有多样性又不会太离谱
            )
            # 从响应对象中提取 AI 的回复文本
            # choices[0] — API 可能返回多个候选回复，我们只取第一个
            # .message.content — 提取消息内容
            # .strip() — 去掉首尾空白字符
            result = response.choices[0].message.content.strip()
            print(f"    [API] 返回 {len(result)} 字符")
            return result
        except Exception as e:
            # 捕获所有异常（网络超时、API 限流、服务器错误等）
            print(f"  API调用失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(3)  # 等 3 秒再重试，给服务器恢复时间
    # 3 次都失败，返回 None，调用方会检查这个值并跳过
    return None


# ============================================================
# Part 1: 定义种子任务（基于 SciAgent benchmark 扩展）
# ============================================================
# "种子"的含义：我们手工写少量高质量题目作为起点，
# 后面通过 API 自动扩展出大量变体题目（就像种子发芽长出更多题）。
#
# 为什么需要种子？直接让 AI 出题容易跑偏或重复，
# 给一个具体的示例做参考，生成的变体质量更高、方向更可控。
#
# 覆盖 8 个科学计算子领域，确保训练数据的多样性。

SEED_CATEGORIES = {
    # 每个类别是一个字典，包含：
    #   description  — 类别的中文描述（会传给 API，帮助它理解出题方向）
    #   seed_prompts — 种子题目列表（人工编写的示例题，用于引导 API 生成变体）

    "equation_solving": {
        "description": "方程求解（代数方程、超越方程、方程组）",
        "seed_prompts": [
            "求解方程 x^3 - 2x - 5 = 0 的所有实数根，画出函数图像",
            "求解线性方程组 3x+2y-z=1, 2x-2y+4z=-2, -x+0.5y-z=0，验证解的正确性",
            "用牛顿迭代法求解超越方程 e^x + x^2 - 2 = 0，初始猜测x0=0，画收敛过程",
            "用sympy符号计算求解参数方程组，并用matplotlib画出解随参数变化的曲线",
        ]
    },
    "ode_solving": {
        "description": "常微分方程数值求解（欧拉法、RK4、scipy.odeint）",
        "seed_prompts": [
            "用龙格-库塔法(RK4)数值求解 dy/dx = -2y + x, y(0)=1, 在x=[0,5]上求解，与解析解对比",
            "用scipy.integrate.solve_ivp求解阻尼振子 x''+0.5x'+4x=cos(t), x(0)=1, x'(0)=0，画相图",
            "用欧拉法和改进欧拉法对比求解 dy/dx=-y+sin(x)，分析步长对精度的影响",
        ]
    },
    "optimization": {
        "description": "单目标/多目标优化（scipy.optimize、NSGA-II、模拟退火）",
        "seed_prompts": [
            "用scipy最小化 Rosenbrock 函数 f(x,y)=(1-x)^2+100(y-x^2)^2，初始点(0,0)，画等高线图",
            "用NSGA-II算法优化双目标问题：最小化 f1=x^2, f2=(x-2)^2, x在[-5,5]范围内，画Pareto前沿",
            "用scipy.optimize.linprog求解线性规划问题：最大化利润并满足资源约束",
            "用模拟退火算法求解旅行商问题(TSP)，随机生成10个城市坐标，画最优路径",
        ]
    },
    "statistics": {
        "description": "统计分析（分布拟合、假设检验、回归）",
        "seed_prompts": [
            "生成1000个正态分布N(50,10)随机数，计算均值、标准差、中位数，画直方图和拟合曲线",
            "用scipy.stats对两组实验数据做t检验，判断均值是否有显著差异，给出p值和置信区间",
            "用sklearn做多元线性回归，计算R^2、残差分析、画预测vs真实值散点图",
        ]
    },
    "signal_processing": {
        "description": "信号处理（FFT频谱分析、滤波、卷积）",
        "seed_prompts": [
            "生成包含50Hz和120Hz的信号（采样率1000Hz），加噪声后用FFT分析频谱，画时域和频域图",
            "设计一个巴特沃斯低通滤波器(截止频率100Hz)，滤除信号中的高频噪声，画滤波前后对比",
            "用小波变换(pywt)分析非平稳信号，画时频图和各层小波系数",
        ]
    },
    "curve_fitting": {
        "description": "曲线拟合（多项式、指数、自定义函数）",
        "seed_prompts": [
            "对数据 x=[0,1,2,3,4,5], y=[2.1,7.7,13.6,27.2,40.9,61.1] 做指数拟合 y=a*exp(b*x)+c",
            "用scipy.optimize.curve_fit拟合高斯函数 y=A*exp(-(x-mu)^2/(2*sigma^2))，从带噪声的数据中恢复参数",
            "用不同阶次(1~6阶)多项式拟合sin(x)数据，画出拟合效果并分析过拟合现象",
        ]
    },
    "linear_algebra": {
        "description": "线性代数（矩阵运算、特征值、SVD）",
        "seed_prompts": [
            "计算矩阵 [[4,2],[1,3]] 的特征值和特征向量，验证 Av = λv",
            "用SVD分解对一张图片做压缩，保留不同数量的奇异值，画出压缩效果对比图",
            "用QR分解求解最小二乘问题 Ax=b（过定系统），并与numpy.linalg.lstsq对比",
        ]
    },
    "numerical_methods": {
        "description": "数值方法（积分、插值、微分）",
        "seed_prompts": [
            "用Simpson法则数值计算积分 ∫_0^π sin(x)dx，与解析解对比误差",
            "用拉格朗日插值和三次样条插值对比，对Runge函数1/(1+25x^2)在[-1,1]上插值",
            "用有限差分法(中心差分)数值求导，分析步长h对精度的影响",
        ]
    },
    "pde_solving": {
        "description": "偏微分方程数值求解（有限差分、热传导方程）",
        "seed_prompts": [
            "用有限差分法求解一维热传导方程 ∂u/∂t = α*∂²u/∂x²，初始条件为正弦分布，画出温度随时间演化图",
            "用Jacobi迭代法求解二维Laplace方程 ∂²u/∂x²+∂²u/∂y²=0，给定边界条件，画等温线图",
        ]
    },
    "data_analysis": {
        "description": "数据分析与可视化（pandas、matplotlib高级用法）",
        "seed_prompts": [
            "用numpy生成模拟实验数据(含异常值)，用Z-score和IQR两种方法检测异常值，对比结果",
            "用主成分分析(PCA)对高维数据降维，画出前两个主成分的散点图和方差解释比",
        ]
    },
}


# ============================================================
# Part 2: 生成 SFT（监督微调）数据
# ============================================================
# SFT 的核心思想：给模型看大量"问题→标准答案"的示例，
# 让它学会模仿这种回答模式。就像做练习题看参考答案。

def generate_sft_variants(category, description, seed_prompt, num_variants=2):
    """
    基于一道种子题，生成多道变体题 + 每道题的高质量解答。

    流程：种子题 --[API扩展]--> N 道变体题 --[API解答]--> N 条 SFT 数据

    参数：
      category     — 类别名称（如 "optimization"），用于标记数据来源
      description  — 类别中文描述（传给 API 帮助理解出题方向）
      seed_prompt  — 种子题目（人工编写的示例题）
      num_variants — 要生成几道变体题（默认 2，主流程中传 3）

    返回：
      SFT 数据列表，每条格式为 {"instruction": 题目, "output": 解答, "category": 类别}
    """

    # ---- Step 1: 让 API 基于种子题生成变体题目 ----
    # 这个 prompt 的设计要点：
    #   - 给 AI 一个角色（"科学计算出题专家"）
    #   - 提供示例题目和类别信息作为参考
    #   - 明确约束（Python编程、本科难度、有数值参数）
    #   - 要求严格 JSON 格式输出，方便程序自动解析
    variant_prompt = f"""你是一个科学计算出题专家。请基于以下示例题目，生成{num_variants}道同类型但不同的科学计算编程题。

类别：{description}
示例题目：{seed_prompt}

要求：
1. 每道题都需要用 Python 编程求解
2. 难度中等，本科生能做
3. 题目要具体、有明确的数值参数
4. 只输出题目，不要解答

请严格按以下JSON格式输出（不要输出其他内容）：
[
  "题目1的完整描述",
  "题目2的完整描述"
]"""

    print(f"  生成 {category} 变体...")
    variants_raw = call_deepseek(variant_prompt)  # 调用 API 获取变体题目
    if not variants_raw:
        return []  # API 3 次都失败了，跳过这个种子题

    # ---- 解析 API 返回的 JSON ----
    # API 有时候会在 JSON 前后加一些多余文字（如 "以下是题目："），
    # 所以需要手动找到 JSON 数组的起止位置 [ ... ]
    try:
        start = variants_raw.find('[')       # 找第一个 [ 的位置
        end = variants_raw.rfind(']') + 1    # 找最后一个 ] 的位置（+1 因为切片是左闭右开）
        if start >= 0 and end > start:
            # 成功找到 JSON 数组部分，截取并解析
            variants = json.loads(variants_raw[start:end])
        else:
            # 没找到方括号，尝试直接解析整个返回值
            variants = json.loads(variants_raw)
    except json.JSONDecodeError:
        # JSON 解析失败（API 返回格式不对），降级使用原始种子题
        print(f"  解析变体失败，使用种子任务")
        variants = [seed_prompt]

    # ---- Step 2: 为每道变体题生成高质量 Python 解答 ----
    sft_data = []
    for variant in variants:
        # 这个 prompt 要求 AI 输出完整可运行的代码 + 原理解释
        solution_prompt = f"""请用 Python 解决以下科学计算问题。给出完整可运行的代码，包含所有import，用print()输出结果。代码前后各用1句话解释原理和结果。

问题：{variant}"""

        solution = call_deepseek(solution_prompt, max_tokens=1024)
        if solution:
            # 组装成标准的 SFT 训练数据格式
            # instruction: 模型的输入（问题）
            # output: 模型应该学会的输出（高质量解答）
            sft_data.append({
                "instruction": variant,
                "output": solution,
                "category": category,
            })
            print(f"    ✓ 生成解答: {variant[:50]}...")
        time.sleep(0.5)  # 每次 API 调用间隔 0.5 秒，避免触发频率限制（rate limit）

    return sft_data


# ============================================================
# Part 3: 生成 DPO（直接偏好优化）偏好数据
# ============================================================
# DPO 的核心思想：给模型看同一个问题的"好回答"和"差回答"，
# 让模型学会判断哪种回答更好，从而提升回答质量。
#
# 类比：考试后老师不仅给你看标准答案（SFT），
# 还给你看一份写得很差的答案让你对比（DPO），这样你更清楚"好"在哪里。

def generate_dpo_pair(instruction, chosen_output):
    """
    基于已有的高质量解答（chosen），生成一个低质量对比回答（rejected）。

    参数：
      instruction   — 题目
      chosen_output — 已经生成好的高质量解答（来自 SFT 阶段）

    返回：
      成功 → {"prompt": 题目, "chosen": 好回答, "rejected": 差回答}
      失败 → None

    注意：chosen_output 参数虽然没有直接传给 API，但保存在返回的数据中，
    和 rejected 配对组成偏好对。
    """

    # 让 API 故意生成一个低质量回答
    # Prompt 中明确列出"低质量特征"，引导 API 生成符合要求的差回答
    rejected_prompt = f"""请用一种**低质量**的方式回答以下科学计算问题。

问题：{instruction}

低质量的特征：
- 解释模糊，不说清楚原理
- 代码不完整或有小错误
- 没有输出关键数值结果
- 不画图或图很粗糙
- 语气随意敷衍

请直接给出这个低质量回答（不要说明这是低质量的）："""

    rejected = call_deepseek(rejected_prompt, max_tokens=1024)
    if rejected:
        # 组装 DPO 标准格式：一个 prompt 对应一个好回答 + 一个差回答
        return {
            "prompt": instruction,       # 题目（模型的输入）
            "chosen": chosen_output,     # 好回答（模型应该偏好的）
            "rejected": rejected,        # 差回答（模型应该避免的）
        }
    return None


# ============================================================
# Part 4: 主流程 — 串联所有步骤，生成完整数据集
# ============================================================

def main():
    """
    主流程：生成 SFT + DPO 训练数据集。

    支持的命令行参数：
      --resume       追加模式：在已有数据上继续生成，不覆盖已有数据
      --sft-only     只生成 SFT 数据，跳过 DPO
      --dpo-only     只为已有 SFT 数据生成 DPO 偏好对
      --variants N   每个种子题生成 N 个变体（默认 8）
      --dpo-ratio R  SFT 数据中生成 DPO 对的比例（默认 0.5，即一半）
    """
    import argparse
    parser = argparse.ArgumentParser(description="SciTune 数据集构建")
    parser.add_argument("--resume", action="store_true",
                        help="追加模式：保留已有数据，继续生成新数据")
    parser.add_argument("--sft-only", action="store_true",
                        help="只生成 SFT 数据，跳过 DPO")
    parser.add_argument("--dpo-only", action="store_true",
                        help="只为已有 SFT 数据生成 DPO 偏好对")
    parser.add_argument("--variants", type=int, default=8,
                        help="每个种子题生成的变体数量（默认 8）")
    parser.add_argument("--dpo-ratio", type=float, default=0.5,
                        help="SFT 数据中生成 DPO 对的比例（默认 0.5）")
    args = parser.parse_args()

    print("=" * 60)
    print("SciTune 数据集构建")
    if args.resume:
        print("（追加模式：保留已有数据）")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)  # 确保 data 目录存在

    sft_path = os.path.join(data_dir, "sft_train.json")
    dpo_path = os.path.join(data_dir, "dpo_train.json")

    # ---- 加载已有数据（追加模式 / DPO-only 模式）----
    sft_all = []
    dpo_all = []
    existing_instructions = set()  # 用于去重

    if args.resume or args.dpo_only:
        if os.path.exists(sft_path):
            with open(sft_path, "r", encoding="utf-8") as f:
                sft_all = json.load(f)
            existing_instructions = {item["instruction"] for item in sft_all}
            print(f"已加载 {len(sft_all)} 条 SFT 数据")
        if os.path.exists(dpo_path):
            with open(dpo_path, "r", encoding="utf-8") as f:
                dpo_all = json.load(f)
            print(f"已加载 {len(dpo_all)} 条 DPO 数据")

    # ---- DPO-only 模式：为已有 SFT 数据补充 DPO 偏好对 ----
    if args.dpo_only:
        existing_dpo_prompts = {item["prompt"] for item in dpo_all}
        # 找出还没有 DPO 对的 SFT 数据
        sft_without_dpo = [
            item for item in sft_all
            if item["instruction"] not in existing_dpo_prompts
        ]
        print(f"\n有 {len(sft_without_dpo)} 条 SFT 数据尚无 DPO 偏好对")

        # 按 dpo-ratio 采样
        import random
        random.shuffle(sft_without_dpo)
        target_count = int(len(sft_without_dpo) * args.dpo_ratio)
        for item in sft_without_dpo[:target_count]:
            dpo_pair = generate_dpo_pair(item["instruction"], item["output"])
            if dpo_pair:
                dpo_pair["category"] = item.get("category", "unknown")
                dpo_all.append(dpo_pair)
                print(f"    ✓ DPO对: {item['instruction'][:50]}...")
            time.sleep(0.5)

    else:
        # ---- 正常模式 / 追加模式：遍历所有类别生成数据 ----
        for category, info in SEED_CATEGORIES.items():
            print(f"\n[{category}] {info['description']}")

            for seed in info["seed_prompts"]:

                # Step A: 生成 SFT 数据
                sft_data = generate_sft_variants(
                    category, info["description"], seed, num_variants=args.variants
                )

                # 去重：跳过已有的题目
                new_sft = [
                    item for item in sft_data
                    if item["instruction"] not in existing_instructions
                ]
                if len(new_sft) < len(sft_data):
                    print(f"    跳过 {len(sft_data) - len(new_sft)} 条重复数据")

                sft_all.extend(new_sft)
                for item in new_sft:
                    existing_instructions.add(item["instruction"])

                # Step B: 生成 DPO 数据（除非 --sft-only）
                if not args.sft_only:
                    dpo_count = max(1, int(len(new_sft) * args.dpo_ratio))
                    for item in new_sft[:dpo_count]:
                        dpo_pair = generate_dpo_pair(item["instruction"], item["output"])
                        if dpo_pair:
                            dpo_pair["category"] = category
                            dpo_all.append(dpo_pair)
                            print(f"    ✓ DPO对: {item['instruction'][:50]}...")
                        time.sleep(0.5)

    # ---- 保存数据到 JSON 文件 ----
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_all, f, ensure_ascii=False, indent=2)

    with open(dpo_path, "w", encoding="utf-8") as f:
        json.dump(dpo_all, f, ensure_ascii=False, indent=2)

    # ---- 打印统计信息 ----
    print("\n" + "=" * 60)
    print("数据集构建完成")
    print("=" * 60)
    print(f"SFT 数据: {len(sft_all)} 条")
    for cat in SEED_CATEGORIES:
        count = sum(1 for d in sft_all if d.get("category") == cat)
        if count > 0:
            print(f"  {cat}: {count} 条")
    print(f"\nDPO 数据: {len(dpo_all)} 对")
    print(f"\n保存到:")
    print(f"  {sft_path}")
    print(f"  {dpo_path}")

    # 提示预期数据量
    total_seeds = sum(len(info["seed_prompts"]) for info in SEED_CATEGORIES.values())
    print(f"\n种子题目总数: {total_seeds}")
    print(f"预期 SFT 上限: ~{total_seeds * args.variants} 条")


# Python 标准入口：只有直接运行此脚本时才执行 main()，
# 如果被其他文件 import 则不会自动执行
if __name__ == "__main__":
    main()
