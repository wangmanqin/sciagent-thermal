"""
Step 5: LoRA微调 Qwen2.5-1.5B —— 让小模型学会科学计算
=====================================================
目标：
  1. 用 4bit 量化 + LoRA 在 8GB 显存上微调 Qwen2.5-1.5B
  2. 用 SciTune 数据集（科学计算QA）做领域自适应 SFT
  3. 对比微调前后在科学计算任务上的表现

与前几步的关系：
  - Step 2 (全量微调GPT-2) → 证明微调有效
  - Step 3 (LoRA微调GPT-2) → 证明LoRA高效
  - Step 5 (本步) → 在真实模型+真实数据上做，连接SciAgent评测
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import torch

# 国内网络访问 HuggingFace 不稳定，自动设置 hf-mirror.com 镜像
# 必须在 import transformers 之前设置环境变量
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("  [镜像] 已设置 HF_ENDPOINT = https://hf-mirror.com")
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# Part 1: 加载数据
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 加载 SciTune SFT 数据集")
print("=" * 60)

data_path = os.path.join(os.path.dirname(__file__), "data", "sft_train.json")
with open(data_path, "r", encoding="utf-8") as f:
    sft_data = json.load(f)

print(f"总数据量: {len(sft_data)} 条")
categories = {}
for item in sft_data:
    cat = item.get("category", "unknown")
    categories[cat] = categories.get(cat, 0) + 1
for cat, count in categories.items():
    print(f"  {cat}: {count} 条")

# 看一条数据
print(f"\n示例:")
print(f"  问题: {sft_data[0]['instruction'][:80]}...")
print(f"  回答: {sft_data[0]['output'][:80]}...")

# ============================================================
# Part 2: 加载 Qwen2.5-1.5B (4bit量化)
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 加载 Qwen2.5-1.5B (4bit量化)")
print("=" * 60)

# 优先使用本地已下载的模型，没有则从 HuggingFace 在线下载
# 【重要】优先使用 Instruct 版本：已经过指令对齐，理解对话格式，微调起点更高
# 这是第一轮实验的关键教训：Base 模型对 ChatML 格式理解不足，导致 DPO 阶段崩溃
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "Qwen", "Qwen2.5-1.5B-Instruct")
LOCAL_BASE_PATH = os.path.join(os.path.dirname(__file__), "models", "Qwen", "Qwen2.5-1.5B")
HUGGINGFACE_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"
HUGGINGFACE_BASE = "Qwen/Qwen2.5-1.5B"

if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_NAME = LOCAL_MODEL_PATH
    print("  → 使用本地 Instruct 模型")
elif os.path.exists(LOCAL_BASE_PATH):
    MODEL_NAME = LOCAL_BASE_PATH
    print("  → 使用本地 Base 模型（建议下载 Instruct 版本以获得更好效果）")
else:
    # 在线下载：优先 Instruct 版本
    MODEL_NAME = HUGGINGFACE_INSTRUCT
    print(f"  → 从 HuggingFace 下载: {MODEL_NAME}")

print(f"\n加载模型: {MODEL_NAME}")
print("使用 4bit 量化 (NF4) 以适配 8GB 显存...")

# 4bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4bit 量化
    bnb_4bit_quant_type="nf4",            # NF4 量化（比FP4更好）
    bnb_4bit_compute_dtype=torch.float16, # 计算时用fp16
    bnb_4bit_use_double_quant=True,       # 双重量化，进一步省显存
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 为 kbit 训练准备模型
model = prepare_model_for_kbit_training(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,} ({total_params/1e9:.2f}B)")
print(f"量化后显存占用: ~{torch.cuda.memory_allocated()/1024**3:.1f} GB")

# ============================================================
# Part 3: 配置 LoRA
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 配置 LoRA")
print("=" * 60)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                  # 秩=16，比GPT-2的r=8更大
    lora_alpha=32,                         # alpha = 2r
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # 覆盖完整的注意力机制：
    # Q决定"问什么"，K决定"被什么匹配"，V决定"传递什么信息"，O决定"输出什么"
    # 四个都微调，比只调Q/V效果更好
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"\n冻结参数: {frozen:,}")
print(f"可训练参数 (LoRA): {trainable:,}")
print(f"占比: {trainable/(frozen+trainable)*100:.3f}%")

# ============================================================
# Part 4: 数据预处理 —— 构造训练格式
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 数据预处理")
print("=" * 60)

MAX_LENGTH = 1024  # Qwen2.5 支持更长的上下文

# 用于定位 assistant 回答起始位置的标记
ASSISTANT_MARKER = "<|im_start|>assistant\n"

def format_example(instruction, output):
    """把问题和回答拼成 Qwen2.5 的对话格式"""
    # Qwen2.5 ChatML 格式
    text = (
        f"<|im_start|>system\n"
        f"You are a scientific computing expert. Solve problems with complete, runnable Python code.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{instruction}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{output}\n"
        f"<|im_end|>"
    )
    return text

def tokenize_sft(examples):
    """
    Tokenize SFT 数据，并做 label masking：
    只在 assistant 回答部分计算 loss，system/user 部分设为 -100（忽略）。
    这样模型只学习"如何回答"，而不是学习复读 system prompt 和用户问题。
    """
    texts = [
        format_example(inst, out)
        for inst, out in zip(examples["instruction"], examples["output"])
    ]
    tokenized = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    )

    # 对 assistant 标记做 tokenize，用于定位 assistant 回答的起始位置
    assistant_tokens = tokenizer(ASSISTANT_MARKER, add_special_tokens=False)["input_ids"]
    assistant_len = len(assistant_tokens)

    all_labels = []
    for input_ids in tokenized["input_ids"]:
        labels = [-100] * len(input_ids)  # 默认全部忽略

        # 从后往前找 assistant 标记的位置（取最后一次出现）
        start_idx = -1
        for i in range(len(input_ids) - assistant_len, -1, -1):
            if input_ids[i:i + assistant_len] == assistant_tokens:
                start_idx = i + assistant_len  # 跳过标记本身，从回答内容开始
                break

        if start_idx >= 0:
            # 从 assistant 回答开始到序列结束，都计算 loss
            labels[start_idx:] = input_ids[start_idx:]

        all_labels.append(labels)

    tokenized["labels"] = all_labels
    return tokenized

# 转换为 HuggingFace Dataset
dataset = Dataset.from_list(sft_data)

# 划分训练集/验证集 (90/10)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"训练集: {len(train_dataset)} 条")
print(f"验证集: {len(eval_dataset)} 条")

# Tokenize
train_dataset = train_dataset.map(
    tokenize_sft, batched=True,
    remove_columns=["instruction", "output", "category"]
)
eval_dataset = eval_dataset.map(
    tokenize_sft, batched=True,
    remove_columns=["instruction", "output", "category"]
)

# 看一个样本的长度
sample_len = len(train_dataset[0]["input_ids"])
print(f"样本token长度示例: {sample_len}")

# ============================================================
# Part 5: 微调前 baseline
# ============================================================
print("\n" + "=" * 60)
print("Part 5: 微调前的生成效果")
print("=" * 60)

test_prompts = [
    "用Python求解方程 x^3 - 6x^2 + 11x - 6 = 0 的所有实数根",
    "用NSGA-II算法求解双目标优化问题：最小化f1=x^2, f2=(x-5)^2",
    "用FFT分析包含30Hz和80Hz成分的信号的频谱",
]

def generate_response(prompt, max_new_tokens=200):
    """用微调前/后的模型生成回答"""
    model.eval()
    messages = (
        f"<|im_start|>system\n"
        f"You are a scientific computing expert. Solve problems with complete, runnable Python code.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{prompt}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(messages, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

print("\n微调前:")
for prompt in test_prompts:
    response = generate_response(prompt, max_new_tokens=150)
    print(f"\n  Q: {prompt}")
    print(f"  A: {response[:300]}")

# ============================================================
# Part 6: LoRA SFT 训练
# ============================================================
print("\n" + "=" * 60)
print("Part 6: 开始 LoRA SFT 训练")
print("=" * 60)

print("""
与 Step 3 (GPT-2 LoRA) 的区别：
  - 模型: GPT-2 (124M) → Qwen2.5-1.5B (1.5B)，大了12倍
  - 数据: WikiText通用文本 → 科学计算专用QA
  - 量化: 无 → 4bit NF4，显存占用从~6GB降到~2GB
  - 目标: 学通用语言风格 → 学科学计算代码生成能力
""")

output_dir = os.path.join(os.path.dirname(__file__), "qwen-sft-lora")

# 根据数据集大小自适应调整训练参数
# 数据少（<50条）→ 更多 epoch、更小学习率；数据多（>200条）→ 更少 epoch
data_size = len(train_dataset)
if data_size < 50:
    num_epochs = 10
    lr = 1e-4
    print(f"  数据较少({data_size}条)，使用更多 epoch={num_epochs}")
elif data_size < 200:
    num_epochs = 5
    lr = 2e-4
    print(f"  数据中等({data_size}条)，epoch={num_epochs}")
else:
    num_epochs = 3
    lr = 2e-4
    print(f"  数据充足({data_size}条)，epoch={num_epochs}")

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=2,         # 4bit量化后batch可以适当开大
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,         # 等效batch_size = 2*2 = 4，加快更新
    warmup_ratio=0.05,
    learning_rate=lr,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=2,
    fp16=True,
    report_to="none",
    optim="paged_adamw_8bit",             # 8bit优化器，进一步省显存
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,          # 训练结束时加载最佳模型
    metric_for_best_model="eval_loss",    # 以验证集 loss 为标准
    greater_is_better=False,              # loss 越小越好
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    # 连续 3 次 eval_loss 不下降则提前终止，防止过拟合
)

print("开始训练...\n")
train_result = trainer.train()

# 保存 LoRA adapter
adapter_path = os.path.join(output_dir, "final_adapter")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"\nLoRA adapter 保存到: {adapter_path}")

# ============================================================
# Part 7: 微调后效果
# ============================================================
print("\n" + "=" * 60)
print("Part 7: 微调后的生成效果")
print("=" * 60)

model.eval()
print("\n微调后:")
for prompt in test_prompts:
    response = generate_response(prompt, max_new_tokens=150)
    print(f"\n  Q: {prompt}")
    print(f"  A: {response[:300]}")

# ============================================================
# Part 8: 训练过程分析
# ============================================================
print("\n" + "=" * 60)
print("Part 8: 训练过程")
print("=" * 60)

log_history = trainer.state.log_history
train_losses = [(log["step"], log["loss"]) for log in log_history if "loss" in log]
eval_losses = [(log["step"], log["eval_loss"]) for log in log_history if "eval_loss" in log]

if train_losses:
    print("\nTraining Loss:")
    for step, loss in train_losses:
        bar = "█" * int(max(0, (5.0 - loss) * 8))
        print(f"  Step {step:4d}: {loss:.4f} {bar}")

if eval_losses:
    print("\nValidation Loss:")
    for step, loss in eval_losses:
        print(f"  Step {step:4d}: {loss:.4f}")

train_time = train_result.metrics.get("train_runtime", 0)
print(f"\n训练耗时: {train_time:.0f}秒 ({train_time/60:.1f}分钟)")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"""
Step 5 完成：在 Qwen2.5-1.5B 上做了科学计算领域的 LoRA SFT

关键技术选择：
  1. 4bit NF4 量化: 1.5B模型只占~2GB显存，8GB卡完全够用
  2. LoRA r=16: 比Step 3的r=8更大，因为1.5B模型的表达空间更大
  3. 8bit AdamW优化器: 进一步压缩显存，和4bit模型配合使用
  4. gradient_accumulation=4: 小batch也能模拟大batch的训练效果
  5. EarlyStoppingCallback: 连续3次eval_loss不下降则提前终止，防止过拟合
  6. Instruct 模型优先: 使用 Qwen2.5-1.5B-Instruct 作为基座，
     天然理解 ChatML 格式，SFT 起点更高，避免 DPO 阶段崩溃
  7. 数据量自适应: 根据数据集大小动态调整 epoch 和学习率

与 Step 3 (GPT-2 LoRA) 的对比：
  | 指标           | Step 3 (GPT-2)  | Step 5 (Qwen2.5)   |
  |---------------|-----------------|---------------------|
  | 模型大小       | 124M            | 1.5B                |
  | 量化           | 无              | 4bit NF4            |
  | LoRA秩         | 8               | 16                  |
  | 数据           | WikiText(通用)   | SciTune(科学计算)    |
  | 可训练参数占比  | 0.24%           | ~0.5%               |
  | 基座模型       | GPT-2 (base)    | Qwen2.5 (Instruct)  |
  | 早停机制       | 无              | patience=3           |

下一步: Step 6 用 DPO 进一步对齐，然后 Step 7 接入 SciAgent 评测
""")
