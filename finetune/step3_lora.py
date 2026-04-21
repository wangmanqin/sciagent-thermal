"""
Step 3: LoRA —— 只改0.2%的参数就能微调大模型
===============================================
目标：理解三个核心问题：
  1. 为什么不需要更新全部参数？(低秩分解的直觉)
  2. LoRA到底在哪里插入了什么？(看得见摸得着)
  3. 和全量微调比，效果差多少？(实验对比)

这是当前工业界最主流的微调方法，面试必问。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ============================================================
# Part 1: 先理解LoRA的直觉 —— 用代码演示低秩分解
# ============================================================
print("=" * 60)
print("Part 1: LoRA的核心思想 —— 低秩分解")
print("=" * 60)

print("""
问题：GPT-2有124M参数，全部微调需要大量显存和算力。
      但微调时真的需要动每一个参数吗？

答案：不需要！研究发现，微调时参数的变化量 ΔW 是"低秩"的。

什么是低秩？看这个例子：
""")

# 用代码演示低秩分解
import torch

# 假设某一层的权重矩阵是 768x768 = 589,824个参数
d = 768
W = torch.randn(d, d)  # 原始权重

# 全量微调: 需要更新全部 768x768 = 589,824 个参数
delta_W_full = torch.randn(d, d)  # 参数变化量
print(f"全量微调 ΔW: {d}x{d} = {d*d:,} 个参数需要更新")

# LoRA: 用两个小矩阵 B(768,r) × A(r,768) 来近似 ΔW
r = 8  # 秩，通常取4/8/16/32
B = torch.randn(d, r)   # 768 x 8
A = torch.randn(r, d)   # 8 x 768
delta_W_lora = B @ A     # 768 x 768，但只用了 768*8*2 个参数！

lora_params = d * r * 2
print(f"LoRA ΔW = B×A: ({d}x{r}) × ({r}x{d}) = {lora_params:,} 个参数")
print(f"参数量减少到: {lora_params/d/d*100:.1f}% !")
print(f"\n直觉: 就像用两个'瘦长'矩阵的乘积，来近似一个'胖方'矩阵的变化")
print(f"      原始权重W不动(frozen)，只训练B和A")
print(f"      推理时: W' = W + B×A")

# ============================================================
# Part 2: 在GPT-2上实际配置LoRA
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 给GPT-2装上LoRA")
print("=" * 60)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# 先看原始模型的参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n原始GPT-2参数量: {total_params:,} ({total_params/1e6:.0f}M)")

# 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,    # 因果语言模型（GPT类）
    r=8,                              # 秩，越大表达能力越强，但参数越多
    lora_alpha=32,                    # 缩放系数，通常设为r的2-4倍
    lora_dropout=0.1,                 # dropout防过拟合
    target_modules=["c_attn"],        # 只在Attention的QKV投影层加LoRA
    # 为什么选c_attn？因为Attention是Transformer的核心
    # c_attn 就是把输入同时投影成 Q, K, V 的那个线性层
)

# 把LoRA应用到模型上
model = get_peft_model(model, lora_config)

# 看看LoRA改了什么
print("\n装上LoRA后的模型结构 (只看第一个Block):")
print(model.base_model.model.transformer.h[0])

# 关键对比：可训练参数量
model.print_trainable_parameters()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"\n详细对比:")
print(f"  冻结参数 (原始GPT-2): {frozen:,}")
print(f"  可训练参数 (LoRA):    {trainable:,}")
print(f"  比例: {trainable/frozen*100:.2f}%")
print(f"\n含义: 只需要训练{trainable/frozen*100:.2f}%的参数，显存占用大幅减少")

# ============================================================
# Part 3: 准备数据 (和Step 2一样)
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 准备数据")
print("=" * 60)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
BLOCK_SIZE = 128

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=False)

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [concatenated[k][i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k in concatenated.keys()
    }
    result["labels"] = result["input_ids"].copy()
    return result

print("处理数据中...")
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
lm_dataset = tokenized.map(group_texts, batched=True)
print(f"训练样本数: {len(lm_dataset['train'])}")

# ============================================================
# Part 4: 微调前的baseline
# ============================================================
print("\n" + "=" * 60)
print("Part 4: LoRA微调前的生成效果")
print("=" * 60)

test_prompts = [
    "The history of artificial intelligence",
    "In the field of quantum computing",
    "The Roman Empire was",
]

model.eval()
print("\n微调前:")
for prompt in test_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=50, do_sample=True,
            temperature=0.7, top_k=50, pad_token_id=tokenizer.eos_token_id,
        )
    print(f"  [{prompt}]")
    print(f"  → {tokenizer.decode(output[0], skip_special_tokens=True)[:200]}\n")

# ============================================================
# Part 5: LoRA微调
# ============================================================
print("=" * 60)
print("Part 5: 开始LoRA微调")
print("=" * 60)

print("""
和Step 2的全量微调对比：
  - Step 2: 更新全部124M参数，用了~8.5分钟
  - Step 3: 只更新LoRA的参数，训练更快，显存更省
  - 训练过程完全一样：前向传播→算loss→反向传播→更新参数
  - 唯一的区别：反向传播时，梯度只更新B和A，不动原始W
""")

training_args = TrainingArguments(
    output_dir="./gpt2-lora",
    num_train_epochs=1,
    per_device_train_batch_size=8,     # LoRA省显存，batch可以开更大
    per_device_eval_batch_size=8,
    warmup_steps=100,
    learning_rate=3e-4,                # LoRA通常用更大的学习率
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    data_collator=data_collator,
)

print("开始训练...\n")
train_result = trainer.train()

# ============================================================
# Part 6: 微调后效果
# ============================================================
print("\n" + "=" * 60)
print("Part 6: LoRA微调后的生成效果")
print("=" * 60)

model.eval()
print("\n微调后:")
for prompt in test_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=50, do_sample=True,
            temperature=0.7, top_k=50, pad_token_id=tokenizer.eos_token_id,
        )
    print(f"  [{prompt}]")
    print(f"  → {tokenizer.decode(output[0], skip_special_tokens=True)[:200]}\n")

# ============================================================
# Part 7: 看LoRA学到了什么
# ============================================================
print("=" * 60)
print("Part 7: 看看LoRA的参数长什么样")
print("=" * 60)

# 提取第一层attention的LoRA参数
for name, param in model.named_parameters():
    if "lora_A" in name and "h.0" in name:
        print(f"\n{name}")
        print(f"  形状: {param.shape}  (r=8, 所以是 8×2304)")
        print(f"  参数范围: [{param.min():.4f}, {param.max():.4f}]")
    if "lora_B" in name and "h.0" in name:
        print(f"\n{name}")
        print(f"  形状: {param.shape}  (2304×8)")
        print(f"  参数范围: [{param.min():.4f}, {param.max():.4f}]")

# ============================================================
# Part 8: 训练过程分析
# ============================================================
print("\n" + "=" * 60)
print("Part 8: 训练过程")
print("=" * 60)

log_history = trainer.state.log_history
train_losses = [(log["step"], log["loss"]) for log in log_history if "loss" in log]

if train_losses:
    print("\nTraining Loss:")
    for step, loss in train_losses:
        bar = "█" * int((5.0 - loss) * 10) if loss < 5.0 else ""
        print(f"  Step {step:4d}: {loss:.4f} {bar}")

train_time = train_result.metrics.get("train_runtime", 0)
print(f"\n训练耗时: {train_time:.0f}秒 ({train_time/60:.1f}分钟)")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"""
LoRA的核心原理你现在应该理解了：

1. 核心公式: W' = W + B×A
   → W是原始权重(冻结)，B和A是两个小矩阵(可训练)
   → B: (d, r), A: (r, d)，r远小于d

2. 为什么有效？
   → 微调的本质是让模型适应特定任务/风格
   → 这种"适应"通常只需要在高维空间中做低维调整
   → LoRA正是利用了这一点：用低秩矩阵捕捉这种调整

3. 实际效果：
   → 可训练参数: {trainable:,} (原来的{trainable/frozen*100:.2f}%)
   → 训练时间: {train_time:.0f}秒
   → 生成质量: 和全量微调接近

4. 面试常问：
   → "LoRA的r怎么选？" → 任务越复杂r越大，通常8-64
   → "为什么target_modules选attention？" → Attention是学习token间关系的核心
   → "LoRA和全量微调哪个好？" → 小数据LoRA更好(不易过拟合)，大数据全量更好

下一步: RLHF (用人类反馈训练模型，ChatGPT的核心)
""")
