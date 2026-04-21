"""
Step 2: 微调GPT-2 —— 让模型学会你想要的风格
=============================================
目标：通过微调理解三个核心概念：
  1. 训练数据怎么喂给Transformer？(数据处理pipeline)
  2. 模型在训练什么？(next token prediction, cross-entropy loss)
  3. 微调前后有什么区别？(pretrain vs finetune)

我们用 wikitext-2 数据集微调GPT-2，让它学会更连贯的百科风格写作
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
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# Part 1: 加载数据集 —— 看看训练数据长什么样
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 训练数据长什么样？")
print("=" * 60)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print(f"\n数据集结构: {dataset}")
print(f"训练集: {len(dataset['train'])} 条")
print(f"验证集: {len(dataset['validation'])} 条")

# 看几条数据
print("\n前5条训练数据:")
for i in range(5):
    text = dataset["train"][i]["text"]
    if text.strip():
        print(f"  [{i}] {text[:100]}...")

# ============================================================
# Part 2: Tokenize —— 把文本变成模型能吃的格式
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 数据预处理 (Tokenization + Chunking)")
print("=" * 60)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2没有pad token，用eos代替

# 关键步骤：把文本tokenize并切成固定长度的块
# 为什么要切块？因为Transformer有最大长度限制(GPT-2是1024)
# 我们用128的块长度，训练更快，你的8GB显存完全够用
BLOCK_SIZE = 128

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=False)

def group_texts(examples):
    """把tokenize后的数字拼接起来，然后切成等长的块

    这一步很关键：
    - 原始文本长短不一，有的一句话，有的一整段
    - 我们把它们全部拼起来，再按BLOCK_SIZE切块
    - 每个块就是一个训练样本
    - labels = input_ids（因为任务就是预测下一个token）
    """
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [concatenated[k][i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k in concatenated.keys()
    }
    result["labels"] = result["input_ids"].copy()  # 标签就是输入本身！
    return result

print("Tokenizing...")
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("Chunking into blocks...")
lm_dataset = tokenized.map(group_texts, batched=True)

print(f"\n处理后的训练样本数: {len(lm_dataset['train'])}")
print(f"每个样本长度: {BLOCK_SIZE} tokens")

# 看一个样本
sample = lm_dataset["train"][0]
print(f"\n一个训练样本 (前50个token):")
print(f"  input_ids: {sample['input_ids'][:50]}...")
print(f"  解码后: {tokenizer.decode(sample['input_ids'][:50])}...")
print(f"\n理解要点: labels和input_ids完全一样！")
print(f"  因为GPT的训练目标就是: 给定前面的token，预测下一个token")
print(f"  loss = CrossEntropy(模型预测的下一个token, 实际的下一个token)")

# ============================================================
# Part 3: 微调前 —— 先看看原始GPT-2生成什么
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 微调前的生成效果 (对照组)")
print("=" * 60)

model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

test_prompts = [
    "The history of artificial intelligence",
    "In the field of quantum computing",
    "The Roman Empire was",
]

print("\n微调前生成:")
model.eval()
for prompt in test_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids, # 输入的数字序列
            max_new_tokens=50, # 最多生成50个新词
            do_sample=True,       # 随机采样，不是贪心
            temperature=0.7,      # 控制随机程度，越小越确定
            top_k=50,             # 只从概率最高的50个token里选
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n  Prompt: {prompt}")
    print(f"  Output: {text[:200]}")

# ============================================================
# Part 4: 开始微调！
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 开始微调 (这是核心)")
print("=" * 60)

print("""
训练过程中发生了什么：
1. 取一个batch的训练样本 (若干个128-token的文本块)
2. 喂入模型，模型对每个位置预测下一个token
3. 用CrossEntropy计算预测和实际的差距 (loss)
4. 反向传播，更新所有参数 (包括Attention的Q/K/V权重)
5. 重复，直到loss下降到满意为止

这和Course 5里讲的训练过程完全一样，只是规模更大
""")

# 训练参数
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",    # 训练结果存在哪个文件夹
    num_train_epochs=1,               # 只跑1个epoch，演示用（实际项目会跑3-5）
    per_device_train_batch_size=4,     # 每批4个样本
    per_device_eval_batch_size=4,
    warmup_steps=100,                  # 学习率预热
    learning_rate=5e-5,               # 微调常用的小学习率
    logging_steps=50,                 # 每50步打印一次loss
    eval_strategy="steps",
    eval_steps=200,                   # 每200步评估一次
    save_strategy="no",               # 不保存checkpoint，节省空间
    fp16=torch.cuda.is_available(),   # 有GPU就用半精度，省显存
    report_to="none",                 # 不上报到wandb
)

# DataCollator: 自动处理batch内的padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # False = Causal LM (GPT风格), True = Masked LM (BERT风格)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    data_collator=data_collator,
)

print("开始训练... (RTX 4060大约需要3-5分钟)\n")
trainer.train()

# ============================================================
# Part 5: 微调后 —— 对比效果
# ============================================================
print("\n" + "=" * 60)
print("Part 5: 微调后的生成效果 (实验组)")
print("=" * 60)

model.eval()
print("\n微调后生成:")
for prompt in test_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n  Prompt: {prompt}")
    print(f"  Output: {text[:200]}")

# ============================================================
# Part 6: 看看训练过程中loss怎么变的
# ============================================================
print("\n" + "=" * 60)
print("Part 6: 训练过程分析")
print("=" * 60)

log_history = trainer.state.log_history
train_losses = [(log["step"], log["loss"]) for log in log_history if "loss" in log]
eval_losses = [(log["step"], log["eval_loss"]) for log in log_history if "eval_loss" in log]

if train_losses:
    print("\nTraining Loss 变化:")
    for step, loss in train_losses:
        bar = "█" * int((5.0 - loss) * 10) if loss < 5.0 else ""
        print(f"  Step {step:4d}: {loss:.4f} {bar}")

if eval_losses:
    print("\nValidation Loss 变化:")
    for step, loss in eval_losses:
        print(f"  Step {step:4d}: {loss:.4f}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结：你刚才做了什么")
print("=" * 60)
print(f"""
你完成了一次完整的LLM微调流程：

1. 数据处理: 文本 → tokenize → 切成{BLOCK_SIZE}长度的块 → labels=input_ids
   → Transformer吃的是固定长度的token序列

2. 训练目标: 给定前面的token，预测下一个token (Causal LM)
   → loss = CrossEntropy(预测分布, 真实token)
   → 这就是GPT系列的核心训练方式

3. 微调 vs 预训练:
   → 预训练: 在海量数据上从零学习语言能力 (需要几千张GPU)
   → 微调: 在预训练基础上，用小数据集调整模型风格 (你的笔记本就能做)
   → 两者的训练过程完全一样，区别只是数据量和起点

4. 关键超参数:
   → learning_rate=5e-5 (微调要用小学习率，别把预训练学到的知识冲掉)
   → batch_size=4, block_size={BLOCK_SIZE} (受限于你的8GB显存)
   → fp16=True (半精度训练，省一半显存)

下一步: 我们会学LoRA (只调一小部分参数的高效微调方法)
        和RLHF (用人类反馈来对齐模型，ChatGPT的核心技术)
""")
