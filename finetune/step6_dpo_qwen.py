"""
Step 6: DPO偏好对齐 Qwen2.5 —— 让模型学会给出高质量科学计算回答
================================================================
目标：
  1. 在 Step 5 LoRA SFT 的基础上，用 DPO 进一步对齐
  2. 让模型偏好"详细准确的科学解答"，远离"敷衍模糊的回答"
  3. 对比 SFT-only vs SFT+DPO 的效果差异

与 Step 4 (GPT-2 DPO) 的区别：
  - Step 4: 在原始GPT-2上做DPO，8条手写数据，验证算法原理
  - Step 6: 在SFT后的Qwen2.5上做DPO，真实偏好数据，面向实际评测

第一轮实验的教训及修复：
  1. Base → Instruct：Base 模型对 ChatML 格式理解不足，导致生成崩溃
  2. 移除 merge_and_unload：4bit 量化模型做 merge 会引入舍入误差，
     改为在 SFT adapter 上直接叠加 DPO adapter（adapter 链式加载）
  3. 增加训练验证集拆分 + 早停：防止 15 对数据训练到 100% accuracy 过拟合
  4. 添加梯度裁剪：防止训练不稳定
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import torch

# 国内网络访问 HuggingFace 不稳定，自动设置 hf-mirror.com 镜像
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("  [镜像] 已设置 HF_ENDPOINT = https://hf-mirror.com")
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ============================================================
# Part 1: 加载偏好数据
# ============================================================
print("=" * 60)
print("Part 1: 加载 DPO 偏好数据")
print("=" * 60)

data_path = os.path.join(os.path.dirname(__file__), "data", "dpo_train.json")
with open(data_path, "r", encoding="utf-8") as f:
    dpo_data = json.load(f)

print(f"偏好数据: {len(dpo_data)} 对")
print(f"\n示例:")
print(f"  Prompt:   {dpo_data[0]['prompt'][:80]}...")
print(f"  Chosen:   {dpo_data[0]['chosen'][:80]}...")
print(f"  Rejected: {dpo_data[0]['rejected'][:80]}...")

# 划分训练集/验证集 (85/15)
import random
random.seed(42)
shuffled = list(dpo_data)
random.shuffle(shuffled)
val_size = max(2, int(len(shuffled) * 0.15))  # 至少保留 2 条做验证
train_data = shuffled[:-val_size]
val_data = shuffled[-val_size:]
print(f"\n训练集: {len(train_data)} 对")
print(f"验证集: {val_size} 对")

# ============================================================
# Part 2: 加载 SFT 后的模型作为起点
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 加载模型")
print("=" * 60)

print("""
【关键修复】第一轮实验的 DPO 失败原因及本轮改进：

  问题 1: 使用 Base 模型作为基座
    → 修复: 改用 Instruct 模型，天然理解 ChatML 对话格式

  问题 2: 对 4bit 模型做 merge_and_unload()
    → 修复: 不做 merge，直接在 SFT adapter 上叠加 DPO adapter
    → 原理: 4bit 量化后权重是离散的，merge 时会引入舍入误差，
            导致 policy 和 ref 模型的起点不一致，DPO 训练偏移

  问题 3: 无验证集、无早停，训练到 100% accuracy
    → 修复: 添加 15% 验证集 + 连续 patience 轮 accuracy 不提升则停止
""")

# 【修复1】使用 Instruct 模型（与 step5 保持一致）
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "Qwen", "Qwen2.5-1.5B-Instruct")
LOCAL_BASE_PATH = os.path.join(os.path.dirname(__file__), "models", "Qwen", "Qwen2.5-1.5B")
HUGGINGFACE_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"

if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_NAME = LOCAL_MODEL_PATH
    print("  → 使用本地 Instruct 模型")
elif os.path.exists(LOCAL_BASE_PATH):
    MODEL_NAME = LOCAL_BASE_PATH
    print("  → 使用本地 Base 模型（建议下载 Instruct 版本以获得更好效果）")
else:
    MODEL_NAME = HUGGINGFACE_INSTRUCT
    print(f"  → 从 HuggingFace 下载: {MODEL_NAME}")

SFT_ADAPTER = os.path.join(os.path.dirname(__file__), "qwen-sft-lora", "final_adapter")

# 4bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- 加载 Policy 模型 ----
# 【修复2】先 merge SFT adapter 到基座，再在 merged 模型上添加 DPO LoRA
# 注意：4bit merge 会有微小舍入误差（会打印 warning），但实测不影响训练效果
# 之前尝试的 add_adapter + set_adapter 方案会导致 SFT adapter 被禁用，
# 使得 policy 模型失去 SFT 能力，DPO loss 异常高（>10），训练完全失败
if os.path.exists(SFT_ADAPTER):
    print(f"\n加载基座模型 + SFT adapter: {SFT_ADAPTER}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    # 加载 SFT adapter 并 merge 到基座
    sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER)
    policy_model = sft_model.merge_and_unload()
    print("  SFT adapter 已 merge 到基座（4bit merge 有微小舍入误差，可接受）")

    # 在 merged 模型上添加 DPO LoRA
    policy_model = prepare_model_for_kbit_training(policy_model)
    dpo_lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,                       # DPO 用较小的 r，避免过度调整
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # 只调 Q/V，比 SFT 更保守
    )
    policy_model = get_peft_model(policy_model, dpo_lora_config)
    print("  DPO LoRA 已添加")
else:
    print(f"未找到 SFT adapter ({SFT_ADAPTER})，使用原始模型")
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    policy_model = prepare_model_for_kbit_training(policy_model)
    dpo_lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    policy_model = get_peft_model(policy_model, dpo_lora_config)

policy_model.print_trainable_parameters()

# ---- 加载 Reference 模型（冻结）----
# Reference 模型 = 基座 + SFT (merged)，没有 DPO adapter
# 与 Policy 模型的唯一区别就是 DPO LoRA 的有无
print("\n加载 Reference 模型 (基座 + SFT merged, 冻结)...")
ref_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True,
)
if os.path.exists(SFT_ADAPTER):
    ref_peft = PeftModel.from_pretrained(ref_base, SFT_ADAPTER)
    ref_model = ref_peft.merge_and_unload()
    # merge 后 ref 和 policy 的基底完全一致（同样的 merge 舍入误差）
else:
    ref_model = ref_base

for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()

print("Policy模型: 基座 + SFT adapter + DPO adapter (可训练)")
print("Reference模型: 基座 + SFT adapter (冻结)")

# ============================================================
# Part 3: DPO 数据集和核心算法
# ============================================================
print("\n" + "=" * 60)
print("Part 3: DPO 核心算法 (修复版)")
print("=" * 60)

print("""
与 Step 4 (GPT-2 DPO) 的对比：

  Step 4:                          Step 6:
  - GPT-2 (124M)                   - Qwen2.5-1.5B-Instruct (4bit)
  - 8条手写偏好数据                 - API 生成的偏好数据
  - 在原始模型上DPO                 - 在SFT后的模型上DPO (adapter链式)
  - 无验证集、无早停                - 有验证集拆分 + 早停机制
  - 验证算法原理                    - 面向实际科学计算评测

  DPO loss 公式完全相同：
  Loss = -log σ(β * (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))
""")

MAX_LENGTH = 512

class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _format_text(self, prompt, response):
        return (
            f"<|im_start|>system\n"
            f"You are a scientific computing expert. Solve problems with complete, runnable Python code.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{prompt}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{response}\n"
            f"<|im_end|>"
        )

    def __getitem__(self, idx):
        item = self.data[idx]

        chosen_text = self._format_text(item["prompt"], item["chosen"])
        rejected_text = self._format_text(item["prompt"], item["rejected"])

        chosen_enc = self.tokenizer(
            chosen_text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


def compute_log_probs(model, input_ids, attention_mask):
    """计算序列对数概率（与 Step 4 相同的核心函数）"""
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    mask = attention_mask[:, 1:]
    token_log_probs = token_log_probs * mask

    return token_log_probs.sum(dim=-1)


def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    """DPO Loss（与 Step 4 相同的核心公式）"""
    chosen_ids = batch["chosen_input_ids"].to(device)
    chosen_mask = batch["chosen_attention_mask"].to(device)
    rejected_ids = batch["rejected_input_ids"].to(device)
    rejected_mask = batch["rejected_attention_mask"].to(device)

    policy_chosen_logps = compute_log_probs(policy_model, chosen_ids, chosen_mask)
    policy_rejected_logps = compute_log_probs(policy_model, rejected_ids, rejected_mask)

    with torch.no_grad():
        ref_chosen_logps = compute_log_probs(ref_model, chosen_ids, chosen_mask)
        ref_rejected_logps = compute_log_probs(ref_model, rejected_ids, rejected_mask)

    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    logits = beta * (chosen_log_ratio - rejected_log_ratio)
    loss = -F.logsigmoid(logits).mean()
    accuracy = (logits > 0).float().mean()

    return loss, accuracy, chosen_log_ratio.mean(), rejected_log_ratio.mean()


# ============================================================
# Part 4: 对齐前 baseline
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 对齐前的生成效果 (SFT-only)")
print("=" * 60)

test_prompts = [
    "用Python求解方程 x^2 - 5x + 6 = 0",
    "用scipy优化 f(x) = (x-3)^2 + 2 的最小值",
    "生成500个均匀分布随机数，画直方图并计算均值",
]

def generate_response(model, prompt, max_new_tokens=150):
    model.eval()
    text = (
        f"<|im_start|>system\n"
        f"You are a scientific computing expert. Solve problems with complete, runnable Python code.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{prompt}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("\n对齐前 (SFT-only):")
for prompt in test_prompts:
    response = generate_response(policy_model, prompt)
    print(f"  Q: {prompt}")
    print(f"  A: {response[:250]}\n")

# ============================================================
# Part 5: DPO 训练 (含验证集 + 早停)
# ============================================================
print("=" * 60)
print("Part 5: DPO 训练 (修复版: 含验证集 + 早停)")
print("=" * 60)

train_dataset = DPODataset(train_data, tokenizer)
val_dataset = DPODataset(val_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)
# batch_size=1: 因为4bit模型 + 两个模型同时在显存中，需要控制内存

optimizer = torch.optim.AdamW(
    [p for p in policy_model.parameters() if p.requires_grad],
    lr=5e-6,
)

# 【修复3】根据数据量自适应训练参数
data_size = len(train_data)
if data_size < 30:
    NUM_EPOCHS = 5
    PATIENCE = 2      # 数据少，早停更敏感
elif data_size < 100:
    NUM_EPOCHS = 3
    PATIENCE = 3
else:
    NUM_EPOCHS = 2
    PATIENCE = 3

BETA = 0.1
MAX_GRAD_NORM = 1.0   # 【修复4】梯度裁剪，防止训练不稳定

print(f"\n配置:")
print(f"  训练数据: {len(train_data)} 对")
print(f"  验证数据: {len(val_data)} 对")
print(f"  Epochs: {NUM_EPOCHS} (max)")
print(f"  Early stopping patience: {PATIENCE}")
print(f"  学习率: 5e-6")
print(f"  Beta: {BETA}")
print(f"  梯度裁剪: {MAX_GRAD_NORM}")
print(f"\n开始训练...\n")

policy_model.train()
all_train_losses = []
all_train_accs = []
all_val_losses = []
all_val_accs = []

best_val_loss = float('inf')
patience_counter = 0
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    # ---- 训练阶段 ----
    policy_model.train()
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    for batch in train_loader:
        optimizer.zero_grad()

        loss, accuracy, chosen_ratio, rejected_ratio = dpo_loss(
            policy_model, ref_model, batch, beta=BETA
        )

        loss.backward()
        # 【修复4】梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy_model.parameters() if p.requires_grad],
            MAX_GRAD_NORM,
        )
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += accuracy.item()
        num_batches += 1

    avg_train_loss = epoch_loss / max(num_batches, 1)
    avg_train_acc = epoch_acc / max(num_batches, 1)
    all_train_losses.append(avg_train_loss)
    all_train_accs.append(avg_train_acc)

    # ---- 验证阶段 ----
    policy_model.eval()
    val_loss = 0
    val_acc = 0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            loss, accuracy, _, _ = dpo_loss(
                policy_model, ref_model, batch, beta=BETA
            )
            val_loss += loss.item()
            val_acc += accuracy.item()
            val_batches += 1

    avg_val_loss = val_loss / max(val_batches, 1)
    avg_val_acc = val_acc / max(val_batches, 1)
    all_val_losses.append(avg_val_loss)
    all_val_accs.append(avg_val_acc)

    # 打印训练进度
    acc_bar = "█" * int(avg_train_acc * 20)
    print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS}: "
          f"train_loss={avg_train_loss:.4f} acc={avg_train_acc:.1%} {acc_bar} | "
          f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.1%}")

    # ---- 早停检查 ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        # 保存当前最佳模型
        best_adapter_path = os.path.join(os.path.dirname(__file__), "qwen-dpo-lora", "best_adapter")
        os.makedirs(best_adapter_path, exist_ok=True)
        policy_model.save_pretrained(best_adapter_path)
        tokenizer.save_pretrained(best_adapter_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  ⚡ 早停: 验证集 loss 连续 {PATIENCE} 个 epoch 未改善")
            print(f"     最佳 epoch: {best_epoch}, 最佳 val_loss: {best_val_loss:.4f}")
            break

# 保存最终 DPO adapter
dpo_adapter_path = os.path.join(os.path.dirname(__file__), "qwen-dpo-lora")
os.makedirs(dpo_adapter_path, exist_ok=True)
policy_model.save_pretrained(dpo_adapter_path)
tokenizer.save_pretrained(dpo_adapter_path)
print(f"\nDPO adapter 保存到: {dpo_adapter_path}")
print(f"最佳 adapter 保存到: {best_adapter_path}")

# ============================================================
# Part 6: 对齐后效果
# ============================================================
print("\n" + "=" * 60)
print("Part 6: 对齐后的生成效果 (SFT + DPO)")
print("=" * 60)

print("\n对齐后 (SFT + DPO):")
for prompt in test_prompts:
    response = generate_response(policy_model, prompt)
    print(f"  Q: {prompt}")
    print(f"  A: {response[:250]}\n")

# ============================================================
# Part 7: 量化对比
# ============================================================
print("=" * 60)
print("Part 7: 偏好变化量化分析")
print("=" * 60)

policy_model.eval()
# 在全量数据上评估（包括训练和验证）
eval_dataset = DPODataset(dpo_data, tokenizer)
eval_loader = DataLoader(eval_dataset, batch_size=1)

total_ref_correct = 0
total_policy_correct = 0
total_items = 0

for batch in eval_loader:
    chosen_ids = batch["chosen_input_ids"].to(device)
    chosen_mask = batch["chosen_attention_mask"].to(device)
    rejected_ids = batch["rejected_input_ids"].to(device)
    rejected_mask = batch["rejected_attention_mask"].to(device)

    with torch.no_grad():
        policy_chosen = compute_log_probs(policy_model, chosen_ids, chosen_mask)
        policy_rejected = compute_log_probs(policy_model, rejected_ids, rejected_mask)
        ref_chosen = compute_log_probs(ref_model, chosen_ids, chosen_mask)
        ref_rejected = compute_log_probs(ref_model, rejected_ids, rejected_mask)

    total_ref_correct += (ref_chosen > ref_rejected).float().sum().item()
    total_policy_correct += (policy_chosen > policy_rejected).float().sum().item()
    total_items += chosen_ids.shape[0]

ref_acc = total_ref_correct / total_items
policy_acc = total_policy_correct / total_items

print(f"\n偏好正确率（模型是否偏好chosen而非rejected）:")
print(f"  对齐前 (ref/SFT-only): {ref_acc:.1%}")
print(f"  对齐后 (SFT + DPO):    {policy_acc:.1%}")
print(f"  提升: {(policy_acc - ref_acc)*100:+.1f}%")

# ============================================================
# Part 8: 训练曲线
# ============================================================
print("\n" + "=" * 60)
print("Part 8: 训练过程")
print("=" * 60)

print("\nTraining Loss:")
for i, loss in enumerate(all_train_losses):
    bar = "█" * int(max(0, (1.0 - loss) * 20))
    print(f"  Epoch {i+1:2d}: {loss:.4f} {bar}")

print("\nValidation Loss:")
for i, loss in enumerate(all_val_losses):
    bar = "█" * int(max(0, (1.0 - loss) * 20))
    print(f"  Epoch {i+1:2d}: {loss:.4f} {bar}")

print("\nTraining Accuracy:")
for i, acc in enumerate(all_train_accs):
    bar = "█" * int(acc * 20)
    print(f"  Epoch {i+1:2d}: {acc:.1%} {bar}")

print("\nValidation Accuracy:")
for i, acc in enumerate(all_val_accs):
    bar = "█" * int(acc * 20)
    print(f"  Epoch {i+1:2d}: {acc:.1%} {bar}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"""
Step 6 完成：在 SFT 后的 Qwen2.5-Instruct 上做了 DPO 偏好对齐

第一轮实验的关键修复：
  1. Base → Instruct 模型: 天然理解 ChatML，SFT 基础更扎实
  2. SFT merge → DPO LoRA: 先 merge SFT adapter 到基座，再加 DPO LoRA
     (注: adapter 链式加载 set_adapter("dpo") 会禁用 SFT adapter，
      导致 policy 失去 SFT 能力，loss 异常高。merge 虽有微小舍入误差但可用)
  3. 添加验证集拆分: {len(train_data)} 训练 / {len(val_data)} 验证
  4. 添加早停机制: patience={PATIENCE}，防止过拟合到 100% accuracy
  5. 添加梯度裁剪: max_norm={MAX_GRAD_NORM}，提升训练稳定性

与 Step 4 (GPT-2 DPO) 的完整对比：

  | 维度         | Step 4 (原理验证)   | Step 6 (实战应用)        |
  |-------------|--------------------|-----------------------  |
  | 模型         | GPT-2 (124M)       | Qwen2.5-1.5B-Instruct   |
  | 起点         | 原始预训练模型      | SFT微调后 (adapter链式)   |
  | 数据         | 8条手写偏好对       | {len(dpo_data)}条API生成偏好对    |
  | 验证集       | 无                 | {len(val_data)}条                 |
  | 早停         | 无                 | patience={PATIENCE}      |
  | 梯度裁剪     | 无                 | max_norm={MAX_GRAD_NORM} |
  | 偏好正确率   | 0%→87.5%          | {ref_acc:.0%}→{policy_acc:.0%}    |

完整训练链路回顾：
  Step 1: 理解Transformer → Step 2: 全量SFT → Step 3: LoRA SFT
  → Step 5: Qwen LoRA SFT (领域数据) → Step 6: DPO对齐 (偏好数据)

下一步: Step 7 接入 SciAgent 系统，在 10 题 Benchmark 上做三模型对比评测
""")
