"""
Step 4: DPO —— 用人类偏好对齐语言模型
======================================
背景：
  经过Step 1-3，你已经会训练一个语言模型了。
  但训练出来的模型只会"续写最可能的下一个token"，它不知道：
    - 哪种回答是人类更喜欢的
    - 哪些内容是有害的、应该拒绝的
    - 怎么做到"有帮助"而非"说废话"

  ChatGPT之所以好用，关键就在于"对齐(Alignment)"这一步。

  对齐的两种主流方法：
    - RLHF: 先训练一个奖励模型，再用PPO强化学习优化 (OpenAI用的)
    - DPO:  直接用偏好数据优化，不需要单独训练奖励模型 (更简单更稳定)

  本脚本实现DPO，因为它更实用、更容易理解，也是当前学术界的主流。

目标：
  1. 理解"偏好数据"长什么样 (chosen vs rejected)
  2. 理解DPO的loss在做什么 (让模型更倾向chosen，远离rejected)
  3. 亲手跑一遍DPO训练，看到对齐前后的变化
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ============================================================
# Part 1: 什么是偏好数据？
# ============================================================
print("=" * 60)
print("Part 1: 偏好数据 —— 对齐的原料")
print("=" * 60)

print("""
RLHF/DPO的核心输入是"偏好数据"：

  给定一个prompt，人类标注者看两个回答，选出更好的那个：
    - chosen:   人类更喜欢的回答 (更有帮助/更准确/更安全)
    - rejected: 人类不喜欢的回答 (废话/错误/有害)

  模型要学的就是：更像chosen，远离rejected
""")

# 构造偏好数据集
# 实际项目中这些数据来自人类标注，这里手动构造演示
preference_data = [
    {
        "prompt": "What is machine learning?",
        "chosen": "What is machine learning? Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
        "rejected": "What is machine learning? Machine learning is a thing that computers do. It is very complicated and hard to explain. You probably wouldn't understand it.",
    },
    {
        "prompt": "Explain gravity simply.",
        "chosen": "Explain gravity simply. Gravity is a fundamental force that attracts objects with mass toward each other. The more massive an object is, the stronger its gravitational pull. This is why we stay on Earth's surface and why planets orbit the Sun.",
        "rejected": "Explain gravity simply. Gravity is gravity. Things fall down because of gravity. That's just how it works. Newton discovered it when an apple fell on his head or something.",
    },
    {
        "prompt": "How does photosynthesis work?",
        "chosen": "How does photosynthesis work? Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. Light energy is captured by chlorophyll in the leaves, driving a series of chemical reactions that produce the energy plants need to grow.",
        "rejected": "How does photosynthesis work? Plants use the sun to make food. They have green stuff in their leaves. It's a biology thing that happens in nature.",
    },
    {
        "prompt": "What causes earthquakes?",
        "chosen": "What causes earthquakes? Earthquakes occur when tectonic plates beneath Earth's surface shift and release built-up stress energy. This sudden release creates seismic waves that propagate through the ground, causing the shaking we feel on the surface.",
        "rejected": "What causes earthquakes? The ground shakes sometimes. It can be really scary. Earthquakes happen a lot in some places and not in others. You should have an emergency kit ready.",
    },
    {
        "prompt": "Why is the sky blue?",
        "chosen": "Why is the sky blue? The sky appears blue because of Rayleigh scattering. When sunlight enters Earth's atmosphere, shorter blue wavelengths of light are scattered more than longer red wavelengths by gas molecules, making the sky appear blue to our eyes.",
        "rejected": "Why is the sky blue? Because it just is. Some people think it has to do with the ocean reflecting but that's not right. It's actually complicated physics stuff.",
    },
    {
        "prompt": "What is DNA?",
        "chosen": "What is DNA? DNA, or deoxyribonucleic acid, is the molecule that carries genetic instructions for all living organisms. It has a double helix structure and contains four chemical bases (adenine, thymine, guanine, cytosine) whose sequence encodes biological information.",
        "rejected": "What is DNA? DNA is the stuff in your cells that makes you who you are. Everyone has different DNA. They use it on crime shows to catch criminals.",
    },
    {
        "prompt": "How do vaccines work?",
        "chosen": "How do vaccines work? Vaccines work by introducing a weakened or inactive form of a pathogen to the immune system. This trains the body to recognize and fight the actual pathogen efficiently if encountered later, providing immunity without causing the disease itself.",
        "rejected": "How do vaccines work? They put stuff in your body that's supposed to help you not get sick. Doctors give them to you when you're a kid. Some people don't like getting shots.",
    },
    {
        "prompt": "What is climate change?",
        "chosen": "What is climate change? Climate change refers to long-term shifts in global temperatures and weather patterns. While natural factors play a role, human activities, particularly burning fossil fuels, have been the primary driver since the industrial revolution, increasing greenhouse gas concentrations in the atmosphere.",
        "rejected": "What is climate change? The weather is changing and it's getting warmer. Some people think it's bad and some people don't. It's been in the news a lot lately.",
    },
]

print(f"偏好数据集大小: {len(preference_data)} 条")
print(f"\n示例:")
print(f"  Prompt:   {preference_data[0]['prompt']}")
print(f"  Chosen:   {preference_data[0]['chosen'][:100]}...")
print(f"  Rejected: {preference_data[0]['rejected'][:100]}...")
print(f"\n区别: chosen回答准确有深度, rejected回答敷衍没信息量")

# ============================================================
# Part 2: DPO的数学原 理 —— 用代码理解
# ============================================================
print("\n" + "=" * 60)
print("Part 2: DPO的核心公式 —— 用代码理解")
print("=" * 60)

print("""
DPO (Direct Preference Optimization) 的核心思想：

  不需要单独训练奖励模型(这是和RLHF的最大区别)，
  直接用一个巧妙的loss函数来优化：

  Loss = -log(σ(β * (log π(chosen) - log π_ref(chosen)
                    - log π(rejected) + log π_ref(rejected))))

  看起来复杂，拆开理解：

  1. log π(chosen):     当前模型给chosen回答的"打分"(对数概率)
  2. log π(rejected):   当前模型给rejected回答的"打分"
  3. log π_ref(chosen):  参考模型(未对齐的原模型)给chosen的打分
  4. log π_ref(rejected): 参考模型给rejected的打分

  β: 温度参数，控制偏离参考模型的程度

  直觉：
  - 让模型给chosen更高的分，给rejected更低的分
  - 但不能偏离参考模型太远(防止模型"忘记"语言能力)
  - σ是sigmoid函数，把分数差映射到0-1之间算loss
""")

# ============================================================
# Part 3: 实现DPO训练
# ============================================================
print("=" * 60)
print("Part 3: 实现DPO")
print("=" * 60)

# 加载tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 加载两个模型：
# 1. policy模型 (要训练的)
# 2. reference模型 (冻结的，作为基准)
print("\n加载模型...")
policy_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# reference模型完全冻结，不参与训练
# 它的作用是防止policy模型在对齐过程中"跑偏"太远
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()

print(f"Policy模型: 可训练")
print(f"Reference模型: 冻结 (用于约束policy不偏离太远)")


class PreferenceDataset(Dataset):
    """偏好数据集

    每条数据包含:
    - prompt: 问题
    - chosen: 人类偏好的回答
    - rejected: 人类不偏好的回答

    tokenize后返回 chosen和rejected的input_ids和attention_mask
    """
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # chosen和rejected都包含完整文本(prompt + response)
        # DPO需要计算整个序列的对数概率
        chosen_enc = self.tokenizer(
            item["chosen"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            item["rejected"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


def compute_log_probs(model, input_ids, attention_mask):
    """计算模型给一段文本的对数概率

    这是DPO的核心计算：
    1. 模型输出每个位置对下一个token的预测分布
    2. 取实际token对应的log概率
    3. 求和得到整个序列的log概率

    返回值越高，说明模型认为这段文本越"合理"
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        # 前向传播，得到每个位置的logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

    # logits[:, :-1] 是位置0到n-1的预测
    # input_ids[:, 1:] 是位置1到n的实际token (也就是"下一个token")
    # 这就是自回归：用位置i的输出预测位置i+1的token
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

    # 取出每个位置上"实际出现的token"的log概率
    # gather: 按input_ids的索引从log_probs中取值
    token_log_probs = log_probs.gather(
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    # 只保留非padding位置的log概率
    mask = attention_mask[:, 1:]
    token_log_probs = token_log_probs * mask

    # 求和 → 整个序列的总log概率
    return token_log_probs.sum(dim=-1)


def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    """DPO Loss 计算

    核心公式:
      loss = -log(σ(β * (log_ratio_chosen - log_ratio_rejected)))

    其中:
      log_ratio_chosen  = log π_policy(chosen)  - log π_ref(chosen)
      log_ratio_rejected = log π_policy(rejected) - log π_ref(rejected)

    直觉：
      - log_ratio_chosen 越大 → policy比ref更喜欢chosen → 好
      - log_ratio_rejected 越小 → policy比ref更不喜欢rejected → 好
      - 两者的差越大，loss越小

    beta控制"偏离参考模型的惩罚"：
      - beta越大，越不允许偏离ref → 更保守
      - beta越小，越允许偏离ref → 更激进
    """
    chosen_ids = batch["chosen_input_ids"].to(device)
    chosen_mask = batch["chosen_attention_mask"].to(device)
    rejected_ids = batch["rejected_input_ids"].to(device)
    rejected_mask = batch["rejected_attention_mask"].to(device)

    # Step 1: 计算policy模型对chosen和rejected的打分
    policy_chosen_logps = compute_log_probs(policy_model, chosen_ids, chosen_mask)
    policy_rejected_logps = compute_log_probs(policy_model, rejected_ids, rejected_mask)

    # Step 2: 计算reference模型对chosen和rejected的打分
    with torch.no_grad():
        ref_chosen_logps = compute_log_probs(ref_model, chosen_ids, chosen_mask)
        ref_rejected_logps = compute_log_probs(ref_model, rejected_ids, rejected_mask)

    # Step 3: 计算log ratio (policy相对于ref的偏好变化)
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    # Step 4: DPO loss
    # logits = β * (chosen比ref好多少 - rejected比ref好多少)
    # 我们希望这个值越大越好(chosen的提升 > rejected的提升)
    logits = beta * (chosen_log_ratio - rejected_log_ratio)
    loss = -F.logsigmoid(logits).mean()

    # 计算accuracy: policy是否正确地偏好chosen
    accuracy = (logits > 0).float().mean()

    return loss, accuracy, chosen_log_ratio.mean(), rejected_log_ratio.mean()


# ============================================================
# Part 4: 微调前baseline
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 对齐前的生成效果")
print("=" * 60)

test_prompts = [
    "What is machine learning?",
    "Explain gravity simply.",
    "How does photosynthesis work?",
]

def generate_response(model, prompt, max_new_tokens=60):
    """用模型生成回答"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("\n对齐前:")
for prompt in test_prompts:
    response = generate_response(policy_model, prompt)
    print(f"  Q: {prompt}")
    print(f"  A: {response[:200]}\n")

# ============================================================
# Part 5: 开始DPO训练
# ============================================================
print("=" * 60)
print("Part 5: DPO训练")
print("=" * 60)

print("""
训练过程：
  1. 取一批偏好数据 (chosen + rejected)
  2. 分别用policy模型和ref模型计算两个回答的log概率
  3. 用DPO loss让policy更偏向chosen、远离rejected
  4. 只更新policy模型，ref模型始终冻结

和普通微调的区别：
  - 普通微调: 只有一份文本，目标是预测下一个token
  - DPO: 有两份文本(好/坏)，目标是学会区分好坏
""")

# 创建数据集和数据加载器
train_dataset = PreferenceDataset(preference_data, tokenizer)
# 数据量小，不shuffle让结果更稳定可观察
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 优化器
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6)  # DPO用很小的学习率

# 训练循环
NUM_EPOCHS = 30  # 数据少，多跑几轮让效果明显
BETA = 0.1       # 温度参数

print(f"\n训练配置:")
print(f"  数据量: {len(preference_data)} 条偏好对")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  学习率: 5e-6")
print(f"  Beta: {BETA}")
print(f"\n开始训练...\n")

policy_model.train()
all_losses = []
all_accuracies = []

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    for batch in train_loader:
        optimizer.zero_grad()

        loss, accuracy, chosen_ratio, rejected_ratio = dpo_loss(
            policy_model, ref_model, batch, beta=BETA
        )

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += accuracy.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches
    all_losses.append(avg_loss)
    all_accuracies.append(avg_acc)

    # 每5个epoch打印一次
    if (epoch + 1) % 5 == 0:
        acc_bar = "█" * int(avg_acc * 20)
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS}: "
              f"loss={avg_loss:.4f}, "
              f"accuracy={avg_acc:.1%} {acc_bar}")

# ============================================================
# Part 6: 对齐后的效果
# ============================================================
print("\n" + "=" * 60)
print("Part 6: 对齐后的生成效果")
print("=" * 60)

print("\n对齐后:")
for prompt in test_prompts:
    response = generate_response(policy_model, prompt)
    print(f"  Q: {prompt}")
    print(f"  A: {response[:200]}\n")

# ============================================================
# Part 7: 量化对比 —— 模型偏好变化
# ============================================================
print("=" * 60)
print("Part 7: 量化分析 —— 模型的偏好发生了什么变化")
print("=" * 60)

print("""
我们来量化看：对齐前后，模型对chosen和rejected的打分变化
""")

policy_model.eval()
eval_dataset = PreferenceDataset(preference_data, tokenizer)
eval_loader = DataLoader(eval_dataset, batch_size=len(preference_data))

for batch in eval_loader:
    chosen_ids = batch["chosen_input_ids"].to(device)
    chosen_mask = batch["chosen_attention_mask"].to(device)
    rejected_ids = batch["rejected_input_ids"].to(device)
    rejected_mask = batch["rejected_attention_mask"].to(device)

    # Policy模型(对齐后)的打分
    with torch.no_grad():
        policy_chosen = compute_log_probs(policy_model, chosen_ids, chosen_mask)
        policy_rejected = compute_log_probs(policy_model, rejected_ids, rejected_mask)

        # Ref模型(对齐前)的打分
        ref_chosen = compute_log_probs(ref_model, chosen_ids, chosen_mask)
        ref_rejected = compute_log_probs(ref_model, rejected_ids, rejected_mask)

    print(f"\n{'Prompt':<35} {'Ref偏好':>10} {'Policy偏好':>12} {'变化':>8}")
    print("-" * 70)

    for i, item in enumerate(preference_data):
        # 偏好分数 = log_prob(chosen) - log_prob(rejected)
        # 正数 = 更偏好chosen，负数 = 更偏好rejected
        ref_pref = (ref_chosen[i] - ref_rejected[i]).item()
        policy_pref = (policy_chosen[i] - policy_rejected[i]).item()
        delta = policy_pref - ref_pref

        # delta > 0 说明对齐后更偏好chosen了
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {item['prompt']:<33} {ref_pref:>+8.1f}  {policy_pref:>+10.1f}  {arrow}{abs(delta):>6.1f}")

    # 总体统计
    ref_correct = (ref_chosen > ref_rejected).float().mean()
    policy_correct = (policy_chosen > policy_rejected).float().mean()
    print(f"\n偏好正确率:")
    print(f"  对齐前(ref):    {ref_correct:.1%}")
    print(f"  对齐后(policy): {policy_correct:.1%}")

# ============================================================
# Part 8: 训练曲线
# ============================================================
print("\n" + "=" * 60)
print("Part 8: 训练过程")
print("=" * 60)

print("\nLoss曲线:")
for i, loss in enumerate(all_losses):
    if (i + 1) % 5 == 0:
        bar = "█" * int((1.0 - loss) * 30) if loss < 1.0 else "▏"
        print(f"  Epoch {i+1:2d}: {loss:.4f} {bar}")

print("\nAccuracy曲线:")
for i, acc in enumerate(all_accuracies):
    if (i + 1) % 5 == 0:
        bar = "█" * int(acc * 20)
        print(f"  Epoch {i+1:2d}: {acc:.1%} {bar}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结：从预训练到对齐的完整链路")
print("=" * 60)
print("""
你现在已经走完了大模型训练的完整流程：

  Step 1: 理解Transformer结构
    → Tokenizer + Embedding + Multi-Head Attention + FFN

  Step 2: 全量微调 (Supervised Fine-Tuning, SFT)
    → 用特定数据让模型学会某种风格/知识

  Step 3: LoRA高效微调
    → W' = W + B×A，只训练0.24%参数

  Step 4: DPO偏好对齐 (本节)
    → 用偏好数据让模型学会区分好坏回答

这正是ChatGPT的训练三阶段：
  1. 预训练 (海量数据学语言能力) → 对应Step 1-2
  2. SFT (有监督微调学对话格式)   → 对应Step 2-3
  3. RLHF/DPO (对齐人类偏好)     → 对应Step 4

DPO vs RLHF:
  - RLHF: 训练奖励模型 → PPO强化学习 (两步，较复杂)
  - DPO:  直接用偏好数据优化 (一步，更简单稳定)
  - 效果相当，DPO是目前的主流趋势 (LLaMA 2/3都用了DPO)

面试必知：
  - "对齐为什么重要？" → 没对齐的模型会生成有害/无用的内容
  - "DPO相比RLHF的优势？" → 不需要训练奖励模型，更稳定，超参更少
  - "beta参数的作用？" → 控制偏离参考模型的程度，防止catastrophic forgetting
""")
