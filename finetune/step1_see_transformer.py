"""
Step 1: 亲手看到Transformer在干什么
==============================
目标：不读论文，直接通过代码理解三个核心问题：
  1. Tokenizer 怎么把文字变成数字？
  2. GPT 模型内部长什么样？（对应你Course 5学的那些层）
  3. 模型怎么一个token一个token地生成文本？
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# ============================================================
# Part 1: Tokenizer —— 文字 → 数字
# ============================================================
# 你在Course 5学过：模型输入是向量，不是文字
# Tokenizer就是干这个转换的

print("=" * 60)
print("Part 1: Tokenizer 把文字变成什么？")
print("=" * 60)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Artificial intelligence is transforming the world"
tokens = tokenizer.tokenize(text)       # 文字 → 子词
token_ids = tokenizer.encode(text)      # 文字 → 数字ID

print(f"\n原文: {text}")
print(f"切分成tokens: {tokens}")
print(f"对应的数字ID: {token_ids}")
print(f"词表大小: {tokenizer.vocab_size}")

# 动手试试：改成你自己的句子看看怎么切分
# text = "你好世界"  # GPT-2是英文模型，中文会被切得很碎，之后我们换中文模型

# ============================================================
# Part 2: 看模型内部结构 —— 对应Course 5的知识
# ============================================================
print("\n" + "=" * 60)
print("Part 2: GPT-2内部长什么样？")
print("=" * 60)

model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")

# 打印模型结构，你会看到：
# - wte: Token Embedding（把数字变成向量，Course 5第一周讲的）
# - wpe: Position Embedding（位置编码，Course 5讲Transformer时提到的）
# - h.0 ~ h.11: 12个Transformer Block（每个都有Attention + FFN）
# - attn: 就是你学的Multi-Head Self-Attention!
# - mlp: 就是Feed Forward Network
print("\n模型结构（仔细看，每一层都对应Course 5的知识）:")
print(model)

# 数一下参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,} ({total_params/1e6:.0f}M)")
print("这就是所谓的 GPT-2 Small (124M参数)")

# ============================================================
# Part 3: 看Attention在做什么
# ============================================================
print("\n" + "=" * 60)
print("Part 3: Attention到底在关注什么？")
print("=" * 60)

inputs = tokenizer(text, return_tensors="pt")

model.config.output_attentions = True
with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions 是每一层的attention权重
# 形状: (batch, num_heads, seq_len, seq_len)
# 这就是Course 5里讲的 softmax(QK^T / sqrt(d_k)) 的结果!

attn = outputs.attentions[0]  # 第一层的attention
print(f"\nAttention权重形状: {attn.shape}")
print(f"  - 1个样本")
print(f"  - {attn.shape[1]}个注意力头 (Multi-Head!)")
print(f"  - {attn.shape[2]}x{attn.shape[3]}的注意力矩阵 (每个token对其他token的关注度)")

# 看第一个head，最后一个token在关注哪些词
last_token_attn = attn[0, 0, -1, :]  # 第1个head, 最后一个token
print(f"\n最后一个token '{tokens[-1]}' 在关注:")
for token, score in zip(tokens, last_token_attn[:len(tokens)]):
    bar = "█" * int(score * 50)
    print(f"  {token:20s} {score:.3f} {bar}")

# ============================================================
# Part 4: 文本生成 —— 一个token一个token往外蹦
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 模型怎么生成文本？")
print("=" * 60)

prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

print(f"\n输入: '{prompt}'")
print(f"开始逐步生成...\n")

# 手动一步步生成，让你看到每一步发生了什么
current_ids = input_ids
for step in range(15):
    with torch.no_grad():
        outputs = model(current_ids)

    # outputs.logits的最后一个位置 = 模型对下一个token的预测
    # 形状: (batch, seq_len, vocab_size=50257)
    next_token_logits = outputs.logits[0, -1, :]

    # 取概率最高的token（贪心解码）
    next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
    next_token = tokenizer.decode(next_token_id[0])

    # 拼接上去，继续生成
    current_ids = torch.cat([current_ids, next_token_id], dim=1)

    generated_so_far = tokenizer.decode(current_ids[0])
    print(f"  Step {step+1:2d}: 预测 '{next_token}' → {generated_so_far}")

print(f"\n最终生成: {tokenizer.decode(current_ids[0])}")

# ============================================================
# 总结：你刚才看到了什么
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
你刚才亲眼看到了：
1. Tokenizer: 文字→子词→数字ID (这就是Embedding的输入)
2. 模型结构: 12层Transformer Block, 每层有Multi-Head Attention + FFN
   → 和Course 5讲的一模一样，只是GPT只用了Decoder部分
3. Attention权重: softmax(QK^T/sqrt(d_k))的实际输出，能看到每个词在关注谁
4. 自回归生成: 每次只预测下一个token，拼上去再预测下一个
   → 这就是GPT的核心: 用过去的所有token预测下一个

下一步：我们会换一个中文模型，然后微调它完成特定任务
""")
