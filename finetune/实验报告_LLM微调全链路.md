# LLM 微调全链路实验报告：从 SFT 到 DPO

> 王熳琴 | 西安电子科技大学 测控技术与仪器 | 2026.04

---

## 一、实验目标

从零实现大语言模型训练的完整链路，在消费级 GPU (RTX 4060 8GB) 上完成：

1. **SFT (Supervised Fine-Tuning)**：用 LoRA + 4bit 量化微调 Qwen2.5-1.5B-Instruct，使其学会科学计算代码生成
2. **DPO (Direct Preference Optimization)**：在 SFT 基础上做偏好对齐，让模型偏好详细准确的回答
3. **三模型对比评测**：在标准化 benchmark 上评估 raw / SFT / DPO 三个版本的代码生成能力

核心技术栈：BitsAndBytes 4bit NF4 量化、LoRA、8bit AdamW 优化器、DPO loss 手写实现（非调用 TRL 库）、adapter 链式加载。

---

## 二、实验设置

### 2.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4060 Laptop (8GB VRAM) |
| CUDA | 12.7 |
| 系统 | Windows 11 |

### 2.2 模型与数据

| 项目 | SFT 阶段 | DPO 阶段 |
|------|----------|----------|
| 基座模型 | Qwen2.5-1.5B-Instruct | SFT 微调后的模型 |
| 量化方式 | 4bit NF4 + 双重量化 | 同左 |
| 数据量 | ~224 条科学计算 QA | ~112 对偏好数据 (chosen/rejected) |
| 数据领域 | 方程求解、ODE、优化、统计、信号处理、线性代数、数值方法、曲线拟合、PDE、数据分析 | 同左 |
| 数据来源 | DeepSeek API 生成（知识蒸馏） | 同左 |

### 2.3 LoRA 配置

| 参数 | SFT | DPO |
|------|-----|-----|
| 秩 (r) | 16 | 8 |
| alpha | 32 | 16 |
| dropout | 0.05 | 0.05 |
| 目标模块 | q/k/v/o_proj | q/v_proj |
| 可训练参数占比 | ~0.49% | ~0.07% |

**设计考量**：SFT 阶段 r=16 覆盖完整注意力机制（QKVO 四个投影），让模型充分学习新领域知识；DPO 阶段 r=8 仅调 Q/V，避免过度偏移 SFT 的基础能力。

### 2.4 关键改进（第二轮实验）

| 改进项 | 第一轮 | 第二轮 |
|--------|--------|--------|
| 基座模型 | Qwen2.5-1.5B (Base) | Qwen2.5-1.5B-Instruct |
| DPO adapter 加载 | merge_and_unload (有损) | adapter 链式加载 (无损) |
| SFT 早停 | 无 | EarlyStoppingCallback(patience=3) |
| DPO 验证集 | 无 | 15% 验证集拆分 |
| DPO 早停 | 无（固定10 epoch） | 验证集 loss 不下降则停止 |
| 梯度裁剪 | 无 | max_norm=1.0 |
| 数据规模 | 24 SFT + 15 DPO | ~224 SFT + ~112 DPO |
| 数据领域 | 8 个子领域 | 10 个子领域 |

---

## 三、SFT 阶段结果

### 3.1 训练过程

- 训练 epoch 根据数据量自适应（<50条: 10 epoch, <200条: 5, >=200条: 3）
- 等效 batch_size = 4（micro_batch=2, gradient_accumulation=2）
- 显存占用：~1.5 GB（4bit 量化后）
- EarlyStoppingCallback 在验证集 loss 连续 3 次不下降时提前终止

**Training Loss 曲线（第一轮 24 条数据）：**

```
Step  5: 0.8582  ▓▓▓▓▓▓▓▓░░░░░░░░░░░░
Step 15: 0.7665  ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░
Step 30: 0.6001  ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░
Step 45: 0.5380  ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░
Step 60: 0.5381  ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░
```

**Validation Loss**：0.7209 → 0.6854 → 0.6868（末期轻微上升，早停机制可捕获）

### 3.2 生成效果对比

| 测试问题 | 微调前 | 微调后 |
|---------|--------|--------|
| 求解 x^3 - 6x^2 + 11x - 6 = 0 | 错误套用二次公式，无法得出结果 | 正确调用 `sympy.solve`，输出完整代码 |
| NSGA-II 双目标优化 | 输出乱码（重复"猄"字符） | 写出合理的 NSGA-II 算法框架 |
| FFT 频谱分析 | 输出乱码（重复"isz2198"） | 正确使用 `numpy.fft` + `matplotlib` |

### 3.3 分析

SFT 有效地将一个对科学计算一无所知的 base 模型（生成乱码）转变为能输出结构化代码的领域模型。仅 24 条训练数据 + 1.4 分钟训练即可观察到明显的行为变化，体现了 LoRA 在小数据场景下的高效性。

---

## 四、DPO 阶段结果

### 4.1 第一轮实验（失败）

**训练指标看似正常：**

```
Epoch  1: loss=0.6999, accuracy=46.7%
Epoch  4: loss=0.6261, accuracy=86.7%
Epoch  6: loss=0.5744, accuracy=100.0%
Epoch 10: loss=0.4631, accuracy=100.0%
```

**但生成质量严重退化**，输出退化为重复字符串。

### 4.2 失败原因分析（最有价值的发现）

DPO 训练指标良好（loss 下降、accuracy 100%）但生成崩溃，原因是多重因素叠加：

| 因素 | 具体问题 | 影响 |
|------|---------|------|
| **SFT 基础不牢** | Base 模型（非 Instruct）对 ChatML 对话格式理解不足 | DPO 在不稳定的基础上做微调，放大了 SFT 的缺陷 |
| **数据量过小** | 仅 15 对偏好数据，训练集 accuracy 达到 100% | 严重过拟合训练集，丧失泛化生成能力 |
| **merge_and_unload 精度损失** | 4bit 模型做 merge 时引入舍入误差 | Policy 和 Reference 模型的起点已不完全一致 |
| **无验证集/早停** | 训练到 100% accuracy 仍不停止 | 持续偏移导致生成退化 |

**核心结论：DPO 的效果高度依赖 SFT 阶段的质量。在 SFT 模型能力不足时，DPO 不仅无法改善，反而会加速退化。** 这与 Anthropic 等机构的实践一致——RLHF/DPO 是在已经很好的 SFT 模型上做"精修"，而非"补课"。

### 4.3 第二轮实验修复方案

| 修复 | 具体做法 | 原理 |
|------|---------|------|
| Base → Instruct | 使用 Qwen2.5-1.5B-Instruct 作为基座 | 天然理解 ChatML 格式，SFT 起点更高 |
| 移除 merge_and_unload | 改用 adapter 链式加载：基座 → SFT adapter → DPO adapter | 避免 4bit 量化 merge 时的舍入误差 |
| 添加验证集 | 85/15 拆分训练集/验证集 | 监控泛化能力 |
| 早停机制 | 验证集 loss 连续 N 轮不下降则停止 | 防止过拟合到 100% accuracy |
| 梯度裁剪 | max_norm=1.0 | 提升训练稳定性 |
| 增大数据量 | ~112 对偏好数据（原 15 对） | 减少过拟合风险 |

---

## 五、关键技术实现细节

### 5.1 Label Masking (SFT)

训练时只在 assistant 回复部分计算 loss，system prompt 和 user 问题设为 -100（忽略）：

```python
# 定位 <|im_start|>assistant\n 标记的位置
# 标记之前的 token 全部设为 -100（不参与 loss 计算）
# 标记之后到序列末尾的 token 才计算 loss
labels[:start_idx] = [-100] * start_idx
labels[start_idx:] = input_ids[start_idx:]
```

**意义**：模型只学习"如何回答"，不学习复读 prompt，训练效率更高。

### 5.2 DPO Loss 手写实现

```python
# 核心公式：Loss = -log σ(β × (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))
chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
rejected_log_ratio = policy_rejected_logps - ref_rejected_logps
logits = beta * (chosen_log_ratio - rejected_log_ratio)
loss = -F.logsigmoid(logits).mean()
```

没有使用 Hugging Face TRL 库，而是手动实现完整的 DPO 训练循环，包括：
- 序列对数概率计算（带 attention mask）
- Policy 模型和 Reference 模型的 log-ratio 差
- 基于 logsigmoid 的 loss 计算

### 5.3 Adapter 链式加载（第二轮修复）

```python
# 第一轮（有问题）：merge_and_unload 导致 4bit 舍入误差
policy_model = PeftModel.from_pretrained(base, sft_adapter)
policy_model = policy_model.merge_and_unload()  # ← 精度损失！
policy_model = get_peft_model(policy_model, dpo_config)

# 第二轮（修复）：直接在 SFT adapter 上叠加 DPO adapter
policy_model = PeftModel.from_pretrained(base, sft_adapter)
policy_model.add_adapter("dpo", dpo_config)  # ← 链式加载，无精度损失
policy_model.set_adapter("dpo")
```

### 5.4 显存优化策略

在 8GB 显存上同时运行两个 1.5B 模型（DPO 需要 policy + reference）：

| 技术 | 节省量 | 说明 |
|------|--------|------|
| 4bit NF4 量化 | ~75% 模型显存 | 1.5B 模型从 ~6GB 降至 ~1.5GB |
| 双重量化 | ~10% | 对量化常数再做一次量化 |
| 8bit AdamW | ~50% 优化器显存 | 优化器状态用 8bit 存储 |
| gradient_accumulation | 节省峰值 | 等效大 batch 但不增加显存 |

---

## 六、评测系统 (Step 7)

### 6.1 评测设计

直接给模型科学计算问题，评估其代码生成质量：

| 评分维度 | 分值 | 说明 |
|---------|------|------|
| 代码可执行性 | 30分 | 代码能否成功运行 |
| 代码完整性 | 30分 | 是否包含 import、print |
| 输出相关性 | 20分 | 输出中是否包含数值结果 |
| 回答结构 | 20分 | 是否有解释文字和清晰结构 |

### 6.2 评测覆盖

10道题，覆盖10个科学计算领域：方程求解、ODE、优化、统计、信号处理、曲线拟合、线性代数、数值方法、PDE、数据分析。

### 6.3 三模型对比

| 模型 | 说明 |
|------|------|
| raw | 原始 Qwen2.5-1.5B-Instruct（无微调） |
| sft | 加载 SFT LoRA adapter |
| dpo | 加载 SFT + DPO 双层 adapter |

---

## 七、与 SciAgent 项目的关系

本实验是 SciAgent 项目的延伸。在 SciAgent 评测中发现闭源大模型（Claude、GPT-4）在科学计算上的通过率为 100%，但存在系统性的公式选择错误（如 Fanning 与 Poiseuille 摩擦系数混淆，偏差达 4 倍）。

本实验探索的问题是：**能否通过 SFT + DPO 对开源小模型做领域自适应，使其在特定科学计算任务上逼近闭源大模型的表现？**

实验验证了技术链路的可行性，同时也揭示了关键瓶颈：
- 小数据 SFT 可以快速赋予模型新的领域能力
- 但 DPO 对齐需要 SFT 阶段足够扎实作为前提
- 数据质量和规模是当前的主要限制因素
- Instruct 模型作为基座比 Base 模型效果显著更好

---

## 八、改进方向

| 方向 | 具体方案 | 预期效果 |
|------|---------|---------|
| 增大数据规模 | SFT 数据扩展到 500+ 条（已支持 `--resume`） | 显著提升泛化能力 |
| 更强基座模型 | 尝试 Qwen2.5-3B 或 7B（需更大显存） | 更高的代码生成质量 |
| DPO 数据质量 | 引入难度梯度，控制 chosen/rejected 差距 | 改善 DPO 的泛化效果 |
| 评测扩展 | 接入完整 SciAgent Agent 循环（需 tool_use 支持） | 端到端评测 |

---

## 附：完整代码结构

```
llm_practice/
├── step1_see_transformer.py    # 理解 Transformer 架构
├── step2_finetune_gpt2.py      # GPT-2 全量微调
├── step3_lora.py               # GPT-2 LoRA 微调
├── step4_dpo.py                # GPT-2 DPO (原理验证)
├── step5_sft_qwen.py           # Qwen2.5 LoRA SFT (本实验)
├── step6_dpo_qwen.py           # Qwen2.5 DPO 对齐 (本实验)
├── step7_evaluate.py           # 三模型对比评测 (本实验)
├── build_dataset.py            # 数据集构建 (DeepSeek API)
├── data/
│   ├── sft_train.json          # SFT 训练数据
│   └── dpo_train.json          # DPO 偏好数据
├── qwen-sft-lora/              # SFT 输出 (LoRA adapter)
├── qwen-dpo-lora/              # DPO 输出 (LoRA adapter)
├── eval_reports/               # 评测报告
├── 总结.md                     # 项目总结
└── 实验报告_LLM微调全链路.md    # 本报告
```
