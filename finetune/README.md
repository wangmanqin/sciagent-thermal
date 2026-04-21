# finetune/ — SciAgent 配套的领域自适应微调与偏好对齐

本目录是 [SciAgent](../README.md) 项目的配套工作,探索:**能否通过领域自适应 LoRA 微调 + DPO 偏好对齐,让开源小模型(Qwen2.5-1.5B)在科学计算任务上逼近闭源大模型的表现?**

所有实验在单卡消费级 GPU(**RTX 4060 Laptop, 8GB**)上完成。

---

## 核心结果

在 SciAgent Benchmark(3 题快速评测子集)上,三模型对比:

| 模型 | 平均分 | 通过率 (≥60) | 相对提升 |
|------|-------:|-------------:|---------:|
| raw (原始 Qwen2.5-1.5B-Instruct) | 45.0 | 0/3 | — |
| SFT (LoRA 微调) | 50.0 | 0/3 | +5 |
| **DPO (SFT merge + DPO LoRA)** | **55.0** | **1/3** | **+10** |

评测报告见 [`eval_reports/scitune_eval_20260415_231349.json`](eval_reports/scitune_eval_20260415_231349.json)。

---

## 整体链路

```
Step 1-4: 原理验证 (GPT-2)
  step1  Transformer 结构理解
  step2  全量 SFT
  step3  LoRA 高效微调
  step4  DPO 偏好对齐  (手写实现,非 TRL 库)

Step 5-7: 领域微调 (Qwen2.5-1.5B-Instruct)
  step5  4bit NF4 量化 + LoRA SFT           9 分钟训练
  step6  SFT merge + DPO LoRA              2 分钟训练
  step7  三模型对比评测                    raw / SFT / DPO

数据构建: build_dataset.py
  DeepSeek API 知识蒸馏 → SFT 250 条 + DPO 129 对
```

---

## 关键技术要点

- **4bit NF4 量化 + 双重量化**:1.5B 模型只占 ~1.5GB 显存,8GB 卡可跑
- **LoRA r=16, QKVO 投影**:可训练参数占比 0.28%
- **Label Masking**:只在 assistant 回复部分计算 loss
- **手写 DPO loss**:`loss = -log σ(β · (log_ratio_chosen - log_ratio_rejected))`,见 [`step6_dpo_qwen.py:270`](step6_dpo_qwen.py#L270)
- **SFT merge + DPO LoRA 方案**:经三轮调试确定,解决了 PEFT adapter 链式加载导致 SFT 能力丢失的问题(详见下方"踩坑记录")

---

## 踩坑记录:PEFT adapter 链式加载陷阱

| 轮次 | 方案 | 结果 |
|------|------|------|
| 第一轮 | Base 模型 + merge_and_unload + 无早停 | 生成崩溃(重复字符) |
| 第二轮 | Instruct + adapter 链式加载 (`add_adapter` + `set_adapter`) | **loss>10, acc=0%** |
| **第三轮** | **Instruct + SFT merge → DPO LoRA(ref 同步 merge)** | **loss 0.62→0.36, acc 97.3%** ✅ |

第二轮失败的根因:`set_adapter("dpo")` 会禁用 SFT adapter (`active_adapters` 从 `['default']` 变为 `['dpo']`),policy 模型失去 SFT 能力,与 ref 的 log prob 差距巨大,DPO loss 发散。

---

## 如何运行

```bash
# 生成/扩充训练数据(需要 DEEPSEEK_API_KEY)
python build_dataset.py --variants 8 --resume

# SFT 训练(~9 分钟,RTX 4060)
python step5_sft_qwen.py

# DPO 训练(~2 分钟)
python step6_dpo_qwen.py

# 三模型对比评测
python step7_evaluate.py            # 完整 10 题
python step7_evaluate.py --quick    # 快速 3 题
```

**模型权重未随仓库上传**(LoRA adapter 加上各 checkpoint 约 140MB,超 GitHub 单仓库建议上限)。按上述脚本运行可本地复现。

---

## 更多文档

- [`总结.md`](总结.md) — 全流程总结与关键技术发现
- [`实验报告_LLM微调全链路.md`](实验报告_LLM微调全链路.md) — 详细实验报告

---

## 技术栈

PyTorch · HuggingFace Transformers · PEFT · bitsandbytes (NF4) · 自研 DPO 实现 · SciAgent Benchmark
