# 简历项目描述 — SciAgent

> 以下提供中文版和英文版，可根据简历语言选用。各版本提供"详细版"和"精简版"两种长度。

---

## 中文版（详细，适合项目经历栏）

**SciAgent：面向微通道散热器领域的自然语言驱动科学计算 Agent**

- 基于 ReAct（Reasoning + Acting）框架，设计并实现了一个科学计算智能体，能够接收自然语言描述的传热学问题，自主生成 Python 代码进行求解，并输出带有可视化图表的分析报告
- 以矩形微通道散热器为应用场景，构建了包含 10 道题目的领域评测基准（Benchmark），覆盖肋片效率、Nusselt 数验证、Shah-London 摩擦关联式、压降计算、一维热阻网络模型、参数敏感性分析、约束优化及 NSGA-II 多目标优化等经典问题，所有题目均配有解析解作为标注答案
- 设计了三维度自动评分系统（验证代码 40 分 + 数值匹配 40 分 + 图表生成 20 分），实现无人工干预的端到端评测流程
- 评测结果：10 题全部通过，平均得分 88.8/100；Agent 在公式计算类任务上接近满分，在多目标优化任务上得分 80 分；分析发现 LLM 在领域公式选择歧义（如 Fanning vs Poiseuille 摩擦系数定义）场景下容易出错
- 技术栈：Python, Claude/DeepSeek API, ReAct Agent, DEAP (NSGA-II), NumPy, SciPy, Matplotlib

**GitHub**: https://github.com/wangmanqin/sciagent-thermal

---

## 中文版（精简，适合一行式项目列表）

**SciAgent：自然语言驱动的科学计算 Agent** — 基于 ReAct 框架实现传热学问题的自动求解，构建 10 题微通道散热器评测基准并设计自动评分系统，评测通过率 100%，平均得分 88.8/100。（Python, LLM API, NSGA-II）

---

## English Version (Detailed)

**SciAgent: Natural Language-Driven Scientific Computing Agent for Microchannel Heat Sink Analysis**

- Designed and implemented a scientific computing agent based on the ReAct (Reasoning + Acting) framework that accepts natural language queries on heat transfer problems, autonomously generates and executes Python code, and produces analysis reports with visualizations
- Built a domain-specific benchmark of 10 problems covering fin efficiency, Nusselt number verification, Shah-London friction correlation, pressure drop calculation, 1D thermal resistance network modeling, parametric sensitivity analysis, constrained optimization, and NSGA-II multi-objective optimization — all with analytical ground-truth solutions
- Developed a three-dimensional auto-scoring system (verification code 40 pts + numerical matching 40 pts + plot generation 20 pts) enabling fully automated end-to-end evaluation
- Achieved 100% pass rate (10/10) with an average score of 88.8/100; identified that LLM agents excel at formula-based computation (near-perfect) but struggle with domain-specific formula disambiguation (e.g., Fanning vs. Poiseuille friction factor conventions)
- Tech stack: Python, Claude/DeepSeek API, ReAct Agent, DEAP (NSGA-II), NumPy, SciPy, Matplotlib

**GitHub**: https://github.com/wangmanqin/sciagent-thermal

---

## English Version (Concise)

**SciAgent: NL-Driven Scientific Computing Agent** — Built a ReAct-based agent that solves heat transfer problems from natural language input; created a 10-problem microchannel heat sink benchmark with auto-scoring; achieved 100% pass rate and 88.8/100 avg score. (Python, LLM API, NSGA-II)

---

## 面试要点提示

如果面试官问到这个项目，以下是几个关键点：

1. **为什么选 ReAct 框架？**
   - ReAct 让 Agent 交替进行"推理"和"执行"，遇到代码报错时能自动观察错误信息并修正，不需要人工干预。实测中 Agent 多次自动修复了运行时错误。

2. **Benchmark 的设计思路是什么？**
   - 10 道题按难度递进（easy → medium → hard），从单公式计算到多目标优化，每道题都有确定的解析解可以自动验证，避免了人工评判的主观性。

3. **最有意思的发现是什么？**
   - LLM 在"选择正确的公式"这件事上会出错。传热领域的摩擦系数有 Fanning、Darcy-Weisbach、Poiseuille number 三种定义，差 4 倍。Agent 在第 4 题选错了公式，导致数值偏差。这说明 LLM 的"领域常识"还不够精确，是未来改进的方向（比如在 prompt 中补充术语规范，或用 RAG 引入教材知识）。

4. **评分系统怎么设计的？**
   - 三个维度：(1) 运行验证代码检查计算逻辑是否正确；(2) 从 Agent 输出文本中正则提取数值，与标注答案做容差匹配；(3) 检查是否生成了图表文件。总分 100 分，>=60 分算通过。
