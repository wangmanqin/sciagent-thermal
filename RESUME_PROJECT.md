# 简历项目描述 — SciAgent

> 以下提供中文版和英文版，可根据简历语言选用。各版本提供"详细版"和"精简版"两种长度。

---

## 中文版（详细，适合项目经历栏，4-5 条 bullet points）

**SciAgent：面向微通道散热器领域的 LLM 科学计算 Agent 与自动化评测框架**

- 基于 ReAct（Reasoning + Acting）推理框架与 Tool-Use 机制，设计并实现了一个面向传热学领域的科学计算智能体。Agent 通过"推理→生成代码→执行→观察结果→再推理"的闭环迭代自主完成求解，支持运行时错误的自动检测与自修复（实测中多次自动修复 RuntimeError 和编码异常）
- 构建了微通道散热器领域的评测基准（Benchmark），包含 10 道由浅入深的测试题：涵盖肋片效率（η = tanh(mH)/mH）、Shah-London 摩擦关联式（Po = 96·f(α)）、Tuckerman-Pease 一维热阻网络、参数敏感性分析以及基于 NSGA-II 的多目标 Pareto 优化等经典问题。所有题目均配有解析解 / 数值验证代码作为 Ground Truth，支持全自动判分
- 设计了多维度自动评分体系：程序化验证（执行标注验证代码判定计算逻辑正确性，40%）+ 数值容差匹配（正则提取 Agent 输出中的数值与期望值做相对/绝对容差比较，40%）+ 可视化完整性检查（检测图表文件生成，20%），实现了无人工干预的端到端评测流程
- 对 Agent 的能力边界进行了系统性分析：Agent 在明确公式的计算任务上接近满分（Easy 类平均 96.7 分），但在存在公式定义歧义的场景下（如 Fanning friction factor vs Poiseuille number，两者相差 4 倍）会选择错误的物理公式（得分 68 分），揭示了 LLM 在垂直领域应用中"领域常识精度不足"的共性问题，提出了通过 Prompt 工程注入术语规范或 RAG 检索教材公式库的改进方案
- 技术栈：Python, Claude/DeepSeek API (Tool Use), ReAct Agent, DEAP (NSGA-II), NumPy/SciPy, Matplotlib

**GitHub**: https://github.com/wangmanqin/sciagent-thermal

---

## 中文版（精简，适合项目列表，2-3 行）

**SciAgent：面向传热学的 LLM 科学计算 Agent 与评测框架**
基于 ReAct 框架与 Tool-Use 机制实现自然语言驱动的科学计算自动求解，构建了微通道散热器领域 10 题评测基准（覆盖解析计算、参数分析到 NSGA-II 多目标优化），设计多维度自动评分体系实现端到端无人工评测。评测通过率 100%，平均 88.8/100 分。通过对比分析揭示了 LLM 在领域公式选择歧义场景下的局限性，并提出 RAG 增强改进方案。（Python, LLM API, ReAct, NSGA-II）

---

## English Version (Detailed)

**SciAgent: LLM-Powered Scientific Computing Agent and Automated Evaluation Framework for Microchannel Heat Sink Analysis**

- Designed and implemented a scientific computing agent based on the ReAct (Reasoning + Acting) framework with Tool-Use capabilities. The agent autonomously solves heat transfer problems through an iterative loop of reasoning → code generation → execution → observation → re-reasoning, with built-in runtime error detection and self-repair mechanisms
- Constructed a domain-specific benchmark of 10 progressively challenging problems in microchannel heat sink thermal analysis, covering fin efficiency (η = tanh(mH)/mH), Shah-London friction correlation (Po = 96·f(α)), Tuckerman-Pease 1D thermal resistance network, parametric sensitivity analysis, and NSGA-II multi-objective Pareto optimization. All problems are equipped with analytical/numerical ground-truth solutions for fully automated grading
- Developed a multi-dimensional auto-scoring system: programmatic verification (executing ground-truth validation code, 40%) + numerical tolerance matching (regex extraction with relative/absolute tolerance comparison, 40%) + visualization completeness check (20%), enabling end-to-end evaluation without human intervention
- Conducted systematic capability analysis: the agent achieves near-perfect scores on explicit-formula tasks (Easy avg. 96.7/100) but fails on formula-disambiguation scenarios (e.g., Fanning friction factor vs. Poiseuille number — a 4× discrepancy, scoring 68/100), revealing a common limitation of LLMs in domain-specific applications — insufficient precision in disciplinary conventions. Proposed improvement via prompt engineering with terminology specifications and RAG-based retrieval from textbook formula databases
- Tech stack: Python, Claude/DeepSeek API (Tool Use), ReAct Agent, DEAP (NSGA-II), NumPy/SciPy, Matplotlib

**GitHub**: https://github.com/wangmanqin/sciagent-thermal

---

## English Version (Concise)

**SciAgent: LLM Scientific Computing Agent & Evaluation Framework**
Built a ReAct-based agent with Tool-Use for automated heat transfer problem solving; constructed a 10-problem microchannel heat sink benchmark (analytical computation to NSGA-II optimization) with multi-dimensional auto-scoring. Achieved 100% pass rate and 88.8/100 avg. score. Identified LLM limitations in domain formula disambiguation and proposed RAG-based improvements. (Python, LLM API, ReAct, NSGA-II)

---

## 面试要点提示

如果面试官问到这个项目，以下是几个关键点：

### Q1: 为什么选 ReAct 框架？跟 Chain-of-Thought 有什么区别？
- CoT 只做推理，不与外部环境交互。ReAct 在推理的基础上引入了"行动"（调用工具执行代码）和"观察"（读取执行结果），形成了闭环。这对科学计算至关重要——Agent 需要真正运行代码才能验证自己的推理是否正确。
- 实测中 Agent 多次在执行后发现结果不对（如数量级异常），主动修改公式重新计算，这是纯 CoT 做不到的。

### Q2: Benchmark 的设计思路是什么？为什么这 10 道题？
- 10 道题按认知难度递进：基础公式代入（Easy）→ 多物理量耦合计算（Medium）→ 多目标优化（Hard）。
- 关键设计原则：每道题必须有确定的解析解或数值标准答案，这样才能实现全自动评分，避免 LLM-as-Judge 的主观性问题。
- 题目覆盖了微通道散热器研究的完整链路：单通道分析 → 散热器级建模 → 参数优化 → 多目标设计。

### Q3: 最有意思/最有价值的发现是什么？
- LLM 在"选择正确的物理公式"这件事上会出错。传热领域的摩擦系数有 Fanning（f）、Darcy-Weisbach（f_D = 4f）、Poiseuille number（Po = f·Re）三种定义方式，数值差 4 倍。Agent 在第 4 题选错了公式，导致压降计算偏差 4 倍。
- 这不是 Agent 的"计算能力"问题，而是"领域常识"问题——LLM 的训练数据中这三种定义都出现过，但它无法判断当前题目语境该用哪个。这个发现指向了一个有研究价值的方向：如何让 LLM 在垂直领域中准确应用专业术语和惯例。

### Q4: 评分系统有什么特别之处？
- 三个维度互补：验证代码检查逻辑正确性（Agent 用的公式对不对）、数值匹配检查结果正确性（算出来的数对不对）、图表检查检查完成度（分析做完了没有）。
- 对比：现有的很多 LLM Benchmark（如 HumanEval、MATH）只检查最终答案对不对。我们的评分能区分"逻辑对但数值格式不匹配"和"逻辑本身就错了"两种失败模式——这在实际分析中很重要。

### Q5: 如果继续做下去，你会怎么改进？
- **短期**：引入 RAG 机制，把传热学教材中的公式库和术语定义建成向量数据库，Agent 在选择公式前先检索确认
- **中期**：扩大 Benchmark 到 50+ 题，覆盖更多传热子领域（相变换热、辐射、多孔介质等），做统计显著性分析
- **长期**：探索 Agent 与仿真软件（COMSOL/Fluent）的 API 对接，让 Agent 不只是写 Python 脚本，还能直接操作仿真软件建模求解
