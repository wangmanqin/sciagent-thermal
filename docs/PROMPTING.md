# Prompt 层设计

`sciagent.prompts` 只做一件事：**把 Agent 在这个领域里该知道的、该
遵守的、该怎么挑工具的，全部塞进 system message**。

## 文件结构

```
sciagent/prompts/
├── __init__.py
└── system.py          # SYSTEM_PROMPT, DOMAIN_HINTS, TOOL_USAGE_GUIDE
```

## 三块内容

### 1. `SYSTEM_PROMPT`（Agent 身份与思考范式）

- **角色定位**：你是一个严谨的工程热物理 Agent，专注微通道散热器。
- **思考范式**：遇到数值题先拆解 → 选工具 → 验证 → 组合 → 给结论。
- **输出约定**：数值一律带单位；超出有效范围要明说；不会就直接承认。
- **红线**：不猜、不幻觉、不编数据。

这一段是写给模型看的"性格"，改它等于换一个模型人格。

### 2. `DOMAIN_HINTS`（领域知识兜底）

写进这里的东西有两类：

1. **关联式的适用范围**：Dittus-Boelter 只对 Re>1e4；Shah-London 只
   对 laminar 矩形通道；Churchill-Bernstein 是圆柱横掠等等。
2. **微通道常见陷阱**：入口效应、粗糙度影响、非圆截面 Dh 的算法。

这部分的核心思想是**让模型在选工具前先"想想看适不适用"**。实测下来
比在 TOOL_USAGE_GUIDE 里塞更多工具更有效。

### 3. `TOOL_USAGE_GUIDE`（工具选型决策表）

一张表，左侧是"遇到什么情况"，右侧是"用哪个工具"。例子：

| 情景 | 首选工具 | 备选 |
|---|---|---|
| 圆管湍流内部对流 | dittus_boelter | gnielinski（更准） |
| 矩形通道层流 | shah_london |  |
| 圆柱横掠 | churchill_bernstein |  |
| 整机热阻分析 | 组合 conduction_resistance_* + convection_resistance + caloric_resistance |  |
| 优化换热-压降 | run_nsga2 + pareto utils |  |

这张表是**人在 10 题 benchmark 上迭代出来的**：每次发现模型选错工
具，就在这里加一行。

## 为什么不用更复杂的 Prompt 框架

试过、放弃了的方案：

- **DSPy / Prompt chains**：对单轮交互帮助大，但 ReAct 循环里每轮
  的输入高度同质，复杂 chain 反而让上下文变重。
- **Few-shot demonstration**：写 3 个 ReAct 完整示范塞 system。
  Token 成本太高，且新工具发布后维护示范成了负担。
- **Output format JSON Schema forcing**：OpenAI 风格的 strict mode
  会让中间"思考"阶段的文字被砍掉，模型失去 thinking-out-loud 的
  余地。

当前方案（纯自然语言 prompt + 工具 schema + 决策表）在 10 题 benchmark
上拿到 100% pass / 88.8/100 均分，属于"成本 / 性能最甜的位置"。

## Prompt 和工具层的边界

**Prompt 不做计算**。
**Prompt 不重复工具 schema 里已经写过的参数说明**。
**Prompt 不嵌入任何 API key / 秘密**。

Prompt 层和工具层的 single source of truth 约定：

- 工具的**技术描述**（参数含义 / 返回单位 / 公式出处）→ 写在工具定义
  的 `description` 字段。
- 工具的**选择原则**（什么时候用什么）→ 写在 `TOOL_USAGE_GUIDE`。

这让两者能独立演化：新增工具时只动 tool 定义，决策逻辑变化时只动
Prompt。

## 多语言处理

- 系统提示整体用**中文**写（用户是中文母语，领域术语也是中英混排）。
- 模型回答跟随用户语言（这条规则写进 SYSTEM_PROMPT）。
- 工具参数描述全部**中英双语**，防止 LLM 误解（例：`temperature_C`
  后面跟"温度（摄氏度）"）。

## 快速排障

| 症状 | 大概率位置 |
|---|---|
| Agent 选错了工具 | `TOOL_USAGE_GUIDE` 缺一行 |
| Agent 超出工具适用范围不 warn | `DOMAIN_HINTS` 没覆盖该关联式的 limits |
| Agent 自己瞎编公式 | `SYSTEM_PROMPT` 里的红线要更突出 |
| Agent 回答太啰嗦 | 在 SYSTEM_PROMPT 加"答案精简、不要罗列过程" |
| Agent 语言不对 | SYSTEM_PROMPT 里的语言规则改表述 |

## 评测驱动迭代

每跑一次 `benchmarks/run_bench.py`，我们会：

1. 收集所有失败和低分的 ReAct log
2. 找规律：是选错工具？越界不 warn？公式搞混？
3. 对应改 `DOMAIN_HINTS` / `TOOL_USAGE_GUIDE` / `SYSTEM_PROMPT` 中
   的一条
4. 重跑 benchmark，看均分是否提升

到目前的版本为止，prompt 层迭代了 7 次。
