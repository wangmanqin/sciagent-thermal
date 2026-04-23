# SciAgent-Thermal 架构设计

## 设计目标

SciAgent-Thermal 是一个面向工程热物理（尤其是微通道散热器）的 Agent。它的
核心要解决的问题是：

> **让大模型在受控环境中调用可验证的科学计算工具，而不是自己"脑补"公式。**

从这个目标出发，整套系统被切成三层，每一层都可以单独测试 / 单独替换。

```
┌─────────────────────────────────────────────────────────┐
│                   User Question                        │
│           (e.g. "80W 1cm² CPU 怎么设计微通道")           │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Prompt Layer  (sciagent/prompts)                      │
│   - SYSTEM_PROMPT  : Agent 身份与思考范式               │
│   - DOMAIN_HINTS   : 微通道 / 热阻网络 / 压降           │
│   - TOOL_USAGE_GUIDE : 工具选型表                       │
└─────────────────────────┬───────────────────────────────┘
                          │  system + user messages
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Workflow Layer  (sciagent/workflow)                   │
│   ReActWorkflow.run():                                  │
│     while not done and n < max_iterations:              │
│       think ── llm.chat(history, tools) ──┐             │
│         │                                 │             │
│         ├─ if tool_calls:  act ──────────▶│             │
│         │    └─ execute_tool(name, args)  │             │
│         │    └─ observe tool_result ──────┘             │
│         └─ else: final answer                           │
└──────┬──────────────────────┬──────────────┬────────────┘
       │                      │              │
       ▼                      ▼              ▼
┌────────────────┐  ┌──────────────────┐ ┌──────────────┐
│  Tool Layer    │  │  Sandbox         │ │  LLM Backend │
│ (44+ tools)    │  │ (AST + subproc)  │ │ DeepSeek /   │
│                │  │                  │ │ Claude / Mock│
│ fluid_props    │  │  ast_whitelist   │ │              │
│ correlations   │  │  - ALLOWED_IMPTS │ │ Unified API: │
│ heat_transfer  │  │  - FORBIDDEN_*   │ │ LLMResponse  │
│ pressure_drop  │  │  subprocess_run  │ │ ToolCall     │
│ geometry       │  │  - 60s timeout   │ │              │
│ properties/*   │  │  - stdout capture│ │              │
│ python_exec ───┼──▶│                  │ │              │
│ plotter        │  └──────────────────┘ └──────────────┘
└────────────────┘
```

## 三层的边界

| 层 | 负责什么 | 不负责什么 |
|---|---|---|
| **Prompt** | 告诉模型它是谁、能做什么、领域里有哪些陷阱 | 任何计算 |
| **Workflow** | 调度 think/act/observe 循环，记录历史 | 具体公式、具体 LLM |
| **Tool** | 给定输入返回可验证结果 | 自由发挥、猜测 |
| **Sandbox** | 拦住危险操作、给脚本代码一块干净地 | 替模型写代码 |
| **LLM** | 生成下一步（thinking + optional tool_calls） | 自己执行工具 |

每一层都可以换：换 LLM 提供商只改 `llm.py`；增加新领域工具只在 `tools/`
下加文件；想换工作流（比如换成 Plan-and-Execute）只需实现一个新的
Workflow 类。

## 为什么是 ReAct 而不是 Plan-and-Execute

微通道散热设计的典型问题链是：

1. 先算水物性（tool A）
2. 再算 Re / Nu（tool B，依赖 A 的 ν, k, Pr）
3. 根据 Nu 算 h（tool C，依赖 B 和几何参数）
4. 组装热阻网络（tool D）
5. 如果温差超限 → 回到 step 1 改条件

这种链条高度依赖中间结果，提前规划容易出错；**"一步一看"的 ReAct 正好
适配**。我们用 `max_iterations=10` 兜底，避免死循环。

## MCP 面的职责

`sciagent/mcp_server/` 把工具层重新暴露成标准 **Model Context Protocol**
服务（JSON-RPC 2.0 over stdio）。这样：

- Claude Desktop / Cursor / Windsurf 等 MCP 客户端可以直接接入；
- 整个科学计算栈解耦，不依赖我们自己的 ReActWorkflow。

MCP handler 三件套：`initialize`, `tools/list`, `tools/call`。详见
[MCP.md](MCP.md)。

## Sandbox 的两道防线

1. **AST 白名单**（`sciagent/sandbox/ast_whitelist.py`）：解析代码树，
   拒绝 `eval/exec/open/__import__`，拒绝 `os.system/subprocess/socket`，
   限制 `import` 到数值计算必需的少数几个包。
2. **子进程隔离 + 超时**：通过白名单的代码会被写入临时文件，用
   独立 Python 解释器运行，60 秒超时，UTF-8 强制编码，stdout/stderr 捕获。

两道防线叠加：第一道拦 90% 的恶意意图，第二道兜住环境破坏和资源耗尽。

## 数据流的"三种粒度"

- **消息级**：`llm.chat(messages, tools)` 的单次问答
- **循环级**：`ReActWorkflow.run(user_query)` 的完整一轮 think-act-observe
- **会话级**：`Agent.run(user_query)` 对外的一次问答

上层不需要知道下层的复杂度——这是为什么 `Agent` 的接口只有两个：`run()`
和 `events` 属性。

## 目录总览

```
sciagent/
├── agent.py            # 门面，组合 LLM+Tools+Prompts+Workflow
├── llm.py              # 统一 LLM 抽象（DeepSeek / Claude / Mock）
├── prompts/            # Prompt 层
│   └── system.py
├── workflow/           # Workflow 层（ReAct）
│   └── react.py
├── tools/              # Tool 层
│   ├── fluid_properties.py
│   ├── correlations.py
│   ├── heat_transfer.py
│   ├── pressure_drop.py
│   ├── geometry.py
│   ├── python_exec.py
│   ├── plotter.py
│   └── properties/
│       ├── water_iapws.py
│       ├── ethylene_glycol.py
│       ├── air.py
│       └── nanofluids.py
├── solvers/            # 求解器（底层数值）
│   ├── thermal_network.py
│   ├── ode.py
│   └── linalg.py
├── optim/              # 优化器
│   ├── nsga2.py
│   ├── single_objective.py
│   └── pareto.py
├── viz/                # 可视化
│   ├── pareto_plot.py
│   ├── convergence.py
│   ├── heatmap.py
│   ├── report.py
│   └── style.py
├── sandbox/            # 沙箱
│   └── ast_whitelist.py
└── mcp_server/         # MCP 服务
    ├── server.py
    └── __main__.py
```

## 和传统"LangChain 一把梭"的区别

|  | SciAgent-Thermal | LangChain 风格 |
|---|---|---|
| 工具定义 | 显式 schema + 纯函数 executor | `@tool` 装饰器混在业务代码里 |
| 工作流 | 手写 ReAct，100 行可读 | AgentExecutor 黑盒 |
| Sandbox | AST + subprocess 两道防线 | 通常只有一层或没有 |
| MCP | 原生支持 | 需要额外适配 |
| 依赖 | 可选（LLM 才需要 requests/anthropic） | 一装全装 |

优先清晰、可测、好解释，而不是"最少代码"。这让它更适合面向博士申请的
作品集——每一层都能在面试中展开讲。
