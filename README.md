# SciAgent: An AI Agent Framework for Scientific Computing (AI4S)

SciAgent is an AI Agent framework that solves microchannel heat sink design
problems from a single natural-language prompt. It follows a
**Prompt–Tool–Workflow** three-layer architecture, exposes its scientific
tools over the **Model Context Protocol (MCP)**, orchestrates multi-tool
collaboration via a **Workflow** layer, and executes model-generated code
inside an **AST whitelist + subprocess** **Sandbox**.

- Benchmark: 10-task microchannel heat-sink evaluation set
- Pass rate: **10/10 (100%)**
- Average score: **88.8 / 100**

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         SciAgent                               │
│                                                                │
│   ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│   │   Prompt    │   │   Workflow   │   │      Tools       │    │
│   │   layer     │──▶│  ReAct loop  │◀─▶│ (via MCP server) │    │
│   │ system.py   │   │   react.py   │   │                  │    │
│   └─────────────┘   └──────┬───────┘   └──────┬───────────┘    │
│                            │                  │                │
│                            ▼                  ▼                │
│                      ┌──────────┐      ┌──────────────┐        │
│                      │   LLM    │      │   Sandbox    │        │
│                      │ llm.py   │      │ ast_whitelist│        │
│                      └──────────┘      │ + subprocess │        │
│                                        └──────────────┘        │
└────────────────────────────────────────────────────────────────┘
```

### Layers

| Layer | Module | Responsibility |
|------|--------|----------------|
| **Prompt** | [sciagent/prompts/](sciagent/prompts/) | System prompt + domain hints + tool-selection guide, assembled on demand. |
| **Tool**  | [sciagent/tools/](sciagent/tools/) | Validated domain tools (water properties, Shah-London Nu/fRe, Dittus-Boelter, hydraulic diameter, fin efficiency) + generic tools (`run_python_code`, `save_xy_plot`). |
| **Workflow** | [sciagent/workflow/react.py](sciagent/workflow/react.py) | ReAct (think → act → observe) loop that orchestrates multi-tool collaboration end-to-end, emits per-step events, caps iterations. |

### MCP server

[sciagent/mcp_server/](sciagent/mcp_server/) publishes the entire Tool layer over the Model Context Protocol (JSON-RPC 2.0 over stdio, `initialize` / `tools/list` / `tools/call`). The same tool set is usable from any MCP host (Claude Desktop, Claude Code, Cursor, …):

```bash
python -m sciagent.mcp_server
```

### Sandbox

[sciagent/sandbox/ast_whitelist.py](sciagent/sandbox/ast_whitelist.py) performs static analysis on every code snippet before execution:
- Imports must be in a whitelist (numpy, scipy, matplotlib, deap, pandas, stdlib math/stats).
- Forbidden names: `eval`, `exec`, `compile`, `__import__`, `open`, `input`.
- Forbidden attr calls: `os.system`, `subprocess.*`, `socket.*`, network libs.
- Dangerous dunder access (`__class__`, `__globals__`, …) is blocked.

Code that passes the static check is then run in an isolated `subprocess` with a 60-second timeout, so a crash or infinite loop in generated code cannot take down the host process.

## Evaluation

Benchmark: 10 microchannel-heat-sink tasks, covering multi-step reasoning,
parameter calculations, and tool-call chains.

| Metric | Result |
|--------|--------|
| Pass rate | **10/10 (100%)** |
| Average score | **88.8 / 100** |
| Avg. iterations per task | 4.0 |
| Avg. time per task | 260.7 s |

Dimensions covered:
- Heat-transfer fundamentals (fin efficiency, Nu=3.66 verification)
- Fluid mechanics (hydraulic diameter, friction factor)
- Microchannel thermal analysis (pressure drop, thermal-resistance network)
- Parametric study (Nu vs aspect ratio)
- Single- and multi-objective optimization (NSGA-II)
- Comprehensive design

See [eval_reports/BENCHMARK_README.md](eval_reports/BENCHMARK_README.md) and the latest report [eval_reports/eval_report_20260409_212803.md](eval_reports/eval_report_20260409_212803.md).

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # put your DEEPSEEK_API_KEY inside

# CLI chat
python main.py

# Run the full benchmark
python evaluate.py

# Boot the MCP server (for Claude Desktop / Cursor / ...)
python -m sciagent.mcp_server
```

## Repository Layout

```
sciagent_2/
├── sciagent/
│   ├── prompts/          # Prompt layer
│   ├── tools/            # Tool layer (7 tools)
│   ├── workflow/         # Workflow layer (ReAct)
│   ├── sandbox/          # AST whitelist + subprocess isolation
│   ├── mcp_server/       # MCP protocol server
│   ├── llm.py            # Unified DeepSeek / Claude / Mock wrapper
│   └── agent.py          # Thin facade composing the three layers
├── benchmark.json        # 10-task evaluation set
├── evaluate.py           # Benchmark runner & scorer
├── main.py               # CLI entry point
└── eval_reports/         # Auto-generated evaluation reports
```

## License

MIT
