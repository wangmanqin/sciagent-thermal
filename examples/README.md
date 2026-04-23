# SciAgent-Thermal Examples

10 个可独立运行的示例，从"单个工具调用"到"完整 Agent + 报告"：

| # | 文件 | 讲什么 |
|---|---|---|
| 01 | `01_basic_heat_transfer.py` | 水物性 + Shah-London + h 计算 |
| 02 | `02_pressure_drop.py` | Colebrook → Darcy-Weisbach → Pump power |
| 03 | `03_thermal_network.py` | 导热 + 并联对流 + 热容 的网络 |
| 04 | `04_nsga2_optim.py` | 4 维变量 × 2 目标的 NSGA-II |
| 05 | `05_agent_react.py` | MockLLM + Agent + ReAct 回放 |
| 06 | `06_fd_conduction.py` | 1D 稳态导热 + Thomas 求解 |
| 07 | `07_nanofluid.py` | Al₂O₃ / CuO / CNT 物性增强 |
| 08 | `08_mcp_client_demo.py` | 用 Python 当 MCP 客户端 |
| 09 | `09_sweep_plot.py` | 参数扫描 + matplotlib 多面板 |
| 10 | `10_full_design_report.py` | 综合：候选方案 + Knee + MD 报告 |

## 怎么跑

```bash
cd /path/to/sciagent_2
pip install -r requirements.txt

# 单个跑
python examples/01_basic_heat_transfer.py

# 全跑一遍（需要 deap 装好）
for f in examples/*.py; do
    echo "=== $f ==="
    python "$f" || echo "(failed, skipping)"
done
```

产物落在 `outputs/examples/`。

## 依赖层级

示例相互独立，但 04/10 需要 DEAP，05 需要 `sciagent.llm`，08 会起
子进程（需要能找到 `python` 可执行文件）。

## 从示例回看架构

| 示例 | 用到的层 |
|---|---|
| 01-03, 06-07, 09 | 只用 Tool Layer + Solver Layer |
| 04, 10 | Tool + Optim + Viz |
| 05 | Prompt + Workflow + LLM + Tool（全层 Agent 栈） |
| 08 | MCP Server（把 Tool Layer 暴露给外部） |

要给面试官演示"Agent 的每一层都能单独工作"——这 10 个例子就是最
直接的证据。
