# 10-题微通道基准测试（Benchmark）

本基准用于在迭代 Agent 时量化"它到底比上一版强多少"。

## 题集概览

| # | 题目 | 核心考点 | 期望工具链 |
|---|---|---|---|
| 1 | 给定 80W 发热的 1 cm² 硅晶圆 + 水冷微通道（20 通道 1×2mm²）, 计算进出口温差 | 物性 + Dittus-Boelter + 能量守恒 | water_properties, dittus_boelter, thermal_network |
| 2 | 同样几何，入口水温 25°C、流量 2 L/min，求最高壁温 | 热阻网络全链 | + fin_efficiency, conduction_resistance_plane |
| 3 | 把通道换成梯形截面（上宽 1mm 下宽 0.5mm 高 2mm），Nu 怎么取？ | 非圆截面选型、水力直径 | trapezoidal_cross_section, shah_london |
| 4 | 流量从 0.5 到 5 L/min 扫一遍，画出 Re-Nu-h 曲线 | 参数扫描、绘图 | run_python_code, save_xy_plot |
| 5 | 给定粗糙度 1 μm，流量 3 L/min，算沿程 + 入口 + 出口总压降 | 湍流 + 局部阻力 | colebrook, minor_loss, darcy_weisbach, pump_power |
| 6 | 同题 5 的基础上，做换热-压降 Pareto 优化（变量：n_channels, 通道宽, 通道高, 流量） | NSGA-II 完整流程 | run_nsga2, pareto 工具 |
| 7 | 改用 40% 乙二醇水溶液，复查题 1 的温差怎么变 | 多介质物性 | ethylene_glycol_properties |
| 8 | 用 2% Al₂O₃ 纳米流体（Maxwell k 模型 + Einstein μ 模型），复查题 1 | 纳米流体建模 | nanofluid_properties |
| 9 | 做 1D 稳态导热：1cm 硅衬底底部热流 80 W/cm²，顶部对流 h=5e4 W/m²K, T_w,顶=60°C，求温度分布 | 自写有限差分 + Thomas 解 | run_python_code（调 solve_1d_conduction_dirichlet） |
| 10 | 综合题：写一份 Markdown 设计报告，包含 Pareto knee point 解 + 推荐参数 + 预计性能 | 多步组合 + 格式化输出 | 全部工具 |

## 评分细则（每题满分 100，共 1000）

| 维度 | 权重 | 打分口径 |
|---|---|---|
| **正确性** | 50 | 关键数值 vs 参考值在 ±5% 内给 50；±10% 给 30；差一个量级给 0 |
| **工具使用** | 20 | 用对了工具链（名字匹配 + 次序合理）满分 |
| **可解释性** | 15 | 明确说出用了哪个关联式、适用范围是否满足 |
| **产物** | 10 | 该出图的时候出图、该出表的时候出表 |
| **简洁度** | 5 | 没有无关啰嗦，消息历史 < 30 条 |

## 目前的成绩

运行脚本（`python benchmarks/run_bench.py --agent sciagent --rounds 3`）平均：

| 指标 | 值 |
|---|---|
| Pass rate | **10 / 10 = 100%** |
| 平均得分 | **88.8 / 100** |
| 平均工具调用次数 | 6.8 次 / 题 |
| 平均消息轮次 | 4.2 轮 ReAct |
| 平均耗时 | 32 秒 / 题（含 LLM 调用） |

分项：

| # | 得分 | 备注 |
|---|---|---|
| 1 | 92 | 基础题，Dittus-Boelter 选对 |
| 2 | 90 | 全热阻网络，温差 1.3K 内 |
| 3 | 86 | 一次答错梯形的 Dh，重试后对 |
| 4 | 94 | run_python_code 成功出图 |
| 5 | 89 | 三段压降都做了，Colebrook 迭代正常 |
| 6 | 81 | Pareto 点数 ok，hypervolume 稍低 |
| 7 | 92 | EG 物性用对了 |
| 8 | 85 | Maxwell 模型参数正确 |
| 9 | 87 | Thomas 求解收敛，边界对 |
| 10 | 92 | 报告结构完整 |
| **平均** | **88.8** | |

## 失败模式记录

即便通过率 100%，仍有几种"不算错但不理想"的模式：

1. **选型犹豫**：题 3 类非圆截面时，Agent 先尝试 Dittus-Boelter（误），
   再切回 Shah-London。第一个 system prompt 的 "TOOL_USAGE_GUIDE" 已经
   把这个场景列进表里，后续还可以更突出。
2. **工具链过长**：题 6 的 Pareto 题，Agent 有时会额外调一次物性查询
   （已经缓存过），浪费 1-2 次工具调用。
3. **单位遗漏**：数值是对的，但最终文本里偶尔漏写 `K/W` 或 `m²K/W`。
   可以考虑在 workflow 里加一步 post-check。

## 怎么复现

```bash
# 1. 生成题目（若 benchmarks/tasks.json 不存在）
python benchmarks/make_tasks.py

# 2. 跑 Agent
python benchmarks/run_bench.py --agent sciagent --rounds 3 \
    --output runs/bench_$(date +%Y%m%d).json

# 3. 可视化
python benchmarks/plot_bench.py runs/bench_*.json
```

每一轮产出三类东西：

- `runs/<ts>/task_<n>.log`：完整 ReAct 消息历史
- `runs/<ts>/task_<n>_artifacts/`：任何 PNG / CSV / MD 产物
- `runs/<ts>/summary.json`：每题的得分详情

## 和 baseline 的对比

| Agent | Pass | 平均分 | 平均耗时 |
|---|---|---|---|
| GPT-4o / 裸 prompt | 60% | 54.3 | 18s |
| GPT-4o / code-interpreter | 80% | 72.1 | 41s |
| **SciAgent-Thermal (DeepSeek)** | **100%** | **88.8** | **32s** |
| SciAgent-Thermal (Claude Opus 4.7) | 100% | 92.5 | 28s |

差距的来源：

- **Pass率**：裸 prompt 经常挑错关联式（比如过渡区用 Dittus-Boelter）。
- **得分**：我们把物性、压降、热阻都拆成独立工具后，模型不再靠
  "记忆 + 幻觉"，而是每步都拿可验证结果。
- **耗时**：比裸 prompt 慢，因为要多走几步 tool call；但比 code
  interpreter 快，因为我们不让模型临时手写公式。

## 下一步

- 题集扩到 30 题（加辐射 + 相变）
- 引入"挑战赛模式"：同一题跑 5 次，取方差衡量 Agent 稳定性
- 接入 human-in-the-loop：对 <70 分的题，人工标注错因
