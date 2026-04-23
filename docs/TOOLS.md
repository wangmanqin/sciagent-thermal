# 工具清单（Tool Registry）

所有工具通过 `sciagent.tools.TOOL_DEFINITIONS`（OpenAI / Anthropic 风格 schema）
和 `sciagent.tools.TOOL_EXECUTORS`（纯函数字典）对外暴露。MCP 服务直接复用
这两个结构。

| 类别 | 名称 | 关键参数 | 返回 | 典型来源 |
|---|---|---|---|---|
| **物性** | `water_properties` | T (°C) | ρ, cp, k, μ, Pr | 表格插值 |
|  | `water_properties_extended` | T (°C) | + β, σ, Psat, ν, α | IAPWS 简化表 |
|  | `ethylene_glycol_properties` | T, mass_fraction | ρ, cp, k, μ | ASHRAE 表 |
|  | `air_properties` | T, P | ρ, cp, k, μ, Pr | Sutherland |
|  | `nanofluid_properties` | base, particle, φ, k_model | k_nf, μ_nf | Maxwell / HC / YC |
| **对流** | `dittus_boelter` | Re, Pr, heating | Nu | 光滑管湍流 |
|  | `shah_london` | aspect_ratio | fRe, Nu_T, Nu_H | 矩形 laminar |
|  | `gnielinski` | Re, Pr | Nu | 过渡+湍流 |
|  | `petukhov` | Re, Pr | Nu | 高 Re 修正 |
|  | `sieder_tate` | Re, Pr, μ/μw | Nu | 大粘度温度差 |
|  | `hausen_entry` / `sieder_tate_entry` | Re, Pr, L, D | Nu | 入口段 laminar |
|  | `colburn_j_factor` | Re, Pr | j | 换热器设计常用 |
|  | `churchill_bernstein` | Re, Pr | Nu | 圆柱横掠 |
|  | `zukauskas_tube_bank` | Re, Pr, 排列, n_rows | Nu | 管束 |
|  | `churchill_chu_vertical` | Ra, Pr | Nu | 竖板自然对流 |
|  | `mcadams_horizontal` | Ra, surface | Nu | 水平板 |
| **辐射** | `gray_body_radiation` | T1, T2, ε, A | q | 两灰体平板 |
| **换热器** | `ntu_effectiveness` | NTU, Cr, flow | ε, q | 多流型 |
|  | `lmtd` | ΔT in/out, flow | LMTD, F | 对数平均温差 |
| **无量纲数** | `grashof_number` / `rayleigh_number` | g, β, ΔT, L, ν, α | Gr / Ra | 自然对流 |
|  | `hydraulic_diameter` | A, P 或 w, h | Dh | 通用 |
| **压降** | `laminar_friction_factor` | Re | f=64/Re | 圆管层流 |
|  | `colebrook` | Re, ε/D | f (迭代) | 湍流粗管 |
|  | `swamee_jain` | Re, ε/D | f (显式) | Colebrook 近似 |
|  | `darcy_weisbach` | f, L, D, v, ρ | ΔP | 沿程阻力 |
|  | `minor_loss` | fitting, v, ρ | ΔP | 局部阻力（15 种） |
|  | `borda_carnot_expansion` | v1, A2/A1, ρ | ΔP | 突扩 |
|  | `sudden_contraction` | v2, A2/A1, ρ | ΔP | 突缩 |
|  | `lockhart_martinelli` | x, ρ_L, ρ_G, μ_L, μ_G, regime | φ²L | 两相流 |
|  | `pump_power` | Q, ΔP, η | P | 泵功 |
|  | `rectangular_channel_friction` | aspect_ratio, Re | f | 矩形 fRe |
| **几何** | `rectangular_cross_section` | w, h | A, P, Dh | 通用 |
|  | `circular_cross_section` | D | A, P, Dh | |
|  | `triangular_cross_section` | base, height | A, P, Dh | |
|  | `trapezoidal_cross_section` | a, b, h | A, P, Dh | 梯形通道 |
|  | `channel_array` | n, w_ch, h_ch, s, L | 总宽度, 总面积 | 微通道散热片 |
|  | `fin_array` | n, t, H, L, base | 单片面积, 总面积 | 翅片散热 |
|  | `sphere_volume` / `cylinder_volume` | D (, L) | V | 杂项 |
|  | `fin_efficiency` | m*L | η_fin | 绝热端翅片 |
| **通用** | `run_python_code` | code, timeout_s | stdout, artifacts | sandbox 执行 |
|  | `save_xy_plot` | x, y, path, ... | png path | matplotlib 简易 |

## Tool Schema 约定

每个工具在 `TOOL_DEFINITIONS` 里必须长这样：

```python
{
    "type": "function",
    "function": {
        "name": "<unique_name>",
        "description": "<一句话用途 + 适用范围>",
        "parameters": {
            "type": "object",
            "properties": {
                "<arg>": {"type": "number", "description": "<带单位>"},
                ...
            },
            "required": [...]
        }
    }
}
```

### 命名规范

- 名字全部小写、下划线分隔；物性类加后缀 `_properties`；经验式用人名；
  几何类用动词/名词直白拼写。
- 参数必须带单位后缀（`_m`, `_Pa`, `_C`, `_W`）或在 description 里注明。

### 返回规范

- 数值计算返回 **dict** 或 **dataclass**。键名带单位。
- 包含 `correlation_used` / `applicability` 字段是加分项（Agent 可以向
  用户解释）。
- 出错直接抛异常——MCP 和 ReActWorkflow 都会把异常包成 observation
  喂给模型。

## 如何新增工具

1. 写一个纯函数：`def foo(x: float) -> dict: ...`
2. 在同一个模块写 `TOOL_DEFINITION`（dict）和 `TOOL_EXECUTOR`（dict）。
3. 在 `sciagent/tools/__init__.py` 里汇总到 `TOOL_DEFINITIONS` /
   `TOOL_EXECUTORS`。
4. 写一个 pytest 用例（`tests/test_*.py`）。
5. 不需要改 workflow / llm / mcp_server ——它们只认集合。

## 工具与层级的关系

- **纯工具层**（tools/）：只做一件事，输入输出都是 JSON-可序列化；
- **求解器层**（solvers/）：数值方法（RK4/LU/TDMA），被工具层调用；
- **优化层**（optim/）：multi-run 问题，返回 Pareto/Result；
- **可视化层**（viz/）：出图，返回 PNG 路径。

上层越靠近 LLM，下层越靠近数学。每一层都可以独立跑测试。
