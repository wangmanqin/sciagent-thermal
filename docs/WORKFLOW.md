# Workflow 层：ReAct 详解

`sciagent.workflow.react.ReActWorkflow` 是把 "LLM thinking ↔ 工具执行 ↔
观察结果" 这个循环具象化成代码的 ~150 行。

## 核心类图

```
┌────────────────────────┐
│  ReActWorkflow         │
│  ────────────────      │
│  - llm                 │
│  - tool_definitions    │
│  - tool_executors      │
│  - max_iterations      │
│  - messages: list      │
│  - events: list        │
│                        │
│  + run(query, on_event)│
│  + _execute_tool(...)  │
└──────────┬─────────────┘
           │ 每步产生
           ▼
     ┌────────────┐
     │ AgentEvent │
     │   ──────   │
     │ event_type │  "thinking" | "tool_call" |
     │ content    │  "tool_result" | "answer" | "error"
     │ metadata   │
     └────────────┘
```

## 单轮循环

```
LLM(messages, tool_definitions)
      │
      ├── tool_calls == []   ──▶ event: answer  → break
      │
      └── tool_calls == [...]
            │
            ├── emit thinking event (content 部分)
            ├── for each tc:
            │     emit tool_call event
            │     result = execute_tool(tc.name, tc.arguments)
            │     emit tool_result event
            │     append to messages as "tool" role
            └── continue to next iteration
```

## AgentEvent 类型

| `event_type` | `content` | `metadata` |
|---|---|---|
| `thinking` | LLM 本轮的自然语言输出 | `{iteration: n}` |
| `tool_call` | 工具名 | `{id, arguments}` |
| `tool_result` | 工具返回（JSON 字符串） | `{tool_name, id, duration_ms}` |
| `answer` | 最终答案 | `{n_tool_calls, n_iterations}` |
| `error` | 错误消息 | `{phase: "tool_call" | "llm" | "sandbox"}` |

这一组事件**就是** Agent 对外能产出的全部"过程"。你在 UI 里看到
的打字流式效果、工具使用提示、动画 loading ——都从订阅这一组事件
衍生出来。

## 为什么有 `on_event` 回调

`ReActWorkflow.run()` 同步执行但支持回调——调用方可以：

```python
wf = ReActWorkflow(llm=llm, tool_definitions=..., tool_executors=...)
events = []
wf.run("优化我的微通道", on_event=events.append)

# 或者 live 打印：
def live(ev):
    print(f"[{ev.event_type}] {ev.content[:80]}")
wf.run("...", on_event=live)
```

这样做的好处：**同一个接口既能做批处理又能做实时 UI**，不用改核心
逻辑。

## 为什么默认 `max_iterations=10`

经验值：微通道题 8 步工具链就封顶，再多说明模型在兜圈子（通常是
把已有结果当"还需要查一下"）。10 给个余量，同时防止无限循环。

达到上限后会 emit `error` 事件并终止——**不会伪装成成功答案**。

## 消息历史的结构

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT + DOMAIN_HINTS + ...},
    {"role": "user", "content": "<user query>"},
    # --- iter 1 ---
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "c1", "content": "<json result>"},
    {"role": "tool", "tool_call_id": "c2", "content": "<json result>"},
    # --- iter 2 ---
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", ...},
    # --- done ---
    {"role": "assistant", "content": "最终答案...", "tool_calls": []},
]
```

这个结构 DeepSeek / OpenAI / Anthropic 都能喂 ——我们只在 `llm.py`
里适配一次格式差异。

## 错误处理策略

**工具报错 → 作为 observation 喂回模型**。我们故意不中断循环，因为
模型通常能看到 error 后自己改参数。

例子：

```
tool_call: water_properties(temperature_C=150)
tool_result: Error: out of valid range [20, 80] °C
assistant: 啊，这个工具不支持 150°C，我改用 water_properties_extended
tool_call: water_properties_extended(temperature_C=150)
tool_result: Error: out of valid range [0, 100] °C
assistant: 那用户说的 150°C 超出了我所有物性工具的有效范围，我应该
           反过来问用户具体场景是什么...
```

**LLM 报错 → 中断，emit `error` 事件**。这种错误往往是 rate limit /
network，重试逻辑让调用方决定。

**Sandbox 拦截 → 同工具报错**。模型通常会换个合法写法重试。

## 设计权衡

| 设计选择 | 原因 |
|---|---|
| 同步 API | 便于 debug、便于接 CLI；需要流式时用 on_event |
| 不用 asyncio | 工具调用基本是 CPU-bound，异步只是增加复杂度 |
| messages 全保留 | context 允许，短对话看得起 |
| 工具并发 = 否 | 同一轮的 tool_calls 顺序执行（简单可预测） |

真上生产需要并发工具调用时再加 `asyncio.gather`，这里不做。
