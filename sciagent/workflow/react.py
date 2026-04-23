"""
Workflow 层：ReAct 风格的多工具编排。

职责：
  - 把 Prompt 层、Tool 层、LLM 层粘在一起
  - 跑 think → act → observe 循环
  - 统计每一步的事件，交给上层（CLI / 评测脚本）展示

之所以独立成一个模块，是为了让 agent.py 尽量薄 —— 未来要换成
Plan-and-Execute、多 Agent 协作或别的 workflow 时，只改这里。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class AgentEvent:
    event_type: str                    # thinking | tool_call | tool_result | answer | error
    content: str
    metadata: dict = field(default_factory=dict)


class ReActWorkflow:
    def __init__(
        self,
        llm,
        tool_definitions: list,
        tool_executors: dict,
        system_prompt: str,
        max_iterations: int = 10,
    ):
        self.llm = llm
        self.tool_definitions = tool_definitions
        self.tool_executors = tool_executors
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        self.messages: list = []
        self.events: list[AgentEvent] = []

    # ---- 事件辅助 ----
    def _emit(self, on_event: Optional[Callable], event_type: str, content: str, **metadata):
        ev = AgentEvent(event_type, content, metadata)
        self.events.append(ev)
        if on_event:
            on_event(ev)

    # ---- 工具执行 ----
    def _execute_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        executor = self.tool_executors.get(name)
        if not executor:
            return f"ERROR: 未知工具 '{name}'", True
        try:
            out = executor(arguments)
            return out, str(out).startswith("ERROR")
        except Exception as e:
            return f"ERROR: 工具执行异常 — {type(e).__name__}: {e}", True

    # ---- 主循环 ----
    def run(self, user_query: str, on_event: Optional[Callable] = None) -> str:
        self.events = []
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]

        self._emit(on_event, "thinking", "正在分析问题...")

        for iteration in range(self.max_iterations):
            response = self.llm.chat(self.messages, tools=self.tool_definitions)

            if response.text:
                evtype = "thinking" if response.has_tool_calls else "answer"
                self._emit(on_event, evtype, response.text)

            if not response.has_tool_calls:
                return response.text

            # assistant 消息：text + tool_use
            assistant_blocks = []
            if response.text:
                assistant_blocks.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            self.messages.append({"role": "assistant", "content": assistant_blocks})

            # 执行工具
            tool_results = []
            for tc in response.tool_calls:
                preview_code = tc.arguments.get("code", "") if isinstance(tc.arguments, dict) else ""
                self._emit(
                    on_event, "tool_call",
                    f"调用工具: {tc.name}",
                    code=preview_code, args=tc.arguments,
                )

                result, is_error = self._execute_tool(tc.name, tc.arguments)
                if is_error:
                    self._emit(on_event, "error", f"{tc.name} 失败，Agent 将尝试修正...\n{result}")
                else:
                    self._emit(on_event, "tool_result", result, tool=tc.name)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                    "is_error": is_error,
                })

            self.messages.append({"role": "user", "content": tool_results})

        self._emit(
            on_event, "error",
            f"Agent 已执行 {self.max_iterations} 轮仍未完成，强制停止。",
        )
        return "抱歉，问题处理未能在限定步数内完成。请尝试简化问题描述。"
