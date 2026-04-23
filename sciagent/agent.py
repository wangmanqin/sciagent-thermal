"""
Agent 门面：把 Prompt / Tool / Workflow 三层组装起来。

对外用法一行：
    Agent().run("你的问题", on_event=print_event)

Agent 本身不含业务逻辑，所有 ReAct 循环在 workflow 层，
所有工具在 tool 层，所有行为约束在 prompt 层。
"""

from __future__ import annotations
from typing import Callable, Optional

from sciagent.llm import create_llm
from sciagent.tools import TOOL_DEFINITIONS, TOOL_EXECUTORS
from sciagent.prompts import build_system_prompt
from sciagent.workflow import ReActWorkflow, AgentEvent

MAX_ITERATIONS = 10


class Agent:
    def __init__(
        self,
        llm_mode: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.llm = create_llm(llm_mode)
        self.system_prompt = system_prompt or build_system_prompt()
        self.workflow = ReActWorkflow(
            llm=self.llm,
            tool_definitions=TOOL_DEFINITIONS,
            tool_executors=TOOL_EXECUTORS,
            system_prompt=self.system_prompt,
            max_iterations=max_iterations,
        )

    @property
    def events(self) -> list[AgentEvent]:
        return self.workflow.events

    @property
    def messages(self) -> list:
        return self.workflow.messages

    def run(self, user_query: str, on_event: Optional[Callable] = None) -> str:
        return self.workflow.run(user_query, on_event=on_event)
