"""
LLM 调用封装层。
支持三种模式：mock（无需 API Key）、deepseek、claude。
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()


class LLMResponse:
    """统一的 LLM 响应格式"""

    def __init__(self, text: str = "", tool_calls: list = None, stop_reason: str = "end_turn"):
        self.text = text
        self.tool_calls = tool_calls or []
        self.stop_reason = stop_reason

    @property
    def has_tool_calls(self):
        return len(self.tool_calls) > 0


class ToolCall:
    """一次工具调用"""

    def __init__(self, id: str, name: str, arguments: dict):
        self.id = id
        self.name = name
        self.arguments = arguments


# ========== Mock 模式 ==========

class MockLLM:
    """Mock LLM，用于在没有 API Key 时测试 Agent 循环。"""

    def __init__(self):
        self._call_count = 0

    def chat(self, messages: list, tools: list = None) -> LLMResponse:
        self._call_count += 1

        if self._call_count == 1:
            code = (
                "import numpy as np\n"
                "\n"
                "primes = [n for n in range(2, 101) if all(n % i != 0 for i in range(2, int(n**0.5)+1))]\n"
                "print(f'2到100之间共有 {len(primes)} 个素数')\n"
                "print(f'素数列表: {primes}')\n"
            )
            return LLMResponse(
                text="我来写一段代码计算素数。",
                tool_calls=[
                    ToolCall(id="mock_1", name="run_python_code", arguments={"code": code})
                ],
                stop_reason="tool_use",
            )

        return LLMResponse(
            text=(
                "代码执行成功！\n\n"
                "**结果分析**：2到100之间共有25个素数。\n\n"
                "（这是 mock 模式的预设回答。接入真实 API 后会根据执行结果生成分析。）"
            ),
            stop_reason="end_turn",
        )


# ========== DeepSeek API 模式（OpenAI 兼容格式） ==========

class DeepSeekLLM:
    """
    调用 DeepSeek API。
    DeepSeek 使用 OpenAI 兼容格式，所以用 openai SDK 调用。
    """

    def __init__(self, model: str = "deepseek-chat"):
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self.model = model

    def chat(self, messages: list, tools: list = None) -> LLMResponse:
        # 转换 tools 为 OpenAI 格式
        oai_tools = None
        if tools:
            oai_tools = []
            for t in tools:
                oai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                })

        # 转换消息格式（把 system 消息保留在 messages 里即可，OpenAI 格式支持）
        oai_messages = []
        for msg in messages:
            if msg["role"] == "system":
                oai_messages.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "user":
                # 可能是普通文字，也可能是 tool_result 列表
                content = msg["content"]
                if isinstance(content, list):
                    # 这是 tool_result，转换为 OpenAI 的 tool message 格式
                    for item in content:
                        if item.get("type") == "tool_result":
                            oai_messages.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item["content"],
                            })
                else:
                    oai_messages.append({"role": "user", "content": content})
            elif msg["role"] == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # assistant 消息可能包含 text + tool_use
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"]),
                                },
                            })
                    assistant_msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    oai_messages.append(assistant_msg)
                else:
                    oai_messages.append({"role": "assistant", "content": content})

        kwargs = {
            "model": self.model,
            "messages": oai_messages,
            "max_tokens": 4096,
            "timeout": 120,  # 120秒超时，防止长时间无响应
        }
        if oai_tools:
            kwargs["tools"] = oai_tools

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # 解析响应
        text = choice.message.content or ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                # DeepSeek 有时返回不合法的 JSON，做容错处理
                raw = tc.function.arguments
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    # 尝试修复：可能是未转义的换行符等
                    args = {"code": raw}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason="tool_calls" if tool_calls else "stop",
        )


# ========== Claude API 模式 ==========

class ClaudeLLM:
    """调用 Claude API（anthropic SDK）"""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def chat(self, messages: list, tools: list = None) -> LLMResponse:
        api_tools = []
        if tools:
            for t in tools:
                api_tools.append({
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["input_schema"],
                })

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if api_tools:
            kwargs["tools"] = api_tools

        system_msg = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                filtered_messages.append(msg)

        if system_msg:
            kwargs["system"] = system_msg
            kwargs["messages"] = filtered_messages

        response = self.client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
        )


# ========== 工厂函数 ==========

def create_llm(mode: str = None):
    """
    创建 LLM 实例。
    - mode="mock"：Mock 模式
    - mode="deepseek"：DeepSeek API
    - mode="claude"：Claude API
    - mode=None（默认）：自动检测可用的 API Key
    """
    if mode == "mock":
        print("[LLM] 使用 Mock 模式")
        return MockLLM()

    if mode == "deepseek":
        print("[LLM] 使用 DeepSeek API")
        return DeepSeekLLM()

    if mode == "claude":
        print("[LLM] 使用 Claude API")
        return ClaudeLLM()

    # 自动检测：优先 DeepSeek（国内可用），其次 Claude
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    if ds_key and ds_key != "your-key-here":
        print("[LLM] 检测到 DeepSeek API Key")
        return DeepSeekLLM()

    cl_key = os.getenv("ANTHROPIC_API_KEY", "")
    if cl_key and cl_key != "your-key-here":
        print("[LLM] 检测到 Claude API Key")
        return ClaudeLLM()

    print("[LLM] 未检测到 API Key，使用 Mock 模式")
    return MockLLM()
