"""
MCP 服务器：把 sciagent.tools 里的工具按 Model Context Protocol 暴露出去。

启动：  python -m sciagent.mcp_server
通过 stdio 跟任何支持 MCP 的宿主（Claude Desktop、Claude Code、Cursor 等）对接。

实现采用 JSON-RPC 2.0 over stdio，遵循 MCP 规范的核心方法：
  - initialize
  - tools/list
  - tools/call

之所以自己实现而不是直接依赖 mcp SDK，是为了减少运行时依赖、
方便在评测环境里直接启动。schema 形状与官方 SDK 保持兼容。
"""

from __future__ import annotations
import sys
import json
import traceback
from typing import Any

from sciagent.tools import TOOL_DEFINITIONS, TOOL_EXECUTORS

# stdio 必须 UTF-8，Windows 默认是 GBK 会把非 ASCII 字段打碎
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "sciagent-thermal"
SERVER_VERSION = "0.2.0"


# ---------------------------------------------------------------------------
# JSON-RPC 基础
# ---------------------------------------------------------------------------

def _write(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _ok(req_id: Any, result: Any) -> None:
    _write({"jsonrpc": "2.0", "id": req_id, "result": result})


def _err(req_id: Any, code: int, message: str, data: Any = None) -> None:
    payload = {"code": code, "message": message}
    if data is not None:
        payload["data"] = data
    _write({"jsonrpc": "2.0", "id": req_id, "error": payload})


# ---------------------------------------------------------------------------
# MCP 方法
# ---------------------------------------------------------------------------

def handle_initialize(params: dict) -> dict:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }


def handle_tools_list(params: dict) -> dict:
    # MCP 的 tools/list 返回 {tools: [{name, description, inputSchema}]}
    tools = []
    for t in TOOL_DEFINITIONS:
        tools.append({
            "name": t["name"],
            "description": t["description"],
            "inputSchema": t["input_schema"],
        })
    return {"tools": tools}


def handle_tools_call(params: dict) -> dict:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if name not in TOOL_EXECUTORS:
        raise ValueError(f"未知工具 '{name}'")

    output = TOOL_EXECUTORS[name](arguments)
    is_error = isinstance(output, str) and output.startswith("ERROR")
    return {
        "content": [{"type": "text", "text": str(output)}],
        "isError": is_error,
    }


DISPATCH = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
}


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def serve() -> None:
    """stdio 上读 JSON-RPC 请求，分发并回写响应。"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _err(None, -32700, f"Parse error: {e}")
            continue

        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        # 通知（没有 id）不需要回包
        is_notification = "id" not in msg

        handler = DISPATCH.get(method)
        if handler is None:
            if not is_notification:
                _err(req_id, -32601, f"Method not found: {method}")
            continue

        try:
            result = handler(params)
            if not is_notification:
                _ok(req_id, result)
        except Exception as e:
            if not is_notification:
                _err(
                    req_id, -32000, f"{type(e).__name__}: {e}",
                    data={"traceback": traceback.format_exc()},
                )


if __name__ == "__main__":
    serve()
