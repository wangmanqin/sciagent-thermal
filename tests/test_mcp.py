"""MCP 服务器 JSON-RPC 协议测试。

这些测试直接调用 handler 函数，不起子进程。

真正的端到端（subprocess stdio）测试放在 tests/test_mcp_e2e.py。
"""

from __future__ import annotations
import json


def _call(method: str, params: dict = None, id_: int = 1) -> dict:
    """直接调用 server 的 handle() 函数。"""
    from sciagent.mcp_server.server import handle_request
    req = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        req["params"] = params
    return handle_request(req)


def test_initialize_returns_protocol_version():
    resp = _call("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0.0.1"},
    })
    assert resp["jsonrpc"] == "2.0"
    assert "result" in resp
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "sciagent-thermal"


def test_tools_list_returns_nonempty():
    resp = _call("tools/list", {})
    assert "result" in resp
    tools = resp["result"]["tools"]
    assert isinstance(tools, list)
    assert len(tools) > 10  # 远大于 10 个工具


def test_tools_list_schema_shape():
    resp = _call("tools/list", {})
    tools = resp["result"]["tools"]
    for t in tools[:5]:
        assert "name" in t
        assert "description" in t
        assert "inputSchema" in t
        assert t["inputSchema"]["type"] == "object"


def test_tools_call_water_properties():
    resp = _call("tools/call", {
        "name": "water_properties",
        "arguments": {"temperature_C": 25},
    })
    assert "result" in resp
    content = resp["result"]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    data = json.loads(content[0]["text"])
    # 应包含 density 等字段
    assert "density" in data or "density_kg_per_m3" in data or "rho" in data


def test_tools_call_unknown_returns_error():
    resp = _call("tools/call", {
        "name": "no_such_tool_xyz",
        "arguments": {},
    })
    # 协议上错误可以是 result.isError 或 error
    if "error" in resp:
        assert resp["error"]["code"] != 0
    else:
        assert resp["result"].get("isError") is True


def test_unknown_method_returns_error():
    resp = _call("nosuch/method", {})
    assert "error" in resp
    # JSON-RPC method not found = -32601
    assert resp["error"]["code"] == -32601


if __name__ == "__main__":
    import sys
    fns = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed.")
    sys.exit(0 if passed == len(fns) else 1)
