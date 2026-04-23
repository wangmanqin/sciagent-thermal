"""
Example 08: 直接用 Python 做 MCP 客户端。

启动 sciagent.mcp_server 作为子进程，通过 stdio JSON-RPC 调工具。

这个示例演示 Claude Desktop / Cursor 内部是怎么调 MCP server 的。
"""

import json
import subprocess
import sys


def _send(proc, req):
    line = json.dumps(req, ensure_ascii=False) + "\n"
    proc.stdin.write(line.encode("utf-8"))
    proc.stdin.flush()


def _recv(proc):
    line = proc.stdout.readline().decode("utf-8", errors="replace")
    return json.loads(line) if line.strip() else None


def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "sciagent.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # 1. initialize
        _send(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "demo", "version": "0.0.1"},
            }
        })
        r = _recv(proc)
        print("initialize:", r.get("result", {}).get("serverInfo"))

        # 2. tools/list
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        r = _recv(proc)
        tools = r["result"]["tools"]
        print(f"{len(tools)} tools available")
        for t in tools[:6]:
            print(f"  - {t['name']}: {t['description'][:50]}...")

        # 3. tools/call water_properties
        _send(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "water_properties",
                "arguments": {"temperature_C": 50.0},
            },
        })
        r = _recv(proc)
        content = r["result"]["content"][0]["text"]
        print("water_properties(50°C):", content[:200])

    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
