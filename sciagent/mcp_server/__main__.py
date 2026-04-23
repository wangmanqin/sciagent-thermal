"""允许 `python -m sciagent.mcp_server` 直接启动 MCP 服务。"""

from sciagent.mcp_server.server import serve

if __name__ == "__main__":
    serve()
