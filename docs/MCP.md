# MCP 接入文档

SciAgent-Thermal 把工具层重新打包成一个符合 **Model Context Protocol**
（Anthropic 2024-11-05 规范）的服务，跑在 stdio 通道上。

## 一分钟上手

```bash
python -m sciagent.mcp_server
```

启动后进程会在 stdin 等 JSON-RPC 2.0 消息，在 stdout 返回响应。

## 协议细节

- **编码**：stdin/stdout 强制 UTF-8（Windows 上容易踩 GBK 的坑，我们
  在 `server.py` 里手动 `reconfigure(encoding="utf-8", errors="replace")`）。
- **帧格式**：每条消息一行（`\n` 结尾），格式为 JSON-RPC 2.0。
- **协议版本**：`2024-11-05`
- **服务信息**：name = `sciagent-thermal`, version = `0.2.0`

### 支持的方法

| Method | Params | Result |
|---|---|---|
| `initialize` | `{protocolVersion, capabilities, clientInfo}` | `{protocolVersion, capabilities, serverInfo}` |
| `tools/list` | `{}` | `{tools: [{name, description, inputSchema}, ...]}` |
| `tools/call` | `{name, arguments}` | `{content: [{type: "text", text: <json>}]}` |

未知方法返回 JSON-RPC 错误码 `-32601`（method not found）。

### `tools/call` 错误约定

- 参数校验失败 / 工具内部异常：返回 `{result: {content: [...], isError: true}}`，
  `text` 里放出错信息，**不**走 `error` 对象（符合 MCP 客户端习惯）。
- 协议级错误（比如 method not found）才走 `error` 对象。

## 在 Claude Desktop 中接入

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`
（Windows：`%APPDATA%\Claude\claude_desktop_config.json`），加入：

```json
{
  "mcpServers": {
    "sciagent-thermal": {
      "command": "python",
      "args": ["-m", "sciagent.mcp_server"],
      "cwd": "/absolute/path/to/sciagent_2"
    }
  }
}
```

重启 Claude Desktop 后，在对话框里就能看到"sciagent-thermal"作为
工具来源。

## 在 Cursor / Windsurf 中接入

编辑 `.cursor/mcp.json` 或 `.codeium/windsurf/mcp.json`：

```json
{
  "mcpServers": {
    "sciagent-thermal": {
      "command": "python",
      "args": ["-m", "sciagent.mcp_server"]
    }
  }
}
```

## 客户端调用示例

初始化握手：

```json
→ {"jsonrpc":"2.0","id":1,"method":"initialize",
   "params":{"protocolVersion":"2024-11-05",
             "capabilities":{},
             "clientInfo":{"name":"demo","version":"0.1"}}}

← {"jsonrpc":"2.0","id":1,
   "result":{"protocolVersion":"2024-11-05",
             "capabilities":{"tools":{}},
             "serverInfo":{"name":"sciagent-thermal","version":"0.2.0"}}}
```

列工具：

```json
→ {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}

← {"jsonrpc":"2.0","id":2,
   "result":{"tools":[
     {"name":"water_properties",
      "description":"查询水在指定温度下的物性...",
      "inputSchema":{"type":"object", ...}},
     ...
   ]}}
```

调工具：

```json
→ {"jsonrpc":"2.0","id":3,"method":"tools/call",
   "params":{"name":"water_properties","arguments":{"temperature_C":25}}}

← {"jsonrpc":"2.0","id":3,
   "result":{"content":[{"type":"text",
     "text":"{\"density\":997.05,\"specific_heat\":4180.3,...}"}]}}
```

## 为什么 MCP 还值得做一层

已经有 `ReActWorkflow` 了，为什么还暴露 MCP？

1. **解耦**：MCP 客户端（Claude Desktop, Cursor）不需要知道 ReAct
   循环、不需要知道我们的 LLM 抽象，只需要拿到"44 个可调函数"。
2. **复用**：任何符合 MCP 的 Agent（比如 Anthropic 自己的 Agent SDK）
   都能直接用上这个工具集。
3. **标准化展示**：面试里展示"这是一个 MCP server"比展示一个自研
   协议更有说服力——它说明我们了解行业当前在推什么。

简而言之：**ReActWorkflow 是"我们这个 Agent 怎么用工具"；MCP 是
"工具如何被任何 Agent 使用"**。

## 已知限制

- 当前 server 不实现 `resources/*` 和 `prompts/*` 能力——只有 tools。
- 不支持双向流式工具调用（MCP spec 里的 progress / cancel）。
- 日志默认写到 stderr，不走 MCP 的 `notifications/message`——这是故意
  保持 server.py 简单。

## 端到端验证

```bash
# 在一个终端启动 server
python -m sciagent.mcp_server

# 在另一个终端（或同一个用管道）发送 JSON-RPC
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
  | python -m sciagent.mcp_server
```

`tests/test_mcp.py` 里用直调 `handle_request()` 的方式测了所有三个
方法，不起子进程，跑得快。
