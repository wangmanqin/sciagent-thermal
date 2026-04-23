"""
命令行入口：`python -m sciagent <subcommand>`

子命令：
  ask <question>       一次性问答（走完整 Agent）
  tools list           列出所有工具
  tools call <name>    调单个工具（JSON 参数从 stdin）
  bench run            跑 benchmark
  mcp                  启动 MCP 服务器
"""

from __future__ import annotations
import argparse
import json
import sys
from typing import List


def _cmd_ask(args) -> int:
    from sciagent.agent import Agent
    from sciagent.llm import create_llm

    llm = create_llm(provider=args.provider, model=args.model)
    agent = Agent(llm=llm, max_iterations=args.max_iter)

    events_seen = 0
    def on_event(ev):
        nonlocal events_seen
        events_seen += 1
        if args.verbose or ev.event_type in ("tool_call", "answer", "error"):
            print(f"[{ev.event_type}] {ev.content[:200] if isinstance(ev.content, str) else ev.content}")

    agent.run(args.question, on_event=on_event)
    print(f"\n(events: {events_seen})")
    return 0


def _cmd_tools_list(args) -> int:
    from sciagent.tools import TOOL_DEFINITIONS
    print(f"# {len(TOOL_DEFINITIONS)} tools registered\n")
    for t in TOOL_DEFINITIONS:
        fn = t["function"]
        props = fn["parameters"].get("properties", {})
        print(f"- **{fn['name']}** ({len(props)} args)")
        print(f"    {fn['description'][:80]}")
    return 0


def _cmd_tools_call(args) -> int:
    from sciagent.tools import TOOL_EXECUTORS

    if args.name not in TOOL_EXECUTORS:
        print(f"Unknown tool: {args.name}", file=sys.stderr)
        return 2
    if args.args_json:
        arguments = json.loads(args.args_json)
    elif not sys.stdin.isatty():
        arguments = json.loads(sys.stdin.read() or "{}")
    else:
        arguments = {}
    try:
        result = TOOL_EXECUTORS[args.name](**arguments)
    except TypeError as e:
        print(f"Argument error: {e}", file=sys.stderr)
        return 2
    # 序列化
    try:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    except TypeError:
        print(str(result))
    return 0


def _cmd_bench_run(args) -> int:
    from benchmarks.tasks import all_tasks
    from benchmarks.runner import run_all
    from sciagent.agent import Agent
    from sciagent.llm import create_llm

    llm = create_llm(provider=args.provider, model=args.model)
    agent = Agent(llm=llm)
    tasks = all_tasks()
    if args.task_id:
        tasks = [t for t in tasks if t.id == args.task_id]
    summary = run_all(agent, tasks, output_dir=args.output_dir, verbose=args.verbose)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_mcp(args) -> int:
    # 简单 re-exec 成 `python -m sciagent.mcp_server`
    from sciagent.mcp_server.server import serve_stdio
    serve_stdio()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sciagent",
        description="SciAgent-Thermal CLI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ask
    pa = sub.add_parser("ask", help="问 Agent 一个问题")
    pa.add_argument("question", help="自然语言问题")
    pa.add_argument("--provider", default="mock", choices=["deepseek", "claude", "mock"])
    pa.add_argument("--model", default=None)
    pa.add_argument("--max-iter", type=int, default=10)
    pa.add_argument("--verbose", action="store_true")
    pa.set_defaults(func=_cmd_ask)

    # tools
    pt = sub.add_parser("tools", help="管理工具")
    pt_sub = pt.add_subparsers(dest="tools_cmd", required=True)
    ptl = pt_sub.add_parser("list", help="列出所有工具")
    ptl.set_defaults(func=_cmd_tools_list)
    ptc = pt_sub.add_parser("call", help="调用一个工具")
    ptc.add_argument("name")
    ptc.add_argument("--args-json", default=None, help="JSON 字符串")
    ptc.set_defaults(func=_cmd_tools_call)

    # bench
    pb = sub.add_parser("bench", help="Benchmark 相关")
    pb_sub = pb.add_subparsers(dest="bench_cmd", required=True)
    pbr = pb_sub.add_parser("run", help="跑 benchmark")
    pbr.add_argument("--provider", default="mock")
    pbr.add_argument("--model", default=None)
    pbr.add_argument("--task-id", type=int, default=None,
                     help="只跑指定题号")
    pbr.add_argument("--output-dir", default="runs/latest")
    pbr.add_argument("--verbose", action="store_true")
    pbr.set_defaults(func=_cmd_bench_run)

    # mcp
    pm = sub.add_parser("mcp", help="启动 MCP stdio 服务")
    pm.set_defaults(func=_cmd_mcp)

    return p


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
