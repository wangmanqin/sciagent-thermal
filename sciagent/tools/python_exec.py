"""
Python 代码执行工具（被动过 sandbox 的壳）。

所有 Agent 写的代码都先走 sciagent.sandbox 做静态审查，审查通过后再
用独立 subprocess 执行。这样做有两层保险：
  1. AST 白名单 — 拦截禁用的模块/调用（os.system、eval、网络库等）
  2. 子进程隔离 — 即便审查漏过，崩溃/超时也不会拖垮主进程
"""

from __future__ import annotations
import os
import sys
import glob
import subprocess

from sciagent.sandbox import check as sandbox_check, SandboxViolation

OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs"
)

# matplotlib 中文字体探测的样板，注入到脚本开头
_FONT_PREAMBLE = (
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.font_manager as fm\n"
    "_cn_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'SimSun',\n"
    "             'STHeiti', 'PingFang SC', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']\n"
    "for _fn in _cn_fonts:\n"
    "    if any(_fn in f.name for f in fm.fontManager.ttflist):\n"
    "        plt.rcParams['font.sans-serif'] = [_fn, 'DejaVu Sans']\n"
    "        plt.rcParams['axes.unicode_minus'] = False\n"
    "        break\n"
    "\n"
)


def run_python_code(code: str, timeout: int = 60) -> str:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 先过沙箱
    try:
        sandbox_check(code)
    except SandboxViolation as e:
        return f"ERROR: Sandbox 拒绝执行 — {e}"

    before = set(glob.glob(os.path.join(OUTPUTS_DIR, "*.png")))

    script_path = os.path.join(OUTPUTS_DIR, "_temp_script.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(_FONT_PREAMBLE + code)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=OUTPUTS_DIR,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: 代码执行超时（{timeout}s 限制）。请检查是否有死循环或计算量过大。"

    parts = []
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    if stdout:
        parts.append(f"STDOUT:\n{stdout}")
    if result.returncode != 0 and stderr:
        parts.append(f"ERROR:\n{stderr}")

    after = set(glob.glob(os.path.join(OUTPUTS_DIR, "*.png")))
    new_pngs = sorted(after - before)
    if new_pngs:
        listing = "\n".join(f"  - {os.path.basename(p)}" for p in new_pngs)
        parts.append(f"GENERATED FILES:\n{listing}")

    return "\n\n".join(parts) if parts else "代码执行成功，无输出。"


TOOL_DEFINITION = {
    "name": "run_python_code",
    "description": (
        "在沙箱中执行一段 Python 代码。"
        "环境已安装 numpy, scipy, matplotlib, deap。"
        "代码会先通过 AST 白名单审查（禁用 os.system / eval / 网络库等）。"
        "需要 print() 输出结果。生成图表用 plt.savefig('name.png')，不要用 plt.show()。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的完整 Python 代码"}
        },
        "required": ["code"],
    },
}


def execute(args: dict) -> str:
    return run_python_code(args["code"])
