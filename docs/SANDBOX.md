# Sandbox 设计文档

`sciagent.sandbox` 的唯一目标是：**当 Agent 拿到一段由 LLM 生成的 Python
代码时，既要能真的执行（科学计算需要 numpy/scipy/matplotlib），又不能
让它破坏系统**。

## 威胁模型

假设 LLM 是"善意但不可信"的。它不会主动攻击，但可能被用户的提示绕进
错误路径（prompt injection），也可能在幻觉驱动下调 `os.system("rm ...")`。

我们不关心内核级的逃逸——目标是"把工程师从看着脚本提心吊胆里解放出来"。

## 两道防线

### 防线 A：AST 白名单（静态）

文件：`sciagent/sandbox/ast_whitelist.py`

关键常量：

```python
ALLOWED_IMPORTS = frozenset({
    "math", "cmath", "random", "statistics", "itertools",
    "functools", "collections", "dataclasses", "typing",
    "numpy", "scipy", "scipy.integrate", "scipy.optimize",
    "scipy.interpolate", "scipy.special", "scipy.linalg",
    "matplotlib", "matplotlib.pyplot", "mpl_toolkits.mplot3d",
    "pandas",
    "deap", "deap.base", "deap.creator", "deap.tools",
    "deap.algorithms",
})
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "__import__", "open", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "breakpoint",
})
FORBIDDEN_ATTRS = frozenset({
    "os.system", "os.popen", "os.remove", "os.rmdir", "os.unlink",
    "os.rename", "os.chmod", "os.fork", "os.exec",
    "subprocess.run", "subprocess.Popen", "subprocess.call",
    "socket.socket", "urllib.request.urlopen",
})
```

walker 逻辑：

- `ast.Import` / `ast.ImportFrom` → 每个模块必须在 ALLOWED 中；
- `ast.Name` → 如果名字在 FORBIDDEN_NAMES 就拒；
- `ast.Attribute` → 拼出 `<value>.<attr>`，在 FORBIDDEN_ATTRS 中就拒。

通过不了白名单 → 抛 `SandboxViolation`，executor 直接返回给 Agent 看。

### 防线 B：子进程执行（动态）

文件：`sciagent/tools/python_exec.py`

通过白名单的代码会：

1. 写入 `outputs/.sandbox_<uuid>.py`
2. 前置 font/backend 预热片段（避免 matplotlib 中文乱码）
3. `subprocess.run(["python", script], timeout=60, encoding="utf-8")`
4. 捕获 stdout/stderr、exit code、新生成的 artifact 文件
5. 结果统一包成 dict 返回

超时 / 非零 exit / OSError 都当普通错误抛出。

## 不在沙箱里做的事

- **网络访问**：白名单里没有 `requests`/`urllib3`；真需要联网的工具
  （比如查物性表）应走独立工具函数，而不是让 LLM 写脚本去 HTTP。
- **文件系统写**：我们允许 matplotlib 保存 PNG，但所有写都落到
  `outputs/` 下；代码里硬写 `/etc/...` 会被 AST 拦掉（没 `open`）。
- **系统资源限制**：Windows 上没有用 `resource.setrlimit`，依赖
  子进程超时兜底。Linux 部署时可以加 cgroup。

## 已验证的拦截案例

这些写法全部会被 AST 层拦住（见 `tests/test_sandbox.py`）：

| 代码 | 为什么被拦 |
|---|---|
| `eval("1+2")` | FORBIDDEN_NAMES |
| `exec("x=1")` | FORBIDDEN_NAMES |
| `open("/etc/passwd")` | FORBIDDEN_NAMES |
| `__import__("os")` | FORBIDDEN_NAMES |
| `import os\nos.system("ls")` | FORBIDDEN_ATTRS |
| `import subprocess` | import not allowed |
| `import socket` | import not allowed |

下列合法写法全部放行：

```python
import numpy as np
a = np.array([1, 2, 3])
print(a.sum())
```

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.savefig("outputs/demo.png")
```

```python
from deap import base, creator, tools
# ... NSGA-II 实现
```

## 为什么不用 RestrictedPython / pyodide

| 方案 | 优点 | 为什么不选 |
|---|---|---|
| RestrictedPython | 成熟 | 不覆盖 numpy/matplotlib 的实际场景 |
| pyodide / wasm | 真·隔离 | 启动慢 + scipy 缺失 |
| docker 容器 | 行业标准 | 对 Windows/macOS 开发者门槛高 |
| **AST+subprocess** | 零依赖、可读、够用 | 选它 |

目标是作品集级别的可演示性，而不是生产级隔离。真要上线会换 firecracker。

## 已知的绕过

白名单不是万无一失的。目前已知的"理论上能绕过去但我们没补"的洞：

- `getattr(__builtins__, "eval")(...)` — 被 FORBIDDEN_NAMES 拦，
  因为 `getattr` 已经在黑名单里；
- 通过 numpy 的 C 扩展触发缓冲区溢出 — 对所有 Python 沙箱都是黑盒风险；
- 把字符串拼起来塞进 eval 再解析 — 前面有 FORBIDDEN_NAMES `eval`。

补洞哲学：**宁愿误杀合法代码（Agent 可以重试），也不放行危险写法**。

## 和 MCP 的关系

MCP `tools/call` 里的 `run_python_code` 走的是同一套 sandbox，所以
外部 MCP 客户端（比如 Claude Desktop）接入时，安全性由服务端保证。
