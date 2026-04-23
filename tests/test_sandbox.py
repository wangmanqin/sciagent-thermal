"""Sandbox 白名单测试：危险操作应被拦截，合法科学计算应放行。"""

from __future__ import annotations
import pytest


def _raises(fn):
    from sciagent.sandbox.ast_whitelist import SandboxViolation
    try:
        fn()
    except SandboxViolation:
        return True
    except Exception as e:
        # 语法错误等也算失败路径，但我们想要 SandboxViolation
        raise AssertionError(f"期望 SandboxViolation，实际 {type(e).__name__}: {e}")
    return False


def test_blocks_eval():
    from sciagent.sandbox.ast_whitelist import check
    assert _raises(lambda: check("eval('1+2')"))


def test_blocks_exec():
    from sciagent.sandbox.ast_whitelist import check
    assert _raises(lambda: check("exec('x = 1')"))


def test_blocks_open():
    from sciagent.sandbox.ast_whitelist import check
    assert _raises(lambda: check("open('/etc/passwd')"))


def test_blocks_os_system():
    from sciagent.sandbox.ast_whitelist import check
    code = "import os\nos.system('ls')"
    assert _raises(lambda: check(code))


def test_blocks_subprocess():
    from sciagent.sandbox.ast_whitelist import check
    code = "import subprocess"
    assert _raises(lambda: check(code))


def test_blocks_socket():
    from sciagent.sandbox.ast_whitelist import check
    code = "import socket"
    assert _raises(lambda: check(code))


def test_blocks_dunder_import():
    from sciagent.sandbox.ast_whitelist import check
    assert _raises(lambda: check("__import__('os')"))


def test_allows_numpy():
    from sciagent.sandbox.ast_whitelist import check
    code = "import numpy as np\na = np.array([1, 2, 3])\nprint(a.sum())"
    check(code)  # 不该抛


def test_allows_matplotlib():
    from sciagent.sandbox.ast_whitelist import check
    code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1,2,3])\n"
    )
    check(code)


def test_allows_math():
    from sciagent.sandbox.ast_whitelist import check
    check("import math\nprint(math.sqrt(2))")


def test_allows_deap():
    from sciagent.sandbox.ast_whitelist import check
    check("from deap import base, creator, tools")


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
