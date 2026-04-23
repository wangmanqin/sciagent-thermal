"""一键跑所有测试，不需要 pytest。

用法：python -m tests.run_all
"""

from __future__ import annotations
import importlib
import sys
import traceback


TEST_MODULES = [
    "tests.test_correlations",
    "tests.test_properties",
    "tests.test_solvers",
    "tests.test_optim",
    "tests.test_sandbox",
    "tests.test_mcp",
    "tests.test_geometry_pressure",
    "tests.test_agent_workflow",
]


def run_module(mod_name: str):
    mod = importlib.import_module(mod_name)
    fns = [(n, f) for n, f in vars(mod).items()
           if n.startswith("test_") and callable(f)]
    passed = 0
    failed = []
    for name, fn in fns:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            failed.append((name, f"AssertionError: {e}"))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            failed.append((name, f"{type(e).__name__}: {e}\n{tb}"))
    return passed, len(fns), failed


def main() -> int:
    total_pass = total = 0
    all_failed = []
    for m in TEST_MODULES:
        print(f"\n=== {m} ===")
        try:
            p, n, failed = run_module(m)
        except Exception as e:
            print(f"  MODULE LOAD ERROR: {e}")
            continue
        total_pass += p
        total += n
        for name, msg in failed:
            print(f"  [FAIL] {name}: {msg}")
            all_failed.append((m, name))
        print(f"  {p}/{n} passed")

    print(f"\n====== TOTAL: {total_pass}/{total} passed ======")
    if all_failed:
        print("Failures:")
        for m, n in all_failed:
            print(f"  - {m}::{n}")
    return 0 if total_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
