"""
基于 AST 的代码白名单审查。

Agent 写的代码在用 subprocess 跑之前，先在这里过一遍静态检查。
这不是 seccomp/容器级别的强隔离，而是"第一道门"：
  - 导入必须在白名单里
  - 禁止 eval / exec / compile / __import__
  - 禁止 os.system / subprocess.* / socket.* 等高危调用
  - 禁止访问私有属性 (__xxx__)

结合 python_exec.py 里的 subprocess 隔离 + 超时，构成两层防线。
"""

from __future__ import annotations
import ast
from typing import Iterable


class SandboxViolation(Exception):
    """代码触犯沙箱规则时抛出。"""


# 允许导入的顶层模块
ALLOWED_IMPORTS = frozenset({
    "math", "cmath", "statistics", "random", "itertools", "functools",
    "collections", "heapq", "bisect", "re", "json", "dataclasses",
    "typing", "enum", "decimal", "fractions",
    "numpy", "scipy", "matplotlib", "deap",
    "pandas",
})

# 禁用的内置名（不管导入与否，一旦出现都报错）
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "__import__",
    "open",          # 文件读写一律走 matplotlib 的 savefig，不让 Agent 自己开文件
    "input",
    "exit", "quit",
})

# 禁用的"模块.方法"调用；第一项是模块前缀，第二项是属性
FORBIDDEN_ATTRS = frozenset({
    ("os", "system"), ("os", "popen"), ("os", "remove"), ("os", "unlink"),
    ("os", "rmdir"), ("os", "removedirs"), ("os", "chmod"),
    ("shutil", "rmtree"),
    ("subprocess", "run"), ("subprocess", "Popen"),
    ("subprocess", "call"), ("subprocess", "check_call"),
    ("subprocess", "check_output"),
    ("socket", "socket"), ("socket", "create_connection"),
    ("urllib", "request"), ("requests", "get"), ("requests", "post"),
})


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.violations: list[str] = []

    def _block(self, msg: str, node: ast.AST):
        self.violations.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in ALLOWED_IMPORTS:
                self._block(f"禁止 import {alias.name}（不在白名单内）", node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        top = module.split(".")[0]
        if top and top not in ALLOWED_IMPORTS:
            self._block(f"禁止 from {module} import ...（不在白名单内）", node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in FORBIDDEN_NAMES:
            self._block(f"禁止使用名称 '{node.id}'", node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # 禁止访问私有/魔术属性
        if node.attr.startswith("__") and node.attr.endswith("__"):
            # 允许一些无害的 dunder（这里我们直接禁掉最危险的一类）
            if node.attr in ("__class__", "__subclasses__", "__globals__",
                             "__dict__", "__getattribute__", "__code__"):
                self._block(f"禁止访问魔术属性 '{node.attr}'", node)

        # 禁止 (module.forbidden_attr)
        if isinstance(node.value, ast.Name):
            pair = (node.value.id, node.attr)
            if pair in FORBIDDEN_ATTRS:
                self._block(f"禁止调用 {pair[0]}.{pair[1]}", node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # 进一步拦截 getattr(obj, 'eval') 这类绕过
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            self._block("禁止使用 getattr（可能绕过白名单）", node)
        self.generic_visit(node)


def check(code: str) -> None:
    """审查通过则返回 None，否则抛 SandboxViolation。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxViolation(f"代码语法错误：{e}") from e

    visitor = _Visitor()
    visitor.visit(tree)
    if visitor.violations:
        raise SandboxViolation("; ".join(visitor.violations))


def list_violations(code: str) -> Iterable[str]:
    """调试用：返回所有违规条目，不抛异常。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    visitor = _Visitor()
    visitor.visit(tree)
    return list(visitor.violations)
