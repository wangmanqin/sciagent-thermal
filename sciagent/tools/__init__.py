"""
Tool 层：对外暴露统一的 TOOL_DEFINITIONS / TOOL_EXECUTORS。

每个子模块各自提供自己的 tool schema 与 executor，本文件把它们汇总。
这样新增工具只要在对应子模块里注册，无需改动 agent / workflow。
"""

from sciagent.tools import fluid_properties as _fp
from sciagent.tools import correlations as _corr
from sciagent.tools import plotter as _plot
from sciagent.tools import python_exec as _pyx


TOOL_DEFINITIONS = [
    _fp.TOOL_DEFINITION,
    *_corr.TOOL_DEFINITIONS,
    _plot.TOOL_DEFINITION,
    _pyx.TOOL_DEFINITION,
]


TOOL_EXECUTORS = {
    _fp.TOOL_DEFINITION["name"]: _fp.execute,
    **_corr.TOOL_EXECUTORS,
    _plot.TOOL_DEFINITION["name"]: _plot.execute,
    _pyx.TOOL_DEFINITION["name"]: _pyx.execute,
}


OUTPUTS_DIR = _pyx.OUTPUTS_DIR

__all__ = ["TOOL_DEFINITIONS", "TOOL_EXECUTORS", "OUTPUTS_DIR"]
