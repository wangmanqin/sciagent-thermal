"""
properties 子包：把水/EG/空气/纳米流体的物性查询统一注册。
"""

from sciagent.tools.properties import water_iapws as _wi
from sciagent.tools.properties import ethylene_glycol as _eg
from sciagent.tools.properties import air as _air
from sciagent.tools.properties import nanofluids as _nf


TOOL_DEFINITIONS = [
    _wi.TOOL_DEFINITION,
    _eg.TOOL_DEFINITION,
    _air.TOOL_DEFINITION,
    _nf.TOOL_DEFINITION,
]


TOOL_EXECUTORS = {
    _wi.TOOL_DEFINITION["name"]: _wi.execute,
    _eg.TOOL_DEFINITION["name"]: _eg.execute,
    _air.TOOL_DEFINITION["name"]: _air.execute,
    _nf.TOOL_DEFINITION["name"]: _nf.execute,
}


__all__ = ["TOOL_DEFINITIONS", "TOOL_EXECUTORS"]
