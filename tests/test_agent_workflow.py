"""Agent + ReAct workflow 集成测试（用 MockLLM，不依赖外部 API）。"""

from __future__ import annotations


def test_mock_llm_emits_tool_call():
    from sciagent.llm import MockLLM
    llm = MockLLM(scripted_responses=[
        {
            "content": "我需要先查一下水的物性",
            "tool_calls": [{
                "id": "call_1",
                "name": "water_properties",
                "arguments": {"temperature_C": 30.0},
            }],
        },
        {"content": "水在 30°C 时密度约 995 kg/m³。", "tool_calls": []},
    ])
    resp = llm.chat([{"role": "user", "content": "水物性"}])
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "water_properties"


def test_react_workflow_completes_with_mock():
    from sciagent.llm import MockLLM
    from sciagent.workflow.react import ReActWorkflow
    from sciagent.tools import TOOL_DEFINITIONS, TOOL_EXECUTORS

    llm = MockLLM(scripted_responses=[
        {
            "content": "需要水物性",
            "tool_calls": [{
                "id": "c1",
                "name": "water_properties",
                "arguments": {"temperature_C": 25.0},
            }],
        },
        {"content": "完成。", "tool_calls": []},
    ])
    wf = ReActWorkflow(
        llm=llm,
        tool_definitions=TOOL_DEFINITIONS,
        tool_executors=TOOL_EXECUTORS,
        max_iterations=5,
    )
    events = []
    wf.run("告诉我水在 25°C 的密度", on_event=events.append)
    kinds = [e.event_type for e in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "answer" in kinds


def test_agent_facade_run():
    from sciagent.agent import Agent
    from sciagent.llm import MockLLM

    llm = MockLLM(scripted_responses=[
        {"content": "直接回答。", "tool_calls": []},
    ])
    agent = Agent(llm=llm)
    events = []
    agent.run("hi", on_event=events.append)
    assert any(e.event_type == "answer" for e in events)


def test_workflow_respects_max_iterations():
    from sciagent.llm import MockLLM
    from sciagent.workflow.react import ReActWorkflow
    from sciagent.tools import TOOL_DEFINITIONS, TOOL_EXECUTORS

    # 永远调用工具，从不给最终答案 → 应该被 max_iterations 截断
    calls = [
        {
            "content": "再查一次",
            "tool_calls": [{"id": f"c{i}", "name": "water_properties",
                            "arguments": {"temperature_C": 25.0}}],
        }
        for i in range(20)
    ]
    llm = MockLLM(scripted_responses=calls)
    wf = ReActWorkflow(
        llm=llm,
        tool_definitions=TOOL_DEFINITIONS,
        tool_executors=TOOL_EXECUTORS,
        max_iterations=3,
    )
    events = []
    wf.run("loop forever", on_event=events.append)
    # 最多 3 次工具调用
    n_calls = sum(1 for e in events if e.event_type == "tool_call")
    assert n_calls <= 3


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
