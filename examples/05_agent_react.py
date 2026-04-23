"""
Example 05: 把 Agent 完整跑一个微通道问题。

用 MockLLM 脚本化回放，方便不接真 API 也能演示 ReAct 流程。
"""

from sciagent.llm import MockLLM
from sciagent.agent import Agent


SCRIPT = [
    # Turn 1: thinking + tool_call
    {
        "content": "用户要 1 L/min 入水温差，先查物性再算能量守恒。",
        "tool_calls": [{
            "id": "c1",
            "name": "water_properties",
            "arguments": {"temperature_C": 25.0},
        }],
    },
    # Turn 2: observed properties, now compute caloric
    {
        "content": "ρ≈997, cp≈4182。调 caloric_resistance 算温升。",
        "tool_calls": [{
            "id": "c2",
            "name": "caloric_resistance",
            "arguments": {
                "mass_flow_kg_per_s": 1.0 / 60,
                "specific_heat_J_per_kgK": 4182,
            },
        }],
    },
    # Turn 3: final answer
    {
        "content": (
            "R_cap = 1/(m·cp) = 1/(0.01667 × 4182) ≈ 0.01435 K/W\n"
            "ΔT = Q·R = 80 × 0.01435 ≈ 1.15 °C"
        ),
        "tool_calls": [],
    },
]


def main():
    llm = MockLLM(scripted_responses=SCRIPT)
    agent = Agent(llm=llm)
    events = []
    agent.run(
        "80W 芯片水冷，流量 1 L/min 入口 25°C，估算进出口温差。",
        on_event=events.append,
    )
    for ev in events:
        typ = ev.event_type
        content = ev.content if isinstance(ev.content, str) else str(ev.content)
        print(f"[{typ}] {content[:100]}")
    print(f"\n共 {len(events)} 个事件")


if __name__ == "__main__":
    main()
