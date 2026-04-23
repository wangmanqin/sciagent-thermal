"""SciAgent-Thermal 的 10 题微通道基准测试。

三个脚本：
  - make_tasks.py : 生成 10 题 JSON 题面 + 参考答案
  - run_bench.py  : 跑 Agent、记录每题 ReAct log
  - score.py      : 对照参考答案评分

测试集合设计原则见 docs/BENCHMARK.md。
"""
