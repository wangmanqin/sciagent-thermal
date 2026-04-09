# SciAgent: Natural Language-Driven Scientific Computing Agent

An AI agent that autonomously solves scientific computing and multi-objective optimization problems from natural language descriptions. Built on the ReAct (Reasoning + Acting) framework, SciAgent can analyze problems, generate code, execute computations, self-correct errors, and deliver publication-quality visualizations — all from a single natural language prompt.

## Demo

**Input:** *"Optimize a rectangular microchannel heat sink using NSGA-II. Channel width 0.1–1.0 mm, depth 0.2–2.0 mm. Minimize thermal resistance and pressure drop."*

**What the Agent does:**
1. Parses the engineering problem and identifies design variables, objectives, and constraints
2. Generates complete NSGA-II optimization code using the DEAP library
3. Executes the code in a sandboxed environment
4. If the code fails, reads the error and automatically fixes it
5. Produces Pareto front visualizations
6. Delivers a natural language analysis with recommended design trade-offs

<!-- TODO: add demo GIF or screenshot here -->

## Architecture

```
User (natural language) → Agent Loop (ReAct) → Final Answer + Plots
                              ↕           ↕
                          LLM API      Code Executor
                         (DeepSeek)    (subprocess)
```

### Core Components

| File | Role |
|------|------|
| `sciagent/agent.py` | **ReAct loop** — orchestrates the think→act→observe cycle |
| `sciagent/llm.py` | **LLM interface** — unified wrapper for DeepSeek / Claude / Mock |
| `sciagent/tools.py` | **Code executor** — sandboxed Python execution via subprocess |
| `sciagent/prompts.py` | **System prompt** — defines agent behavior and constraints |
| `app.py` | **Web UI** — Streamlit frontend with real-time agent process display |

### How the ReAct Loop Works

```
Initialize: messages = [system_prompt, user_query]

Loop (max 10 iterations):
  1. Send messages + tool_definitions to LLM API
  2. If LLM returns text only       → task complete, return answer
     If LLM returns tool_call        → continue to step 3
  3. Execute the requested tool (run Python code)
  4. Append tool result to messages
  5. Go to step 1
```

Self-correction happens naturally: when code execution fails, the error message is appended to the conversation history. The LLM sees the error in the next iteration and generates corrected code.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| subprocess for code execution | Process isolation — agent-generated code runs in a separate process, preventing crashes from affecting the main application |
| 60-second execution timeout | Prevents infinite loops in generated code from hanging the system |
| Max 10 iteration cap | Prevents the agent from entering endless error-correction cycles |
| Unified LLMResponse format | Enables seamless switching between LLM providers (DeepSeek, Claude) without modifying the agent logic |
| Cumulative message history | LLMs are stateless — full conversation history must be sent with every API call |
| English-only plot labels | matplotlib lacks CJK font support by default; enforced via system prompt |

## Evaluation Results

Tested on 10 scientific computing tasks spanning 7 categories (basic computation, equation solving, optimization, multi-objective optimization, statistics, signal processing, curve fitting):

| Metric | Result |
|--------|--------|
| First-attempt success | **10/10** |
| Success after self-correction | 0/10 |
| Failed | 0/10 |
| Average time per task | 66.5s |
| Average iterations per task | 1.6 |

| # | Task | Difficulty | Iterations | Time(s) |
|---|------|-----------|-----------|---------|
| 1 | Fibonacci sequence + plot | Easy | 1 | 28.8 |
| 2 | Cubic equation root finding | Easy | 2 | 91.9 |
| 3 | Rosenbrock optimization | Medium | 3 | 97.4 |
| 4 | Bi-objective NSGA-II | Medium | 1 | 65.5 |
| 5 | Normal distribution analysis | Easy | 1 | 46.0 |
| 6 | ODE solving (RK4) | Medium | 1 | 63.1 |
| 7 | Linear system solving | Easy | 3 | 95.5 |
| 8 | TSP (simulated annealing) | Hard | 1 | 55.3 |
| 9 | FFT signal analysis | Medium | 1 | 63.3 |
| 10 | Exponential curve fitting | Medium | 2 | 58.6 |

**Key Findings:**
- **Multiple iterations ≠ errors.** Tasks with 2-3 iterations (e.g., Rosenbrock, linear system) were the agent proactively decomposing the task (compute → visualize → analyze), not fixing bugs.
- **The agent consistently over-delivers.** For the ODE task, it autonomously verified convergence order. For TSP, it ran 10 independent trials to validate solution stability. These were not requested — the agent decided they were scientifically appropriate.
- **Current benchmark did not trigger self-correction.** Future work should include adversarial tests (incorrect formulas, uncommon libraries, edge cases) to probe failure boundaries.

## Quick Start

### Prerequisites
- Python 3.10+
- A DeepSeek API key ([platform.deepseek.com](https://platform.deepseek.com))

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/sciagent.git
cd sciagent
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY
```

### Run (CLI)

```bash
python main.py
```

### Run (Web UI)

```bash
python -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Tech Stack

- **LLM**: DeepSeek API (OpenAI-compatible, with Function Calling)
- **Agent Framework**: Custom ReAct implementation
- **Optimization**: DEAP (NSGA-II), SciPy
- **Visualization**: matplotlib
- **Frontend**: Streamlit
- **Language**: Python

## Motivation

As a researcher working on microchannel heat sink optimization using CFD and NSGA-II, I extensively use LLMs as daily research tools. This project explores whether LLMs can go beyond text generation to autonomously execute scientific computing workflows — from problem formulation to code generation, execution, debugging, and result analysis.

The project also serves as a systematic investigation into the capabilities and limitations of LLM-based agents in scientific computing contexts, examining where they succeed, where they fail, and why.

## License

MIT
