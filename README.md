# ReflectionAgents

> A declarative agentic workflow framework built with Grok API + Ray + reflection loops.

Autonomous agents that plan, execute in parallel, critique each other, and synthesize results — all described with the same high-level DSL regardless of backend (CPU, GPU, Ray, or future Cerebras WSE-3).

![Agent workflow graph](examples/agent_graph_cpu.png)

---

## Why This Project Exists

This project was created to explore an idea: if modern AI tools can dramatically accelerate the way we design, run, and analyze complex computational experiments — such as quantum error correction simulations.

I wanted to see how far a Reflexion-style agentic approach could go when combined with real parallel execution (Ray), multiple hardware backends, and actual simulation code (Stim). The result is a lightweight, educational framework that turns exploratory scientific workflows into adaptive, self-improving loops.

It is not meant to replace massive production frameworks like those used at CERN or Fermilab. Instead, it demonstrates what a single person (with heavy AI assistance) can build quickly today, and it provides a clean starting point for others who want to experiment with agentic + parallel scientific computing.

The goal is educational and exploratory: to show what is already possible, to make these ideas accessible, and to invite collaboration from people working in quantum error correction, high-performance computing, and agentic AI.

---

## Project Goals

- **Single DSL, multiple backends** — write once, run on CPU threads, GPU (Triton), distributed Ray clusters, or Cerebras CSL without changing your workflow code
- **Real agentic pipelines** — Planner → parallel specialist agents → reflection / critique loops → Synthesizer, all calling the real Grok API
- **Serious workloads** — agents process the full text of *Pride and Prejudice* (~750 kB) split across 4 parallel workers to demonstrate real throughput
- **Confidence-gated loops** — agents self-score their output (0.0–1.0 via JSON); reflection rounds continue only until average confidence crosses a threshold
- **Tool use** — agents receive keyword extraction and word-count results before calling the API, grounding their analysis in facts
- **Persistent memory** — optional `--debug` flag writes all round results to JSON + SQLite for inspection and replay
- **Visualization** — every workflow saves a PNG graph of its agent network

---

## Features

| Feature | Details |
|---------|---------|
| Same DSL for all backends | `ParallelDSL(backend="cpu" / "ray" / "gpu" / "csl")` |
| Real Grok API integration | `grok-3` via xAI's OpenAI-compatible endpoint |
| Parallel agents | 4 specialist agents run simultaneously via `dsl.map()` or Ray remote tasks |
| Reflection loops | Each agent critiques and rewrites its own prior output |
| Confidence gating | Loops stop early when avg agent confidence ≥ threshold |
| Tool use | Keyword extraction + word count injected into every agent prompt |
| Persistent memory | Ray actor stores results in-memory; `--debug` writes JSON + SQLite |
| Graph visualization | `networkx` + `matplotlib` PNG saved after every run |
| Interactive CLI | Natural-language command interface via `agent.py` |

---

## API Key Setup

Examples 06–08 require an [xAI Grok API key](https://console.x.ai). Example 05 needs nothing.

**The key is never in any Python file.** All code uses `os.getenv("XAI_API_KEY")` exclusively.

### On Replit
1. Click the **🔒 Secrets** tab in the left sidebar
2. **New Secret** → Name: `XAI_API_KEY` → Value: your key (starts with `xai-`)
3. Replit injects it automatically — no `.env` file needed

### Local / after cloning from GitHub
```bash
# create .env (already in .gitignore — never commit it)
echo "XAI_API_KEY=xai-your-key-here" > .env

# load for this session
export $(cat .env)
```
Get a key at: **https://console.x.ai** → sign in → **Create API Key**

---

## Installation

```bash
pip install -e .
pip install ray        # needed for examples 07, 08, 09, 10
pip install stim       # needed for examples 09, 10 (QEC simulation)
pip install matplotlib # needed for example 10 (noise curve plot)
```

---

## How to Run Each Example

All commands from the workspace root.

### Example 05 — CPU agents, no API key required
```bash
python3 examples/05_agentic_workflow.py
```
Pure Python parallel agents. A Planner decomposes a task, three workers (Researcher, Analyst, Critic) run in parallel via `ThreadPoolExecutor`, and a Synthesizer combines results. Graph saved to `examples/agent_graph_cpu.png`.

---

### Example 06 — Grok API + CPU backend
```bash
python3 examples/06_agentic_grok_workflow.py
```
Downloads *Pride and Prejudice* (~750 kB), splits it into 4 chunks, and runs 4 specialist agents in parallel against the real Grok API. A Synthesizer writes a literary analysis report. Runtime ~15–20 s. Graph saved to `examples/agent_graph_grok_cpu.png`.

---

### Example 07 — Grok API + Ray + reflection loops
```bash
python3 examples/07_agentic_grok_ray_reflection.py
```
Same as 06 but Ray distributes agents across CPU cores. A second reflection round has each agent critique its own prior output. Ray warnings are suppressed automatically. Runtime ~30–40 s. Graph saved to `examples/agent_graph_grok_ray.png`.

---

### Example 08 — Confidence-gated reflection + tool use + memory
```bash
# default: in-memory only, no files written
python3 examples/08_confidence_tools_memory.py

# --debug: also writes JSON + SQLite after every round
python3 examples/08_confidence_tools_memory.py --debug
```
Most advanced workflow. Key parameters at the top of the file:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `CONFIDENCE_THRESHOLD` | `0.88` | Stop reflection early when avg confidence reaches this |
| `MAX_ROUNDS` | `4` | Hard cap on rounds regardless of confidence |

Agents receive keyword extraction + word count tool results before calling Grok, return structured JSON (`analysis`, `confidence`, `gaps`), and the loop stops as soon as the average crosses the threshold. Runtime ~20–50 s depending on rounds. Graph saved to `examples/agent_graph_confidence.png`.

With `--debug`, two extra files are written to `examples/`:
- `memory_store.json` — full store, all rounds, human-readable
- `memory_store.db` — SQLite, one row per agent per round

Query the database:
```bash
python3 -c "
import sqlite3
con = sqlite3.connect('examples/memory_store.db')
for r in con.execute('SELECT round_key, agent, confidence FROM agent_results'):
    print(r)
"
```

---

### Example 10 — Adaptive QEC noise sweep (dynamic loop + plot)
```bash
python3 examples/10_qec_adaptive_loop.py
```
A **Grok navigator agent** drives the experiment: after each Stim measurement it decides which noise level to probe next (filling gaps, hunting the threshold region) until it is confident the curve is well-characterised or a round cap is hit.

What happens step by step:
1. **Seed** — one Stim measurement at noise=0.02 (no API call)
2. **Adaptive loop** — each round: Navigator (Grok) sees all data, picks the next noise value, Stim runs it; loop stops when `confidence >= 0.85` or 8 rounds
3. **Plot** — log-log error rate vs noise PNG with measured points, annotation order, and theoretical 3p² reference line → `examples/qec_noise_curve.png`
4. **Agent graph** → `examples/agent_graph_qec_adaptive.png`
5. **Synthesizer** — Grok writes a final report identifying the failure threshold and top recommendation

Results saved to `examples/qec_adaptive_results.json`. Runtime ~30–60 s.

Requires `pip install stim matplotlib`.

---

### Example 09 — Quantum Error Correction simulation (Stim + Ray + Grok)
```bash
python3 examples/09_qec_agentic_simulation.py
```
Runs a **noise-sweep experiment** on a 3-qubit bit-flip repetition code using [Stim](https://github.com/quantumlib/Stim). Four Ray agents each simulate a different noise level (0.5 %, 1 %, 2 %, 4 %), call Grok for scientific analysis, then critique each other in a reflection round. A Synthesizer writes a final scientific report.

What happens step by step:
1. **Planner** — Grok designs the experiment
2. **Local sanity check** — Stim runs all 4 noise levels locally and prints raw error rates before any API call
3. **Parallel agents** — 4 Ray agents each run their own Stim circuit and call Grok for analysis
4. **Reflection** — each agent critiques prior results and suggests improvements
5. **Graph** saved to `examples/agent_graph_qec.png`
6. **Synthesizer** writes the final report with concrete recommendations

Results always saved to `examples/qec_results.json`. Runtime ~40–60 s.

Expected noise-scaling output (confirms the code suppresses single-qubit errors as ~noise²):
```
noise=0.005  →  logical_error_rate=0.00010
noise=0.010  →  logical_error_rate=0.00030
noise=0.020  →  logical_error_rate=0.00140
noise=0.040  →  logical_error_rate=0.00560
```

Requires `pip install stim`.

---

### Interactive CLI
```bash
python3 agent.py
```
Natural-language interface. Type commands like `run cpu`, `run ray`, or `visualize`.

---

## Inspiration

The reflection and critic loops in this framework are inspired by the **Reflexion** paper:

> Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023).
> **Reflexion: Language Agents with Verbal Reinforcement Learning.**
> arXiv:2303.11366. https://arxiv.org/abs/2303.11366

Reflexion introduced the idea of agents generating verbal self-critiques and using them to improve future outputs — the same mechanism driving the reflection rounds in examples 07, 08, 09, and 10. Rather than updating model weights, agents reflect on their prior output in natural language and produce a better answer on the next pass.

---

## How Reflection Works in This Framework

The reflection/critic loops are inspired by the **Reflexion** (Shinn et al., 2023) paper. Instead of updating model weights, agents generate verbal self-critiques in natural language and use them to improve future outputs.

In the examples, this manifests as an **adaptive experimental loop**:
- A "Navigator" agent sees all previous simulation results.
- It reasons in natural language about gaps in the data, threshold regions, and scientific priorities.
- It chooses the next noise level to probe.
- Critic agents review the results and suggest improvements to the search strategy.

### Comparison with Classical Optimization Algorithms

| Aspect | Classical Optimizers (SGD, Bayesian Optimization, Grid Search, CMA-ES, etc.) | Agentic Reflexion Approach (this framework) | Advantage |
|---|---|---|---|
| **Decision Making** | Purely numerical / statistical | Natural language reasoning + domain knowledge | Reflexion |
| **Interpretability** | Low (black-box next-point suggestions) | High (explains *why* it chose the next point) | Reflexion |
| **Incorporating Scientific Intuition** | Difficult (must be encoded mathematically) | Natural ("probe near expected threshold") | Reflexion |
| **Self-Improvement of Strategy** | Static algorithm | Dynamic via reflection/critic loops | Reflexion |
| **Handling Qualitative Goals** | Poor | Excellent ("find where the code starts failing badly") | Reflexion |
| **Sample Efficiency** | Usually better | Lower, but more "intelligent" per sample | Classical |
| **Speed & Cost** | Very fast and cheap | Slower (API calls) | Classical |
| **Best Use Case** | Fine-tuning known objectives, high-dimensional problems | Exploratory science, hypothesis generation, complex reasoning | — |

**Summary**: This agentic Reflexion approach is not trying to replace classical optimizers. It shines in **exploratory research** where understanding, reasoning, and strategic experiment design matter more than pure sample efficiency. It behaves like a small team of collaborating scientists rather than a blind numerical search algorithm.

This makes it particularly valuable for quantum error correction studies, where knowing *why* a code fails at certain noise levels is often more important than finding the absolute minimum error rate.

---

## Reflexion in the Wild — Popularity and Adoption

The original Reflexion paper (2023) is now considered **foundational** in agentic AI. By 2025–2026, self-reflection and critic loops have moved from "new research idea" to **standard design pattern** — built into almost every major agentic framework.

### How Reflexion Compares to Other Reasoning Techniques

| Technique | Core Idea | Strengths | Weaknesses | When Reflexion Wins |
|---|---|---|---|---|
| **ReAct** | Reason → Act → Observe loop | Simple, tool-friendly | No explicit self-correction | When you need critique |
| **Chain-of-Thought (CoT)** | Step-by-step reasoning | Easy to implement | One-shot, no self-fix | When iteration helps |
| **Tree-of-Thoughts (ToT)** | Explore multiple reasoning branches | Good exploration | Expensive, no learning from failure | When verbal critique is key |
| **Self-Consistency** | Sample multiple answers & vote | Reduces hallucinations | No structured critique | When you want learning |
| **Reflexion / Reflection** | Generate → Critique → Improve | **Strong self-correction & learning** | Extra LLM calls (higher cost/latency) | **Exploratory & high-stakes tasks** |

Reflexion's real edge is **verbal self-critique + memory of past failures**. Agents learn from mistakes in natural language without fine-tuning the model — especially powerful for exploratory science, qualitative reasoning, and tasks with human-readable audit requirements.

### Frameworks and Companies Using It (as of 2026)

**Major frameworks:**
- **LangGraph** (LangChain) — official "Reflection & Reflexion" examples and built-in support
- **CrewAI** — critic agents and reflection loops out of the box
- **AutoGen** — conversational self-critique and iterative refinement
- **LlamaIndex Agents** — reflection as a standard pattern

**Real-world adoption:**
Production deployments include code generation & review agents, incident root-cause analysis, legal/contract review, research summarization, and logistics optimization (documented 10–15 % gains from reflection). Large labs (OpenAI, Anthropic) have moved to o1-style "thinking" models that perform implicit self-reflection at scale.

**Bottom line**: Reflexion is not the single most popular technique (ReAct and planning loops still dominate simple agents), but it is one of the most respected advanced patterns for building agents that genuinely improve over time.

---

## What Makes ReflectionAgents Unique

Most existing frameworks implement some form of reflection — but ReflectionAgents combines several capabilities that individually exist elsewhere but are rarely found together.

| Feature | Most Existing Frameworks (LangGraph, CrewAI, AutoGen, etc.) | ReflectionAgents | Why It Matters |
|---|---|---|---|
| **Declarative DSL** | Mostly imperative / graph configuration | Single clean high-level DSL | Much easier to learn and teach |
| **Multiple Backends** | Usually one execution engine (mostly single-node) | CPU, Triton GPU, Ray (distributed), future Cerebras | True hardware flexibility |
| **Parallel Execution** | Limited or manual | Native parallel agents via Ray | Scales to real workloads |
| **Built-in Reflection Loops** | Optional add-on or plugin | Core primitive, every example uses it | Reflection is first-class, not bolted on |
| **Domain-specific simulation** | Generic text / tool agents | Stim QEC integration — real physics simulation | Agents reason over real scientific data |
| **Adaptive experimentation** | Rare | Navigator agent drives the experiment loop | Autonomous scientific discovery |
| **Audit trail** | Often missing | Every decision logged + JSON + SQLite | Full reproducibility and traceability |
| **Visualization** | Basic or absent | NetworkX agent graph PNG per example | Immediate insight into workflow structure |

### The Real Unique Value Proposition

ReflectionAgents is one of the few frameworks that successfully combines:

1. **True declarative programming for agentic workflows** — you write *what* you want, not *how* to orchestrate it
2. **First-class reflection/critic loops as a core primitive** — not an afterthought
3. **Real parallel & distributed execution** — via Ray, not simulated concurrency
4. **Hardware-backend-agnostic design** — CPU → GPU → future wafer-scale (Cerebras WSE-3)
5. **Domain-grounded agents** — real physics (Stim QEC) rather than toy text tasks

Most popular frameworks are either very easy but not parallel/distributed (CrewAI), or very powerful but complex and low-level (LangGraph). ReflectionAgents sits in the middle: high-level enough to be readable, low-level enough to run real distributed science.

---

## Agentic Workflow Pattern

```
          ┌─────────┐
          │ Planner │
          └────┬────┘
    ┌──────────┼──────────┬──────────┐
    ▼          ▼          ▼          ▼
Researcher  Analyst    Critic   Historian   ← parallel, real Grok API
    │          │          │          │
    └──────────┼──────────┴──────────┘
               │  reflection round (if confidence < threshold)
    ┌──────────┼──────────┬──────────┐
    ▼          ▼          ▼          ▼
  R_R2       A_R2      C_R2      H_R2       ← agents improve own output
    └──────────┼──────────┴──────────┘
          ┌────┴──────┐
          │Synthesizer│
          └───────────┘
```

---

## Backends

| Backend | Status | Requires |
|---------|--------|---------|
| CPU | ✅ Always available | Nothing |
| GPU (Triton) | ✅ With CUDA | `pip install torch triton` |
| Ray distributed | ✅ Available | `pip install ray` |
| Cerebras CSL | 🔧 Stub / sketch | Cerebras SDK + WSE-3 |

---

## Project Structure

```
ReflectionAgents/
├── dsl/
│   ├── base_dsl.py              # ParallelDSL class, backend dispatch
│   ├── dataflow_dsl.py          # Dataflow graph DSL
│   ├── graph_planner.py         # networkx task graph planner
│   ├── visualizer.py            # matplotlib PNG renderer
│   └── backends/
│       ├── cpu_backend.py       # ThreadPoolExecutor
│       ├── gpu_backend.py       # Triton kernel (falls back to CPU)
│       ├── ray_backend.py       # Ray remote tasks (falls back to CPU)
│       └── cerebras_csl.py      # CSL skeleton generator
├── examples/
│   ├── 01_simple_map.py
│   ├── 02_dataflow_pipeline.py
│   ├── 03_triton_gpu_kernel.py
│   ├── 04_ray_distributed.py
│   ├── 05_agentic_workflow.py           # CPU, no API
│   ├── 06_agentic_grok_workflow.py      # Grok + CPU
│   ├── 07_agentic_grok_ray_reflection.py  # Grok + Ray + reflection
│   ├── 08_confidence_tools_memory.py    # Confidence-gated + tools + memory
│   ├── 09_qec_agentic_simulation.py     # QEC — Stim + Ray + Grok
│   ├── 10_qec_adaptive_loop.py          # Adaptive loop + noise curve plot
│   └── agent_graph_cpu.png              # sample visualization
├── agent.py                     # natural-language CLI
├── instructions-agent-grok-ray.md
├── setup.py
├── LICENSE
└── README.md
```

---

## Security Notes

- `XAI_API_KEY` is read only from environment variables — never from files or code
- `.env` is in `.gitignore` — never commit it
- No credentials, tokens, or secrets appear anywhere in the source tree
- Ray CPU warnings are suppressed via env vars set in code, not via any secret

---

## License

MIT — see [LICENSE](LICENSE)
