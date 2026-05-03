# Running the Agentic Workflow Examples

All commands assume you are in the project root:

```bash
cd artifacts/declarative-parallel-dsl
```

---

## Example 05 — CPU agents, no API required

Pure Python parallel agents using the CPU backend. No API key needed.

```bash
python3 examples/05_agentic_workflow.py
```

What it does:
- A Planner agent decomposes a task into subtasks
- Three worker agents (Researcher, Analyst, Critic) run in parallel
- A Synthesizer combines the results

Graph saved to: `examples/agent_graph_cpu.png`

---

## Example 06 — Grok API + CPU backend

Four parallel agents call the real Grok API to analyze *Pride and Prejudice*.

```bash
python3 examples/06_agentic_grok_workflow.py
```

Requirements:
- `XAI_API_KEY` set in Replit Secrets (Secrets tab → New Secret → `XAI_API_KEY`)

What it does:
- Fetches the full text of *Pride and Prejudice* from Project Gutenberg
- Splits it into 4 chunks, one per agent
- Planner, Researcher, Analyst, Critic, and Historian agents call Grok in parallel
- A Synthesizer writes a final literary analysis report
- Typical runtime: ~15–20 seconds

Graph saved to: `examples/agent_graph_grok_cpu.png`

---

## Example 07 — Grok API + Ray distributed backend + reflection loops

Ray distributes agents across CPU cores. Agents critique each other's outputs in a fixed second reflection round. Ray CPU warnings are suppressed automatically.

```bash
python3 examples/07_agentic_grok_ray_reflection.py
```

Requirements:
- `XAI_API_KEY` set in Replit Secrets
- `ray` installed (already done if you followed setup)

What it does:
- Fetches *Pride and Prejudice* and splits into 4 chunks
- Round 1: four agents (Researcher, Analyst, Critic, Historian) run in parallel via Ray
- Round 2 (reflection): each agent critiques the previous round's output
- Synthesizer writes a final report combining all outputs
- Ray initializes a local cluster automatically on startup
- Typical runtime: ~30–40 seconds

Graph saved to: `examples/agent_graph_grok_ray.png`

Note: the two Ray warning env vars are set automatically inside the script:
```
RAY_DISABLE_DOCKER_CPU_WARNING=1
RAY_USE_MULTIPROCESSING_CPU_COUNT=1
```

---

## Example 08 — Confidence-gated reflection + tool use + Ray memory (most advanced)

Agents self-score their own confidence as JSON (0.0–1.0). Reflection rounds repeat automatically until the average confidence exceeds the threshold or the maximum number of rounds is reached. Agents also receive real tool output (keyword extraction, word count) to ground their analysis. Results are persisted in a Ray remote object store across rounds.

```bash
python3 examples/08_confidence_tools_memory.py
```

Requirements:
- `XAI_API_KEY` set in Replit Secrets
- `ray` installed

Key parameters at the top of the file — edit to taste:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `CONFIDENCE_THRESHOLD` | `0.88` | Stop early when avg agent confidence reaches this |
| `MAX_ROUNDS` | `4` | Hard cap on reflection rounds regardless of confidence |

What it does:
- Runs keyword extraction and word count on each chunk before any API call (fast, local tools)
- Each agent returns structured JSON with `analysis`, `confidence`, and `gaps` fields
- After each round, average confidence is printed — if above threshold, stops early
- All round results are stored in a Ray remote `MemoryStore` actor
- The graph is drawn after the run so it shows exactly how many rounds actually executed
- Typical runtime: 1–2 rounds, ~20–50 seconds

Graph saved to: `examples/agent_graph_confidence.png`

---

## Visualization graphs

Each example saves a PNG diagram of its agent workflow graph to the `examples/` directory:

| Example | Backend | Graph file |
|---------|---------|------------|
| 05 | CPU (no API) | `examples/agent_graph_cpu.png` |
| 06 | Grok + CPU | `examples/agent_graph_grok_cpu.png` |
| 07 | Grok + Ray | `examples/agent_graph_grok_ray.png` |
| 08 | Grok + Ray + confidence | `examples/agent_graph_confidence.png` |

The graphs show how data flows from Planner through parallel agents to Synthesizer, including reflection rounds and the Ray MemoryStore actor in example 08.

---

## Interactive agent CLI

You can also run the natural-language command interface:

```bash
python3 agent.py
```

Type commands like `run cpu`, `run ray`, or `visualize` to interact with the DSL directly.
