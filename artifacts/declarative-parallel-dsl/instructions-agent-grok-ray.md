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

The most advanced workflow: Ray distributes agents across cores, and agents critique each other's outputs in a second reflection round.

```bash
python3 examples/07_agentic_grok_ray_reflection.py
```

Requirements:
- `XAI_API_KEY` set in Replit Secrets
- `ray` package installed (`pip install ray`)

What it does:
- Fetches *Pride and Prejudice* and splits into 4 chunks
- Round 1: four agents (Researcher, Analyst, Critic, Historian) run in parallel via Ray
- Round 2 (reflection): each agent critiques the previous round's output
- Synthesizer writes a final report combining all outputs
- Ray initializes a local cluster automatically on startup
- Typical runtime: ~30–40 seconds

Graph saved to: `examples/agent_graph_grok_ray.png`

---

## Visualization graphs

Each example saves a PNG diagram of its agent workflow graph:

| Example | Backend | Graph file |
|---------|---------|------------|
| 05 | CPU (no API) | `examples/agent_graph_cpu.png` |
| 06 | Grok + CPU | `examples/agent_graph_grok_cpu.png` |
| 07 | Grok + Ray | `examples/agent_graph_grok_ray.png` |

The graphs show how data flows from the Planner through parallel agents to the Synthesizer, including reflection rounds in example 07.

---

## Interactive agent CLI

You can also run the natural-language command interface:

```bash
python3 agent.py
```

Type commands like `run cpu`, `run ray`, or `visualize` to interact with the DSL directly.
