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

### Persistent memory files (--debug only)

By default no files are written. Pass `--debug` to enable disk persistence:

```bash
python3 examples/08_confidence_tools_memory.py --debug
```

When `--debug` is active, two files are written to `examples/` after every round:

**`examples/memory_store.json`** — the complete store, all rounds, human-readable:
```json
{
  "round_1": [ { "agent": "Researcher", "confidence": 0.91, ... }, ... ],
  "round_2": [ ... ],
  "synthesizer": [ ... ]
}
```
Open it in any text editor. Prior runs are loaded back in and extended — not overwritten.

**`examples/memory_store.db`** — SQLite, one row per agent per round. Query it from the shell:
```bash
python3 -c "
import sqlite3
con = sqlite3.connect('examples/memory_store.db')
for r in con.execute('SELECT round_key, agent, confidence FROM agent_results'):
    print(r)
"
```

You can filter by round, sort by confidence, compare runs — standard SQL. Both files are created on the first `--debug` run if they don't exist, or appended to if they do. Without `--debug`, the memory store lives only in Ray's object store for the duration of the script.

---

## Visualization graphs

Each example saves a PNG diagram of its agent workflow graph to the `examples/` directory:

| Example | Backend | Graph file |
|---------|---------|------------|
| 05 | CPU (no API) | `examples/agent_graph_cpu.png` |
| 06 | Grok + CPU | `examples/agent_graph_grok_cpu.png` |
| 07 | Grok + Ray | `examples/agent_graph_grok_ray.png` |
| 08 | Grok + Ray + confidence | `examples/agent_graph_confidence.png` |
| 09 | Stim + Ray + Grok (QEC) | `examples/agent_graph_qec.png` |
| 10 | Adaptive loop + plot (QEC) | `examples/agent_graph_qec_adaptive.png` |

The graphs show how data flows from Planner through parallel agents to Synthesizer, including reflection rounds and the Ray MemoryStore actor in example 08.

---

## Example 09 — Quantum Error Correction (QEC) simulation + Grok + Ray

A noise-sweep experiment on a 3-qubit bit-flip repetition code using [Stim](https://github.com/quantumlib/Stim), with 4 parallel Ray agents analyzing different noise levels, a reflection/critique round, and a Grok-written final scientific report.

```bash
python3 examples/09_qec_agentic_simulation.py
```

Requirements:
- `XAI_API_KEY` set in Replit Secrets
- `stim` installed (`pip install stim`)
- `ray` installed

What it does:
- **Step 1 — Planner**: Grok designs the noise-sweep experiment
- **Step 2 — Local sanity check**: Stim runs all 4 noise levels locally before dispatching agents
- **Step 3 — Parallel agents**: 4 Ray agents each run their own Stim simulation (noise: 0.5%, 1%, 2%, 4%) and call Grok for scientific analysis
- **Step 4 — Reflection**: each agent critiques the prior results and suggests improvements
- **Step 5 — Graph**: agent network saved to `examples/agent_graph_qec.png`
- **Step 6 — Synthesizer**: Grok writes a final scientific report with recommendations
- Results always saved to `examples/qec_results.json`
- Typical runtime: ~40–60 seconds

Graph saved to: `examples/agent_graph_qec.png`
Results saved to: `examples/qec_results.json`

The noise-sweep output from the local sanity check looks like:
```
noise=0.005  →  logical_error_rate=0.0002
noise=0.010  →  logical_error_rate=0.0012
noise=0.020  →  logical_error_rate=0.0050
noise=0.040  →  logical_error_rate=0.0196
```
As expected for a repetition code: logical error rate scales roughly as noise², confirming the code suppresses single-qubit errors.

---

## Example 10 — Adaptive QEC noise sweep (dynamic loop + plot)

Grok acts as a **Navigator agent** that steers the experiment: it sees all data collected so far and picks the most informative noise level to probe next, until it is confident the error-rate curve is well-characterised.

```bash
python3 examples/10_qec_adaptive_loop.py
```

Requirements:
- `XAI_API_KEY` set in Replit Secrets
- `stim` and `matplotlib` installed

What it does:
- **Seed** — one Stim run at noise=0.02 (no API call)
- **Adaptive loop** — each round: Navigator (Grok) chooses the next noise value based on gaps and the threshold region; Stim runs it; loop stops when `confidence >= 0.85` or 8 rounds reached
- **Log-log plot** — error rate vs noise PNG with measured points, probe order annotation, and theoretical 3p² reference line → `examples/qec_noise_curve.png`
- **Agent graph** → `examples/agent_graph_qec_adaptive.png`
- **Synthesizer** — final report with estimated failure threshold and top recommendation
- All history saved to `examples/qec_adaptive_results.json`

Typical runtime: ~30–60 s (sequential rounds, one Grok call each).

---

## Interactive agent CLI

You can also run the natural-language command interface:

```bash
python3 agent.py
```

Type commands like `run cpu`, `run ray`, or `visualize` to interact with the DSL directly.
