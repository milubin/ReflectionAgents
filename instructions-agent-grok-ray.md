# Running the Agentic Workflow Examples

All commands assume you are in the project root (`/home/runner/workspace`).

## Inspiration

The reflection and critic loops are inspired by the **Reflexion** paper (Shinn et al., 2023):

> **Reflexion: Language Agents with Verbal Reinforcement Learning**
> Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
> arXiv:2303.11366 — https://arxiv.org/abs/2303.11366

Rather than updating model weights, Reflexion agents reflect on their prior outputs verbally and produce improved answers on the next pass. This is exactly the mechanism in examples 07–11: each reflection round the agent reads its own previous analysis (or episode results) and is asked to critique and improve it.

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
| 11 | GameEnv + Ray + Grok (enemy AI) | `examples/agent_graph_enemy_ai.png` |

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

## Example 11 — Adaptive Enemy AI (GameEnv + Ray + Grok Reflexion)

A dungeon RPG enemy that **learns to beat the player** across episodes via a Reflexion-style agentic loop. The enemy starts with a dumb baseline policy (slow pursuit, weak attacks) and improves across up to 7 rounds of multi-agent critique until a Navigator meta-agent is confident the strategy is strong enough.

```bash
python3 examples/11_adaptive_enemy_ai.py
```

Requirements:
- `XAI_API_KEY` set in Replit Secrets
- `ray` and `matplotlib` installed
- No pygame display needed — `GameEnv` is pure Python

### GameEnv adapter interface

The `GameEnv` class is the bridge between the game simulation and the agent prompts:

```python
class GameEnv:
    def reset()                      → GameState dict
    def step(state, policy)          → (new_state, reward, done)
    def serialize_state(state)       → str   # text grid injected into agent prompts
    def parse_action(text)           → dict  # agent output → policy update
    def run_episode(policy, seed)    → EpisodeResult
    def evaluate_policy(policy, n)   → metrics dict
```

`serialize_state` renders the dungeon as a text grid with player `P` and enemy `E` positions, HP bars, and Manhattan distance — everything the Grok agents need to reason about the enemy's situation without any game engine.

### Enemy policy parameters

All 6 parameters are tuned by the agents each round:

| Parameter | Range | Effect |
|---|---|---|
| `speed` | 1–2 | BFS steps per turn (1 = walk, 2 = sprint) |
| `attack_bonus` | 0–15 | Extra flat ATK from learned aggression |
| `crit_chance` | 0.05–0.35 | Enemy critical hit probability |
| `flank_weight` | 0.0–1.0 | 0 = direct BFS chase, 1 = always try to flank |
| `burst_threshold` | 0.0–0.5 | Sprint 2 steps when player HP ratio < this |
| `retreat_hp_pct` | 0.0–0.5 | Retreat when own HP ratio < this |

### Simulation physics (verified)

The dungeon is a 12×9 snake-corridor map. Enemy always pursues via BFS (no aggro gate). Each turn the enemy moves `speed` steps, then attacks if adjacent (Manhattan dist ≤ 1). Player counter-attacks every turn.

- BFS path length from enemy spawn to player spawn: **15 steps**
- Baseline policy: win rate **~2%**, avg damage **~71**
- Fully-tuned policy: win rate **~100%**, avg damage **~111**

### What happens step by step

1. **Baseline** — `GameEnv.evaluate_policy()` runs 12 episodes with the default policy; enemy win rate is ~2%
2. **4 parallel Ray/Grok analysts** dispatch simultaneously:
   - **Tactician** — optimises `attack_bonus` and `crit_chance` (combat timing)
   - **Pathfinder** — optimises `speed` and `flank_weight` (movement patterns)
   - **Predictor** — models player behaviour patterns and suggests counter-strategies
   - **Historian** — reviews the last 4 rounds of episode logs, identifies what worked
3. **Navigator** — meta-agent sees all 4 analyses, resolves conflicts, selects the 1–2 highest-leverage parameter changes, rates its own confidence
4. **Reflection round** — one critic agent per proposed change validates or overrides it
5. **Policy applied** — `apply_changes()` clamps all values to `POLICY_BOUNDS`, then `evaluate_policy()` re-runs 12 episodes
6. Loop repeats until `Navigator.confidence ≥ 0.82` **and** `win_rate ≥ 0.4`, or after 7 rounds

### Stop condition and confidence gating

```python
CONFIDENCE_THRESHOLD = 0.82
MAX_ROUNDS           = 7

if nav["stop"] or nav["confidence"] >= CONFIDENCE_THRESHOLD:
    print("Navigator satisfied — stopping.")
    break
```

The Navigator sets `"stop": true` in its JSON when it believes the policy is strong enough; the outer loop also hard-stops at 7 rounds.

### Outputs

| File | Contents |
|---|---|
| `examples/enemy_learning_curve.png` | Win rate + avg damage per round (bar), Navigator confidence (line) |
| `examples/agent_graph_enemy_ai.png` | Full multi-round agent graph (analysts → Navigator → critics → GameEnv) |
| `examples/enemy_ai_results.json` | Complete log: all policies, metrics, analyst suggestions, critiques, final report |

### Key implementation details

- All Ray remote functions re-import `openai` and `os` locally (no closure capture across workers)
- `_parse_json()` strips markdown fences before `json.loads` — handles Grok's occasional ```json wrapping
- `apply_changes()` applies `POLICY_BOUNDS` clamping so agents can't produce out-of-range values
- `history_summary()` feeds only the last 4 rounds to agents to keep prompts short
- The agent graph is built with `networkx` and rendered via `dsl.visualizer.visualize_graph`

Typical runtime: ~60–120 s (each round: 4 parallel analyst calls + 1 Navigator + N critic calls + 12 sim episodes).

---

## Interactive agent CLI

You can also run the natural-language command interface:

```bash
python3 agent.py
```

Type commands like `run cpu`, `run ray`, or `visualize` to interact with the DSL directly.
