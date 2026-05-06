"""
examples/11_adaptive_enemy_ai.py

Adaptive Enemy AI via Reflexion-style Agentic Loop
  - Pure Python grid simulation of dungeon combat (no display needed)
  - Enemy policy has 6 tunable parameters: aggro range, attack bonus,
    crit chance, flanking weight, burst threshold, retreat HP fraction
  - Each episode: player vs enemy in a corridor dungeon, 10 trials/policy
  - 4 parallel Grok agents (Tactician, Pathfinder, Predictor, Historian)
    analyse episode results and propose policy improvements
  - Navigator meta-agent picks the highest-leverage change each round
  - Reflection / critic round validates suggestions before applying
  - Confidence gating stops the loop when agents agree policy is strong
  - Learning curve saved → examples/enemy_learning_curve.png
  - Agent graph saved   → examples/agent_graph_enemy_ai.png
  - Results saved       → examples/enemy_ai_results.json

The enemy literally learns to beat you across episodes.

Requires:
  pip install ray matplotlib
  XAI_API_KEY set in Replit Secrets

Run: python3 examples/11_adaptive_enemy_ai.py
"""
import os
import logging
os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import json
import re
import time
import copy
import random
import ray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from openai import OpenAI

from dsl.base_dsl import ParallelDSL
from dsl.visualizer import visualize_graph

# ── Config ─────────────────────────────────────────────────────────────────────
GROK_MODEL           = "grok-3"
EPISODES_PER_POLICY  = 12     # sim trials per policy evaluation
MAX_TURNS_PER_EP     = 60     # hard cap on turns per episode
CONFIDENCE_THRESHOLD = 0.82
MAX_ROUNDS           = 7
DIVIDER              = "─" * 60

CURVE_PATH   = "examples/enemy_learning_curve.png"
GRAPH_PATH   = "examples/agent_graph_enemy_ai.png"
RESULTS_PATH = "examples/enemy_ai_results.json"

# ── Dungeon map (12 × 9) ────────────────────────────────────────────────────────
# Compact snake-corridor dungeon: enemy reaches player in ~14 BFS steps
# '.' walkable  '#' wall  'P' player spawn  'E' enemy spawn
SIM_MAP = [
    "############",
    "#P.........#",
    "#.########.#",
    "#..........#",
    "#.########.#",
    "#..........#",
    "#.########.#",
    "#.........E#",
    "############",
]

WALKABLE = set('.PE')

def _map_cells():
    cells = {}
    for r, row in enumerate(SIM_MAP):
        for c, ch in enumerate(row):
            cells[(c, r)] = ch
    return cells

MAP_CELLS   = _map_cells()
MAP_COLS    = len(SIM_MAP[0])
MAP_ROWS    = len(SIM_MAP)
PLAYER_START = next((c, r) for (c, r), ch in MAP_CELLS.items() if ch == 'P')
ENEMY_START  = next((c, r) for (c, r), ch in MAP_CELLS.items() if ch == 'E')


# ── Policy & stats ─────────────────────────────────────────────────────────────

BASELINE_POLICY = {
    "speed":           1,     # BFS steps per turn (1=walk, 2=sprint)
    "attack_bonus":    0,     # flat ATK bonus from learned aggression
    "crit_chance":     0.10,  # enemy critical hit probability
    "flank_weight":    0.0,   # 0=direct BFS chase, 1.0=always try to flank
    "burst_threshold": 0.0,   # sprint 2 steps when player HP ratio < this (0=never)
    "retreat_hp_pct":  0.0,   # retreat when own HP ratio < this (0=never)
}

PLAYER_STATS = {"hp": 100, "atk": 12, "defense": 5, "magic": 10}
ENEMY_STATS  = {"hp": 65,  "atk": 14, "defense": 4}   # zombie-tier baseline


# ── BFS pathfinder ─────────────────────────────────────────────────────────────

def bfs_next_step(start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[Tuple[int,int]]:
    """Return the first step on the shortest walkable path from start to goal."""
    if start == goal:
        return start
    visited = {start}
    queue   = deque([(start, [])])
    while queue:
        pos, path = queue.popleft()
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nb = (pos[0]+dx, pos[1]+dy)
            if nb in visited:
                continue
            if MAP_CELLS.get(nb, '#') not in WALKABLE:
                continue
            visited.add(nb)
            new_path = path + [nb]
            if nb == goal:
                return new_path[0] if new_path else start
            queue.append((nb, new_path))
    return None   # no path (shouldn't happen in connected map)


def flank_step(enemy: Tuple[int,int], player: Tuple[int,int]) -> Optional[Tuple[int,int]]:
    """Try to move perpendicular to the player direction (flanking manoeuvre)."""
    dx = player[0] - enemy[0]
    dy = player[1] - enemy[1]
    perp = [(-dy, dx), (dy, -dx)]   # two perpendicular directions
    random.shuffle(perp)
    for pdx, pdy in perp:
        nb = (enemy[0] + (1 if pdx > 0 else -1 if pdx < 0 else 0),
              enemy[1] + (1 if pdy > 0 else -1 if pdy < 0 else 0))
        if MAP_CELLS.get(nb, '#') in WALKABLE:
            return nb
    return bfs_next_step(enemy, player)


# ── Episode result ─────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    enemy_won:        bool
    turns:            int
    damage_to_player: int
    player_hp_end:    int
    enemy_hp_end:     int
    crits:            int


# ── GameEnv ────────────────────────────────────────────────────────────────────

class GameEnv:
    """
    Pure Python dungeon combat simulation.
    No pygame, no display — runs in milliseconds.
    Follows the adapter interface described in the design notes.
    """

    def reset(self):
        return {
            "player_pos": PLAYER_START,
            "enemy_pos":  ENEMY_START,
            "player_hp":  PLAYER_STATS["hp"],
            "enemy_hp":   ENEMY_STATS["hp"],
            "turn":       0,
        }

    def serialize_state(self, state: Dict) -> str:
        px, py = state["player_pos"]
        ex, ey = state["enemy_pos"]
        dist   = abs(px - ex) + abs(py - ey)
        lines  = []
        for r, row in enumerate(SIM_MAP):
            rendered = ""
            for c, ch in enumerate(row):
                if (c, r) == (px, py): rendered += "P"
                elif (c, r) == (ex, ey): rendered += "E"
                else: rendered += ch
            lines.append(rendered)
        grid = "\n".join(lines)
        return (
            f"Turn {state['turn']}  |  "
            f"Player HP: {state['player_hp']}/{PLAYER_STATS['hp']}  "
            f"Enemy HP: {state['enemy_hp']}/{ENEMY_STATS['hp']}  "
            f"Manhattan dist: {dist}\n{grid}"
        )

    def parse_action(self, text: str) -> Dict:
        """Extract policy parameter overrides from agent text (used for future extensions)."""
        return {}

    def step(self, state: Dict, policy: Dict) -> Tuple[Dict, float, bool]:
        """Advance one turn. Returns (new_state, reward, done)."""
        s      = copy.deepcopy(state)
        px, py = s["player_pos"]
        ex, ey = s["enemy_pos"]

        # ── Enemy movement — always pursues, policy controls HOW ─────────────
        e_hp_ratio = s["enemy_hp"] / ENEMY_STATS["hp"]
        p_hp_ratio = s["player_hp"] / PLAYER_STATS["hp"]

        # Retreat if own HP is critically low
        retreating = (policy["retreat_hp_pct"] > 0 and
                      e_hp_ratio < policy["retreat_hp_pct"])

        # Burst: sprint 2 steps when player is weak
        bursting   = (policy["burst_threshold"] > 0 and
                      p_hp_ratio < policy["burst_threshold"])

        steps_this_turn = 2 if (bursting or policy["speed"] >= 2) else 1

        for _ in range(steps_this_turn):
            if retreating:
                # Move one step away from player (toward corner furthest from player)
                candidates = []
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nb = (ex+dx, ey+dy)
                    if MAP_CELLS.get(nb, '#') in WALKABLE:
                        dist_nb = abs(nb[0]-px) + abs(nb[1]-py)
                        candidates.append((nb, dist_nb))
                if candidates:
                    ex, ey = max(candidates, key=lambda x: x[1])[0]
            else:
                if (policy["flank_weight"] > 0 and
                        random.random() < policy["flank_weight"]):
                    step = flank_step((ex, ey), (px, py))
                else:
                    step = bfs_next_step((ex, ey), (px, py))
                if step and step != (px, py):
                    ex, ey = step

        s["enemy_pos"] = (ex, ey)

        reward  = 0.0
        done    = False
        new_dist = abs(px - ex) + abs(py - ey)

        # ── Combat (adjacent = dist 1) ──────────────────────────────────────
        if new_dist <= 1:
            # Enemy attacks
            e_atk  = ENEMY_STATS["atk"] + policy["attack_bonus"]
            crit   = random.random() < policy["crit_chance"]
            e_dmg  = max(1, e_atk + random.randint(-2, 3) - PLAYER_STATS["defense"])
            if crit:
                e_dmg = int(e_dmg * 1.7)
            s["player_hp"] = max(0, s["player_hp"] - e_dmg)
            reward += e_dmg * 0.1

            # Player counter-attacks
            p_dmg = max(1, PLAYER_STATS["atk"] + random.randint(-2, 4) - ENEMY_STATS["defense"])
            if random.random() < 0.15:
                p_dmg = int(p_dmg * 1.8)
            s["enemy_hp"] = max(0, s["enemy_hp"] - p_dmg)

        if s["player_hp"] <= 0:
            done = True; reward += 5.0
        if s["enemy_hp"]  <= 0:
            done = True; reward -= 3.0

        s["turn"] += 1
        if s["turn"] >= MAX_TURNS_PER_EP:
            done = True

        return s, reward, done

    def run_episode(self, policy: Dict, seed: int = 0) -> EpisodeResult:
        random.seed(seed)
        state   = self.reset()
        total_dmg = 0
        crits     = 0
        p_start   = state["player_hp"]

        while True:
            old_php = state["player_hp"]
            state, _, done = self.step(state, policy)
            dmg = old_php - state["player_hp"]
            if dmg > 0:
                total_dmg += dmg
                if dmg > (ENEMY_STATS["atk"] + policy["attack_bonus"]) * 1.5:
                    crits += 1
            if done:
                break

        return EpisodeResult(
            enemy_won        = state["player_hp"] <= 0,
            turns            = state["turn"],
            damage_to_player = total_dmg,
            player_hp_end    = state["player_hp"],
            enemy_hp_end     = state["enemy_hp"],
            crits            = crits,
        )

    def evaluate_policy(self, policy: Dict, n: int = EPISODES_PER_POLICY) -> Dict:
        results = [self.run_episode(policy, seed=i) for i in range(n)]
        wins    = sum(1 for r in results if r.enemy_won)
        return {
            "win_rate":         round(wins / n, 3),
            "avg_damage":       round(sum(r.damage_to_player for r in results) / n, 1),
            "avg_turns":        round(sum(r.turns for r in results) / n, 1),
            "avg_player_hp_end": round(sum(r.player_hp_end for r in results) / n, 1),
            "avg_crits":        round(sum(r.crits for r in results) / n, 2),
            "episodes":         n,
        }


# ── Grok helpers ───────────────────────────────────────────────────────────────

def _grok_client():
    key = os.getenv("XAI_API_KEY")
    if not key:
        raise ValueError("XAI_API_KEY not set — add it in Replit Secrets.")
    return OpenAI(api_key=key, base_url="https://api.x.ai/v1")

def _parse_json(raw: str) -> Dict:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"raw": raw, "confidence": 0.4}


# ── Parallel analyst agents ────────────────────────────────────────────────────

@ray.remote
def analyst_agent(
    role: str,
    focus: str,
    policy: Dict,
    metrics: Dict,
    history_summary: str,
    round_num: int,
) -> Dict:
    import os, time, re, json
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    start  = time.time()

    policy_str = json.dumps(policy, indent=2)
    metrics_str = json.dumps(metrics, indent=2)

    prompt = f"""You are a {role} for an adaptive enemy AI system in a dungeon RPG.

CURRENT ENEMY POLICY (parameters you can tune):
{policy_str}

Parameter ranges:
  speed:            1–2   (BFS steps per turn: 1=walk, 2=sprint — key pursuit speed)
  attack_bonus:     0–15  (extra flat ATK from learned aggression)
  crit_chance:      0.05–0.35 (enemy critical hit probability)
  flank_weight:     0.0–1.0 (0=direct BFS chase, 1=always try to flank)
  burst_threshold:  0.0–0.5 (sprint 2 steps when player HP < this fraction; 0=never)
  retreat_hp_pct:   0.0–0.5 (retreat when own HP < this fraction; 0=never)

EPISODE RESULTS (round {round_num}):
{metrics_str}

EPISODE HISTORY:
{history_summary}

Your focus: {focus}

The enemy wants to WIN more (increase win_rate) and deal more damage.
Current win_rate={metrics['win_rate']} avg_damage={metrics['avg_damage']}

Analyse from your specialised perspective and suggest ONE OR TWO specific parameter changes
that would most improve enemy performance.

Reply ONLY with valid JSON:
{{
  "analysis":    "<2-3 sentence analysis from your role's perspective>",
  "suggestions": [
    {{"param": "<param_name>", "new_value": <value>, "reason": "<why>"}}
  ],
  "confidence":  <float 0.0–1.0 how confident you are these help>
}}"""

    resp = client.chat.completions.create(
        model="grok-3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.55,
        max_tokens=500,
    )
    parsed = _parse_json(resp.choices[0].message.content)
    return {
        "role":        role,
        "analysis":    parsed.get("analysis", ""),
        "suggestions": parsed.get("suggestions", []),
        "confidence":  float(parsed.get("confidence", 0.5)),
        "time":        round(time.time() - start, 2),
    }


# ── Navigator ──────────────────────────────────────────────────────────────────

@ray.remote
def navigator_agent(
    all_suggestions: List[Dict],
    policy: Dict,
    metrics: Dict,
    round_num: int,
) -> Dict:
    import os, time, json
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    start  = time.time()

    sug_text = ""
    for a in all_suggestions:
        sug_text += f"\n[{a['role']} | conf={a['confidence']:.2f}]\n"
        sug_text += f"  Analysis: {a['analysis']}\n"
        for s in a.get("suggestions", []):
            sug_text += f"  → {s['param']} = {s['new_value']}  ({s['reason']})\n"

    prompt = f"""You are a Navigator meta-agent coordinating an adaptive enemy AI experiment.

Current policy: {json.dumps(policy)}
Current metrics: {json.dumps(metrics)}
Round: {round_num}

Agent suggestions:
{sug_text}

Your job:
1. Select the BEST 1-3 parameter changes to apply this round (highest expected impact)
2. Resolve conflicts if agents disagree
3. Rate your confidence (0–1) that the selected changes will improve win_rate
4. Note: win_rate={metrics['win_rate']}, aim > 0.5 eventually

Stop condition: if confidence >= 0.82 AND win_rate >= 0.4, declare the policy good.

Reply ONLY with valid JSON:
{{
  "apply": [
    {{"param": "<param_name>", "new_value": <value>, "reason": "<why selected>"}}
  ],
  "rationale":  "<1-2 sentence strategy summary>",
  "confidence": <float 0.0–1.0>,
  "stop":       <true if policy is now strong enough, false otherwise>
}}"""

    resp = client.chat.completions.create(
        model="grok-3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35,
        max_tokens=400,
    )
    parsed = _parse_json(resp.choices[0].message.content)
    return {
        "apply":      parsed.get("apply", []),
        "rationale":  parsed.get("rationale", ""),
        "confidence": float(parsed.get("confidence", 0.5)),
        "stop":       bool(parsed.get("stop", False)),
        "time":       round(time.time() - start, 2),
    }


# ── Critic / reflection ────────────────────────────────────────────────────────

@ray.remote
def critic_agent(role: str, suggestion: Dict, metrics: Dict, policy: Dict) -> Dict:
    import os, time
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    start  = time.time()

    prompt = f"""You are a critic reviewing a proposed enemy AI policy change.

Proposed change: {json.dumps(suggestion)}
Current metrics: {json.dumps(metrics)}
Current policy:  {json.dumps(policy)}

Is this change well-reasoned? Will it actually improve enemy win rate?
Check for:
- Conflicting parameters (e.g. retreat + burst simultaneously)
- Out-of-range values
- Diminishing returns (already near ceiling)

Reply ONLY with valid JSON:
{{
  "verdict":   "approve" | "modify" | "reject",
  "comment":   "<1-2 sentence critique>",
  "override":  {{"param": "<name>", "new_value": <val>}} or null
}}"""

    resp = client.chat.completions.create(
        model="grok-3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=250,
    )
    parsed = _parse_json(resp.choices[0].message.content)
    return {
        "critic":   role,
        "verdict":  parsed.get("verdict", "approve"),
        "comment":  parsed.get("comment", ""),
        "override": parsed.get("override"),
        "time":     round(time.time() - start, 2),
    }


# ── Policy application ─────────────────────────────────────────────────────────

POLICY_BOUNDS = {
    "speed":           (1,    2),
    "attack_bonus":    (0,   15),
    "crit_chance":     (0.05, 0.35),
    "flank_weight":    (0.0, 1.0),
    "burst_threshold": (0.0, 0.5),
    "retreat_hp_pct":  (0.0, 0.5),
}

def apply_changes(policy: Dict, changes: List[Dict]) -> Dict:
    p = copy.deepcopy(policy)
    for ch in changes:
        param = ch.get("param")
        val   = ch.get("new_value")
        if param in p and val is not None:
            lo, hi = POLICY_BOUNDS[param]
            p[param] = max(lo, min(hi, type(p[param])(val)))
    return p


def history_summary(log: List[Dict]) -> str:
    if not log:
        return "No prior episodes."
    lines = []
    for i, entry in enumerate(log[-4:]):   # last 4
        m = entry["metrics"]
        lines.append(
            f"  Round {entry['round']}: win_rate={m['win_rate']}  "
            f"avg_dmg={m['avg_damage']}  avg_turns={m['avg_turns']}"
        )
    return "\n".join(lines)


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_learning_curve(log: List[Dict]) -> None:
    rounds    = [e["round"] for e in log]
    win_rates = [e["metrics"]["win_rate"] for e in log]
    avg_dmgs  = [e["metrics"]["avg_damage"] for e in log]
    conf      = [e.get("navigator_confidence", 0) for e in log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(rounds, win_rates, "o-", color="#E53935", linewidth=2, markersize=7,
             label="Enemy win rate")
    ax1.axhline(0.5, linestyle="--", color="#888", linewidth=1, label="50% threshold")
    for i, (r, w) in enumerate(zip(rounds, win_rates)):
        ax1.annotate(f"{w:.0%}", (r, w), textcoords="offset points",
                     xytext=(4, 6), fontsize=8)
    ax1.set_ylabel("Enemy win rate", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title("Adaptive Enemy AI — Learning Curve\n"
                  "(Reflexion-style agentic loop: Tactician · Pathfinder · Predictor · Historian)",
                  fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.bar(rounds, avg_dmgs, color="#1565C0", alpha=0.75, label="Avg damage to player")
    ax2_r = ax2.twinx()
    ax2_r.plot(rounds, conf, "s--", color="#6A1B9A", linewidth=1.5,
               markersize=5, label="Navigator confidence")
    ax2_r.set_ylim(0, 1.1)
    ax2_r.set_ylabel("Navigator confidence", fontsize=10, color="#6A1B9A")
    ax2.set_xlabel("Episode round", fontsize=11)
    ax2.set_ylabel("Avg damage dealt", fontsize=11)
    ax2.set_xticks(rounds)
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Learning curve saved → {CURVE_PATH}")


# ── Agent graph ────────────────────────────────────────────────────────────────

def build_agent_graph(n_rounds: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("GameEnv\n(Sim)")
    prev = "GameEnv\n(Sim)"
    for i in range(1, n_rounds + 1):
        nav   = f"Navigator\nR{i}"
        nodes = [f"Tactician\nR{i}", f"Pathfinder\nR{i}",
                 f"Predictor\nR{i}",  f"Historian\nR{i}"]
        critics = [f"Critic\nR{i}a", f"Critic\nR{i}b"]
        for n in nodes:
            G.add_node(n)
            G.add_edge(prev, n)
            G.add_edge(n, nav)
        G.add_node(nav)
        for c in critics:
            G.add_node(c)
            G.add_edge(nav, c)
        sim = f"GameEnv\nR{i}→"
        G.add_node(sim)
        for c in critics:
            G.add_edge(c, sim)
        prev = sim
    G.add_node("Final Policy")
    G.add_edge(prev, "Final Policy")
    return G


# ── Main loop ──────────────────────────────────────────────────────────────────

def adaptive_enemy_loop():
    env     = GameEnv()
    policy  = copy.deepcopy(BASELINE_POLICY)
    log     : List[Dict] = []
    round_n = 0

    agent_defs = [
        ("Tactician",  "Combat Tactician",
         "Optimise attack timing, crit chance, and burst behaviour during combat"),
        ("Pathfinder", "Movement Pathfinder",
         "Optimise enemy movement: aggro range, flanking, and retreat decisions"),
        ("Predictor",  "Player Behaviour Predictor",
         "Predict how the player plays and suggest counter-strategies"),
        ("Historian",  "Episode Historian",
         "Review what worked in past episodes and identify repeating patterns"),
    ]

    # ── Baseline episode ──────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("BASELINE — evaluating default policy")
    print(DIVIDER)
    print(f"  Policy: {policy}")
    metrics = env.evaluate_policy(policy)
    print(f"  Results: win_rate={metrics['win_rate']}  "
          f"avg_dmg={metrics['avg_damage']}  avg_turns={metrics['avg_turns']}")
    log.append({"round": 0, "policy": copy.deepcopy(policy),
                "metrics": metrics, "navigator_confidence": 0.0})

    while round_n < MAX_ROUNDS:
        round_n += 1
        print(f"\n{DIVIDER}")
        print(f"ROUND {round_n} — 4 analyst agents in parallel")
        print(DIVIDER)

        # ── Step 1: 4 parallel analysts ───────────────────────────────────────
        hist = history_summary(log)
        futures = [
            analyst_agent.remote(role, focus, policy, metrics, hist, round_n)
            for role, _, focus in agent_defs
        ]
        analysts = ray.get(futures)

        print(f"\n  Analyst results:")
        for a in analysts:
            print(f"\n  [{a['role']}] conf={a['confidence']:.2f}  ({a['time']}s)")
            print(f"    {a['analysis']}")
            for s in a.get("suggestions", []):
                print(f"    → {s['param']} = {s['new_value']}  ({s.get('reason','')})")

        # ── Step 2: Navigator picks the best changes ──────────────────────────
        print(f"\n  Navigator deciding best changes...")
        nav = ray.get(navigator_agent.remote(analysts, policy, metrics, round_n))
        print(f"  Navigator conf={nav['confidence']:.3f}  stop={nav['stop']}")
        print(f"  Rationale: {nav['rationale']}")
        for ch in nav["apply"]:
            print(f"  Apply: {ch['param']} = {ch['new_value']}  ({ch.get('reason','')})")

        # ── Step 3: Critic / reflection ───────────────────────────────────────
        print(f"\n  Reflection — critics validate proposed changes...")
        critic_futures = [
            critic_agent.remote(
                f"Critic-{agent_defs[i % len(agent_defs)][0]}",
                ch, metrics, policy
            )
            for i, ch in enumerate(nav["apply"])
        ]
        critiques = ray.get(critic_futures) if critic_futures else []

        # Apply overrides from critics
        final_changes = list(nav["apply"])
        for crit, ch in zip(critiques, final_changes):
            print(f"  [{crit['critic']}] verdict={crit['verdict']}  {crit['comment']}")
            if crit["verdict"] == "modify" and crit.get("override"):
                ov = crit["override"]
                for fc in final_changes:
                    if fc["param"] == ov["param"]:
                        fc["new_value"] = ov["new_value"]
                        print(f"    Override: {ov['param']} → {ov['new_value']}")

        # ── Step 4: Apply policy, evaluate ───────────────────────────────────
        policy  = apply_changes(policy, final_changes)
        metrics = env.evaluate_policy(policy)
        log.append({
            "round":                round_n,
            "policy":               copy.deepcopy(policy),
            "metrics":              metrics,
            "analyst_suggestions":  analysts,
            "navigator":            nav,
            "critiques":            critiques,
            "navigator_confidence": nav["confidence"],
        })

        print(f"\n  Updated policy: {policy}")
        print(f"  New metrics: win_rate={metrics['win_rate']}  "
              f"avg_dmg={metrics['avg_damage']}  avg_turns={metrics['avg_turns']}")

        if nav["stop"] or nav["confidence"] >= CONFIDENCE_THRESHOLD:
            print(f"\n  ✅ Navigator satisfied (conf={nav['confidence']:.3f}) — stopping.")
            break

    return log, policy, metrics


# ── Final report ───────────────────────────────────────────────────────────────

@ray.remote
def final_report_agent(log: List[Dict], final_policy: Dict) -> Dict:
    import os, re, json, time
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    start  = time.time()

    progression = "\n".join(
        f"  Round {e['round']}: win_rate={e['metrics']['win_rate']}  "
        f"avg_dmg={e['metrics']['avg_damage']:.1f}  policy={e['policy']}"
        for e in log
    )

    prompt = f"""You are writing a final analysis of an adaptive enemy AI experiment.

An enemy in a dungeon RPG started with a dumb policy (straight-line chase, random attacks)
and improved over {len(log)-1} reflection rounds guided by 4 Grok agents:
  Tactician, Pathfinder, Predictor, Historian.

Win rate progression and policies:
{progression}

Final policy: {json.dumps(final_policy, indent=2)}

Write a concise report (4-6 sentences) covering:
1. How much the enemy improved (win rate change)
2. Which parameter changes had the biggest impact and why
3. What kind of player the final enemy is most dangerous against
4. One further improvement that would make it even stronger

Reply ONLY with valid JSON:
{{
  "report":           "<full report>",
  "biggest_impact":   "<most important policy change>",
  "player_weakness":  "<what player behaviour this enemy exploits best>"
}}"""

    resp = client.chat.completions.create(
        model="grok-3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=600,
    )
    parsed = _parse_json(resp.choices[0].message.content)
    return {**parsed, "time": round(time.time() - start, 2)}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False,
             object_store_memory=200 * 1024 * 1024,
             logging_level=logging.ERROR)

    print("=" * 60)
    print("🎮  Adaptive Enemy AI — Reflexion-style Agentic Loop")
    print("    GameEnv · Ray · Grok · Reflection · Confidence gating")
    print("    The enemy learns to beat you across episodes")
    print("=" * 60)

    total_start = time.time()

    # ── Show baseline state serialization ─────────────────────────────────────
    env = GameEnv()
    state0 = env.reset()
    print(f"\n{DIVIDER}")
    print("Initial dungeon state (serialized for agent prompts):")
    print(DIVIDER)
    print(env.serialize_state(state0))

    # ── Run adaptive loop ──────────────────────────────────────────────────────
    log, final_policy, final_metrics = adaptive_enemy_loop()

    # ── Learning curve plot ───────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"Plotting learning curve → {CURVE_PATH}")
    print(DIVIDER)
    plot_learning_curve(log)

    # ── Agent graph ───────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"Building agent graph → {GRAPH_PATH}")
    print(DIVIDER)
    n_rounds = max(1, len(log) - 1)
    G = build_agent_graph(n_rounds)
    visualize_graph(
        G,
        title="Adaptive Enemy AI — Reflexion Loop (Tactician · Pathfinder · Predictor · Historian)",
        output_path=GRAPH_PATH,
    )

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("Final report agent...")
    print(DIVIDER)
    report = ray.get(final_report_agent.remote(log, final_policy))

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump({"log": log, "final_policy": final_policy,
                   "final_metrics": final_metrics, "report": report},
                  f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────────────
    baseline = log[0]["metrics"]
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print("="*60)
    print(report.get("report", report))
    print(f"\n  Biggest impact : {report.get('biggest_impact', '')}")
    print(f"  Player weakness: {report.get('player_weakness', '')}")
    print(f"\n  Win rate  : {baseline['win_rate']} → {final_metrics['win_rate']}"
          f"  ({'+' if final_metrics['win_rate'] >= baseline['win_rate'] else ''}"
          f"{round(final_metrics['win_rate'] - baseline['win_rate'], 3):+.3f})")
    print(f"  Avg damage: {baseline['avg_damage']} → {final_metrics['avg_damage']}")
    print(f"  Rounds run: {n_rounds}")
    print(f"  Final policy: {final_policy}")
    print(f"\n  Curve  → {CURVE_PATH}")
    print(f"  Graph  → {GRAPH_PATH}")
    print(f"  Results→ {RESULTS_PATH}")
    print(f"  Time   : {round(time.time()-total_start, 1)}s")
    print("=" * 60)
