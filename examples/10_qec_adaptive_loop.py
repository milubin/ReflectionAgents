"""
examples/10_qec_adaptive_loop.py

Adaptive QEC Experiment Loop
  - 3-qubit bit-flip repetition code using Stim
  - Grok agent decides the NEXT noise level to probe based on prior results
  - Loop runs until the agent is satisfied (confidence >= threshold) or
    a max iteration cap is reached
  - Error rate vs noise curve saved → examples/qec_noise_curve.png
  - Agent workflow graph saved       → examples/agent_graph_qec_adaptive.png
  - Full history saved               → examples/qec_adaptive_results.json

How the adaptive loop works:
  1. Start with one seed measurement at noise=0.02
  2. A Grok "Navigator" agent sees all results so far and decides:
       - which noise level to probe next  (float, 0.001 – 0.20)
       - its confidence that the curve is well-characterised (0–1)
       - a brief scientific rationale
  3. We run that Stim simulation, append the point, repeat
  4. When confidence >= CONFIDENCE_THRESHOLD or MAX_ROUNDS reached → stop
  5. Plot the full curve and save the PNG

Requires:
  pip install stim ray matplotlib
  XAI_API_KEY set in Replit Secrets

Run: python3 examples/10_qec_adaptive_loop.py
"""
import os
import logging
os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import json
import re
import time
import ray
import numpy as np
import stim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any, Tuple
from openai import OpenAI

from dsl.base_dsl import ParallelDSL
from dsl.visualizer import visualize_graph

# ── Config ─────────────────────────────────────────────────────────────────────
GROK_MODEL           = "grok-3"
SHOTS                = 20_000          # per simulation (more = less noise in estimates)
CONFIDENCE_THRESHOLD = 0.85            # stop when Navigator is this confident
MAX_ROUNDS           = 8              # hard cap
SEED_NOISE           = 0.02           # first measurement
NOISE_MIN            = 0.001
NOISE_MAX            = 0.20

DIVIDER      = "─" * 60
CURVE_PATH   = "examples/qec_noise_curve.png"
GRAPH_PATH   = "examples/agent_graph_qec_adaptive.png"
RESULTS_PATH = "examples/qec_adaptive_results.json"


# ── Grok client ────────────────────────────────────────────────────────────────

def create_grok_client() -> OpenAI:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError(
            "XAI_API_KEY not set.\n"
            "Replit: Secrets tab -> New Secret -> XAI_API_KEY"
        )
    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


# ── Stim simulation ─────────────────────────────────────────────────────────────

def run_repetition_code(noise: float, shots: int = SHOTS) -> Dict:
    """
    3-qubit bit-flip repetition code.
    R -> DEPOLARIZE1 -> M 0 1 2, majority-vote decoding.
    """
    circuit = stim.Circuit(f"""
        R 0 1 2
        DEPOLARIZE1({noise}) 0 1 2
        M 0 1 2
    """)
    sampler      = circuit.compile_sampler()
    measurements = sampler.sample(shots=shots)
    errors       = int(np.sum(np.sum(measurements, axis=1) >= 2))
    return {
        "noise":              noise,
        "shots":              shots,
        "logical_errors":     errors,
        "logical_error_rate": round(errors / shots, 7),
    }


# ── Navigator agent (decides next noise level) ─────────────────────────────────

@ray.remote
def navigator_agent(history: List[Dict], round_num: int) -> Dict:
    """
    Sees all (noise, error_rate) pairs collected so far.
    Returns: next_noise, confidence, rationale.
    """
    import os, json, re, time
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    start  = time.time()

    history_text = "\n".join(
        f"  noise={r['noise']:.4f}  logical_error_rate={r['logical_error_rate']:.6f}"
        for r in sorted(history, key=lambda x: x["noise"])
    )

    prompt = f"""You are a quantum error correction experiment navigator.
You are designing a noise sweep for a 3-qubit bit-flip repetition code.
The code suppresses errors quadratically: logical_error_rate ≈ 3 * noise^2 (for small noise).

Current data collected (round {round_num}):
{history_text}

Your goal: choose the NEXT noise value to probe to best characterise the error-rate curve.
Priorities:
- Fill large gaps in the data
- Probe the threshold region (where the code starts failing badly, ~noise > 0.10)
- Avoid duplicating points already measured (within 0.002 tolerance)
- Keep noise in [{NOISE_MIN}, {NOISE_MAX}]

Rate your confidence (0.0–1.0) that the curve is already well-characterised:
- < 0.5: many gaps remain
- 0.7: reasonable coverage
- 0.85+: curve is well-sampled across low, mid, and high noise regimes

Reply ONLY with valid JSON (no markdown):
{{
  "next_noise": <float {NOISE_MIN}–{NOISE_MAX}>,
  "confidence": <float 0.0–1.0>,
  "rationale":  "<1-2 sentence scientific justification>"
}}"""

    resp = client.chat.completions.create(
        model="grok-3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300,
    )
    raw   = resp.choices[0].message.content.strip()
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(clean)
        next_noise  = float(parsed["next_noise"])
        confidence  = float(parsed["confidence"])
        rationale   = str(parsed.get("rationale", ""))
        # clamp
        next_noise = max(NOISE_MIN, min(NOISE_MAX, next_noise))
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        next_noise = 0.05
        confidence = 0.3
        rationale  = "(parse error — fallback)"

    return {
        "round":      round_num,
        "next_noise": round(next_noise, 4),
        "confidence": round(confidence, 3),
        "rationale":  rationale,
        "time":       round(time.time() - start, 2),
    }


# ── Synthesizer agent ──────────────────────────────────────────────────────────

@ray.remote
def synthesizer_agent(history: List[Dict]) -> Dict:
    """Writes a concise final scientific summary of the full noise sweep."""
    import os, json, re, time
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    start  = time.time()

    data_text = "\n".join(
        f"  noise={r['noise']:.4f}  logical_error_rate={r['logical_error_rate']:.6f}  "
        f"({r['logical_errors']}/{r['shots']} errors)"
        for r in sorted(history, key=lambda x: x["noise"])
    )

    prompt = f"""You are a quantum computing scientist writing a final summary report.

Noise sweep results for a 3-qubit bit-flip repetition code ({SHOTS:,} shots each):
{data_text}

Write a concise scientific report (4-6 sentences) covering:
1. Whether the quadratic suppression (logical_error_rate ≈ 3 * noise^2) holds
2. Where the code starts to fail (identify the threshold region)
3. Key recommendations for improving the code (e.g., more qubits, better decoder)

Reply ONLY with valid JSON:
{{
  "report":          "<full report text>",
  "threshold_noise": <estimated noise value where code begins to fail>,
  "recommendation":  "<single most impactful improvement>"
}}"""

    resp = client.chat.completions.create(
        model="grok-3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=600,
    )
    raw   = resp.choices[0].message.content.strip()
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(clean)
    except Exception:
        parsed = {"report": raw, "threshold_noise": None, "recommendation": ""}

    return {**parsed, "time": round(time.time() - start, 2)}


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_noise_curve(history: List[Dict], navigator_log: List[Dict]) -> None:
    pts     = sorted(history, key=lambda x: x["noise"])
    noises  = [p["noise"] for p in pts]
    rates   = [p["logical_error_rate"] for p in pts]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Measured points
    ax.scatter(noises, rates, color="#2196F3", zorder=5, s=80, label="Measured")
    ax.plot(noises, rates, color="#2196F3", linewidth=1.5, alpha=0.6)

    # Theoretical quadratic reference: 3 * p^2
    p_ref  = np.linspace(min(noises) * 0.5, max(noises) * 1.1, 200)
    r_ref  = 3 * p_ref ** 2
    ax.plot(p_ref, r_ref, "--", color="#FF5722", linewidth=1.2,
            label=r"Theory: $3p^2$ (repetition code)")

    # Annotate order of probing
    for i, p in enumerate(pts):
        ax.annotate(str(i + 1), (p["noise"], p["logical_error_rate"]),
                    textcoords="offset points", xytext=(4, 6),
                    fontsize=7, color="#555555")

    ax.set_xlabel("Physical noise rate  $p$", fontsize=12)
    ax.set_ylabel("Logical error rate", fontsize=12)
    ax.set_title("3-Qubit Bit-Flip Repetition Code\nAdaptive Noise Sweep (Stim + Grok navigator)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)

    # Confidence progression sidebar text
    conf_text = "Navigator confidence per round:\n" + "\n".join(
        f"  R{n['round']}: {n['confidence']:.2f}  (→ noise={n['next_noise']:.4f})"
        for n in navigator_log
    )
    fig.text(0.02, 0.02, conf_text, fontsize=6.5, family="monospace",
             verticalalignment="bottom", color="#444444")

    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {CURVE_PATH}")


# ── Agent graph ────────────────────────────────────────────────────────────────

def build_adaptive_graph(n_rounds: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("Seed\n(noise=0.02)")
    prev = "Seed\n(noise=0.02)"
    for i in range(1, n_rounds + 1):
        nav  = f"Navigator\nR{i}"
        sim  = f"Stim\nR{i}"
        G.add_node(nav)
        G.add_node(sim)
        G.add_edge(prev, nav)
        G.add_edge(nav, sim)
        prev = sim
    G.add_node("Synthesizer")
    G.add_edge(prev, "Synthesizer")
    G.add_node("Plot\n(PNG)")
    G.add_edge("Synthesizer", "Plot\n(PNG)")
    return G


# ── Main adaptive loop ─────────────────────────────────────────────────────────

def adaptive_qec_loop():
    print(f"\n{DIVIDER}")
    print("SEED — first measurement at noise=0.02")
    print(DIVIDER)
    seed = run_repetition_code(SEED_NOISE)
    print(f"  noise={seed['noise']:.4f}  "
          f"logical_error_rate={seed['logical_error_rate']:.6f}  "
          f"({seed['logical_errors']}/{seed['shots']} errors)")

    history       : List[Dict] = [seed]
    navigator_log : List[Dict] = []
    round_num = 0

    while round_num < MAX_ROUNDS:
        round_num += 1
        print(f"\n{DIVIDER}")
        print(f"ROUND {round_num} — Navigator chooses next noise level")
        print(DIVIDER)

        nav_result = ray.get(navigator_agent.remote(history, round_num))
        navigator_log.append(nav_result)

        conf      = nav_result["confidence"]
        next_n    = nav_result["next_noise"]
        rationale = nav_result["rationale"]

        print(f"  Confidence so far : {conf:.3f}  (threshold={CONFIDENCE_THRESHOLD})")
        print(f"  Next noise chosen : {next_n:.4f}")
        print(f"  Rationale         : {rationale}")
        print(f"  Navigator time    : {nav_result['time']}s")

        if conf >= CONFIDENCE_THRESHOLD:
            print(f"\n  ✅ Confidence {conf:.3f} >= {CONFIDENCE_THRESHOLD} — stopping loop.")
            break

        # Check for near-duplicate
        already_measured = any(abs(h["noise"] - next_n) < 0.002 for h in history)
        if already_measured:
            print(f"  ⚠️  noise={next_n:.4f} already measured — nudging +0.005")
            next_n = round(min(NOISE_MAX, next_n + 0.005), 4)

        print(f"\n  Running Stim simulation  noise={next_n:.4f}  shots={SHOTS:,}...")
        sim = run_repetition_code(next_n)
        history.append(sim)
        print(f"  logical_error_rate={sim['logical_error_rate']:.6f}  "
              f"({sim['logical_errors']}/{sim['shots']} errors)")
    else:
        print(f"\n  ⏹  Max rounds ({MAX_ROUNDS}) reached.")

    return history, navigator_log


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False,
             object_store_memory=200 * 1024 * 1024,
             logging_level=logging.ERROR)

    print("=" * 60)
    print("🔬  Adaptive QEC Noise Sweep")
    print("    Stim · Ray · Grok navigator · adaptive loop · plot")
    print("    3-Qubit Bit-Flip Repetition Code")
    print("=" * 60)

    total_start = time.time()

    # ── Adaptive loop ──────────────────────────────────────────────────────────
    history, navigator_log = adaptive_qec_loop()

    # ── Agent graph ────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"Building agent graph → {GRAPH_PATH}")
    print(DIVIDER)
    G = build_adaptive_graph(len(navigator_log))
    visualize_graph(
        G,
        title="Adaptive QEC Loop — Navigator + Stim + Grok",
        output_path=GRAPH_PATH,
    )

    # ── Plot ───────────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"Plotting noise curve → {CURVE_PATH}")
    print(DIVIDER)
    plot_noise_curve(history, navigator_log)

    # ── Synthesizer ────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("Synthesizer — final scientific report")
    print(DIVIDER)
    final = ray.get(synthesizer_agent.remote(history))
    print(f"\n  Time: {final['time']}s")

    # ── Save ───────────────────────────────────────────────────────────────────
    payload = {
        "history":       history,
        "navigator_log": navigator_log,
        "final_report":  final,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL QEC REPORT")
    print("="*60)
    print(final.get("report", final))
    if final.get("threshold_noise"):
        print(f"\n  Estimated failure threshold : noise ≈ {final['threshold_noise']}")
    if final.get("recommendation"):
        print(f"  Top recommendation          : {final['recommendation']}")

    print(f"\n  Data points collected : {len(history)}")
    print(f"  Rounds run            : {len(navigator_log)}")
    print(f"  Noise values probed   : {sorted(round(h['noise'],4) for h in history)}")
    print(f"\n  Curve plot  → {CURVE_PATH}")
    print(f"  Agent graph → {GRAPH_PATH}")
    print(f"  Results     → {RESULTS_PATH}")
    print(f"  Total time  : {round(time.time()-total_start, 1)}s")
    print("=" * 60)
