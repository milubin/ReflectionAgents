"""
examples/07_agentic_grok_ray_reflection.py

Agentic workflow with:
- Real Grok API (client recreated safely inside each Ray remote task)
- Ray distributed parallelism
- Reflection / critic loops (agents critique each other iteratively)
- Large public text (Pride and Prejudice)
- Graph visualization  →  examples/agent_graph_grok_ray.png

Requires: XAI_API_KEY environment variable set in Replit Secrets.

Run: python3 examples/07_agentic_grok_ray_reflection.py
"""
import os
import logging
os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
import time
import ray
import requests
from typing import List, Dict, Any
from openai import OpenAI

from dsl.base_dsl import ParallelDSL
from dsl.visualizer import visualize_graph
import networkx as nx

GROK_MODEL = "grok-3"
DIVIDER = "─" * 60
GRAPH_PATH = "examples/agent_graph_grok_ray.png"


def create_grok_client() -> OpenAI:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ XAI_API_KEY not set.\n"
            "   Add it in Replit: click the 🔒 Secrets tab → New Secret → XAI_API_KEY"
        )
    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


def fetch_pride_and_prejudice() -> str:
    url = "https://www.gutenberg.org/ebooks/1342.txt.utf-8"
    print("📥 Fetching Pride and Prejudice (~750 kB)...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    start = text.find("CHAPTER I.")
    end = text.find("END OF THE PROJECT GUTENBERG EBOOK")
    return text[start:end].strip() if start != -1 and end != -1 else text


def split_into_chunks(text: str, num_chunks: int = 4) -> List[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in normalized.split("\n\n") if len(p.strip()) > 100]
    chunk_size = max(1, len(paragraphs) // num_chunks)
    return ["\n\n".join(paragraphs[i:i + chunk_size])
            for i in range(0, len(paragraphs), chunk_size)][:num_chunks]


def preview(text: str, chars: int = 120) -> str:
    snippet = text.replace("\n", " ").strip()[:chars]
    return snippet + "..." if len(text) > chars else snippet


@ray.remote
def remote_agent_act(agent_name: str, role: str, task: str, context: str = "") -> Dict[str, Any]:
    """Ray-safe: Grok client recreated inside each remote task (avoids pickling issues)."""
    import os, time
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    start = time.time()
    prompt = (
        f"You are {role}.\n"
        f"Task: {task}\n"
        f"Context:\n{context[:10000]}\n\n"
        f"Reply with a clear, concise analysis (3-5 sentences)."
    )
    resp = client.chat.completions.create(
        model="grok-3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=600,
    )
    return {
        "agent": agent_name,
        "role": role,
        "output": resp.choices[0].message.content,
        "time": round(time.time() - start, 2),
    }


def build_reflection_graph(agent_defs: List[tuple], rounds: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("Planner")
    G.add_node("Synthesizer")

    for name, _ in agent_defs:
        G.add_node(name)
        G.add_edge("Planner", name)

    for r in range(1, rounds):
        for name, _ in agent_defs:
            prev = name if r == 1 else f"{name}\nCritic R{r}"
            critic_node = f"{name}\nCritic R{r+1}"
            G.add_node(critic_node)
            G.add_edge(prev, critic_node)

    for node in G.nodes:
        if node != "Synthesizer":
            G.add_edge(node, "Synthesizer")

    return G


def agentic_grok_ray_reflection_workflow(rounds: int = 2):
    full_text = fetch_pride_and_prejudice()
    chunks = split_into_chunks(full_text, num_chunks=4)
    print(f"✅ Loaded {len(full_text):,} chars → {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, 1):
        print(f"   Chunk {i}: {len(chunk):,} chars — \"{preview(chunk, 80)}\"")

    agent_defs = [
        ("Researcher", "Plot and Character Summarizer"),
        ("Analyst",    "Themes and Motifs Expert"),
        ("Critic",     "Social Commentary Detector"),
        ("Historian",  "Regency-Era Context Expert"),
    ]

    # Visualize full graph (initial agents + reflection rounds + synthesizer)
    print(f"\n{DIVIDER}")
    print(f"Building agent graph ({rounds} rounds) → {GRAPH_PATH}")
    print(DIVIDER)
    G = build_reflection_graph(agent_defs, rounds)
    visualize_graph(
        G,
        title=f"Ray + Grok + Reflection ({rounds} rounds) — Pride and Prejudice",
        output_path=GRAPH_PATH,
    )

    # ── Round 1: initial parallel analysis ────────────────────────
    print(f"\n{DIVIDER}")
    print("ROUND 1 — Parallel Grok agents via Ray")
    print(DIVIDER)
    print("\n🚀 Dispatching agents in parallel via Ray...\n")
    for (name, role), chunk in zip(agent_defs, chunks):
        print(f"  ┌─ [{name}] {role}")
        print(f"  │  📄 \"{preview(chunk, 100)}\"")
        print(f"  └─ submitted to Ray ✓")

    futures = [
        remote_agent_act.remote(name, role, "Deeply analyze this passage.", chunk)
        for (name, role), chunk in zip(agent_defs, chunks)
    ]
    results = ray.get(futures)

    print("\n  Results received:")
    for r in results:
        print(f"\n  [{r['agent']} — {r['role']}] ({r['time']}s)")
        print(f"  {r['output']}")

    # ── Reflection rounds ──────────────────────────────────────────
    for round_num in range(1, rounds):
        print(f"\n{DIVIDER}")
        print(f"REFLECTION ROUND {round_num + 1} — agents critique previous outputs")
        print(DIVIDER)
        print("\n🔄 Dispatching critic agents via Ray...\n")

        critique_futures = [
            remote_agent_act.remote(
                f"{r['agent']}_Critic_R{round_num + 1}",
                "critic",
                f"Critique and improve this analysis:\n{r['output'][:4000]}",
                ""
            )
            for r in results
        ]
        critique_results = ray.get(critique_futures)

        print("  Critique results:")
        for r in critique_results:
            print(f"\n  [{r['agent']}] ({r['time']}s)")
            print(f"  {r['output']}")

        results.extend(critique_results)

    # ── Synthesizer ────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("FINAL STEP — Synthesizer combines all outputs")
    print(DIVIDER)
    combined = "\n\n---\n\n".join(r["output"] for r in results)
    print(f"\n📝 Sending {len(results)} agent outputs to Synthesizer...\n")
    final = ray.get(remote_agent_act.remote(
        "Synthesizer", "synthesizer",
        "Write a coherent, high-quality literary analysis report from all agent outputs.",
        combined
    ))
    print(f"  [Synthesizer] ✓ {final['time']}s")

    return {"all_results": results, "final_report": final}


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False,
             object_store_memory=200 * 1024 * 1024,
             logging_level=logging.ERROR)

    print("=" * 60)
    print("🔥  Ray + Grok + Reflection Agentic Workflow")
    print("    Real API · Distributed · Self-critique loops")
    print("=" * 60)
    total_start = time.time()
    result = agentic_grok_ray_reflection_workflow(rounds=2)

    print(f"\n{'='*60}")
    print("📖  FINAL LITERARY ANALYSIS (after reflection loops)")
    print("="*60)
    print(result["final_report"]["output"])
    print(f"\n⏱️  Total time : {round(time.time()-total_start, 1)}s")
    print(f"Agent outputs (incl. reflections): {len(result['all_results'])}")
    print(f"\n🎉  Complete! Graph saved → {GRAPH_PATH}")
    print("=" * 60)
