"""
examples/07_agentic_grok_ray_reflection.py

Agentic workflow with:
- Real Grok API (client recreated safely inside each Ray remote task)
- Ray distributed parallelism
- Reflection / critic loops (agents critique each other iteratively)
- Large public text (Pride and Prejudice)
- Graph visualization

Requires: XAI_API_KEY environment variable set in Replit Secrets.
          pip install ray[default] openai requests

Run: python examples/07_agentic_grok_ray_reflection.py
"""
import os
import time
import ray
import requests
from typing import List, Dict, Any
from openai import OpenAI

from dsl.base_dsl import ParallelDSL
from dsl.visualizer import visualize_graph
import networkx as nx

GROK_MODEL = "grok-3"


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


@ray.remote
def remote_agent_act(agent_name: str, role: str, task: str, context: str = "") -> Dict[str, Any]:
    """Ray-safe: recreates the Grok client inside the remote task (no pickling issues)."""
    client = create_grok_client()
    start = time.time()
    prompt = (
        f"You are {role}.\n"
        f"Task: {task}\n"
        f"Context:\n{context[:10000]}\n\n"
        f"Reply with a clear, concise analysis (3-5 sentences)."
    )
    resp = client.chat.completions.create(
        model=GROK_MODEL,
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


def build_reflection_graph(agents: List[tuple], rounds: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("Planner")
    G.add_node("Synthesizer")

    prev_nodes = []
    for name, role in agents:
        G.add_node(name)
        G.add_edge("Planner", name)
        prev_nodes.append(name)

    for r in range(1, rounds):
        curr_nodes = []
        for name, _ in agents:
            critic_node = f"{name}\n(Critic R{r+1})"
            G.add_node(critic_node)
            G.add_edge(name if r == 1 else f"{name}\n(Critic R{r})", critic_node)
            curr_nodes.append(critic_node)

    for node in G.nodes:
        if node != "Synthesizer":
            G.add_edge(node, "Synthesizer")

    return G


def agentic_grok_ray_reflection_workflow(rounds: int = 2):
    full_text = fetch_pride_and_prejudice()
    chunks = split_into_chunks(full_text, num_chunks=4)
    print(f"✅ Loaded {len(full_text):,} chars → {len(chunks)} chunks\n")

    agent_defs = [
        ("Researcher", "Plot and Character Summarizer"),
        ("Analyst",    "Themes and Motifs Expert"),
        ("Critic",     "Social Commentary Detector"),
        ("Historian",  "Regency-Era Context Expert"),
    ]

    # Visualize the full workflow graph including reflection rounds
    G = build_reflection_graph(agent_defs, rounds)
    visualize_graph(G, title=f"Ray + Grok + Reflection ({rounds} rounds)", output_path="agent_graph_07.png")

    # Round 1: parallel initial analysis
    print("🚀 Round 1: Parallel Grok agents via Ray...\n")
    futures = [
        remote_agent_act.remote(name, role, "Deeply analyze this passage.", chunk)
        for (name, role), chunk in zip(agent_defs, chunks)
    ]
    results = ray.get(futures)
    for r in results:
        print(f"  [{r['agent']}] ✓ {r['time']}s")

    # Reflection rounds: agents critique previous outputs
    for round_num in range(1, rounds):
        print(f"\n🔄 Reflection Round {round_num + 1}: agents critique each other...\n")
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
        for r in critique_results:
            print(f"  [{r['agent']}] ✓ {r['time']}s")
        results.extend(critique_results)

    # Synthesizer
    print("\n📝 Final synthesis...\n")
    combined = "\n\n---\n\n".join(r["output"] for r in results)
    final = ray.get(remote_agent_act.remote(
        "Synthesizer", "synthesizer",
        "Write a coherent, high-quality literary analysis from all agent outputs.",
        combined
    ))

    return {"all_results": results, "final_report": final}


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    print("🔥 Ray + Grok + Reflection Agentic Workflow\n")
    total_start = time.time()
    result = agentic_grok_ray_reflection_workflow(rounds=2)

    print("\n" + "="*70)
    print("📖 FINAL LITERARY ANALYSIS (after reflection loops)")
    print("="*70)
    print(result["final_report"]["output"])
    print(f"\n⏱️  Total: {round(time.time()-total_start, 1)}s")
    print(f"Total agent outputs (incl. reflections): {len(result['all_results'])}")
    print("\n🎉 Full agentic workflow: Ray + Grok API + self-critique complete!")
