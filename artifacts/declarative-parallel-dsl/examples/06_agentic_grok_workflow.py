"""
examples/06_agentic_grok_workflow.py

Agentic workflow with REAL Grok API + large public text (Pride and Prejudice).
Requires: XAI_API_KEY environment variable set in Replit Secrets.

Run: python3 examples/06_agentic_grok_workflow.py
"""
import os
import time
from typing import List, Dict, Any

import requests
from openai import OpenAI

from dsl.base_dsl import ParallelDSL
from dsl.visualizer import visualize_graph
import networkx as nx

GROK_MODEL = "grok-3"
DIVIDER = "─" * 60


def get_grok_client() -> OpenAI:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ XAI_API_KEY not set.\n"
            "   Add it in Replit: click the 🔒 Secrets tab → New Secret → XAI_API_KEY"
        )
    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


def fetch_pride_and_prejudice() -> str:
    url = "https://www.gutenberg.org/ebooks/1342.txt.utf-8"
    print("📥 Fetching Pride and Prejudice (~750 kB, public domain)...")
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
    chunks = ["\n\n".join(paragraphs[i:i + chunk_size])
              for i in range(0, len(paragraphs), chunk_size)]
    return chunks[:num_chunks]


def preview(text: str, chars: int = 200) -> str:
    """Return the first `chars` characters of text, cleaned up."""
    snippet = text.replace("\n", " ").strip()[:chars]
    return snippet + "..." if len(text) > chars else snippet


class GrokAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def act(self, task: str, context: str = "") -> Dict[str, Any]:
        print(f"\n  ┌─ [{self.name}] {self.role}")
        if context:
            print(f"  │  📄 Processing: \"{preview(context, 120)}\"")
        print(f"  │  ⏳ Calling Grok API...", flush=True)

        client = get_grok_client()
        start = time.time()
        prompt = (
            f"You are {self.role}.\n"
            f"Task: {task}\n"
            f"Context (excerpt):\n{context[:8000]}\n\n"
            f"Reply with a clear, concise analysis (3-5 sentences)."
        )
        resp = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        output = resp.choices[0].message.content
        elapsed = round(time.time() - start, 2)

        print(f"  │  ✅ Done in {elapsed}s")
        print(f"  │  💬 Response: \"{preview(output, 180)}\"")
        print(f"  └{'─'*56}")

        return {"agent": self.name, "role": self.role, "output": output, "time": elapsed}


def build_workflow_graph(agents: List[GrokAgent]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("Planner")
    G.add_node("Synthesizer")
    for agent in agents:
        G.add_node(agent.name)
        G.add_edge("Planner", agent.name)
        G.add_edge(agent.name, "Synthesizer")
    return G


def agentic_grok_workflow():
    full_text = fetch_pride_and_prejudice()
    chunks = split_into_chunks(full_text, num_chunks=4)
    print(f"✅ Loaded {len(full_text):,} characters → split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, 1):
        print(f"   Chunk {i}: {len(chunk):,} chars — \"{preview(chunk, 80)}\"")

    # ── Step 1: Planner ───────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("STEP 1 — Planner decomposes the task")
    print(DIVIDER)
    planner = GrokAgent("Planner", "Task Decomposition Specialist")
    plan = planner.act(
        "Decompose a literary analysis of Pride and Prejudice into 4 parallel subtasks.",
        "Analyze themes, characters, social commentary, and narrative style."
    )
    print(f"\n📋 Full planner output:\n{plan['output']}\n")

    # ── Step 2: Parallel agents ────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("STEP 2 — Parallel agents (running simultaneously via DSL)")
    print(DIVIDER)

    agents = [
        GrokAgent("Researcher", "Plot and Character Summarizer"),
        GrokAgent("Analyst",    "Themes and Motifs Expert"),
        GrokAgent("Critic",     "Social Commentary and Irony Detector"),
        GrokAgent("Historian",  "Regency-Era Context Expert"),
    ]

    G = build_workflow_graph(agents)
    visualize_graph(G, title="Grok Agentic Workflow — CPU + Pride and Prejudice", output_path="examples/agent_graph_grok_cpu.png")

    dsl = ParallelDSL(backend="cpu")
    pairs = list(zip(agents, chunks))

    def run_grok_agent(pair):
        agent, chunk = pair
        return agent.act("Deeply analyze this passage.", chunk)

    print("\n🚀 Dispatching 4 agents in parallel...\n")
    results = dsl.map(run_grok_agent, pairs)

    # Print all results in full after parallel run
    print(f"\n{DIVIDER}")
    print("STEP 2 — Individual agent results (full)")
    print(DIVIDER)
    for r in results:
        print(f"\n[{r['agent']} — {r['role']}] ({r['time']}s)")
        print(r["output"])

    # ── Step 3: Synthesizer ────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("STEP 3 — Synthesizer combines all outputs")
    print(DIVIDER)
    synthesizer = GrokAgent("Synthesizer", "Final Report Writer")
    combined = "\n\n---\n\n".join(r["output"] for r in results)
    final = synthesizer.act("Write a coherent literary analysis from these agent outputs.", combined)

    return {"plan": plan, "results": results, "final": final}


if __name__ == "__main__":
    print("=" * 60)
    print("🔥  Agentic Grok Workflow")
    print("    Real Grok API + Pride and Prejudice")
    print("=" * 60)
    total_start = time.time()
    result = agentic_grok_workflow()

    print(f"\n{'='*60}")
    print("📖  FINAL LITERARY ANALYSIS REPORT")
    print("="*60)
    print(result["final"]["output"])
    print(f"\n⏱️  Total time : {round(time.time()-total_start, 1)}s")
    print(f"Parallel agents: {len(result['results'])}")
    print("\n🎉  Grok API agentic workflow complete!")
    print("    Graph saved → examples/agent_graph_grok_cpu.png")
    print("=" * 60)
