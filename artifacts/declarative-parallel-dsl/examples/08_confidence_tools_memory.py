"""
examples/08_confidence_tools_memory.py

Advanced agentic workflow adding:
  1. Confidence-based dynamic stopping  — agents self-score 0.0–1.0 via JSON;
     reflection rounds continue until avg confidence > CONFIDENCE_THRESHOLD or
     MAX_ROUNDS is reached.
  2. Tool use                           — agents can call a keyword-extraction
     tool and a word-count tool; results are injected back into the prompt.
  3. Persistent memory                  — a Ray remote object store keeps
     round results alive across reflection loops without re-fetching text.

Graph saved to: examples/agent_graph_confidence.png

Requires: XAI_API_KEY in Replit Secrets, ray installed.
Run:  python3 examples/08_confidence_tools_memory.py
"""
import os
os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
os.environ.setdefault("RAY_USE_MULTIPROCESSING_CPU_COUNT", "1")

import json
import time
import re
import ray
import requests
from typing import List, Dict, Any
from openai import OpenAI

from dsl.visualizer import visualize_graph
import networkx as nx

GROK_MODEL          = "grok-3"
CONFIDENCE_THRESHOLD = 0.88
MAX_ROUNDS          = 4
GRAPH_PATH          = "examples/agent_graph_confidence.png"
DIVIDER             = "─" * 60


# ── Tools available to agents ─────────────────────────────────────────────────

def tool_keyword_extract(text: str, top_n: int = 8) -> List[str]:
    """Return the top_n most frequent meaningful words (simple local tool)."""
    stopwords = {
        "the","a","an","and","or","but","in","on","at","to","of","for",
        "is","are","was","were","it","its","this","that","with","as","by",
        "he","she","they","we","you","i","his","her","their","our","not",
        "had","have","has","from","be","been","which","who","what","so",
        "if","then","when","all","one","more","said","very","would","could",
        "should","than","my","your","him","them","me","do","did","no","but",
    }
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq, key=lambda w: freq[w], reverse=True)
    return sorted_words[:top_n]


def tool_word_count(text: str) -> int:
    return len(text.split())


def run_tools(context: str) -> Dict[str, Any]:
    """Run all available tools on the context and return results."""
    return {
        "keywords":   tool_keyword_extract(context, top_n=8),
        "word_count": tool_word_count(context),
    }


def preview(text: str, chars: int = 120) -> str:
    snippet = text.replace("\n", " ").strip()[:chars]
    return snippet + "..." if len(text) > chars else snippet


# ── Ray remote store (persistent memory across rounds) ────────────────────────

@ray.remote
class MemoryStore:
    """Simple key-value store that lives in the Ray object graph."""
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def put(self, key: str, value: Any):
        self._store[key] = value

    def get(self, key: str, default=None):
        return self._store.get(key, default)

    def all(self) -> Dict[str, Any]:
        return dict(self._store)


# ── Ray remote agent (confidence-aware, tool-using) ──────────────────────────

@ray.remote
def remote_agent_act(
    agent_name: str,
    role: str,
    task: str,
    context: str,
    tool_results: Dict[str, Any],
    prior_output: str = "",
) -> Dict[str, Any]:
    """
    Calls Grok and asks it to return JSON:
      {
        "analysis":   "<3-5 sentence analysis>",
        "confidence": 0.0-1.0,
        "gaps":       "<what is still uncertain>"
      }
    """
    import os, time, json, re
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")

    prior_section = (
        f"\nYour previous analysis (improve on this):\n{prior_output[:2000]}\n"
        if prior_output else ""
    )

    prompt = f"""You are {role}.
Task: {task}

Tool results (use these to ground your analysis):
  - Top keywords in the passage: {tool_results.get('keywords', [])}
  - Word count: {tool_results.get('word_count', 0):,}
{prior_section}
Context (passage excerpt):
{context[:8000]}

Reply ONLY with valid JSON in exactly this format:
{{
  "analysis":   "<your 3-5 sentence analysis>",
  "confidence": <float 0.0-1.0 reflecting how complete/certain your analysis is>,
  "gaps":       "<brief note on what remains uncertain or could be improved>"
}}"""

    start = time.time()
    resp = client.chat.completions.create(
        model="grok-3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=700,
    )
    raw = resp.choices[0].message.content.strip()
    elapsed = round(time.time() - start, 2)

    # Robust JSON parse — strip markdown fences if present
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {"analysis": raw, "confidence": 0.5, "gaps": "JSON parse failed"}

    return {
        "agent":      agent_name,
        "role":       role,
        "output":     parsed.get("analysis", raw),
        "confidence": float(parsed.get("confidence", 0.5)),
        "gaps":       parsed.get("gaps", ""),
        "time":       elapsed,
    }


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_dynamic_graph(agent_defs: List[tuple], rounds_run: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("Planner")
    G.add_node("Synthesizer")
    G.add_node("MemoryStore")
    for name, _ in agent_defs:
        G.add_node(name)
        G.add_edge("Planner", name)
        G.add_edge("MemoryStore", name)
    for r in range(1, rounds_run):
        for name, _ in agent_defs:
            prev = name if r == 1 else f"{name}\nR{r}"
            curr = f"{name}\nR{r+1}"
            G.add_node(curr)
            G.add_edge(prev, curr)
            G.add_edge("MemoryStore", curr)
            G.add_edge(curr, "Synthesizer")
    for name, _ in agent_defs:
        G.add_edge(name, "Synthesizer")
    return G


# ── Main workflow ─────────────────────────────────────────────────────────────

def fetch_pride_and_prejudice() -> str:
    url = "https://www.gutenberg.org/ebooks/1342.txt.utf-8"
    print("📥 Fetching Pride and Prejudice...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    start = text.find("CHAPTER I.")
    end   = text.find("END OF THE PROJECT GUTENBERG EBOOK")
    return text[start:end].strip() if start != -1 and end != -1 else text


def split_into_chunks(text: str, n: int = 4) -> List[str]:
    normalized  = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs  = [p.strip() for p in normalized.split("\n\n") if len(p.strip()) > 100]
    chunk_size  = max(1, len(paragraphs) // n)
    return ["\n\n".join(paragraphs[i:i + chunk_size])
            for i in range(0, len(paragraphs), chunk_size)][:n]


def confidence_workflow():
    full_text = fetch_pride_and_prejudice()
    chunks    = split_into_chunks(full_text, n=4)
    print(f"✅ {len(full_text):,} chars → {len(chunks)} chunks")
    for i, c in enumerate(chunks, 1):
        print(f"   Chunk {i}: {len(c):,} chars — \"{preview(c, 70)}\"")

    agent_defs = [
        ("Researcher", "Plot and Character Summarizer"),
        ("Analyst",    "Themes and Motifs Expert"),
        ("Critic",     "Social Commentary Detector"),
        ("Historian",  "Regency-Era Context Expert"),
    ]

    # Pre-run tools on each chunk (local, fast)
    print(f"\n{DIVIDER}")
    print("TOOLS — running keyword extraction + word count on each chunk")
    print(DIVIDER)
    chunk_tools = []
    for i, chunk in enumerate(chunks, 1):
        tr = run_tools(chunk)
        chunk_tools.append(tr)
        print(f"  Chunk {i}: {tr['word_count']:,} words | keywords: {tr['keywords']}")

    # Start Ray persistent memory store
    memory = MemoryStore.remote()

    results      = []
    rounds_run   = 0
    prior_by_agent: Dict[str, str] = {name: "" for name, _ in agent_defs}

    for round_num in range(1, MAX_ROUNDS + 1):
        rounds_run = round_num
        print(f"\n{DIVIDER}")
        print(f"ROUND {round_num} — {'Initial analysis' if round_num == 1 else 'Reflection (confidence-based)'}")
        print(DIVIDER)

        futures = [
            remote_agent_act.remote(
                name, role,
                "Deeply analyze this passage using the tool results to ground your claims.",
                chunk,
                tool_results,
                prior_by_agent[name],
            )
            for (name, role), chunk, tool_results in zip(agent_defs, chunks, chunk_tools)
        ]
        round_results = ray.get(futures)

        # Persist this round to memory store
        ray.get(memory.put.remote(f"round_{round_num}", round_results))

        for r in round_results:
            prior_by_agent[r["agent"]] = r["output"]
            flag = "✅" if r["confidence"] >= CONFIDENCE_THRESHOLD else "🔄"
            print(f"\n  {flag} [{r['agent']}] confidence={r['confidence']:.2f} ({r['time']}s)")
            print(f"     {r['output']}")
            if r["gaps"]:
                print(f"     Gaps: {r['gaps']}")

        avg_conf = sum(r["confidence"] for r in round_results) / len(round_results)
        print(f"\n  → Average confidence: {avg_conf:.2f} (threshold: {CONFIDENCE_THRESHOLD})")

        results = round_results
        if avg_conf >= CONFIDENCE_THRESHOLD:
            print(f"  🎯 Threshold reached after round {round_num} — stopping early!")
            break
        elif round_num < MAX_ROUNDS:
            print(f"  ↩  Below threshold — running reflection round {round_num + 1}...")

    # Save graph (now we know how many rounds ran)
    print(f"\n{DIVIDER}")
    print(f"Building graph ({rounds_run} round(s) run) → {GRAPH_PATH}")
    print(DIVIDER)
    G = build_dynamic_graph(agent_defs, rounds_run)
    visualize_graph(
        G,
        title=f"Confidence-Gated Reflection — {rounds_run} round(s)",
        output_path=GRAPH_PATH,
    )

    # Final synthesis
    print(f"\n{DIVIDER}")
    print("SYNTHESIZER — combining all agent outputs")
    print(DIVIDER)
    combined = "\n\n---\n\n".join(
        f"[{r['agent']} | conf={r['confidence']:.2f}]\n{r['output']}" for r in results
    )
    final = ray.get(remote_agent_act.remote(
        "Synthesizer", "Final Report Writer",
        "Write a polished literary analysis report from these agent outputs.",
        combined,
        {},
        "",
    ))
    print(f"  [Synthesizer] ✓ {final['time']}s")

    # Dump full memory store
    all_memory = ray.get(memory.all.remote())
    print(f"\n  📦 Memory store has {len(all_memory)} round(s) of results.")

    return {"rounds_run": rounds_run, "results": results, "final": final}


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    print("=" * 60)
    print("🔥  Confidence-Gated Agentic Workflow")
    print(f"    Threshold={CONFIDENCE_THRESHOLD} · Max rounds={MAX_ROUNDS}")
    print("    Tools · Ray memory · Dynamic stopping")
    print("=" * 60)

    total_start = time.time()
    result = confidence_workflow()

    print(f"\n{'='*60}")
    print("📖  FINAL LITERARY ANALYSIS")
    print("=" * 60)
    print(result["final"]["output"])
    print(f"\n⏱️  Total time  : {round(time.time()-total_start, 1)}s")
    print(f"Rounds run     : {result['rounds_run']} / {MAX_ROUNDS}")
    print(f"Graph saved    : {GRAPH_PATH}")
    print("=" * 60)
