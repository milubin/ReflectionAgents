"""
examples/05_agentic_workflow.py

Agentic workflow using the declarative DSL — no external API required.
Demonstrates: task decomposition → parallel agents → synthesis → graph visualization.
"""
from dsl.base_dsl import ParallelDSL
from dsl.visualizer import visualize_graph
import networkx as nx
import time
from typing import List, Dict, Any


class Agent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def act(self, task: str) -> Dict[str, Any]:
        print(f"  [{self.name} | {self.role}] → {task[:70]}...")
        time.sleep(0.2)
        return {
            "agent": self.name,
            "role": self.role,
            "output": f"[{self.role}] Analysis complete for: '{task[:50]}'",
            "confidence": 0.85,
        }


def build_workflow_graph(workers: List[Agent], subtasks: List[str]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("Planner")
    G.add_node("Synthesizer")
    for worker, task in zip(workers, subtasks):
        label = f"{worker.name}\n{task[:25]}..."
        G.add_node(label)
        G.add_edge("Planner", label)
        G.add_edge(label, "Synthesizer")
    return G


def agentic_workflow(main_task: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"🧠 Task: {main_task}")
    print('='*60)

    # 1. Planner decomposes the task
    planner = Agent("Planner", "Task Decomposition")
    subtasks = [
        "Gather background data and key facts",
        "Analyze patterns and emerging trends",
        "Identify weaknesses, risks, and open questions",
    ]
    planner.act(main_task)
    print("\n📋 Subtasks:")
    for i, st in enumerate(subtasks, 1):
        print(f"  {i}. {st}")

    # 2. Parallel worker agents
    workers = [
        Agent("Researcher", "researcher"),
        Agent("Analyst",    "analyst"),
        Agent("Critic",     "critic"),
    ]

    # Visualize the graph before running
    G = build_workflow_graph(workers, subtasks)
    visualize_graph(G, title=f"Agent Graph — CPU: {main_task[:50]}", output_path="examples/agent_graph_cpu.png")

    # Run in parallel via the DSL
    dsl = ParallelDSL(backend="cpu")
    pairs = list(zip(workers, subtasks))

    def run_agent(pair):
        worker, task = pair
        return worker.act(task)

    print("\n🚀 Running parallel agents via DSL...\n")
    results = dsl.map(run_agent, pairs)

    # 3. Synthesizer combines results
    synthesizer = Agent("Synthesizer", "synthesizer")
    combined = " | ".join(r["output"] for r in results)
    final = synthesizer.act(f"Synthesize: {combined}")

    print("\n=== FINAL REPORT ===")
    print(final["output"])
    print(f"\nParallel agents used: {len(results)}")
    return {"subtasks": subtasks, "results": results, "final": final}


if __name__ == "__main__":
    agentic_workflow("Write a report on the future of wafer-scale AI hardware")
