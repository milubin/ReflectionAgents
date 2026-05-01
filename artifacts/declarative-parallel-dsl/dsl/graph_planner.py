import networkx as nx
from typing import List, Dict, Any


class GraphPlanner:
    def plan_from_nl(self, command: str, data: List[Any] = None) -> Dict:
        """Turn natural language into a real dependency graph using networkx."""
        G = nx.DiGraph()

        cmd = command.lower()

        if "square" in cmd or "gpu" in cmd:
            G.add_node("square", func=lambda x: x * x, backend="gpu")
        if "sum" in cmd:
            G.add_node("sum", func=sum, backend="cpu")

        if not G.nodes:
            G.add_node("identity", func=lambda x: x, backend="cpu")

        if data:
            G.add_node("input_data", value=data)
            first_compute = next(
                (n for n in G.nodes if n != "input_data"), None
            )
            if first_compute:
                G.add_edge("input_data", first_compute)

        if len(data or []) > 10000:
            G.graph["parallelism"] = "high"

        print(f"📊 Built graph with {len(G.nodes)} nodes, {len(G.edges)} edges")
        return {"graph": G, "data": data or []}
