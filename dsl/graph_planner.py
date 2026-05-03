import networkx as nx
from typing import List, Dict, Any, Tuple, Callable
import re


class GraphPlanner:
    def plan_from_nl(self, command: str, data: List[Any] = None) -> Dict:
        """Turn natural language into a real dependency graph using networkx."""
        G = nx.DiGraph()
        cmd = command.lower()

        ops = self._extract_ops(cmd)

        G.add_node("input", data=data or [])
        prev = "input"
        for i, (name, func, backend) in enumerate(ops):
            node_id = f"{name}_{i}"
            G.add_node(node_id, func=func, backend=backend)
            G.add_edge(prev, node_id)
            prev = node_id

        if len(data or []) > 10000:
            G.graph["parallelism"] = "high"

        order = list(nx.topological_sort(G))
        print(f"📊 Graph: {' → '.join(order)}")
        return {"graph": G, "data": data or []}

    def _extract_ops(self, cmd: str) -> List[Tuple[str, Callable, str]]:
        """Extract ordered operations from the natural-language command."""
        ops = []

        if re.search(r"square|x\*x|x\^2|\*\*2", cmd):
            ops.append(("square", lambda x: x * x, "gpu" if "gpu" in cmd else "cpu"))
        if re.search(r"double|multiply by 2|\* ?2|times 2", cmd):
            ops.append(("double", lambda x: x * 2, "cpu"))
        if re.search(r"increment|add 1|\+1", cmd):
            ops.append(("increment", lambda x: x + 1, "cpu"))
        if re.search(r"negate|negative|flip sign", cmd):
            ops.append(("negate", lambda x: -x, "cpu"))

        if not ops:
            ops.append(("identity", lambda x: x, "cpu"))

        return ops
