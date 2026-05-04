from dsl.base_dsl import ParallelDSL
from dsl.graph_planner import GraphPlanner
import networkx as nx
import time
import re


class ParallelAgent:
    def __init__(self):
        self.planner = GraphPlanner()
        self.current_dsl = ParallelDSL(backend="cpu")

    def run(self, command: str, data: list = None):
        print(f"\n🤖 Agent received: '{command}'")

        graph = self.planner.plan_from_nl(command, data)

        backend = self._choose_backend(command)
        self.current_dsl = ParallelDSL(backend=backend)

        start = time.time()
        result = self._execute_graph(graph)
        duration = time.time() - start

        if duration > 1.5 and backend == "ray":
            print("⚡ Runtime rewrite: detected slowness → would increase parallelism")

        print(f"✅ Done in {duration:.2f}s | backend={backend} | preview={result[:5]}")
        return result

    def _choose_backend(self, command: str) -> str:
        cmd = command.lower()
        if re.search(r"ray|distributed|cluster|multi-node", cmd):
            if not self._is_available("ray"):
                print("⚠️  Ray is not installed. To use distributed execution, run:")
                print("       pip install ray[default]")
                print("   Falling back to CPU.")
                return "cpu"
            return "ray"
        if re.search(r"gpu|triton|cuda", cmd):
            if not self._is_available("gpu"):
                return "cpu"
            return "gpu"
        return "cpu"

    @staticmethod
    def _is_available(backend: str) -> bool:
        if backend == "gpu":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("⚠️  GPU backend requires a CUDA-capable GPU and these packages:")
                    print("       pip install torch triton")
                    print("   No GPU detected on this machine. Falling back to CPU.")
                    return False
                import triton
                return True
            except ImportError as e:
                missing = str(e).split("'")[1] if "'" in str(e) else str(e)
                print(f"⚠️  GPU backend requires packages that are not installed.")
                print(f"   Missing: {missing}")
                print(f"   Install with: pip install torch triton")
                print(f"   Falling back to CPU.")
                return False
        if backend == "ray":
            try:
                import ray
                return True
            except ImportError:
                return False
        return True

    def _execute_graph(self, graph: dict) -> list:
        """Traverse the networkx graph topologically and apply each node's function."""
        G = graph["graph"]
        current = graph["data"]

        for node in nx.topological_sort(G):
            node_data = G.nodes[node]
            if "func" in node_data:
                current = self.current_dsl.map(node_data["func"], current)

        return current


if __name__ == "__main__":
    agent = ParallelAgent()
    data = list(range(20))

    print("=" * 50)
    print("ReflectionAgents — Interactive Agent")
    print("=" * 50)
    print("Commands you can try:")
    print("  square the numbers on CPU")
    print("  double the numbers on CPU")
    print("  increment then square on CPU")
    print("  make this run distributed on Ray")
    print("  build a pipeline that squares on GPU")
    print("Type 'quit' to exit.")
    print(f"Working data: {data}")
    print("=" * 50)

    while True:
        try:
            command = input("\nEnter command: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not command:
            continue
        if command.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        agent.run(command, data)
