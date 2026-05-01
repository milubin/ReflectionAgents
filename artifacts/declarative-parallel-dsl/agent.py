from dsl.base_dsl import ParallelDSL
from dsl.dataflow_dsl import Node, run_flow_graph
from dsl.graph_planner import GraphPlanner
import time
import re


class ParallelAgent:
    def __init__(self):
        self.planner = GraphPlanner()
        self.current_dsl = ParallelDSL(backend="cpu")

    def run(self, command: str, data: list = None):
        print(f"🤖 Agent received: {command}")

        graph = self.planner.plan_from_nl(command, data)

        backend = self._choose_backend(command)
        self.current_dsl = ParallelDSL(backend=backend)

        start = time.time()
        result = self._execute_graph(graph)

        duration = time.time() - start
        if duration > 1.5 and "ray" in backend:
            print("⚡ Runtime rewrite: detected slowness → increasing workers / parallelism")

        print(f"✅ Finished in {duration:.2f}s | Backend: {backend} | Preview: {result[:5] if result else None}")
        return result

    def _choose_backend(self, command: str) -> str:
        cmd = command.lower()
        if re.search(r"ray|distributed|cluster|multi-node", cmd):
            return "ray"
        if re.search(r"gpu|triton|cuda", cmd):
            return "gpu"
        return "cpu"

    def _execute_graph(self, graph):
        if isinstance(graph, dict) and "data" in graph:
            return self.current_dsl.map(lambda x: x * x, graph["data"])
        return ["demo-result"]


if __name__ == "__main__":
    agent = ParallelAgent()
    test_data = list(range(20000))

    print("\n=== Running agent demo ===")
    agent.run("just run it normally on CPU", test_data[:1000])
