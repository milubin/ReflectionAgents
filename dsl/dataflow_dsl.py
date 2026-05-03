from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, List


class Node:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func
        self.successors: List["Node"] = []

    def then(self, next_node: "Node"):
        self.successors.append(next_node)
        return next_node


def run_flow_graph(source_data: List[Any], start_node: Node) -> List[Any]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(start_node.func, item): item for item in source_data}
        return [f.result() for f in futures]
