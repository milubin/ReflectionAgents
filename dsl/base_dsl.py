from typing import Callable, List, Any


class ParallelDSL:
    def __init__(self, max_workers: int = 4, backend: str = "cpu"):
        backend = backend.lower()
        if backend == "gpu":
            from .backends.triton_gpu import TritonGPUBackend
            self.backend = TritonGPUBackend()
        elif backend == "ray":
            from .backends.ray_distributed import RayBackend
            self.backend = RayBackend()
        elif backend == "csl":
            from .backends.cerebras_csl import CerebrasCSLBackend
            self.backend = CerebrasCSLBackend()
        else:
            from .backends.cpu import CPUBackend
            self.backend = CPUBackend(max_workers)

    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """Declarative parallel map — works on CPU / GPU / Ray / (future) Cerebras"""
        return self.backend.map(func, items)
