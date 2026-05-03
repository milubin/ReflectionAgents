from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any


class CPUBackend:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(func, item): i for i, item in enumerate(items)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results
