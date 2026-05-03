import torch
import triton
import triton.language as tl
from typing import Callable, List, Any


@triton.jit
def square_kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * x
    tl.store(y_ptr + offsets, y, mask=mask)


class TritonGPUBackend:
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        if not items:
            return []
        data = torch.tensor(items, device="cuda", dtype=torch.float32)
        result = torch.empty_like(data)
        n = data.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        square_kernel[grid](data, result, n, BLOCK_SIZE=1024)
        return result.cpu().tolist()
