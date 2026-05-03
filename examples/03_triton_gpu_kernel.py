from dsl.base_dsl import ParallelDSL
import time


def heavy_compute(x):  # placeholder — backend uses Triton kernel
    return x * x


if __name__ == "__main__":
    data = list(range(1_000_000))
    dsl = ParallelDSL(backend="gpu")
    start = time.time()
    result = dsl.map(heavy_compute, data)
    print(f"GPU time: {time.time()-start:.3f}s | First 5: {result[:5]}")
