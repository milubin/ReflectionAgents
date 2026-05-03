from dsl.base_dsl import ParallelDSL
import time


def heavy_compute(x):
    return x * x


if __name__ == "__main__":
    data = list(range(1_000_000))
    dsl = ParallelDSL(backend="ray")   # auto-starts Ray cluster
    start = time.time()
    result = dsl.map(heavy_compute, data)
    print(f"Ray distributed time: {time.time()-start:.3f}s | First 5: {result[:5]}")
