from dsl.base_dsl import ParallelDSL
import time


def heavy_compute(x):
    return x * x


if __name__ == "__main__":
    data = list(range(100_000))
    dsl = ParallelDSL(backend="cpu")
    start = time.time()
    result = dsl.map(heavy_compute, data)
    print(f"CPU time: {time.time()-start:.3f}s | First 5: {result[:5]}")
