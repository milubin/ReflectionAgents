# How to Run the CLI

## 1. Open the Shell
In the Replit interface, find the **Shell** tab (usually at the bottom or in the tools panel). That's your terminal.

## 2. Navigate to the project
```bash
cd artifacts/declarative-parallel-dsl
```

## 3. Run an example
```bash
python3 examples/01_simple_map.py
```
This runs a parallel map over 100,000 numbers using the CPU backend. You'll see output like:
```
CPU time: 1.757s | First 5: [0, 1, 4, 9, 16]
```

## 4. Run the dataflow pipeline example
```bash
python3 examples/02_dataflow_pipeline.py
```

## 5. Run the tests
```bash
python3 tests/test_dsl.py
```

---

## Quick reference — which examples need what

| Example | Command | Requires |
|---|---|---|
| `01_simple_map.py` | CPU parallel map | Nothing extra |
| `02_dataflow_pipeline.py` | Dataflow graph | Nothing extra |
| `03_triton_gpu_kernel.py` | GPU kernel | CUDA GPU |
| `04_ray_distributed.py` | Distributed cluster | Ray installed |

Start with examples 01 and 02 — they work out of the box right now.
