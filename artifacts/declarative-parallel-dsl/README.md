# declarative-parallel-dsl + Agent

Minimal, student-friendly declarative DSL for parallel programming — now with a **full AI agent** that turns natural language into parallel flow graphs.

Example agent commands:
- `"make this run distributed on Ray"`
- `"build a dataflow pipeline that squares numbers on GPU"`
- `"just run it normally on CPU"`

The agent uses networkx for dependency graphs and can dynamically rewrite the plan at runtime (Grok-like behavior).

Inspired by `std::execution::par`, TBB Flow Graph, NVIDIA stdexec, and Cerebras-style dataflow.

## Features
- Same clean API across CPU, Triton GPU, Ray (distributed cluster), and a CSL sketch for Cerebras WSE
- Natural-language agent (`agent.py`) that routes to the right backend automatically
- Real dependency graph planning via networkx (`dsl/graph_planner.py`)
- Grok-like runtime rewrite sketch (observe → plan → rewrite)
- Zero-boilerplate for students
- Easy to extend

## Install
```bash
pip install -e .
```

## Quick Start

### Run the agent
```bash
python agent.py
```

### Run examples
```bash
python examples/01_simple_map.py          # CPU parallel map
python examples/02_dataflow_pipeline.py   # dataflow graph
python examples/03_triton_gpu_kernel.py   # GPU (requires CUDA)
python examples/04_ray_distributed.py     # multi-node / cluster
```

### Run tests
```bash
python tests/test_dsl.py
```

## How It Works

```
Natural language command
        ↓
  GraphPlanner (networkx DiGraph)
        ↓
  Backend selection (CPU / GPU / Ray / CSL)
        ↓
  Parallel execution
        ↓
  Runtime rewrite if needed (Grok-like)
```

## Future Cerebras CSL Backend
The `CerebrasCSLBackend` already generates valid layout + PE code skeletons. When you have access to the Cerebras SDK it will compile and run directly on WSE-3 (both training and inference).

## Optional: Real Grok API
Get a free API key at [x.ai](https://x.ai) and set `GROK_API_KEY` as an environment variable to upgrade the NL parser from keyword-based to LLM-powered.

Happy teaching & hacking!
