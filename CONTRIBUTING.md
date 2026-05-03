# Contributing to Declarative-Parallel-Dsl

Thank you for your interest in contributing! This project is designed as an educational and prototyping tool for declarative parallel and agentic AI systems.

## Ways to Contribute

- **New examples** (especially agentic patterns, tool use, or domain-specific workflows)
- **New backends** (Cerebras CSL, additional GPU kernels, cloud providers, etc.)
- **Improvements to core DSL** (better error handling, configuration, visualization)
- **Documentation** (tutorials, more comments, blog-style explanations)
- **Bug fixes and performance tweaks**
- **Ideas / Issues** — even just suggestions are valuable

## Development Setup

```bash
git clone https://github.com/milubin/Declarative-Parallel-Dsl.git
cd Declarative-Parallel-Dsl
pip install -e .
pip install ray        # for distributed examples (07, 08)
```

Set your API key (required for examples 06–08):
```bash
export XAI_API_KEY=xai-your-key-here
```

Run a quick smoke test (no API key needed):
```bash
python3 examples/05_agentic_workflow.py
```

## Project Structure

```
dsl/                     # Core DSL and backends
examples/                # Runnable examples (01–08)
agent.py                 # Natural-language CLI
instructions-agent-grok-ray.md   # Detailed walkthrough
```

## Adding a New Example

1. Create `examples/NN_your_example.py` following the existing numbering
2. Import from `dsl.base_dsl` and use `ParallelDSL(backend=...)` as the entry point
3. Save any visualization to `examples/agent_graph_your_name.png`
4. Add a row to the examples table in `README.md`
5. If it uses an API key or extra dependency, note it clearly at the top of the file

## Adding a New Backend

1. Create `dsl/backends/your_backend.py` with a class that implements `.map(fn, items)`
2. Register it in `dsl/base_dsl.py` under `ParallelDSL.__init__`
3. Add a graceful fallback to CPU if the dependency is not installed (see `gpu_backend.py` for the pattern)
4. Add an example that exercises the new backend

## Code Style

- Python 3.10+ compatible
- No external secrets or API keys in source files — always use `os.getenv()`
- Prefer explicit progress output (`print`) so users can see what is happening
- Keep examples self-contained and runnable with a single `python3` command

## Submitting a Pull Request

1. Fork the repo and create a feature branch: `git checkout -b feature/my-thing`
2. Make your changes and test them locally
3. Open a PR with a short description of what you added or fixed
4. If it touches the DSL core, include at least one example that exercises the change

## Reporting Issues

Open a GitHub Issue with:
- Python version (`python3 --version`)
- Ray version if relevant (`python3 -c "import ray; print(ray.__version__)"`)
- The full error output
- The command you ran

---

Questions? Open a Discussion or an Issue — all feedback is welcome.
