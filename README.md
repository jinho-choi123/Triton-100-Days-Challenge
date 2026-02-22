# Triton 100 Days Challenge

This is a 100-day challenge to learn CUDA programming in Triton.

## Setup

This repo assumes the CUDA version 12.6 or higher is installed and configured.

```bash
# Create a virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Running Tests
If you run the following test, it will also create a folder named `bench_results` in the root directory and store the benchmark results in it.
```bash
uv run pytest
```

If you want to run the benchmark only, you can run the following command:
```bash
uv run pytest benchmark/
```

If you want to run the tests only, you can run the following command:
```bash
uv run pytest tests/
```
