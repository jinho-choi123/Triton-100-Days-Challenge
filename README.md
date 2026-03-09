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

## List of Kernels

| Challenge    | Kernel Name | Description | Kernel Code | Test Code | Benchmark Code |
| -------- | ------- | ------- | ------- | ------- | ------- |
| day001  |   vector addition  | Kernel for adding two vectors | [Kernel Code](triton_kernels/day001/vector_add.py) | [Test Code](tests/test_vector_add.py) | [Benchmark Code](benchmark/bench_vector_add.py) |
| day002  |   matrix addition  | Kernel for adding two matrices | [Kernel Code](triton_kernels/day002/matrix_add.py) | [Test Code](tests/test_matrix_add.py) | [Benchmark Code](benchmark/bench_matrix_add.py) |
| day003  | matrix vector multiplication    | Kernel for multiplying a matrix with a vector | [Kernel Code](triton_kernels/day003/matrix_vector_mul.py) | [Test Code](tests/test_matrix_vector_mul.py) | [Benchmark Code](benchmark/bench_matrix_vector_mul.py) |
| day004 | prefix sum | Kernel for computing the prefix sum of an array | [Kernel Code](triton_kernels/day004/prefix_sum.py) | [Test Code](tests/test_prefix_sum.py) | [Benchmark Code](benchmark/bench_prefix_sum.py) |
| day005 | layer normalization | Kernel for performing layer normalization | [Kernel Code](triton_kernels/day005/layer_norm.py) | [Test Code](tests/test_layer_norm.py) | [Benchmark Code](benchmark/bench_layer_norm.py) |
| day006 | matrix transpose | Kernel for transposing a matrix | [Kernel Code](triton_kernels/day006/matrix_transpose.py) | [Test Code](tests/test_matrix_transpose.py) | [Benchmark Code](benchmark/bench_matrix_transpose.py) |
| day007 | vector 1d convolution | Kernel for performing 1D convolution on a vector | [Kernel Code](triton_kernels/day007/vector_conv1d.py) | [Test Code](tests/test_vector_conv1d.py) | [Benchmark Code](benchmark/bench_vector_conv1d.py) |
| day008 | matrix 2d convolution | Kernel for performing 2D convolution on a matrix | [Kernel Code](triton_kernels/day008/matrix_conv2d.py) | [Test Code](tests/test_matrix_conv2d.py) | [Benchmark Code](benchmark/bench_matrix_conv2d.py) |
| day009 | softmax | Kernel for performing softmax on a matrix | [Kernel Code](triton_kernels/day009/softmax.py) | [Test Code](tests/test_softmax.py) | [Benchmark Code](benchmark/bench_softmax.py) |
| day010 | relu | Kernel for performing relu on an vector | [Kernel Code](triton_kernels/day010/relu.py) | [Test Code](tests/test_relu.py) | [Benchmark Code](benchmark/bench_relu.py) |
