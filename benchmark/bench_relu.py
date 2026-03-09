import torch
import triton

import triton_kernels.day010.relu as relu


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_elements"],
        x_vals=[2**i for i in range(10)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch Native", "Custom Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="execution_time(ms)",
        y_log=True,
        plot_name="relu-benchmark",
        args={},
    )
)
def run_relu_benchmark(n_elements, provider):
    input = torch.randn(n_elements, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.relu(input), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: relu.relu(input), quantiles=quantiles
        )

    return ms, min_ms, max_ms


def bench_relu(bench_result_dir):
    relu_bench_result_dir = bench_result_dir / "relu"
    relu_bench_result_dir.mkdir(exist_ok=True)
    run_relu_benchmark.run(save_path=relu_bench_result_dir, print_data=True)
