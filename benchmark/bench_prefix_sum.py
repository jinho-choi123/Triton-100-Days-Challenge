import torch
import triton
import triton_kernels.day004.prefix_sum as prefix_sum


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["vector_size"],
        x_vals=[2**i for i in range(0, 12)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch Native", "Custom Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="execution_time(ms)",
        y_log=True,
        plot_name="prefix-sum-benchmark",
        args={},
    )
)
def run_prefix_sum_benchmark(vector_size, provider):
    x = torch.randn(vector_size, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.cumsum(x, dim=0), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: prefix_sum.prefix_sum(x), quantiles=quantiles
        )

    return ms, min_ms, max_ms


def bench_prefix_sum(bench_result_dir):
    prefix_sum_bench_result_dir = bench_result_dir / "prefix_sum"
    prefix_sum_bench_result_dir.mkdir(exist_ok=True)
    run_prefix_sum_benchmark.run(save_path=prefix_sum_bench_result_dir, print_data=True)
