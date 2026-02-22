import torch
import triton
import triton_kernels.day001.vector_add as vector_add


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["vector_size"],
        x_vals=[2**i for i in range(0, 24)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch Native", "Custom Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="execution_time(ms)",
        y_log=True,
        plot_name="vector-add-benchmark",
        args={},
    )
)
def run_vector_add_benchmark(vector_size, provider):
    x = torch.randn(vector_size, device="cuda")
    y = torch.randn(vector_size, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add.vector_add(x, y), quantiles=quantiles
        )

    return ms, min_ms, max_ms


def bench_vector_add(bench_result_dir):
    vector_add_bench_result_dir = bench_result_dir / "vector_add"
    vector_add_bench_result_dir.mkdir(exist_ok=True)
    run_vector_add_benchmark.run(save_path=vector_add_bench_result_dir, print_data=True)
