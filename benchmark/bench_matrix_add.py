import torch
import triton
import triton_kernels.day002.matrix_add as matrix_add


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["matrix_size"],
        x_vals=[2 ** (i * 2) for i in range(0, 12)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch Native", "Custom Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="execution_time(ms)",
        y_log=True,
        plot_name="matrix-add-benchmark",
        args={},
    )
)
def run_matrix_add_benchmark(matrix_size, provider):
    x = torch.randn(int(matrix_size**0.5), int(matrix_size**0.5), device="cuda")
    y = torch.randn(int(matrix_size**0.5), int(matrix_size**0.5), device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matrix_add.matrix_add(x, y), quantiles=quantiles
        )

    return ms, min_ms, max_ms


def bench_matrix_add(bench_result_dir):
    matrix_add_bench_result_dir = bench_result_dir / "matrix_add"
    matrix_add_bench_result_dir.mkdir(exist_ok=True)
    run_matrix_add_benchmark.run(save_path=matrix_add_bench_result_dir, print_data=True)
