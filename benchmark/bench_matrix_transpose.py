import torch
import triton
import triton_kernels.day006.matrix_transpose as matrix_transpose


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
        plot_name="matrix-transpose-benchmark",
        args={},
    )
)
def run_matrix_transpose_benchmark(matrix_size, provider):
    M = int(matrix_size**0.5)
    N = M
    x = torch.randn(M, N, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.transpose(x, 0, 1).contiguous(), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matrix_transpose.matrix_transpose(x), quantiles=quantiles
        )

    return ms, min_ms, max_ms


def bench_matrix_transpose(bench_result_dir):
    matrix_transpose_bench_result_dir = bench_result_dir / "matrix_transpose"
    matrix_transpose_bench_result_dir.mkdir(exist_ok=True)
    run_matrix_transpose_benchmark.run(
        save_path=matrix_transpose_bench_result_dir, print_data=True
    )
