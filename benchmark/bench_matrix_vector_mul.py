import torch
import triton
import triton_kernels.day003.matrix_vector_mul as matrix_vector_mul


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "K"],
        x_vals=[(2**i, 2**i) for i in range(0, 10)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch Native", "Custom Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="execution_time(ms)",
        y_log=True,
        plot_name="matrix-vector-mul-benchmark",
        args={},
    )
)
def run_matrix_vector_mul_benchmark(M, K, provider):
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    x = torch.randn((K,), device="cuda", dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(A, x), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matrix_vector_mul.matrix_vector_mul(A, x), quantiles=quantiles
        )

    return ms, min_ms, max_ms


def bench_matrix_vector_mul(bench_result_dir):
    matrix_vector_mul_bench_result_dir = bench_result_dir / "matrix_vector_mul"
    matrix_vector_mul_bench_result_dir.mkdir(exist_ok=True)
    run_matrix_vector_mul_benchmark.run(
        save_path=matrix_vector_mul_bench_result_dir, print_data=True
    )
