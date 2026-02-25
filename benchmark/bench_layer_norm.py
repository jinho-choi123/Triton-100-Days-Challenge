import torch
import triton
import triton_kernels.day005.layer_norm as layer_norm


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=[(2**i, 2**i) for i in range(0, 12)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch Native", "Custom Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="execution_time(ms)",
        y_log=True,
        plot_name="layer-norm-benchmark",
        args={},
    )
)
def run_layer_norm_benchmark(M, N, provider):
    input = torch.randn((M, N), device="cuda")
    gamma = torch.randn((N,), device="cuda")
    beta = torch.randn((N,), device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.layer_norm(input, (N,), gamma, beta),
            quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layer_norm.layer_norm(input, gamma, beta), quantiles=quantiles
        )

    return ms, min_ms, max_ms


def bench_layer_norm(bench_result_dir):
    layer_norm_bench_result_dir = bench_result_dir / "layer_norm"
    layer_norm_bench_result_dir.mkdir(exist_ok=True)
    run_layer_norm_benchmark.run(save_path=layer_norm_bench_result_dir, print_data=True)
