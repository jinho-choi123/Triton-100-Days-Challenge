import torch
import triton
import triton_kernels.day007.conv1d as conv1d


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N", "K"],
        x_vals=[(2 ** (i + 1), 2**i) for i in range(10)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch Native", "Custom Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="execution_time(ms)",
        y_log=True,
        plot_name="conv1d-benchmark",
        args={},
    )
)
def run_conv1d_benchmark(N, K, provider):
    # create random tensors
    input = torch.randn(N, device="cuda")
    kernel = torch.randn(K, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    # compute conv1d
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.conv1d(
                input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)
            ).squeeze(),
            quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: conv1d.conv1d(input, kernel), quantiles=quantiles
        )

    return ms, min_ms, max_ms


def bench_conv1d(bench_result_dir):
    conv1d_bench_result_dir = bench_result_dir / "conv1d"
    conv1d_bench_result_dir.mkdir(exist_ok=True)

    run_conv1d_benchmark.run(save_path=conv1d_bench_result_dir, print_data=True)
