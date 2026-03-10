import torch
import triton

import triton_kernels.day011.online_softmax as online_softmax


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
        plot_name="online-softmax-benchmark",
        args={},
    )
)
def run_online_softmax_benchmark(M, N, provider):
    input = torch.randn(M, N, device="cuda", dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.softmax(input, dim=1),
            quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: online_softmax.online_softmax(input),
            quantiles=quantiles,
        )

    return ms, min_ms, max_ms


def bench_online_softmax(bench_result_dir):
    online_softmax_bench_result_dir = bench_result_dir / "online_softmax"
    online_softmax_bench_result_dir.mkdir(exist_ok=True)
    run_online_softmax_benchmark.run(
        save_path=online_softmax_bench_result_dir, print_data=True
    )
