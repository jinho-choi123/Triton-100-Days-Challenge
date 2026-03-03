import torch
import triton

import triton_kernels.day008.matrix_conv2d as matrix_conv2d


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["InputRows", "InputCols", "KernelRows", "KernelCols"],
        x_vals=[(2 ** (i + 2), 2 ** (i + 2), 2**i, 2**i) for i in range(10)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch Native", "Custom Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="execution_time(ms)",
        y_log=True,
        plot_name="conv2d-benchmark",
        args={},
    )
)
def run_matrix_conv2d_benchmark(InputRows, InputCols, KernelRows, KernelCols, provider):
    # Generate random input and kernel tensors
    input = torch.randn((InputRows, InputCols), device="cuda", dtype=torch.float32)
    kernel = torch.randn((KernelRows, KernelCols), device="cuda", dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    # compute conv1d
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.conv2d(
                input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)
            ).squeeze(),
            quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matrix_conv2d.matrix_conv2d(input, kernel), quantiles=quantiles
        )

    return ms, min_ms, max_ms


def bench_matrix_conv2d(bench_result_dir):
    matrix_conv2d_bench_result_dir = bench_result_dir / "matrix_conv2d"
    matrix_conv2d_bench_result_dir.mkdir(exist_ok=True)

    run_matrix_conv2d_benchmark.run(
        save_path=matrix_conv2d_bench_result_dir, print_data=True
    )
