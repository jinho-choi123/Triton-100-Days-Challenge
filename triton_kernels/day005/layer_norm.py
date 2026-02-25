import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": tl.constexpr(16)}),
        triton.Config(kwargs={"BLOCK_SIZE": tl.constexpr(32)}),
        triton.Config(kwargs={"BLOCK_SIZE": tl.constexpr(64)}),
        triton.Config(kwargs={"BLOCK_SIZE": tl.constexpr(128)}),
        triton.Config(kwargs={"BLOCK_SIZE": tl.constexpr(256)}),
        triton.Config(kwargs={"BLOCK_SIZE": tl.constexpr(512)}),
        triton.Config(kwargs={"BLOCK_SIZE": tl.constexpr(1024)}),
    ],
    key=["M", "N"],
)
@triton.jit
def layer_norm_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    gamma_ptr,
    beta_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # get the program id
    # single program normalize single row.
    pid = tl.program_id(axis=0)

    # allocate memory to store the sum and variance
    # allocate fp64 to prevent overflow
    sum = tl.zeros((1,), dtype=tl.float64)
    variance = tl.zeros((1,), dtype=tl.float64)

    # loop over the row
    for i in range(0, N, BLOCK_SIZE):
        # load the input
        offsets = pid * N + i + tl.arange(0, BLOCK_SIZE)
        masks = (i + tl.arange(0, BLOCK_SIZE)) < N
        partial_input = tl.load(input_ptr + offsets, mask=masks)

        # compute the sum
        sum += tl.sum(partial_input)

    # calculate the mean
    mean = sum / N

    # loop over the row again to compute the variance
    for i in range(0, N, BLOCK_SIZE):
        # load the input
        offsets = pid * N + i + tl.arange(0, BLOCK_SIZE)
        masks = (i + tl.arange(0, BLOCK_SIZE)) < N
        partial_input = tl.load(input_ptr + offsets, mask=masks, other=mean)

        # compute the variance
        variance += tl.sum((partial_input - mean) * (partial_input - mean))

    variance = variance / N

    # calculate the variance
    # set epsilon as 1e-5 to prevent division by zero
    std = tl.sqrt(variance + 1e-5)

    # loop over the row again to compute the output
    for i in range(0, N, BLOCK_SIZE):
        # load the input
        offsets = pid * N + i + tl.arange(0, BLOCK_SIZE)
        masks = (i + tl.arange(0, BLOCK_SIZE)) < N
        partial_input = tl.load(input_ptr + offsets, mask=masks)

        # load gamma and beta
        gamma = tl.load(gamma_ptr + i + tl.arange(0, BLOCK_SIZE), mask=masks)
        beta = tl.load(beta_ptr + i + tl.arange(0, BLOCK_SIZE), mask=masks)

        # compute the output
        partial_output = ((partial_input - mean) * gamma / std) + beta

        # store the output
        tl.store(output_ptr + offsets, partial_output, mask=masks)


def layer_norm(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    # validate devices
    assert input.device == DEVICE and gamma.device == DEVICE and beta.device == DEVICE

    # check if input, gamma, beta are all contiguous
    assert input.is_contiguous() and gamma.is_contiguous() and beta.is_contiguous(), (
        "Inputs must be contiguous"
    )

    # get the shape of input
    M, N = input.shape

    # check the shape of gamma and beta
    assert gamma.shape == (N,), "Gamma shape must be (N,)"
    assert beta.shape == (N,), "Beta shape must be (N,)"

    # allocate memory for output vector
    output = torch.zeros_like(input)

    # Define grid generator. It takes metadata, and generates the grid
    grid = lambda meta: (M,)

    # Call the kernel
    layer_norm_kernel[grid](input, output, M, N, gamma, beta)

    return output
