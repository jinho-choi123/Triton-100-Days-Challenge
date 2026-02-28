import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 4}),
        triton.Config(kwargs={"BLOCK_SIZE": 8}),
        triton.Config(kwargs={"BLOCK_SIZE": 16}),
        triton.Config(kwargs={"BLOCK_SIZE": 32}),
        triton.Config(kwargs={"BLOCK_SIZE": 64}),
        triton.Config(kwargs={"BLOCK_SIZE": 128}),
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
    ],
    key=["N", "K"],
)
@triton.jit
def conv1d_kernel(
    input_ptr,  # pointer to input tensor
    kernel_ptr,  # pointer to kernel tensor
    output_ptr,  # pointer to output tensor
    N,  # size of input vector
    K,  # size of kernel vector
    BLOCK_SIZE: tl.constexpr,  # size of block
):
    # get the program id
    pid = tl.program_id(axis=0)

    # allocate memory for accumulation
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # compute the input offsets
    input_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_masks = input_offsets < N

    for kernel_offset in range(0, K):
        weight = tl.load(kernel_ptr + kernel_offset)
        inputs = tl.load(
            input_ptr + input_offsets + kernel_offset, mask=input_masks, other=0.0
        )
        acc += weight * inputs

    # write the acc back to output ptr
    output_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_masks = output_offsets < (N - K + 1)
    tl.store(output_ptr + output_offsets, acc, mask=output_masks)


def conv1d(input: torch.Tensor, kernel: torch.Tensor):
    # validate devices
    assert input.device == DEVICE
    assert kernel.device == DEVICE

    # check if input is contiguous
    assert input.is_contiguous(), "Input must be contiguous"
    assert kernel.is_contiguous(), "Kernel must be contiguous"

    # get the shape of input
    N = input.shape[0]

    # get the shape of kernel
    K = kernel.shape[0]

    assert N >= K, "Input size must be greater than kernel size"

    # allocate memory for output vector
    input_dtype = input.dtype
    output = torch.zeros((N - K + 1), dtype=input_dtype, device=DEVICE)

    # check if output is contiguous
    assert output.is_contiguous(), "Output must be contiguous"

    # Define grid generator. It takes metadata, and generates the grid
    grid = lambda meta: (triton.cdiv(N - K + 1, meta["BLOCK_SIZE"]),)

    # Call the kernel
    conv1d_kernel[grid](input, kernel, output, N, K)

    return output
