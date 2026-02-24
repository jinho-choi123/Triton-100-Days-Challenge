import torch

import triton
import triton.language as tl

MAX_BLOCKS = 16
DEVICE = triton.runtime.driver.active.get_active_torch_device()
BLOCK_SIZE = 512


@triton.jit
def block_prefix_sum_kernel(
    input_ptr,
    output_ptr,
    block_sums_ptr,
    n_elements,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """This kernel computes the local prefix sum of a block of elements(BLOCK_SIZE)."""
    # get the program id
    pid = tl.program_id(axis=0)

    # validate the pid is less than MAX_BLOCKS
    assert pid < MAX_BLOCKS, "The program id must be less than MAX_BLOCKS"

    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask)

    block_prefix_sum = tl.cumsum(input)

    tl.store(output_ptr + offsets, block_prefix_sum, mask=mask)

    tl.store(block_sums_ptr + pid, tl.sum(input))


@triton.jit
def cumsum_block_sums_kernel(block_sums_ptr, num_blocks, MAX_BLOCKS: tl.constexpr):
    """This kernel computes the cumulative sum of the block sums. The number of program is 1."""

    offsets = tl.arange(0, MAX_BLOCKS)
    mask = offsets < num_blocks

    block_sums = tl.load(block_sums_ptr + offsets, mask=mask)
    block_prefix_sum = tl.cumsum(block_sums)
    tl.store(block_sums_ptr + offsets, block_prefix_sum, mask=mask)


@triton.jit
def global_prefix_sum_kernel(
    output_ptr,
    block_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """This kernel computes the global prefix sum of a block of elements(BLOCK_SIZE).

    We assume that the block_sums are already computed and stored in the block_sums_ptr.
    """
    # get the program id
    pid = tl.program_id(axis=0)

    # load the block sum for the previous block
    if pid == 0:
        prev_block_sum = 0.0
    else:
        prev_block_sum = tl.load(block_sums_ptr + pid - 1)

    # calculate the offset
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    intermediate_output = tl.load(output_ptr + offsets, mask=mask)

    output = intermediate_output + prev_block_sum

    tl.store(output_ptr + offsets, output, mask=mask)


def prefix_sum(input: torch.Tensor):
    # get the shape of the input
    (N,) = input.shape

    # allocate memory for the output
    output = torch.zeros_like(input)

    # allocate memory for the block sums
    block_sums = torch.zeros(MAX_BLOCKS, device=DEVICE)

    # validate devices
    assert (
        input.device == DEVICE
        and output.device == DEVICE
        and block_sums.device == DEVICE
    )

    # check if the input is contiguous
    assert input.is_contiguous(), "Input must be contiguous"

    # define grid generator
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    # call the kernel
    block_prefix_sum_kernel[grid](
        input, output, block_sums, N, tl.constexpr(MAX_BLOCKS), tl.constexpr(BLOCK_SIZE)
    )

    # call the kernel
    cumsum_block_sums_kernel[(1,)](block_sums, N, tl.constexpr(MAX_BLOCKS))

    # call the kernel
    global_prefix_sum_kernel[grid](output, block_sums, N, tl.constexpr(BLOCK_SIZE))

    return output
