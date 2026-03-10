import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}),
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["M, N"],
)
@triton.jit
def safe_softmax_kernel(
    input_ptr,  # pointer to the input matrix
    output_ptr,  # pointer to the output matrix
    M,  # number of rows in the input matrix
    N,  # number of columns in the input matrix
    BLOCK_SIZE: tl.constexpr,
):
    # get block id
    # we are going to process softmax for pid'th row of the input matrix
    pid = tl.program_id(0)

    # allocate memory for global max value
    global_max_value = -float("inf")

    # get the max value for the current row
    for col_start_offset in range(0, N, BLOCK_SIZE):
        # compute the column offset for the current block
        col_offsets = col_start_offset + tl.arange(0, BLOCK_SIZE)

        offsets = pid * N + col_offsets

        # compute the masks
        masks = col_offsets < N

        # load the input values
        input_block = tl.load(input_ptr + offsets, mask=masks, other=-float("inf"))

        # compute the max value for the current block
        block_max_value = tl.max(input_block, axis=0)

        # update the global max value
        global_max_value = tl.where(
            block_max_value > global_max_value, block_max_value, global_max_value
        )

    # allocate memory for the sum of exponentials
    sum_exp = 0.0
    # compute the sum of exponentials for the current row
    for col_start_offset in range(0, N, BLOCK_SIZE):
        # compute the column offset for the current block
        col_offsets = col_start_offset + tl.arange(0, BLOCK_SIZE)

        offsets = pid * N + col_offsets

        # compute the masks
        masks = col_offsets < N

        # load the input values
        input_block = tl.load(input_ptr + offsets, mask=masks, other=-float("inf"))

        # compute the exponentials and sum them up
        exp_block = tl.exp(input_block - global_max_value)
        sum_exp += tl.sum(exp_block, axis=0)

        # store the exponentials in the output buffer temporarily
        tl.store(output_ptr + offsets, exp_block, mask=masks)

    # compute the final softmax values for the current row
    for col_start_offset in range(0, N, BLOCK_SIZE):
        # compute the column offset for the current block
        col_offsets = col_start_offset + tl.arange(0, BLOCK_SIZE)

        offsets = pid * N + col_offsets

        # compute the masks
        masks = col_offsets < N

        # load the exponentials from the output buffer
        exp_block = tl.load(output_ptr + offsets, mask=masks, other=0.0)

        # compute the softmax values and store them in the output buffer
        softmax_block = exp_block / sum_exp
        tl.store(output_ptr + offsets, softmax_block, mask=masks)


def safe_softmax(input: torch.Tensor) -> torch.Tensor:
    # validate devices
    assert input.device == DEVICE, (
        f"Input tensor must be on the same device as the kernel. Expected {DEVICE}, but got {input.device}."
    )

    # check if input is contiguous
    assert input.is_contiguous(), "Input tensor must be contiguous."

    # get the shape
    M, N = input.shape

    # allocate output tensor
    output = torch.empty_like(input)

    # define grid generator
    grid = lambda meta: (M,)

    # launch the kernel
    safe_softmax_kernel[grid](input, output, M, N)

    return output
