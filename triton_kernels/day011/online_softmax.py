import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 16}),
        triton.Config({"BLOCK_SIZE": 32}),
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["M", "N"],
)
@triton.jit
def online_softmax_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Shape of the matrices:
    - input: MxN
    - output: MxN

    We will process softmax for pid'th row of the input matrix.
    """
    # get the program id
    pid = tl.program_id(axis=0)

    # allocate shared memory for max value and sum of exponentials
    max_value = -float("inf")
    sum_of_exps = 0.0

    # loop over the row
    for col_start_offset in range(0, N, BLOCK_SIZE):
        # compute the input offsets
        input_offsets = pid * N + col_start_offset + tl.arange(0, BLOCK_SIZE)
        input_masks = (col_start_offset + tl.arange(0, BLOCK_SIZE)) < N

        # load the input values
        input_block = tl.load(
            input_ptr + input_offsets, mask=input_masks, other=-float("inf")
        )

        # compute the max value for the current block
        block_max_value = tl.max(input_block, axis=0)

        prev_max_value = max_value

        max_value = tl.where(block_max_value > max_value, block_max_value, max_value)

        sum_of_exps = sum_of_exps * tl.exp(prev_max_value - max_value) + tl.sum(
            tl.exp(input_block - max_value)
        )

    # compute the final softmax values for the current row
    for col_start_offset in range(0, N, BLOCK_SIZE):
        # compute the input offsets
        input_offsets = pid * N + col_start_offset + tl.arange(0, BLOCK_SIZE)
        input_masks = (col_start_offset + tl.arange(0, BLOCK_SIZE)) < N

        # load the input values
        input_block = tl.load(
            input_ptr + input_offsets, mask=input_masks, other=-float("inf")
        )

        # compute the softmax values and store them in the output buffer
        softmax_block = tl.exp(input_block - max_value) / sum_of_exps

        # store the softmax values in the output buffer
        tl.store(output_ptr + input_offsets, softmax_block, mask=input_masks)


def online_softmax(input: torch.Tensor) -> torch.Tensor:
    # validate devices
    assert input.device == DEVICE, (
        "Input must be on the same device as the current device"
    )

    # check if input is contiguous
    assert input.is_contiguous(), "Input must be contiguous"

    # get the shape of input
    M, N = input.shape

    # allocate memory for output
    output = torch.empty_like(input)

    # define grid generator
    grid = lambda meta: (M,)

    # launch the kernel
    online_softmax_kernel[grid](input, output, M, N)

    return output
