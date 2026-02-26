import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 8}),
        triton.Config(kwargs={"BLOCK_SIZE": 16}),
        triton.Config(kwargs={"BLOCK_SIZE": 32}),
        triton.Config(kwargs={"BLOCK_SIZE": 64}),
    ],
    key=["M", "N"],
)
@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    # get the program id
    x_pid = tl.program_id(axis=0)
    y_pid = tl.program_id(axis=1)

    # compute the block offset
    M_offsets = x_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    N_offsets = y_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    M_masks = M_offsets < M
    N_masks = N_offsets < N

    offsets = M_offsets[:, None] * N + N_offsets[None, :]

    masks = M_masks[:, None] & N_masks[None, :]

    # load the block
    sub_matrix = tl.load(input_ptr + offsets, mask=masks, other=0.0)

    # transpose the block
    transposed_sub_matrix = tl.trans(sub_matrix)

    # compute the output offsets
    # output is N x M (transposed), so row indices come from y_pid, col indices from x_pid
    output_N_offsets = y_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_M_offsets = x_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    output_offsets = output_N_offsets[:, None] * M + output_M_offsets[None, :]

    output_N_masks = output_N_offsets < N
    output_M_masks = output_M_offsets < M

    output_masks = output_N_masks[:, None] & output_M_masks[None, :]

    # store the block
    tl.store(output_ptr + output_offsets, transposed_sub_matrix, mask=output_masks)


def matrix_transpose(input: torch.Tensor):
    # validate devices
    assert input.device == DEVICE

    # check if input is contiguous
    assert input.is_contiguous(), "Input must be contiguous"

    # get the shape of input
    M, N = input.shape

    # allocate memory for output vector
    output = torch.zeros_like(input).transpose(0, 1).contiguous()

    # check if output is contiguous
    assert output.is_contiguous(), "Output must be contiguous"

    # Define grid generator. It takes metadata, and generates the grid
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )

    # Call the kernel
    matrix_transpose_kernel[grid](input, output, M, N)

    return output
