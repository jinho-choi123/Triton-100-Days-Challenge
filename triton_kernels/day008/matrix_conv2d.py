import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}),
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
    ],
    key=["Irow, Icol, Krow, Kcol, Orow, Ocol"],
)
@triton.jit
def matrix_conv2d_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    Irow,
    Icol,
    Krow,
    Kcol,
    Orow,
    Ocol,
    BLOCK_SIZE: tl.constexpr,
):
    # get block id
    block_row_idx = tl.program_id(axis=0)
    block_col_idx = tl.program_id(axis=1)

    # allocate sram for accumulation
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # iterate over kernel
    for k_row in tl.range(Krow):
        for k_col in tl.range(Kcol):
            # compute input and kernel offsets
            input_row_offsets = (
                block_row_idx * BLOCK_SIZE + k_row + tl.arange(0, BLOCK_SIZE)
            )
            input_row_masks = input_row_offsets < Irow
            input_col_offsets = (
                block_col_idx * BLOCK_SIZE + k_col + tl.arange(0, BLOCK_SIZE)
            )
            input_col_masks = input_col_offsets < Icol

            input_offsets = (
                input_row_offsets[:, None] * Icol + input_col_offsets[None, :]
            )
            input_masks = input_row_masks[:, None] & input_col_masks[None, :]

            kernel_offset = k_row * Kcol + k_col

            # load input and kernel values
            # NOTE: Use caching policy to optimize memory access
            input_block = tl.load(
                input_ptr + input_offsets,
                mask=input_masks,
                other=0.0,
                cache_modifier=".ca",
            )
            kernel_value = tl.load(kernel_ptr + kernel_offset)

            # perform multiplication and accumulate
            acc += input_block * kernel_value

    # compute the output offset
    output_row_offsets = block_row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_row_masks = output_row_offsets < Orow
    output_col_offsets = block_col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_col_masks = output_col_offsets < Ocol
    output_offsets = output_row_offsets[:, None] * Ocol + output_col_offsets[None, :]
    output_masks = output_row_masks[:, None] & output_col_masks[None, :]

    # store the result
    tl.store(output_ptr + output_offsets, acc, mask=output_masks)


def matrix_conv2d(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    # validate devices
    assert input.device == DEVICE
    assert kernel.device == DEVICE

    # check if input is contiguous
    assert input.is_contiguous(), "Input must be contiguous"
    assert kernel.is_contiguous(), "Kernel must be contiguous"

    # get shape
    Irow, Icol = input.shape
    Krow, Kcol = kernel.shape

    assert Irow >= Krow, "Input rows must be greater than or equal to kernel rows"
    assert Icol >= Kcol, "Input cols must be greater than or equal to kernel cols"

    # allocate memory for output
    Orow = Irow - Krow + 1
    Ocol = Icol - Kcol + 1
    output = torch.empty((Orow, Ocol), device=DEVICE, dtype=torch.float32)

    # define grid generator
    grid = lambda meta: (
        triton.cdiv(Orow, meta["BLOCK_SIZE"]),
        triton.cdiv(Ocol, meta["BLOCK_SIZE"]),
    )

    # launch kernel
    matrix_conv2d_kernel[grid](
        input_ptr=input,
        kernel_ptr=kernel,
        output_ptr=output,
        Irow=Irow,
        Icol=Icol,
        Krow=Krow,
        Kcol=Kcol,
        Orow=Orow,
        Ocol=Ocol,
    )

    return output
