import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def matrix_add_kernel(
    x_ptr,  # pointer to the first input matrix
    y_ptr,  # pointer to the second input matrix
    output_ptr,  # pointer to the output matrix
    n_elements,  # number of elements in the matrix
    BLOCK_SIZE: tl.constexpr,  # size of the block
):
    # get the program id
    pid = tl.program_id(axis=0)

    # compute the block start
    block_start = pid * BLOCK_SIZE

    # compute the offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # create a mask to avoid going out of bounds
    mask = offsets < n_elements

    # load the data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # compute the output
    output = x + y

    # store the output
    tl.store(output_ptr + offsets, output, mask=mask)


def matrix_add(x: torch.Tensor, y: torch.Tensor):
    # allocate memory for output vector
    output = torch.zeros_like(x)

    # get number of elements
    n_elements = output.numel()

    # validate devices
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    # check if x and y are all contiguous
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"

    # Define grid generator. It takes metadata, and generates the grid
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Call the kernel
    matrix_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=tl.constexpr(1024))

    return output
