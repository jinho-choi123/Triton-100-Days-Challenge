import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def vector_add_kernel(x_ptr, # pointer to the first input vector
y_ptr, # pointer to the second input vector
output_ptr, # pointer to the output vector
n_elements, # number of elements in the vectors
BLOCK_SIZE: tl.constexpr # size of the block
):
    # get the program id
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor):

    # allocate memory for output vector
    output = torch.zeros_like(x)

    # get number of elements
    n_elements = output.numel()

    # validate devices
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    # Define grid generator. It takes metadata, and generates the grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),) 

    # Call the kernel
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output




    

