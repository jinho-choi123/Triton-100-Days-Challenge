import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def matrix_conv2d(input: torch.Tensor, kernel: torch.Tensor):
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