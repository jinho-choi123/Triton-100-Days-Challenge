import pytest
import torch
from loguru import logger

import triton_kernels.day010.relu as relu


@pytest.mark.parametrize(
    "n_elements",
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 513, 1024, 2048, 4096, 8192, 8194],
)
def test_relu(n_elements):
    logger.info(f"Testing relu with {n_elements} elements")

    # create random input tensor
    input = torch.randn(n_elements, device="cuda")

    # compute relu
    output = relu.relu(input)

    # compute reference
    ref_output = torch.nn.functional.relu(input)

    # check if the output is correct
    assert torch.allclose(output, ref_output)

    logger.info("Relu test passed!")
