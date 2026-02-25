import triton_kernels.day004.prefix_sum as prefix_sum
import torch
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "n_elements", [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
)
def test_prefix_sum(n_elements):
    logger.info(f"Testing prefix_sum with {n_elements} elements")
    # create random tensors
    input = torch.randn(n_elements, device="cuda")
    output = prefix_sum.prefix_sum(input)

    # compute reference
    ref_output = torch.cumsum(input, dim=0)

    # check if the output is correct
    assert torch.allclose(output, ref_output, rtol=1e-3, atol=1e-4)

    logger.info("Prefix sum test passed!")
