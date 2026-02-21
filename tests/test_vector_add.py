import triton_kernels.day001.vector_add as vector_add
import torch
from loguru import logger
import pytest


@pytest.mark.parametrize("n_elements", [1024, 2048, 4096])
def test_vector_add(n_elements):
    logger.info(f"Testing vector_add with {n_elements} elements")
    # create random tensors
    x = torch.randn(n_elements, device="cuda")
    y = torch.randn(n_elements, device="cuda")

    # compute vector add
    output = vector_add.vector_add(x, y)

    # check if the output is correct
    assert torch.allclose(output, x + y)

    logger.info("Vector add test passed!")
