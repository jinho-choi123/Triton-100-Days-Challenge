import triton_kernels.day007.vector_conv1d as vector_conv1d

import torch
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "N, K",
    [
        (4, 2),
        (4, 4),
        (32, 4),
        (33, 5),
        (256, 16),
        (257, 17),
        (1024, 32),
        (1025, 31),
    ],
)
def test_vector_conv1d(N, K):
    logger.info("Testing vector_conv1d with N={} and K={}", N, K)

    # create random tensors
    input = torch.randn(N, device="cuda")
    kernel = torch.randn(K, device="cuda")

    # compute vector_conv1d
    output = vector_conv1d.vector_conv1d(input, kernel)

    # compute expected output
    expected_output = torch.nn.functional.conv1d(
        input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)
    ).squeeze()

    # check if output is correct
    assert torch.allclose(output, expected_output), "Output is not correct"

    logger.info("Test passed!")
