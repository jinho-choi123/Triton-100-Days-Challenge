import triton_kernels.day007.conv1d as conv1d

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
def test_conv1d(N, K):
    logger.info("Testing conv1d with N={} and K={}", N, K)

    # create random tensors
    input = torch.randn(N, device="cuda")
    kernel = torch.randn(K, device="cuda")
    logger.debug("Input: {}", input)
    logger.debug("Kernel: {}", kernel)

    # compute conv1d
    output = conv1d.conv1d(input, kernel)
    logger.debug("Output: {}", output)

    # compute expected output
    expected_output = torch.nn.functional.conv1d(
        input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)
    ).squeeze()
    logger.debug("Expected Output: {}", expected_output)

    # check if output is correct
    assert torch.allclose(output, expected_output), "Output is not correct"

    logger.info("Test passed!")
