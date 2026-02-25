import triton_kernels.day005.layer_norm as layer_norm
import torch
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "M,N",
    [
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (5, 9),
        (15, 213),
    ],
)
def test_layer_norm(M, N):
    logger.info(f"Testing layer_norm with M={M}, N={N}")
    # create random tensors
    input = torch.randn((M, N), device="cuda")
    gamma = torch.randn((N,), device="cuda")
    beta = torch.randn((N,), device="cuda")

    # compute layer norm
    output = layer_norm.layer_norm(input, gamma, beta)

    # compute reference
    ref_output = torch.nn.functional.layer_norm(input, (N,), gamma, beta)

    # check if the output is correct
    assert torch.allclose(output, ref_output, rtol=1e-4, atol=1e-5)

    logger.info("Layer norm test passed!")
