import pytest
import torch
from loguru import logger

import triton_kernels.day011.online_softmax as online_softmax


@pytest.mark.parametrize(
    "M, N",
    [
        (32, 32),
        (64, 64),
        (65, 65),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (4000, 4000),
    ],
)
def test_online_softmax(M, N):
    logger.info(f"Testing online softmax with M={M}, N={N}")
    # Create random input tensor
    input_tensor = torch.randn(M, N, device="cuda")

    # Compute softmax using the custom kernel
    output_custom = online_softmax.online_softmax(input_tensor)

    # Compute softmax using PyTorch's built-in function for comparison
    output_torch = torch.nn.functional.softmax(input_tensor, dim=1)

    # Check if the outputs are close enough
    assert torch.allclose(output_custom, output_torch), "Softmax outputs do not match!"

    logger.info("Online softmax test passed!")
