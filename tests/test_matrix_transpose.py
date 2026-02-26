import triton_kernels.day006.matrix_transpose as matrix_transpose
import torch
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "M,N",
    [
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8),
        (16, 16),
        (16, 32),
        (33, 33),
        (32, 16),
        (32, 32),
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
        (127, 127),
        (256, 256),
        (256, 512),
        (512, 256),
        (512, 512),
        (1024, 1024),
        (1024, 2048),
        (2048, 1024),
        (2048, 2048),
        (4096, 4096),
    ],
)
def test_matrix_transpose(M, N):
    logger.info(f"Testing matrix_transpose with M={M}, N={N}")
    # create a random matrix
    input = torch.randn(M, N, device="cuda")

    # compute the transpose
    output = matrix_transpose.matrix_transpose(input)

    # compute reference
    ref_output = torch.transpose(input, 0, 1)

    # check if the output is correct
    assert torch.allclose(output, ref_output)

    logger.info("Matrix transpose test passed!")
