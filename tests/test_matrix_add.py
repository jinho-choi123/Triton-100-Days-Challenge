import triton_kernels.day002.matrix_add as matrix_add

import torch
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "matrix_shape",
    [
        (1,),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (1024, 1024),
        (32, 32, 32),
        (1025, 1026),
    ],
)
def test_matrix_add(matrix_shape):
    logger.info(f"Testing matrix_add with shape {matrix_shape}")
    # create random tensors
    x = torch.randn(matrix_shape, device="cuda")
    y = torch.randn(matrix_shape, device="cuda")

    # compute matrix add
    output = matrix_add.matrix_add(x, y)

    # check if the output is correct
    assert torch.allclose(output, x + y)

    logger.info("Matrix add test passed!")
