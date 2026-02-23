import triton_kernels.day003.matrix_vector_mul as matrix_vector_mul

import torch
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "M,K",
    [(4, 4), (16, 16), (32, 32), (64, 64), (128, 128), (1024, 1024), (1025, 1026)],
)
def test_matrix_vector_mul(M, K):
    logger.info(f"Testing matrix_vector_mul with M={M}, K={K}")
    # create random tensors
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    x = torch.randn((K,), device="cuda", dtype=torch.float32)

    # compute matrix vector mul
    output = matrix_vector_mul.matrix_vector_mul(A, x)

    # compute reference
    ref_output = torch.matmul(A, x)

    # check if the output is correct
    assert torch.allclose(output, ref_output, rtol=1e-4, atol=1e-4)

    logger.info("Matrix vector mul test passed!")
