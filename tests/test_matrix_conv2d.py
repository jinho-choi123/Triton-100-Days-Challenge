import pytest
import torch
from loguru import logger

import triton_kernels.day008.matrix_conv2d as matrix_conv2d


@pytest.mark.parametrize(
    "Icol, Irow, Kcol, Krow",
    [
        (16, 16, 3, 3),
        (32, 32, 5, 5),
        (64, 64, 7, 7),
        (128, 128, 9, 9),
        (256, 256, 11, 11),
        (512, 512, 13, 13),
        (1024, 1024, 15, 15),
        (2048, 2048, 17, 17),
        (4096, 4096, 19, 19),
    ],
)
def test_matrix_conv2d(Icol, Irow, Kcol, Krow):
    logger.info(
        f"Testing matrix_conv2d with Icol={Icol}, Irow={Irow}, Kcol={Kcol}, Krow={Krow}"
    )

    # Create random input and kernel tensors
    input = torch.randn((Irow, Icol), dtype=torch.float32, device="cuda")
    kernel = torch.randn((Krow, Kcol), dtype=torch.float32, device="cuda")

    # compute matrix_conv2d
    output = matrix_conv2d.matrix_conv2d(input, kernel)

    # compute expected output using PyTorch's conv2d
    expected_output = torch.nn.functional.conv2d(
        input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)
    ).squeeze()

    # check if output is correct
    assert torch.allclose(output, expected_output), (
        "Output does not match expected output"
    )

    logger.info("Test passed!")
