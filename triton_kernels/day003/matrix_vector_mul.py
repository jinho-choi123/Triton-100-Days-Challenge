import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": tl.constexpr(16)}),
        triton.Config(kwargs={"BLOCK_SIZE": tl.constexpr(32)}),
        triton.Config(kwargs={"BLOCK_SIZE": tl.constexpr(64)}),
    ],
    key=["M", "K"],
)
@triton.jit
def matrix_vector_mul_kernel(
    A_ptr,
    x_ptr,
    output_ptr,
    M,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    # get the program id
    pid = tl.program_id(axis=0)

    m_offset_start = pid * BLOCK_SIZE
    m_offsets = m_offset_start + tl.arange(0, BLOCK_SIZE)
    m_masks = m_offsets < M

    # allocate shared memory for accumulation
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # create a loop to compute the dot product
    for k_offset_start in range(0, K, BLOCK_SIZE):
        k_offsets = k_offset_start + tl.arange(0, BLOCK_SIZE)
        k_masks = k_offsets < K

        offsets = k_offsets + m_offsets[:, None] * K
        masks = k_masks & m_masks[:, None]

        # shape of partial_A is (BLOCK_SIZE, BLOCK_SIZE). The data is loaded in shared memory.
        partial_A = tl.load(A_ptr + offsets, mask=masks, other=0.0)
        # shape of partial_x is (BLOCK_SIZE,). The data is loaded in shared memory.
        partial_x = tl.load(x_ptr + k_offsets, mask=k_masks, other=0.0)

        # compute the dot product
        dot_product = tl.dot(partial_A, partial_x[:, None], input_precision="ieee")
        dot_product = tl.reshape(dot_product, (BLOCK_SIZE,))

        # accumulate the dot product
        acc = acc + dot_product

    # store the output
    tl.store(output_ptr + m_offsets, acc, mask=m_masks)


def matrix_vector_mul(A: torch.Tensor, x: torch.Tensor):
    # Get the shape of A
    M, K = A.shape

    # Get the dtype of A
    dtype = A.dtype

    # allocate memory for output vector
    output = torch.zeros((M,), dtype=dtype, device=DEVICE)

    # validate devices
    assert A.device == DEVICE and x.device == DEVICE and output.device == DEVICE

    # check if A and x are all contiguous
    assert A.is_contiguous() and x.is_contiguous(), "Inputs must be contiguous"

    # Define grid generator. It takes metadata, and generates the grid
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]),)

    # Call the kernel
    matrix_vector_mul_kernel[grid](A, x, output, M, K)

    return output
