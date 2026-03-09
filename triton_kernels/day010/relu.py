import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}),
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["n_elements"],
)
@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # get the program id
    pid = tl.program_id(axis=0)

    # calculate the block start
    block_start = pid * BLOCK_SIZE

    # calculate the offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # calculate the masks
    masks = offsets < n_elements

    # load the input
    input = tl.load(input_ptr + offsets, mask=masks)

    # calculate the output value
    output = tl.where(input > 0, input, 0.0)

    # store the output
    tl.store(output_ptr + offsets, output, mask=masks)


def relu(input: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(input)
    n_elements = input.shape[0]

    # define grid generator
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](input, output, n_elements)

    return output
