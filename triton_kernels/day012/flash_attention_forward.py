from math import sqrt
from typing import Optional
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Get the device properties.
device_props = torch.cuda.get_device_properties(DEVICE)

# Get the maximum shared memory per SM.
sm_shared_mem = device_props.shared_memory_per_block

def previous_power_of_2(n):
    """Returns the previous power of 2 less than or equal to n."""
    if n <= 1:
        return 1
    # Subtract 1, get the bit length, and shift 1 by that amount
    return 1 << ((n).bit_length() - 1)


@triton.jit
def flash_attention_forward_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    attn_mask_ptr,
    Q_row,
    KV_row,
    d: tl.constexpr,
    B_r: tl.constexpr,
    B_c: tl.constexpr,
):
    # get the program id
    pid = tl.program_id(axis=0)

    # compute the block row start of Q, O, m, l blocks
    qoml_row_start = pid * B_r

    # Calculate the query offsets
    q_row_offsets = qoml_row_start + tl.arange(0, B_r)
    q_row_masks = q_row_offsets < Q_row
    q_col_offsets = tl.arange(0, d)
    q_masks = q_row_masks[:, None]
    q_offsets = q_row_offsets[:, None] * d + q_col_offsets[None, :]

    # load q_block
    q_block = tl.load(Q_ptr + q_offsets, mask=q_masks, other=0.0)

    # Calculate the scale factor.
    scale_factor = 1.0 / (d ** 0.5)

    # Create a buffer for the O block
    o_buffer = tl.zeros((B_r, d), dtype=tl.float32)

    # Create a buffer for the m block and l block
    m_buffer = tl.full((B_r,), float("-inf"), dtype=tl.float32)
    l_buffer = tl.zeros((B_r,), dtype=tl.float32)

    # Loop over KV blocks
    for kv_row_start in tl.range(0, KV_row, B_c):
        # Calculate the offsets for the KV block
        kv_row_offsets = kv_row_start + tl.arange(0, B_c)
        kv_row_masks = kv_row_offsets < KV_row
        kv_col_offsets = tl.arange(0, d)
        kv_masks = kv_row_masks[:, None]

        kv_offsets = kv_row_offsets[:, None] * d + kv_col_offsets[None, :]

        k_block = tl.load(K_ptr + kv_offsets, mask=kv_masks, other=0.0)

        # transpose k_block
        k_block = tl.trans(k_block)

        # load v_block
        v_block = tl.load(V_ptr + kv_offsets, mask=kv_masks, other=0.0)

        # Load attn mask block
        attnmask_row_offsets = qoml_row_start + tl.arange(0, B_r)
        attnmask_row_masks = attnmask_row_offsets < Q_row
        attnmask_col_offsets = kv_row_start + tl.arange(0, B_c)
        attnmask_col_masks = attnmask_col_offsets < KV_row
        attnmask_offsets = attnmask_row_offsets[:, None] * KV_row + attnmask_col_offsets[None, :]
        attnmask_masks = attnmask_row_masks[:, None] & attnmask_col_masks[None, :]
        attnmask_block = tl.load(attn_mask_ptr + attnmask_offsets, mask=attnmask_masks, other=-float("inf"))

        # Calculate the dot product of q_block and k_block
        s = tl.dot(q_block, k_block, input_precision="ieee")

        # Scale the dot product
        s = s * scale_factor

        # Apply the attn mask.
        s += attnmask_block

        # Compute the max, exp(s-m), and sum(exp(s-m)).
        # This is the online softmax computation.
        m = tl.max(s, axis=1)
        

        # Check if the m is greater than previous max value.
        m_new = tl.maximum(m, m_buffer)

        p = tl.exp(s - m_new[:, None])
        l = tl.sum(p, axis=1)

        alpha = tl.exp(m_buffer - m_new)


        l_new = alpha * l_buffer + l

        # Update the output
        o_buffer *= alpha[:, None]
        o_buffer +=  tl.dot(p, v_block, input_precision="ieee")

        # Store the new values
        m_buffer = m_new
        l_buffer = l_new

    # Calculate the offsets for the O block
    o_buffer /= l_buffer[:, None]

    o_row_offsets = qoml_row_start + tl.arange(0, B_r)
    o_row_masks = o_row_offsets < Q_row
    o_col_offsets = tl.arange(0, d)
    o_masks = o_row_masks[:, None]
    o_offsets = o_row_offsets[:, None] * d + o_col_offsets[None, :]
    tl.store(O_ptr + o_offsets, o_buffer, mask=o_masks)


def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    print(f"Shared memory per block: {sm_shared_mem}")

    # validate devices
    assert Q.device == DEVICE, "Q must be on the same device as the current device"
    assert K.device == DEVICE, "K must be on the same device as the current device"
    assert V.device == DEVICE, "V must be on the same device as the current device"

    # check if inputs are contiguous
    assert Q.is_contiguous(), "Q must be contiguous"
    assert K.is_contiguous(), "K must be contiguous"
    assert V.is_contiguous(), "V must be contiguous"

    # get the shape of inputs
    Q_row, Q_col = Q.shape
    K_row, K_col = K.shape
    V_row, V_col = V.shape

    d = Q_col

    # Check if d is a power of 2.
    assert d & (d - 1) == 0, "d must be a power of 2"

    # If attnmask is given
    if attn_mask is not None:
        assert attn_mask.device == DEVICE, "Attn mask must be on the same device as the current device"
        assert attn_mask.is_contiguous(), "Attn mask must be contiguous"
        assert attn_mask.shape == (Q_row, K_row), "Attn mask must have the same shape as Q and K"
    else:
        # Create an all-true attn mask.
        attn_mask = torch.zeros((Q_row, K_row), device=DEVICE)

    # Assert Q_col, K_col, V_col are all equal to d.
    assert Q_col == K_col == V_col, "Q, K, and V must have the same number of columns"

    assert K_row == V_row, "K and V must have the same number of rows"

    # allocate memory for output
    O = torch.empty_like(Q)

    # Calculate B_r and B_c.
    B_r = previous_power_of_2(min(sm_shared_mem // (16 * d), d))
    B_c = previous_power_of_2(min(sm_shared_mem // (16 * d), K_row))

    print(f"B_r: {B_r}, B_c: {B_c}")

    # define grid generator
    grid = lambda meta: ((Q_row - 1) // B_r + 1,)

    # launch kernel
    flash_attention_forward_kernel[grid](Q, K, V, O, attn_mask, Q_row, K_row, d, B_r, B_c)

    return O

    
