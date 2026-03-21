from typing import Optional
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Get the device properties.
device_props = torch.cuda.get_device_properties(DEVICE)

# Get the maximum shared memory per SM.
sm_shared_mem = device_props.shared_memory_per_block

def next_power_of_2(n):
    """Returns the next power of 2 greater than or equal to n."""
    if n <= 1:
        return 1
    # Subtract 1, get the bit length, and shift 1 by that amount
    return 1 << (n - 1).bit_length()

@triton.jit
def flash_attention_forward_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    m_ptr,
    l_ptr,
    attn_mask_ptr,
    Q_row,
    KV_row,
    B_r: tl.constexpr,
    B_c: tl.constexpr,
    d: tl.constexpr,
):

    # Loop over KV blocks
    for j in tl.range(0, KV_row, B_c):
        # Calculate the offsets for the KV block
        kv_row_offsets = j + tl.arange(0, B_c)
        kv_row_masks = kv_row_offsets < KV_row
        kv_col_offsets = tl.arange(0, d)

        kv_offsets = kv_row_offsets[:, None] * d + kv_col_offsets[None, :]

        k_block = tl.load(K_ptr + kv_offsets, mask=kv_row_masks[:, None], other=0.0)

        # transpose k_block
        k_block = tl.trans(k_block, (1, 0))

        v_block = tl.load(V_ptr + kv_offsets, mask=kv_row_masks[:, None], other=0.0)

        # Loop over Q, O, m, l blocks
        for i in tl.range(0, Q_row, B_r):
            # Load attn mask blocks
            attn_mask_row_offsets = i + tl.arange(0, B_r)
            attn_mask_row_masks = attn_mask_row_offsets < Q_row
            attn_mask_col_offsets = j + tl.arange(0, B_c)
            attn_mask_col_masks = attn_mask_col_offsets < KV_row
            attn_mask_offsets = attn_mask_row_offsets[:, None] * KV_row + attn_mask_col_offsets[None, :]
            attn_mask_masks = attn_mask_row_masks[:, None] & attn_mask_col_masks[None, :]
            attn_mask_block = tl.load(attn_mask_ptr + attn_mask_offsets, mask=attn_mask_masks, other=0.0)

            # Calculate the offsets
            qo_row_offsets = i + tl.arange(0, B_r)
            qo_row_masks = qo_row_offsets < Q_row
            qo_col_offsets = tl.arange(0, d)

            qo_offsets = qo_row_offsets[:, None] * d + qo_col_offsets[None, :]

            q_block = tl.load(Q_ptr + qo_offsets, mask=qo_row_masks[:, None], other=0.0)
            o_block = tl.load(O_ptr + qo_offsets, mask=qo_row_masks[:, None], other=0.0)

            # Load m, l blocks
            ml_offsets = i + tl.arange(0, B_r)
            ml_masks = ml_offsets < Q_row

            m_prev = tl.load(m_ptr + ml_offsets, mask=ml_masks, other=0.0)
            l_prev = tl.load(l_ptr + ml_offsets, mask=ml_masks, other=0.0)

            # Calculate the dot product of q_block and k_block
            s = tl.dot(q_block, k_block, input_precision="ieee")

            # Scale the dot product by 1/sqrt(d).
            s = s / (d ** 0.5)

            # Apply the attn mask.
            s = tl.where(attn_mask_block == 0.0, -float("inf"), s)

            # Compute the max, exp(s-m), and sum(exp(s-m)).
            # This is the online softmax computation.
            m = tl.max(s, axis=1)
            p = tl.exp(s - m[:, None])
            l = tl.sum(p, axis=1)

            # Check if the m is greater than previous max value.
            m_new = tl.where(m > m_prev, m, m_prev)
            l_new = tl.exp(m_prev - m_new) * l_prev + tl.exp(m - m_new) * l

            # Update the output
            o_new = (1.0 / l_new[:, None]) * (l_prev[:, None] * tl.exp(m_prev - m_new) * o_block + tl.exp(m - m_new) * tl.dot(p, v_block, input_precision="ieee"))

            # Store the new values
            tl.store(m_ptr + ml_offsets, m_new, mask=ml_masks)
            tl.store(l_ptr + ml_offsets, l_new, mask=ml_masks)
            tl.store(O_ptr + qo_offsets, o_new, mask=qo_row_masks[:, None])


def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
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

    # If attnmask is given
    if attn_mask is not None:
        assert attn_mask.device == DEVICE, "Attn mask must be on the same device as the current device"
        assert attn_mask.is_contiguous(), "Attn mask must be contiguous"
        assert attn_mask.shape == (Q_row, K_row), "Attn mask must have the same shape as Q and K"
    else:
        # Create an all-true attn mask.
        attn_mask = torch.ones((Q_row, K_row), device=DEVICE)

    # Assert Q_col, K_col, V_col are all equal to d.
    assert Q_col == K_col == V_col, "Q, K, and V must have the same number of columns"

    assert Q_row == K_row == V_row, "Q, K, and V must have the same number of rows"

    # allocate memory for output
    O = torch.empty_like(Q)
    l = torch.zeros((Q_row,), device=DEVICE)
    m = torch.full((Q_row,), float('-inf'), device=DEVICE)

    # Calculate B_r and B_c.
    B_r = next_power_of_2(min(sm_shared_mem // (4 * d), d))
    B_c = next_power_of_2(min(sm_shared_mem // (4 * d), d))

    # define grid generator
    grid = lambda meta: (1,)

    # launch kernel
    flash_attention_forward_kernel[grid](Q, K, V, O, m, l, attn_mask, Q_row, K_row, B_r, B_c, d)

    return O

    
