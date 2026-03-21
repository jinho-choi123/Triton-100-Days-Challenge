import triton_kernels.day012.flash_attention_forward as flash_attention_forward
import torch
from loguru import logger
import pytest
import torch.nn.functional as F

@pytest.mark.parametrize(
    "query_len, kv_len, d",
    [
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
    ],
)
def test_flash_attention_forward(query_len, kv_len, d):
    logger.info(f"Testing flash_attention_forward with query_len={query_len}, kv_len={kv_len}, d={d}")
    # create random tensors
    query = torch.randn((query_len, d), device="cuda")
    key = torch.randn((kv_len, d), device="cuda")
    value = torch.randn((kv_len, d), device="cuda")

    attn_mask = torch.ones((query_len, kv_len), device="cuda")

    # compute reference
    ref_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)

    # compute flash attention forward
    output = flash_attention_forward.flash_attention_forward(query, key, value, attn_mask)

    assert torch.allclose(output, ref_output, rtol=1e-4, atol=1e-5)

    logger.info("Flash attention forward test passed!")
