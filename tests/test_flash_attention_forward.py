import triton_kernels.day012.flash_attention_forward as flash_attention_forward
import torch
from loguru import logger
import pytest
import torch.nn.functional as F
from .helpers import compare_tensors

@pytest.mark.parametrize(
    "query_len, kv_len, d",
    [
        (16, 16, 16),
        (32, 32, 32),
        (33, 33, 32),
        (32, 32, 64),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 128),
        (512, 512, 128),
        (512, 128, 128),
        (1024, 1024, 128),
        (2048, 2048, 128),
        (2000, 2000, 128),
    ],
)
def test_flash_attention_forward(query_len, kv_len, d):
    logger.info(f"Testing flash_attention_forward with query_len={query_len}, kv_len={kv_len}, d={d}")
    # create random tensors
    query = torch.randn((query_len, d), device="cuda", dtype=torch.float32)
    key = torch.randn((kv_len, d), device="cuda", dtype=torch.float32)
    value = torch.randn((kv_len, d), device="cuda", dtype=torch.float32)

    attn_mask = torch.zeros((query_len, kv_len), device="cuda", dtype=torch.float32)

    # compute reference
    ref_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)

    # compute flash attention forward
    output = flash_attention_forward.flash_attention_forward(query, key, value, attn_mask)

    compare_tensors(output, ref_output, atol=1e-5)

    logger.info("Flash attention forward test passed!")
