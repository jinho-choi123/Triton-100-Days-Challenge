import torch
from loguru import logger

def compare_tensors(
    output: torch.Tensor,
    ref_output: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    max_mismatches_to_log: int = 20,
) -> None:
    """Compare two tensors for shape and numerical closeness.

    Args:
        output (torch.Tensor): Actual tensor produced by the implementation under test.
        ref_output (torch.Tensor): Reference tensor to compare against.
        rtol (float, optional): Relative tolerance for closeness check. Defaults to 1e-5.
        atol (float, optional): Absolute tolerance for closeness check. Defaults to 1e-8.
        equal_nan (bool, optional): Whether NaN values at the same locations are treated as equal. Defaults to False.
        max_mismatches_to_log (int, optional): Maximum number of mismatched locations to log. Defaults to 20.

    Raises:
        AssertionError: If tensor shapes differ or values are not close under the given tolerances.
    """
    logger.info("Comparing tensors with shape output={} ref={}", output.shape, ref_output.shape)
    assert output.shape == ref_output.shape, (
        f"Tensors have different shapes: output={tuple(output.shape)} ref={tuple(ref_output.shape)}"
    )

    if torch.allclose(output, ref_output, rtol=rtol, atol=atol, equal_nan=equal_nan):
        logger.info("Tensors are close enough (rtol={}, atol={})", rtol, atol)
        return

    # Compute elementwise closeness mask using same tolerance rule as allclose.
    abs_diff = (output - ref_output).abs()
    tol = atol + rtol * ref_output.abs()
    close_mask = abs_diff <= tol

    if equal_nan:
        both_nan = torch.isnan(output) & torch.isnan(ref_output)
        close_mask = close_mask | both_nan

    mismatch_idx = torch.nonzero(~close_mask, as_tuple=False)
    mismatch_count = int(mismatch_idx.shape[0])

    max_abs_diff = float(abs_diff.max().item())
    mean_abs_diff = float(abs_diff.mean().item())
    logger.error(
        "Tensors are not close: mismatches={} max_abs_diff={:.6e} mean_abs_diff={:.6e}",
        mismatch_count,
        max_abs_diff,
        mean_abs_diff,
    )

    if mismatch_count > 0:
        # Log worst mismatches first.
        mismatch_abs_diff = abs_diff[~close_mask]
        topk = min(max_mismatches_to_log, mismatch_count)
        top_vals, top_pos = torch.topk(mismatch_abs_diff, k=topk)

        # Move to CPU for stable logging.
        mismatch_idx_cpu = mismatch_idx.detach().cpu()
        top_vals_cpu = top_vals.detach().cpu()
        top_pos_cpu = top_pos.detach().cpu()

        logger.error("Showing top {} mismatches by absolute difference:", topk)
        for rank in range(topk):
            pos_in_mismatch = int(top_pos_cpu[rank].item())
            idx = tuple(int(x) for x in mismatch_idx_cpu[pos_in_mismatch].tolist())

            out_val = float(output[idx].detach().cpu().item())
            ref_val = float(ref_output[idx].detach().cpu().item())
            diff_val = float(top_vals_cpu[rank].item())
            rel_val = diff_val / (abs(ref_val) + 1e-12)

            logger.error(
                "idx={} output={:.6e} ref={:.6e} abs_diff={:.6e} rel_diff={:.6e}",
                idx,
                out_val,
                ref_val,
                diff_val,
                rel_val,
            )

    raise AssertionError(
        f"Tensors are not close (rtol={rtol}, atol={atol}); mismatched elements: {mismatch_count}"
    )