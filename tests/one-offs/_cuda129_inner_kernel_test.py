"""Inner kernel test for cu129 x cu128-SA-wheel compatibility (Issue #32).

Runs INSIDE the slim test venv created by test_cuda129_compat.py.
Not meant to be run directly against the project venv.

Verifies:
  1. torch loads and reports the expected CUDA build (12.9)
  2. CUDA device is available (the RTX 5090)
  3. `import sageattention` succeeds with no ABI/load error
  4. a real SageAttention kernel runs on GPU tensors
  5. its output matches torch SDPA within a tolerance (correctness, not just non-crash)

Exit code 0 = PASS, non-zero = FAIL. Emits machine-readable RESULT: lines.
"""
import sys
import traceback


def log(msg):
    print(msg, flush=True)


def main():
    # --- 1. torch + CUDA build ---
    try:
        import torch
    except Exception as e:
        log(f"RESULT: FAIL import_torch {e!r}")
        return 1

    torch_ver = torch.__version__
    cuda_build = torch.version.cuda
    log(f"INFO: torch={torch_ver} torch.version.cuda={cuda_build}")

    # Optional expected CUDA build (e.g. "12.9" or "13.2") passed as argv[1];
    # only a sanity WARN, never fatal to the kernel test.
    expected_cuda = sys.argv[1] if len(sys.argv) > 1 else None
    if expected_cuda and cuda_build != expected_cuda:
        log(f"RESULT: WARN cuda_build_unexpected expected={expected_cuda} got={cuda_build}")

    # --- 2. CUDA device ---
    if not torch.cuda.is_available():
        log("RESULT: FAIL cuda_not_available")
        return 1
    dev_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    log(f"INFO: device='{dev_name}' compute_capability=sm_{cap[0]}{cap[1]}")

    # --- 3. import sageattention ---
    try:
        import sageattention
        from sageattention import sageattn
    except Exception as e:
        log(f"RESULT: FAIL import_sageattention {e!r}")
        traceback.print_exc()
        return 1
    sa_ver = getattr(sageattention, "__version__", "unknown")
    log(f"INFO: sageattention imported, version={sa_ver}")

    # --- 4 + 5. run kernel and compare to SDPA ---
    try:
        import torch.nn.functional as F

        torch.manual_seed(0)
        batch, heads, seq, head_dim = 1, 8, 256, 128
        # HND layout = (batch, heads, seq, dim), which is what SDPA expects too
        q = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float16)

        # Reference: standard scaled dot-product attention
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # SageAttention kernel (HND layout to match the reference tensor shape)
        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        torch.cuda.synchronize()

        if out.shape != ref.shape:
            log(f"RESULT: FAIL shape_mismatch sa={tuple(out.shape)} ref={tuple(ref.shape)}")
            return 1

        out_f = out.float()
        ref_f = ref.float()
        if not torch.isfinite(out_f).all():
            log("RESULT: FAIL non_finite_output")
            return 1

        # Cosine similarity across the flattened tensors (SA is int8-quantized,
        # so expect close-but-not-exact; cosine sim is the robust metric)
        cos = F.cosine_similarity(out_f.flatten(), ref_f.flatten(), dim=0).item()
        max_abs = (out_f - ref_f).abs().max().item()
        mean_abs = (out_f - ref_f).abs().mean().item()
        ref_scale = ref_f.abs().mean().item()
        log(f"INFO: cosine_sim={cos:.5f} max_abs_err={max_abs:.4f} "
            f"mean_abs_err={mean_abs:.4f} ref_mean_abs={ref_scale:.4f}")

        # SageAttention vs full-precision SDPA typically lands >0.99 cosine sim
        if cos >= 0.99:
            log("RESULT: PASS kernel_correct")
            return 0
        else:
            log(f"RESULT: FAIL kernel_inaccurate cosine_sim={cos:.5f}")
            return 1

    except Exception as e:
        log(f"RESULT: FAIL kernel_run {e!r}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
