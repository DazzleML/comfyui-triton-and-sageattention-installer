"""Kernel benchmark: SageAttention2++ vs SageAttention3 (FP4) vs torch SDPA.

Investigation for a possible `--sage-version 3`. Answers three questions with
numbers, on one GPU, with the kernel as the only variable:

  1. Is SA3 actually faster than SA2++ on this box?
  2. What does it cost in accuracy? (cosine vs full-precision SDPA)
  3. What does it cost in VRAM?

SA2++ is NOT a separate package -- it is what `sageattention.sageattn()` already
dispatches to on sm120 when CUDA >= 12.8 (`pv_accum_dtype="fp32+fp16"`). So the
comparison is "the thing users already have" vs "the thing that needs a source
build", which is the decision actually on the table.

Requires a single environment holding BOTH packages, so torch/driver/clocks are
identical across measurements:
    python >= 3.13, torch >= 2.8.0 + CUDA >= 12.8, Blackwell (sm_100/120/121)
    pip install <SA2 wheel>          # sageattention
    cd SageAttention/sageattention3_blackwell && pip install .   # sageattn3

Usage:
    python bench_sa2pp_vs_sa3.py            # default shape sweep
    python bench_sa2pp_vs_sa3.py --quick    # one shape, fast
"""
import argparse
import sys

import torch
import torch.nn.functional as F

# (batch, heads, seq_len, head_dim) -- HND layout, which both kernels accept.
# Sequence lengths chosen to bracket image (~4-16k) and video (~32k+) diffusion.
SHAPES = [
    (1, 24, 4096, 128),    # SDXL-ish image attention
    (1, 24, 16384, 128),   # large image / short video
    (1, 24, 32768, 128),   # video (WAN-scale)
]
QUICK_SHAPES = [(1, 24, 16384, 128)]

WARMUP = 5
REPS = 20


def have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def timed(fn, warmup=WARMUP, reps=REPS):
    """Median wall time in ms, CUDA-synchronized, plus peak VRAM in MB."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return times[len(times) // 2], peak_mb


def cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--causal", action="store_true", help="benchmark the causal mask path")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("RESULT: FAIL no_cuda")
        return 1

    cap = torch.cuda.get_device_capability(0)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"device      : {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
    print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}")
    print(f"dtype       : {args.dtype}   causal={args.causal}")

    has_sa2 = have("sageattention")
    has_sa3 = have("sageattn3")
    print(f"sageattention (SA2++): {'yes' if has_sa2 else 'NO'}")
    print(f"sageattn3     (SA3)  : {'yes' if has_sa3 else 'NO'}")
    if not (has_sa2 or has_sa3):
        print("RESULT: FAIL nothing_to_benchmark")
        return 1

    if has_sa2:
        from sageattention import sageattn
        from sageattention.core import get_cuda_version, _cuda_archs
        arch = _cuda_archs[0]
        cv = get_cuda_version()
        mode = "fp32+fp16 (SA2++)" if (arch == "sm120" and cv >= (12, 8)) else "fp32 (NOT SA2++)"
        print(f"SA2 dispatch : {arch}, CUDA {cv} -> {mode}")
    if has_sa3:
        from sageattn3 import sageattn3_blackwell

    shapes = QUICK_SHAPES if args.quick else SHAPES
    print()
    hdr = f"{'shape':>26} | {'kernel':<12} | {'ms':>8} | {'vs SDPA':>8} | {'peak MB':>9} | {'cosine':>8}"
    print(hdr)
    print("-" * len(hdr))

    for (b, h, s, d) in shapes:
        torch.manual_seed(0)
        q = torch.randn(b, h, s, d, device="cuda", dtype=dtype)
        k = torch.randn(b, h, s, d, device="cuda", dtype=dtype)
        v = torch.randn(b, h, s, d, device="cuda", dtype=dtype)
        label = f"({b},{h},{s},{d})"

        # Reference: full-precision-accumulate SDPA.
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=args.causal)
        sdpa_ms, sdpa_mb = timed(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=args.causal))
        print(f"{label:>26} | {'torch SDPA':<12} | {sdpa_ms:8.3f} | {'1.00x':>8} | "
              f"{sdpa_mb:9.1f} | {'--':>8}")

        if has_sa2:
            try:
                out = sageattn(q, k, v, tensor_layout="HND", is_causal=args.causal)
                ms, mb = timed(
                    lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=args.causal))
                print(f"{'':>26} | {'SA2++':<12} | {ms:8.3f} | {sdpa_ms/ms:7.2f}x | "
                      f"{mb:9.1f} | {cosine(out, ref):8.5f}")
            except Exception as e:
                print(f"{'':>26} | {'SA2++':<12} | {'ERROR':>8}   {type(e).__name__}: {str(e)[:40]}")

        if has_sa3:
            try:
                out3 = sageattn3_blackwell(q, k, v, is_causal=args.causal)
                ms, mb = timed(lambda: sageattn3_blackwell(q, k, v, is_causal=args.causal))
                print(f"{'':>26} | {'SA3 (FP4)':<12} | {ms:8.3f} | {sdpa_ms/ms:7.2f}x | "
                      f"{mb:9.1f} | {cosine(out3, ref):8.5f}")
            except Exception as e:
                print(f"{'':>26} | {'SA3 (FP4)':<12} | {'ERROR':>8}   {type(e).__name__}: {str(e)[:40]}")

        del q, k, v, ref
        torch.cuda.empty_cache()
        print()

    print("Notes:")
    print("  - 'vs SDPA' > 1.00x means faster than torch's own attention.")
    print("  - cosine is against full-precision SDPA on identical inputs; SA2 is")
    print("    typically ~0.999. Upstream states SA2 is MORE accurate than SA3.")
    print("  - Kernel-level timing is an upper bound on real-world gain: a diffusion")
    print("    step is more than attention, so end-to-end s/it will improve less.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
