"""End-to-end test: CUDA 12.9 PyTorch x cu128 SageAttention wheel (Issue #32).

Question this answers empirically:
  Does a SageAttention wheel built for cu128 work on a PyTorch built for cu129?
  (CUDA 12.x minor-version compatibility says it should -- we TEST, not assume.)

What it does (no system CUDA change -- the cu129 runtime ships inside the torch wheel):
  1. Create a slim throwaway venv (NOT the project/ComfyUI venv)
  2. Install torch 2.9.0+cu129 from the PyTorch cu129 index
  3. Install SA 2.2.0.post3 cu128torch2.9.0 wheel (same torch ver, CUDA minor 128 vs 129)
  4. Run _cuda129_inner_kernel_test.py inside that venv (import + real kernel + SDPA compare)
  5. Report PASS/FAIL; tear down venv unless --keep

Usage:
  python tests/one-offs/test_cuda129_compat.py            # default run, cleans up
  python tests/one-offs/test_cuda129_compat.py --keep     # keep venv for inspection
  python tests/one-offs/test_cuda129_compat.py --venv-path C:/code/cu129_test_venv

Requires: Windows, an NVIDIA GPU, internet (downloads ~3GB torch). Python 3.10-3.13.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Defaults chosen to isolate the exact question: identical torch version (2.9.0),
# only the CUDA minor differs (cu129 torch vs cu128 SA wheel).
TORCH_SPEC = "torch==2.9.0+cu129"
TORCH_INDEX = "https://download.pytorch.org/whl/cu129"
# SageAttention imports triton + numpy at module load; torch 2.9 -> triton 3.5.x
# (per the installer's PyTorch/Triton constraint table). These are real runtime
# deps of SA, NOT torch deps, so installing them does not perturb the cu129 torch.
TRITON_SPEC = "triton-windows>=3.5,<3.6"
EXTRA_DEPS = ["numpy"]
SA_WHEEL_URL = (
    "https://github.com/woct0rdho/SageAttention/releases/download/"
    "v2.2.0-windows.post3/"
    "sageattention-2.2.0+cu128torch2.9.0.post3-cp39-abi3-win_amd64.whl"
)

INNER_TEST = Path(__file__).parent / "_cuda129_inner_kernel_test.py"
DEFAULT_VENV = Path("C:/code/comfyui-triton-sageattention-installer/cu129_test_venv")


def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def venv_python(venv_path: Path) -> Path:
    return venv_path / "Scripts" / "python.exe"


def main():
    ap = argparse.ArgumentParser(description="cu129 x cu128-SA compatibility test")
    ap.add_argument("--venv-path", type=Path, default=DEFAULT_VENV,
                    help=f"where to create the slim test venv (default: {DEFAULT_VENV})")
    ap.add_argument("--torch-spec", default=TORCH_SPEC)
    ap.add_argument("--torch-index", default=TORCH_INDEX)
    ap.add_argument("--triton-spec", default=TRITON_SPEC)
    ap.add_argument("--sa-wheel", default=SA_WHEEL_URL)
    ap.add_argument("--keep", action="store_true", help="keep the venv after the run")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse an existing venv at --venv-path if present (skip create+install)")
    args = ap.parse_args()

    if sys.platform != "win32":
        print("This test targets Windows (cp39-abi3 win_amd64 SA wheels).")
        return 2

    venv = args.venv_path.resolve()
    created = False

    try:
        # Create venv unless reusing an existing one. Installs below are
        # idempotent (pip skips already-satisfied requirements), so a reused
        # venv with torch already cached re-runs cheaply.
        if not (args.reuse and venv_python(venv).exists()):
            if venv.exists():
                print(f"Removing stale venv at {venv}")
                shutil.rmtree(venv, ignore_errors=True)
            print(f"Creating slim venv at {venv}")
            r = run([sys.executable, "-m", "venv", str(venv)])
            if r.returncode != 0:
                print("RESULT: FAIL venv_create")
                return 1
            created = True
        else:
            print(f"Reusing existing venv at {venv}")

        vpy = venv_python(venv)
        run([str(vpy), "-m", "pip", "install", "-q", "--upgrade", "pip"])

        # torch from cu129 index (the whole point -- torch.version.cuda == 12.9)
        r = run([str(vpy), "-m", "pip", "install", "-q",
                 args.torch_spec, "--index-url", args.torch_index])
        if r.returncode != 0:
            print("RESULT: FAIL torch_install")
            return 1

        # SageAttention's runtime deps (triton + numpy) -- NOT torch deps, so
        # they don't perturb the cu129 torch. Without these, `import sageattention`
        # fails with ModuleNotFoundError before any kernel can run.
        r = run([str(vpy), "-m", "pip", "install", "-q", args.triton_spec, *EXTRA_DEPS])
        if r.returncode != 0:
            print("RESULT: FAIL triton_numpy_install")
            return 1

        # SageAttention cu128 wheel with --no-deps so pip can't pull a CPU torch
        # over our cu129 build (its deps are torch+triton, already satisfied).
        r = run([str(vpy), "-m", "pip", "install", "-q", "--no-deps", args.sa_wheel])
        if r.returncode != 0:
            print("RESULT: FAIL sa_wheel_install")
            return 1

        vpy = venv_python(venv)
        print("\n" + "=" * 64)
        print("Running inner kernel test inside the cu129 venv")
        print("=" * 64)
        r = run([str(vpy), str(INNER_TEST)])
        inner_rc = r.returncode

        print("\n" + "=" * 64)
        if inner_rc == 0:
            print("OVERALL: PASS  -- cu128 SA wheel works on cu129 torch")
        else:
            print(f"OVERALL: FAIL  -- inner test returned {inner_rc}")
        print("=" * 64)
        return inner_rc

    finally:
        if created and not args.keep:
            print(f"\nCleaning up venv at {venv}")
            shutil.rmtree(venv, ignore_errors=True)
        elif args.keep:
            print(f"\nKeeping venv at {venv} (--keep)")


if __name__ == "__main__":
    sys.exit(main())
