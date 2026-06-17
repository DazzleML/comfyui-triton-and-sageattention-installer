"""End-to-end test: CUDA 13.2 PyTorch x cu130 SageAttention post5 wheel.

Companion to test_cuda129_compat.py, for the second tested CUDA-minor gap:
  Does a SageAttention wheel built for cu130 (and torch 2.10 ABI) work on a
  PyTorch built for cu132 + torch 2.12? CUDA 13.x minor-version compatibility
  says it should -- this verifies it (import + real kernel vs torch SDPA).

This also exercises the post5 wheel line, which is cp310-abi3 (Python 3.10+)
rather than post3/post4's cp39, and the "132" -> "130" CUDA alias.

What it does (no system CUDA change -- the cu132 runtime ships in the torch wheel):
  1. Create a slim throwaway venv (NOT the project/ComfyUI venv)
  2. Install torch 2.12.0+cu132 from the PyTorch cu132 index
  3. Install triton-windows (3.6.x for torch 2.12) + numpy
  4. Install SA 2.2.0.post5 cu130torch2.10.0andhigher wheel (cp310-abi3)
  5. Run _cuda129_inner_kernel_test.py inside that venv (import + real kernel + SDPA)
  6. Report PASS/FAIL; tear down venv unless --keep

Usage:
  python tests/one-offs/test_cuda132_compat.py            # default run, cleans up
  python tests/one-offs/test_cuda132_compat.py --keep     # keep venv for inspection
  python tests/one-offs/test_cuda132_compat.py --reuse    # reuse existing venv (fast)

Requires: Windows, an NVIDIA GPU, internet (downloads ~3GB torch). Python 3.10-3.13.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# torch 2.12 + cu132 is the gap; cu130 post5 (torch 2.10 ABI, cp310) is the wheel.
TORCH_SPEC = "torch==2.12.0+cu132"
TORCH_INDEX = "https://download.pytorch.org/whl/cu132"
TRITON_SPEC = "triton-windows>=3.6,<4"
EXTRA_DEPS = ["numpy"]
SA_WHEEL_URL = (
    "https://github.com/woct0rdho/SageAttention/releases/download/"
    "v2.2.0-windows.post5/"
    "sageattention-2.2.0+cu130torch2.10.0andhigher.post5-cp310-abi3-win_amd64.whl"
)
EXPECTED_CUDA = "13.2"

INNER_TEST = Path(__file__).parent / "_cuda129_inner_kernel_test.py"
DEFAULT_VENV = Path("C:/code/comfyui-triton-sageattention-installer/cu132_test_venv")


def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def venv_python(venv_path: Path) -> Path:
    return venv_path / "Scripts" / "python.exe"


def main():
    ap = argparse.ArgumentParser(description="cu132 x cu130-post5 compatibility test")
    ap.add_argument("--venv-path", type=Path, default=DEFAULT_VENV)
    ap.add_argument("--torch-spec", default=TORCH_SPEC)
    ap.add_argument("--torch-index", default=TORCH_INDEX)
    ap.add_argument("--triton-spec", default=TRITON_SPEC)
    ap.add_argument("--sa-wheel", default=SA_WHEEL_URL)
    ap.add_argument("--expected-cuda", default=EXPECTED_CUDA)
    ap.add_argument("--keep", action="store_true", help="keep the venv after the run")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse an existing venv at --venv-path if present (skip create+install)")
    args = ap.parse_args()

    if sys.platform != "win32":
        print("This test targets Windows (abi3 win_amd64 SA wheels).")
        return 2

    venv = args.venv_path.resolve()
    created = False

    try:
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

        r = run([str(vpy), "-m", "pip", "install", "-q",
                 args.torch_spec, "--index-url", args.torch_index])
        if r.returncode != 0:
            print("RESULT: FAIL torch_install")
            return 1

        r = run([str(vpy), "-m", "pip", "install", "-q", args.triton_spec, *EXTRA_DEPS])
        if r.returncode != 0:
            print("RESULT: FAIL triton_numpy_install")
            return 1

        r = run([str(vpy), "-m", "pip", "install", "-q", "--no-deps", args.sa_wheel])
        if r.returncode != 0:
            print("RESULT: FAIL sa_wheel_install")
            return 1

        print("\n" + "=" * 64)
        print("Running inner kernel test inside the cu132 venv")
        print("=" * 64)
        r = run([str(vpy), str(INNER_TEST), args.expected_cuda])
        inner_rc = r.returncode

        print("\n" + "=" * 64)
        if inner_rc == 0:
            print("OVERALL: PASS  -- cu130 post5 wheel works on cu132 torch 2.12")
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
