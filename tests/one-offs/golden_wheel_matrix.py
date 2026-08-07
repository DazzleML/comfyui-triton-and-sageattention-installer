"""Golden-output snapshot of the wheel/constraint decision surface.

Purpose: prove a refactor changed NO behavior in the data-driven core.

The 158 unit tests assert specific scenarios. This instead sweeps the WHOLE
decision surface (every CUDA code x torch version x python version x
experimental flag) and records what the installer decides for each. Capture it
before a refactor, capture it after, and diff:

    python tests/one-offs/golden_wheel_matrix.py > before.txt
    ...refactor...
    python tests/one-offs/golden_wheel_matrix.py > after.txt
    diff before.txt after.txt        # must be EMPTY

Byte-identical output means the wheel-selection, CUDA-alias, ABI-floor and
Triton-constraint logic all behave exactly as before -- which is the v0.8.6
--dryrun-predicts-install invariant, checked exhaustively rather than by sample.

Pure in-process function calls: no subprocess, no network, no GPU. Runs in ~1s.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from comfyui_triton_sageattention import ComfyUIInstaller

# Every CUDA code the installer could plausibly see: shipped wheels, the tested
# aliases (129->128, 132->130), untested gaps, and future/unknown ones.
CUDA_CODES = ["118", "121", "124", "126", "127", "128", "129", "130", "131", "132", "140", "150"]

# Torch versions spanning every configured line plus unknown-future ones.
TORCH_VERSIONS = [
    "2.4.0", "2.5.0", "2.5.1", "2.6.0", "2.7.0", "2.7.1", "2.8.0",
    "2.9.0", "2.9.1", "2.10.0", "2.11.0", "2.12.0", "2.12.1", "2.13.0",
    "2.14.0", "3.0.0",
]

PYTHON_VERSIONS = ["39", "310", "311", "312", "313", "314"]


def installer():
    """Bare instance -- these methods are pure w.r.t. the wheel config table."""
    return ComfyUIInstaller.__new__(ComfyUIInstaller)


def emit_wheel_matching(inst, out):
    out.append("=" * 78)
    out.append("SECTION 1: _find_matching_wheel(cuda, torch, python, experimental)")
    out.append("=" * 78)
    for cuda in CUDA_CODES:
        for torch in TORCH_VERSIONS:
            for py in PYTHON_VERSIONS:
                for exp in (False, True):
                    try:
                        m = inst._find_matching_wheel(
                            cuda_ver=cuda, torch_ver=torch, python_ver=py,
                            include_experimental=exp,
                        )
                    except Exception as e:
                        out.append(f"cu{cuda} t{torch} py{py} exp={int(exp)} -> EXC {type(e).__name__}: {e}")
                        continue
                    if m is None:
                        out.append(f"cu{cuda} t{torch} py{py} exp={int(exp)} -> None")
                    else:
                        out.append(
                            f"cu{cuda} t{torch} py{py} exp={int(exp)} -> "
                            f"sa={m['sage_version']} wheel_cuda={m['cuda']} "
                            f"abi3={m['is_abi3']} exp_wheel={m['is_experimental']} "
                            f"url={m['wheel_url'].rsplit('/', 1)[-1]}"
                        )


def emit_cuda_alias(inst, out):
    out.append("")
    out.append("=" * 78)
    out.append("SECTION 2: _cuda_matches(wheel_cuda, detected_cuda)")
    out.append("=" * 78)
    for wheel in CUDA_CODES:
        for detected in CUDA_CODES:
            out.append(f"wheel={wheel} detected={detected} -> {inst._cuda_matches(wheel, detected)}")


def emit_abi3(inst, out):
    out.append("")
    out.append("=" * 78)
    out.append("SECTION 3: _abi3_cp(py_spec)")
    out.append("=" * 78)
    for spec in [None, "39", "310", "311", "312", "313"]:
        out.append(f"py_spec={spec} -> {inst._abi3_cp(spec)}")


def emit_triton(inst, out):
    out.append("")
    out.append("=" * 78)
    out.append("SECTION 4: Triton constraint + bidirectional compatibility")
    out.append("=" * 78)
    for torch in TORCH_VERSIONS:
        out.append(f"constraint(torch {torch}) -> {inst._get_triton_version_constraint(torch)!r}")
    triton_versions = ["3.1.0", "3.2.0", "3.3.1", "3.4.0", "3.5.1", "3.6.0", "3.7.0", "4.0.0"]
    for triton in triton_versions:
        for torch in TORCH_VERSIONS:
            ok, msg = inst._check_triton_pytorch_compatibility(triton, torch)
            out.append(f"compat(triton {triton}, torch {torch}) -> {ok} | {msg}")


def emit_sa_versions(inst, out):
    out.append("")
    out.append("=" * 78)
    out.append("SECTION 5: _get_available_sa2_versions (availability report)")
    out.append("=" * 78)
    for cuda in ["124", "126", "128", "129", "130", "132"]:
        for torch in ["2.5.1", "2.7.1", "2.9.0", "2.10.0", "2.12.0", "2.13.0"]:
            for py in ["39", "310", "312"]:
                info = inst._get_available_sa2_versions(cuda, torch, py, include_experimental=False)
                for ver in sorted(info):
                    d = info[ver]
                    out.append(
                        f"cu{cuda} t{torch} py{py} :: {ver} compatible={d.get('compatible')} "
                        f"reason={d.get('reason', '')!r}"
                    )


def emit_sa_version_match(inst, out):
    out.append("")
    out.append("=" * 78)
    out.append("SECTION 6: _sageattention_version_matches / major parsing")
    out.append("=" * 78)
    installed = ["1.0.6", "2.2.0.post3", "2.2.0+cu128torch2.7.1.post3",
                 "2.2.0+cu130torch2.10.0andhigher.post6", "3.0.0", "-", ""]
    targets = ["1.0.6", "2.2.0.post3", "2.2.0.post4", "2.2.0.post5", "2.2.0.post6", "2.2.0"]
    for i in installed:
        out.append(f"major({i!r}) -> {inst._parse_sageattention_major_version(i)}")
        for t in targets:
            out.append(f"matches({i!r}, {t!r}) -> {inst._sageattention_version_matches(i, t)}")


def emit_setuptools(inst, out):
    out.append("")
    out.append("=" * 78)
    out.append("SECTION 7: _parse_setuptools_constraint (Issue #34)")
    out.append("=" * 78)
    cases = [
        [], None, ["setuptools"], ["setuptools<82"], ["setuptools>=77.0.3"],
        ["setuptools<82", "setuptools>=77.0.3"], ["setuptools<82", "setuptools<82"],
        ['setuptools; extra == "dev"'], ['setuptools>=70; extra == "build"'],
        ['setuptools>=77.0.3; python_version < "3.12"'], ["SetupTools<82"],
        ["setuptools[core]>=70"], ["numpy>=1.0"], ["  setuptools  <82  "],
    ]
    for c in cases:
        out.append(f"{c!r} -> {ComfyUIInstaller._parse_setuptools_constraint(c)!r}")


def emit_cuda_format(inst, out):
    out.append("")
    out.append("=" * 78)
    out.append("SECTION 8: _format_cuda_version")
    out.append("=" * 78)
    for code in CUDA_CODES + ["1210", "9", "", "cpu"]:
        out.append(f"{code!r} -> {inst._format_cuda_version(code)!r}")


def main():
    inst = installer()
    out = []
    out.append("GOLDEN WHEEL/CONSTRAINT DECISION MATRIX")
    out.append("Diff two runs of this file to prove a refactor changed no behavior.")
    out.append("")
    for fn in (emit_wheel_matching, emit_cuda_alias, emit_abi3, emit_triton,
               emit_sa_versions, emit_sa_version_match, emit_setuptools, emit_cuda_format):
        fn(inst, out)
    text = "\n".join(out)
    print(text)
    print(f"\n# total decision rows: {len(out)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
