"""Live end-to-end check: setuptools constraint handling (Issue #34).

Reproduces the reported bug against a REAL torch install, then verifies the fixed
`upgrade_pip_setuptools()` both (a) does not break a good environment and
(b) repairs one already broken by the old unconditional upgrade.

The bug: `pip install --upgrade pip setuptools` pushes setuptools past bounds
declared by installed packages. torch 2.12.x requires setuptools<82, so pip prints:

    ERROR: pip's dependency resolver does not currently take into account all the
    packages that are installed. ...
    torch 2.12.1 requires setuptools<82, but you have setuptools 83.0.0

pip exits 0, so the install "works anyway" -- but `pip check` reports the env broken.

Uses a CPU-only torch (~200MB) rather than a CUDA build: the constraint comes from
torch's metadata, so no GPU is needed to exercise it.

Usage:
  python tests/one-offs/test_setuptools_constraint_live.py
  python tests/one-offs/test_setuptools_constraint_live.py --keep
  python tests/one-offs/test_setuptools_constraint_live.py --reuse   # skip re-download
"""
import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import comfyui_triton_sageattention as installer_mod

# torch 2.12.1 declares setuptools<82 -- the version from the Issue #34 report.
TORCH_SPEC = "torch==2.12.1"
DEFAULT_VENV = Path("C:/code/comfyui-triton-sageattention-installer/setuptools_repro_venv")


def venv_python(venv: Path) -> Path:
    sub = "Scripts" if sys.platform == "win32" else "bin"
    return venv / sub / ("python.exe" if sys.platform == "win32" else "python")


def pip(py: Path, *args, capture=True):
    return subprocess.run([str(py), "-m", "pip", *args],
                          capture_output=capture, text=True)


def setuptools_version(py: Path) -> str:
    out = pip(py, "show", "setuptools").stdout or ""
    for line in out.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return "(absent)"


def pip_check(py: Path) -> str:
    r = pip(py, "check")
    return (r.stdout or r.stderr or "").strip()


def build_installer(py: Path):
    """Minimal ComfyUIInstaller wired to the target venv (bypasses heavy __init__)."""
    log = logging.getLogger("live-test")
    handler_cls = (installer_mod.WindowsHandler if sys.platform == "win32"
                   else installer_mod.LinuxHandler)
    handler = handler_cls.__new__(handler_cls)
    handler.python_path = py
    handler.logger = log
    inst = installer_mod.ComfyUIInstaller.__new__(installer_mod.ComfyUIInstaller)
    inst.handler = handler
    inst.logger = log
    return inst


def main():
    ap = argparse.ArgumentParser(description="live setuptools constraint check")
    ap.add_argument("--venv-path", type=Path, default=DEFAULT_VENV)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    venv = args.venv_path.resolve()
    created = False
    failures = []

    try:
        if not (args.reuse and venv_python(venv).exists()):
            if venv.exists():
                shutil.rmtree(venv, ignore_errors=True)
            print(f"Creating venv at {venv}")
            if subprocess.run([sys.executable, "-m", "venv", str(venv)]).returncode != 0:
                print("RESULT: FAIL venv_create")
                return 1
            created = True
            py = venv_python(venv)
            print(f"Installing {TORCH_SPEC} (CPU) ...")
            if pip(py, "install", "-q", TORCH_SPEC).returncode != 0:
                print("RESULT: FAIL torch_install")
                return 1
        else:
            print(f"Reusing venv at {venv}")

        py = venv_python(venv)
        inst = build_installer(py)

        # --- 1. discovery finds torch's declared bound ---
        raw = inst._get_installed_setuptools_requirements()
        spec = inst._parse_setuptools_constraint(raw)
        print(f"\n[1] discovered: raw={raw} -> spec={spec!r}")
        if spec != "setuptools<82":
            failures.append(f"expected 'setuptools<82', discovered {spec!r}")

        # --- 2. break the env exactly as the old code did ---
        pip(py, "install", "-q", "--upgrade", "setuptools")
        broken_ver, broken_check = setuptools_version(py), pip_check(py)
        print(f"[2] after unconditional upgrade: setuptools={broken_ver}")
        print(f"    pip check: {broken_check}")
        if "No broken requirements" in broken_check:
            print("    (note: environment not actually broken -- torch may have "
                  "dropped its upper bound; the rest of this test is then moot)")

        # --- 3. the fix repairs it ---
        print("\n[3] running fixed upgrade_pip_setuptools() ...")
        inst.upgrade_pip_setuptools()
        fixed_ver, fixed_check = setuptools_version(py), pip_check(py)
        print(f"    setuptools={fixed_ver}")
        print(f"    pip check: {fixed_check}")
        if "No broken requirements" not in fixed_check:
            failures.append(f"env still broken after fix: {fixed_check}")

        # --- 4. running again is stable (no oscillation) ---
        print("\n[4] running again (idempotence) ...")
        inst.upgrade_pip_setuptools()
        again_check = pip_check(py)
        if "No broken requirements" not in again_check:
            failures.append(f"env broken on second run: {again_check}")

        # --- 5. FRESH-INSTALL ordering: is #34 also fixed when torch isn't there yet? ---
        # install() runs upgrade_pip_setuptools() BEFORE install_pytorch(). On a
        # torch-less env, discovery finds no constraint and setuptools upgrades
        # unbounded -- and only THEN does torch (declaring setuptools<82) install.
        # Verified: pip BACKTRACKS setuptools while resolving torch, so the env
        # ends up consistent. pip will not retroactively fix an already-installed
        # package's constraint (which is why the upgrade case broke), but it does
        # respect constraints of a package it is currently resolving.
        # This guards that behavior in case pip's resolver ever changes.
        print("\n[5] fresh-install ordering (no torch -> unbounded upgrade -> torch lands) ...")
        fresh = venv.parent / "freshinstall_probe_venv"
        shutil.rmtree(fresh, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "venv", str(fresh)],
                       capture_output=True)
        fpy = venv_python(fresh)
        pip(fpy, "install", "-q", "--upgrade", "pip", "setuptools")
        unbounded = setuptools_version(fpy)
        pip(fpy, "install", "-q", TORCH_SPEC)
        after_torch, fresh_check = setuptools_version(fpy), pip_check(fpy)
        print(f"    setuptools before torch: {unbounded}")
        print(f"    setuptools after  torch: {after_torch}")
        print(f"    pip check: {fresh_check}")
        if "No broken requirements" not in fresh_check:
            failures.append(
                f"FRESH-INSTALL case broken (#34 not fixed for new users): {fresh_check}")
        shutil.rmtree(fresh, ignore_errors=True)

        print("\n" + "=" * 60)
        if failures:
            for f in failures:
                print(f"RESULT: FAIL {f}")
            print("OVERALL: FAIL")
            return 1
        print("OVERALL: PASS -- constraint respected, env consistent, idempotent")
        print("=" * 60)
        return 0

    finally:
        if created and not args.keep:
            print(f"\nCleaning up {venv}")
            shutil.rmtree(venv, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
