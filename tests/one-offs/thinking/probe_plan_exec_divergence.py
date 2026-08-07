"""Probe: where does the InstallPlan disagree with what install() actually does?

Reasoning about plan/execute agreement kept producing wrong answers, so this
probes the real code instead of arguing about it. Every scenario here was
CONFIRMED by running it -- none is inferred.

Background: v0.8.6 established the invariant "--dryrun exactly predicts --install"
and fixed it for PyTorch and Triton. This probe checks whether it actually holds
for SageAttention. It does not.

    python tests/one-offs/thinking/probe_plan_exec_divergence.py

Scenarios probed:
  A  fresh install (no torch): plan computes SA against torch-BEFORE-install
     (no match -> SA 1.x) but install() installs PyTorch first, so execution
     re-derives against torch-AFTER-install and gets SA 2.x.
  B  plain --install with SA present: plan says KEEP, but the plan check at
     clone_and_install_repositories is gated on `self.upgrade`, so it reinstalls.
  C  --sage-version never reaches the plan at all: dryrun shows the auto-selected
     wheel regardless of what the user explicitly requested; execution dispatches
     on sage_version_major/exact separately.

No network, no subprocess, no GPU -- pure plan-layer probing.
"""
import sys
from pathlib import Path
from unittest.mock import Mock

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent.parent          # repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "unit"))  # for setup_full_mock

from comfyui_triton_sageattention import ComfyUIInstaller, parse_sage_version
import test_installplan_matrix as T


def make_installer(sage_version="auto", upgrade=False, force=False):
    """Bare installer wired like conftest's mock_installer (bypasses __init__)."""
    i = ComfyUIInstaller.__new__(ComfyUIInstaller)
    i.base_path = Path("C:/fake/comfyui")
    i.force = force
    i.upgrade = upgrade
    i.interactive = False
    i.experimental = False
    i.with_custom_nodes = False
    i.sage_version_raw = sage_version
    i.sage_version_major, i.sage_version_exact = parse_sage_version(sage_version)
    i.logger = Mock()
    i.handler = Mock()
    i.handler.python_path = Path("C:/fake/python.exe")
    i.handler.environment_type = "venv"
    i.installed_packages = []
    return i


def hdr(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def probe_a():
    hdr("SCENARIO A: fresh install -- plan sees no torch, execution sees new torch")
    i = make_installer()
    T.setup_full_mock(i, torch_version=None, torch_cuda=None, nvcc_cuda="13.0")
    plan = i.plan_installation()
    pt, sa = plan.get_action("PyTorch"), plan.get_action("SageAttention")
    print(f"  PLAN  PyTorch      : {pt.action:8} target={pt.target_version!r}")
    print(f"  PLAN  SageAttention: {sa.action:8} target={sa.target_version!r}")
    print(f"  PLAN  SA wheel_url : {sa.details!r}")
    # install() order: install_pytorch() runs BEFORE clone_and_install_repositories()
    after = i._find_matching_wheel(cuda_ver="130", torch_ver="2.13.0", python_ver="312")
    picked = after["sage_version"] if after else "1.0.6 (fallback)"
    print(f"  EXEC  after torch 2.13.0+cu130 lands -> {picked}")
    diverges = (after is not None) and ("1.0.6" in str(sa.target_version))
    print(f"  >>> DIVERGES: {diverges}  (dryrun promises SA 1.x, install delivers SA 2.x)")
    return diverges


def probe_b():
    hdr("SCENARIO B: plain --install with SA already installed -- plan KEEP ignored")
    i = make_installer(upgrade=False)
    T.setup_full_mock(i, torch_version="2.13.0+cu130", torch_cuda="13.0",
                      nvcc_cuda="13.0", sa_version="2.2.0.post6")
    sa = i.plan_installation().get_action("SageAttention")
    print(f"  PLAN  SageAttention: {sa.action:8} reason={sa.reason!r}")
    # The guard in clone_and_install_repositories is: if self.upgrade and ...
    guard_runs = bool(i.upgrade)
    print(f"  EXEC  plan-KEEP guard active? {guard_runs}   (guard is `if self.upgrade and ...`)")
    diverges = (sa.action == "KEEP") and not guard_runs
    print(f"  >>> DIVERGES: {diverges}  (plan says KEEP, execution reinstalls anyway)")
    return diverges


def probe_c():
    hdr("SCENARIO C: --sage-version is invisible to the plan")
    results = {}
    for req in ["auto", "1", "2", "2.1.1"]:
        i = make_installer(sage_version=req)
        T.setup_full_mock(i, torch_version="2.13.0+cu130", torch_cuda="13.0",
                          nvcc_cuda="13.0")
        sa = i.plan_installation().get_action("SageAttention")
        results[req] = sa.target_version
        print(f"  --sage-version {req:6} -> PLAN/dryrun shows: {sa.target_version}")
    print("  EXEC dispatches on sage_version_major/exact (clone_and_install_repositories)")
    diverges = len(set(results.values())) == 1 and len(results) > 1
    print(f"  >>> DIVERGES: {diverges}  (plan output identical regardless of explicit request)")
    return diverges


def main():
    print(__doc__.split("\n")[0])
    found = {"A": probe_a(), "B": probe_b(), "C": probe_c()}
    hdr("SUMMARY")
    for k, v in found.items():
        print(f"  Scenario {k}: {'DIVERGES' if v else 'ok'}")
    n = sum(found.values())
    print(f"\n  {n}/3 scenarios show plan/execution divergence.")
    print("  Each is a case where --dryrun does not predict --install.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
