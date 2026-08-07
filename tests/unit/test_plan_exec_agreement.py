"""Plan/execution agreement tests — the invariant the 158-test suite cannot see.

v0.8.6 established that `--dryrun` must predict `--install`, and fixed it for
PyTorch and Triton. It was never true for SageAttention, and nothing could detect
that: the existing matrices test plan *generation* against a fully-populated
environment (23 SAGEATTENTION_MATRIX rows, 0 with torch_ver=None) and never run an
execution path.

The invariant these tests defend is the achievable restatement -- the literal
"dryrun exactly predicts install" is impossible, because install_pytorch runs
`pip install "torch>=2.5.0"` against a live index and pip picks the version:

    Dryrun must never make a claim that install contradicts.

Each test was written to FAIL against the defect it names, then flipped to a hard
failure as its fix landed. They all pass now; a regression turns one red.

Several were added only after running the installer for real against a fresh
ComfyUI checkout -- the fresh-install path (no PyTorch present yet) is where the
plan and the execution disagree most, and no amount of matrix testing against a
fully-populated environment can see it.

Defect IDs (D1, D2, ...) refer to the release's design analysis, kept with the
project's maintainer notes as
`2026-08-07__16-12-25__DISCUSS_Rnd4_FINAL_ASSESSMENT_(version-constraint-generalization).md`.
Each test states its own expected behavior, so the ID is a cross-reference only.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root
sys.path.insert(0, str(Path(__file__).parent))                # sibling test modules

from comfyui_triton_sageattention import (
    ComfyUIInstaller,
    PlatformHandler,
    WindowsHandler,
    parse_sage_version,
)
from test_installplan_matrix import setup_full_mock


def request_sage(installer, raw):
    """Apply a --sage-version request the way __init__ does."""
    installer.sage_version_raw = raw
    installer.sage_version_major, installer.sage_version_exact = parse_sage_version(raw)
    return installer


def sa_action(installer):
    return installer.plan_installation().get_action("SageAttention")


# ---------------------------------------------------------------------------
# D1 — the explicit request must reach the plan
# ---------------------------------------------------------------------------

def test_sage_version_request_changes_the_plan(mock_installer):
    """Different --sage-version requests must not produce identical plans.

    Today all four produce "2.2.0.post6 - SA 2.x", because the planner only ever
    computes the auto target. A user asking for 1.x is shown the 2.x wheel.
    """
    targets = {}
    for raw in ["auto", "1", "2", "2.1.1"]:
        inst = request_sage(mock_installer, raw)
        setup_full_mock(inst, "2.13.0+cu130", "13.0", "13.0", sa_version=None)
        targets[raw] = sa_action(inst).target_version

    assert len(set(targets.values())) > 1, (
        f"plan is identical regardless of the request: {targets}"
    )


def test_explicit_sa1_request_is_not_reported_as_nothing_to_do(mock_installer):
    """--sage-version 1 with SA 2.x installed must not plan KEEP.

    Reproduced against the real installer: dryrun printed
    "Nothing to do - all components are already up to date" and the subsequent
    real run uninstalled 2.2.0.post6 and installed 1.0.6.
    """
    inst = request_sage(mock_installer, "1")
    setup_full_mock(inst, "2.13.0+cu130", "13.0", "13.0",
                    sa_version="2.2.0+cu130torch2.10.0andhigher.post6")
    action = sa_action(inst)

    assert action.action != "KEEP", (
        f"plan says {action.action} ({action.reason!r}) but execution installs SA 1.0.6"
    )


# ---------------------------------------------------------------------------
# D6 — the "newer than we know" shortcut must not pre-empt an explicit request
# ---------------------------------------------------------------------------

def test_sa3_user_with_explicit_request_is_planned(mock_installer):
    """SA 3.x + --sage-version 2 must plan the downgrade, not KEEP.

    The `current_major > 2` early return fires first, so the plan claims KEEP
    while execution dispatches on major == 2 and downgrades.
    """
    inst = request_sage(mock_installer, "2")
    setup_full_mock(inst, "2.13.0+cu130", "13.0", "13.0", sa_version="3.0.0")
    action = sa_action(inst)

    assert action.action != "KEEP", (
        f"plan says KEEP ({action.reason!r}) but execution would downgrade SA 3.0.0"
    )


# ---------------------------------------------------------------------------
# D2 — the plan must not claim a wheel it computed against the wrong torch
# ---------------------------------------------------------------------------

def test_fresh_install_claim_is_not_contradicted(mock_installer):
    """On a fresh install the plan must not confidently claim SA 1.0.6.

    install() installs PyTorch *before* SageAttention, so execution re-derives
    against the newly-installed torch and picks a 2.x wheel. The plan may say
    "determined after PyTorch installs" -- it may not assert the 1.x fallback.
    """
    setup_full_mock(mock_installer, torch_version=None, torch_cuda=None, nvcc_cuda="13.0")
    action = sa_action(mock_installer)

    # What execution will actually see once PyTorch has landed.
    would_install = mock_installer._find_matching_wheel(
        cuda_ver="130", torch_ver="2.13.0", python_ver="312"
    )
    assert would_install is not None, "fixture assumption: a 2.x wheel exists for cu130"

    assert "1.0.6" not in str(action.target_version), (
        f"plan claims {action.target_version!r} but execution installs "
        f"{would_install['sage_version']}"
    )


def test_fresh_install_wheel_details_are_not_contradicted(mock_installer):
    """The wheel-availability fields must not assert the 1.x fallback either.

    Found by running a real fresh install: the ComponentAction target correctly
    said "determined after PyTorch installs", but `plan.sa_wheel_available` was
    still computed against the absent torch, so the same preview printed
    "Will install: SA 1.0.6 from PyPI" and a matching warning two sections
    lower. The honest target is worthless if the display contradicts it.
    """
    setup_full_mock(mock_installer, torch_version=None, torch_cuda=None, nvcc_cuda="13.0")
    plan = mock_installer.plan_installation()

    assert plan.sa_wheel_available is None, (
        f"sa_wheel_available={plan.sa_wheel_available!r} asserts a definite answer; "
        f"the wheel depends on the torch version pip resolves at execution time"
    )
    fallback_warnings = [w for w in plan.warnings if "SA 1.x fallback" in w]
    assert not fallback_warnings, (
        f"preview warns {fallback_warnings!r} but execution installs a 2.x wheel"
    )


def test_fresh_install_triton_claim_is_not_contradicted(mock_installer):
    """Same defect, Triton: the plan must not imply an unconstrained install.

    Execution re-derives the constraint from the torch install_pytorch just put in
    place (e.g. '>=3.6,<4'). The plan cannot know that value -- pip chooses the torch
    version at execution time -- so under the chosen design it names the condition
    rather than asserting either a specific constraint or none at all.
    """
    setup_full_mock(mock_installer, torch_version=None, torch_cuda=None, nvcc_cuda="13.0")
    action = mock_installer.plan_installation().get_action("Triton")
    exec_constraint = mock_installer._get_triton_version_constraint("2.13.0")
    assert exec_constraint, "fixture assumption: torch 2.13 yields a real constraint"

    shown = f"{action.target_version} {action.details}"
    # Honest outcomes: state the actual constraint, or say it isn't known yet.
    # Dishonest outcome: silently advertise an unconstrained install.
    assert (exec_constraint in shown) or (mock_installer.PENDING_TORCH_NOTE in shown), (
        f"plan shows {shown!r}, which implies an unconstrained install, "
        f"but execution pins {exec_constraint!r}"
    )


# ---------------------------------------------------------------------------
# D5 — --force must reach every component, not just PyTorch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component", ["Triton", "SageAttention"])
def test_force_reaches_all_components(mock_installer, component):
    """--install --force must not be a silent no-op for Triton or SageAttention.

    Both execution paths consult the plan and return early on KEEP, so a missing
    force branch in the planner made --force do nothing for them (D5).
    """
    mock_installer.force = True
    setup_full_mock(mock_installer, "2.13.0+cu130", "13.0", "13.0",
                    triton_version="3.7.1", sa_version="2.2.0.post6")
    action = mock_installer.plan_installation().get_action(component)

    assert action.action != "KEEP", (
        f"--force left {component} at KEEP; only PyTorch honors it"
    )


# ---------------------------------------------------------------------------
# Incomplete PyTorch indexes -- the preview must name the index install uses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("detected,expected,why", [
    ("13.2", "130", "cu132 publishes no torchaudio on any platform"),
    ("132",  "130", "already-normalized tag is aliased too"),
    ("13.0", "130", "complete index, unchanged"),
    ("12.9", "129", "complete index -- must NOT borrow CUDA_WHEEL_ALIASES' 129->128"),
    ("12.8", "128", "complete index, unchanged"),
])
def test_pytorch_index_alias(detected, expected, why):
    """Only indexes measured to be incomplete may be redirected.

    The 129->128 case is the point of this test. CUDA_WHEEL_ALIASES maps it
    because no cu129 *SageAttention* wheel exists, but the cu129 *PyTorch* index
    is complete. Sharing one table would silently move CUDA 12.9 users onto
    cu128 torch builds for no reason.
    """
    assert PlatformHandler.pytorch_index_cuda(detected) == expected, why


def test_pytorch_url_uses_complete_index():
    """The URL builder must emit the aliased index, not the detected one."""
    h = object.__new__(WindowsHandler)
    assert h.get_pytorch_install_url("13.2").endswith("/cu130")
    assert h.get_pytorch_install_url("13.0").endswith("/cu130")
    assert h.get_pytorch_install_url("12.8").endswith("/cu128")
    assert h.get_pytorch_install_url("cpu").endswith("/cpu")


def test_fresh_install_cuda_claim_matches_index(mock_installer):
    """On CUDA 13.2 the preview must not advertise a CUDA the install won't use.

    Found by running a real fresh install on a 13.2 machine: pip was handed
    --index-url .../cu132, which has torch and torchvision but zero torchaudio,
    so the whole transaction failed and nothing was installed. The install now
    uses cu130; the preview has to say cu130 too, or it makes a claim the
    install contradicts.
    """
    setup_full_mock(mock_installer, torch_version=None, torch_cuda=None, nvcc_cuda="13.2")
    action = mock_installer.plan_installation().get_action("PyTorch")

    shown = f"{action.target_version} {action.reason}"
    assert "13.0" in shown, f"preview shows {shown!r}, but install uses the cu130 index"
    assert action.target_version.count("13.2") == 0, (
        f"target_version {action.target_version!r} advertises CUDA 13.2; "
        f"pip is pointed at cu130"
    )


# ---------------------------------------------------------------------------
# D7 — execution must use the plan's CUDA, not a second nvcc probe
# ---------------------------------------------------------------------------

def test_force_reinstall_uses_torch_cuda_not_nvcc(mock_installer):
    """When torch's CUDA and nvcc disagree, the plan must follow torch.

    This is the Issue #18 bug class. In v0.6.8 the installer compared
    torch.version.cuda (13.0, bundled with a Portable install) against
    `nvcc --version` (12.6, a stale system toolkit), declared a mismatch, and
    "fixed" it by downgrading a working PyTorch 2.9.1+cu130 to 2.7.0+cu126 --
    taking Triton and SageAttention down with it.

    v0.6.10 made the KEEP/UPGRADE decision torch-first, but `install()` still
    called the nvcc-only probe for the CUDA it handed to install_pytorch(). That
    value only reaches a real `pip install --index-url` under --force, which is
    why it survived: the one path where nvcc could still overwrite a working,
    deliberately-matched install.
    """
    mock_installer.force = True
    setup_full_mock(mock_installer, "2.9.1+cu128", torch_cuda="12.8", nvcc_cuda="13.0",
                    sa_version="2.2.0.post6")
    plan = mock_installer.plan_installation()

    assert plan.cuda_for_wheels == "128", (
        f"plan resolved CUDA {plan.cuda_for_wheels!r} from nvcc; execution would "
        f"reinstall torch from the cu130 index over a working cu128 install (Issue #18)"
    )


def test_missing_nvcc_does_not_downgrade_working_cuda_torch(mock_installer):
    """No nvcc on PATH must not turn a --force reinstall into a CPU install.

    detect_and_setup_cuda() returns "cpu" when `nvcc --version` cannot run, which
    is ordinary for Portable/bundled-CUDA setups with no system-wide toolkit.
    Sourcing execution's CUDA from that probe meant --force could install
    CPU-only PyTorch over a working CUDA one.
    """
    mock_installer.force = True
    setup_full_mock(mock_installer, "2.13.0+cu130", torch_cuda="13.0", nvcc_cuda=None,
                    sa_version="2.2.0.post6")
    plan = mock_installer.plan_installation()

    assert plan.cuda_for_wheels == "130", (
        f"plan resolved {plan.cuda_for_wheels!r} with nvcc absent; a working CUDA "
        f"install must not be reinstalled as CPU-only"
    )


# ---------------------------------------------------------------------------
# Build tools — the last mutation that was not gated by the plan
# ---------------------------------------------------------------------------

def test_no_op_install_does_not_touch_build_tools(mock_installer):
    """A plan of all-KEEP must not upgrade pip/setuptools.

    Found on a live `--install` against a real ComfyUI: the plan printed
    KEEP/KEEP/KEEP, and the run still uninstalled setuptools 82.0.1 and
    installed 83.0.0 (plus a pip bump), while `--dryrun` mentioned neither.
    pip and setuptools are build tooling -- if nothing is being built, touching
    them can only put a working environment at risk.
    """
    mock_installer.force = False
    setup_full_mock(mock_installer, "2.10.0+cu130", "13.0", "13.2",
                    triton_version="3.6.0.post26",
                    sa_version="2.2.0+cu130torch2.9.0andhigher.post4")
    plan = mock_installer.plan_installation()

    assert not plan.has_changes(), "fixture assumption: this environment is fully up to date"
    assert plan.build_tools_upgrade_needed is False, (
        "plan would upgrade pip/setuptools on a run where nothing is installed"
    )


def test_real_install_still_upgrades_build_tools(mock_installer):
    """When something IS being installed, the build tooling is still prepared."""
    setup_full_mock(mock_installer, torch_version=None, torch_cuda=None, nvcc_cuda="13.0")
    plan = mock_installer.plan_installation()

    assert plan.has_changes(), "fixture assumption: a fresh install has work to do"
    assert plan.build_tools_upgrade_needed is True, (
        "build tooling must be prepared before installing packages"
    )


def test_force_still_upgrades_build_tools(mock_installer):
    """--force reaches the build tooling too, consistent with the components."""
    mock_installer.force = True
    setup_full_mock(mock_installer, "2.13.0+cu130", "13.0", "13.0",
                    triton_version="3.7.1", sa_version="2.2.0.post6")
    plan = mock_installer.plan_installation()

    assert plan.build_tools_upgrade_needed is True


# ---------------------------------------------------------------------------
# A no-op install must not write to disk either
# ---------------------------------------------------------------------------

def _make_dev_files(root, libs):
    """Build a python dev-files layout: headers plus the given .lib names."""
    (root / "include").mkdir(parents=True, exist_ok=True)
    for header in ("Python.h", "pyconfig.h", "object.h"):
        (root / "include" / header).write_text("/* stub */", encoding="utf-8")
    (root / "libs").mkdir(parents=True, exist_ok=True)
    for name in libs:
        (root / "libs" / name).write_text("stub", encoding="utf-8")
    return root / "include", root / "libs"


@pytest.mark.parametrize("libs,expected,why", [
    (["python3.lib", "python312.lib"], True,
     "exactly what `python -m venv` produces -- the common case"),
    (["python3.lib", "python313.lib"], True, "any Python version"),
    (["python3.lib", "python312.lib", "_socket.lib", "_ssl.lib", "_bz2.lib", "_lzma.lib"],
     True, "portable python_embeded carries more, still fine"),
    ([], False, "no import library -- genuinely incomplete"),
    (["_socket.lib", "_ssl.lib"], False, "no python*.lib -- genuinely incomplete"),
])
def test_dev_files_detected_in_a_venv(mock_installer, tmp_path, libs, expected, why):
    """A venv's two .lib files must count as complete.

    The check required at least 5 .lib files, but `python -m venv` creates
    exactly two (python3.lib and pythonXY.lib). It therefore returned False for
    every venv on every run, so the dev-files zip was re-downloaded and
    re-extracted over the venv each time -- including on installs that changed
    nothing -- and "Python development files already present" was unreachable.
    Portable's python_embeded ships more .lib files, hiding it.
    """
    include_dir, libs_dir = _make_dev_files(tmp_path, libs)
    result = mock_installer._check_python_dev_files(include_dir, libs_dir)
    assert result is expected, why


def test_dev_files_missing_headers_still_detected(mock_installer, tmp_path):
    include_dir, libs_dir = _make_dev_files(tmp_path, ["python3.lib", "python312.lib"])
    (include_dir / "Python.h").unlink()
    assert mock_installer._check_python_dev_files(include_dir, libs_dir) is False


def test_no_op_install_keeps_an_existing_run_script(mock_installer, tmp_path):
    """A run that changes nothing must not rewrite the launcher.

    run_nvidia_gpu.bat is a plain script users routinely edit -- extra flags, a
    different port, custom paths. It was rewritten from template on every
    install, including ones where all three components reported KEEP, silently
    discarding those edits.
    """
    from comfyui_triton_sageattention import WindowsHandler

    script = tmp_path / "run_nvidia_gpu.bat"
    script.write_text("REM user's customized launcher\n", encoding="utf-8")

    # Build the plan first -- setup_full_mock configures the Mock handler, so the
    # real methods have to be grafted on afterwards.
    mock_installer.force = False
    setup_full_mock(mock_installer, "2.13.0+cu130", "13.0", "13.0",
                    triton_version="3.7.1.post27", sa_version="2.2.0.post6")
    mock_installer._current_plan = mock_installer.plan_installation()

    mock_installer.handler.base_path = tmp_path
    mock_installer.handler.RUN_SCRIPT_NAME = WindowsHandler.RUN_SCRIPT_NAME
    mock_installer.handler.get_run_script_path = (
        lambda: WindowsHandler.get_run_script_path(mock_installer.handler)
    )
    def fail(*a, **k):
        raise AssertionError("create_run_script() was called on a no-op install")
    mock_installer.handler.create_run_script = fail
    assert not mock_installer._current_plan.has_changes(), "fixture: nothing to do"

    mock_installer.create_run_script("13.0")
    assert script.read_text(encoding="utf-8") == "REM user's customized launcher\n", (
        "the user's edited run script was overwritten by a no-op install"
    )


# ---------------------------------------------------------------------------
# Asking a question must not build an environment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("may_create,expect_created,why", [
    (False, False, "reporting/backup commands discover, never create"),
    (True, True, "install/upgrade must still create one when absent"),
])
def test_env_is_only_created_when_the_command_needs_it(tmp_path, may_create,
                                                       expect_created, why):
    """`--backup-clean` in a non-ComfyUI directory used to create a venv there.

    The handler called _setup_python_environment() from its constructor, so
    merely constructing the installer built an environment -- regardless of
    whether the command was going to install anything. Reported from a real run:
    `--backup-clean` created a 17 MB venv and then printed "No backups found".
    """
    import logging
    from comfyui_triton_sageattention import WindowsHandler

    handler = object.__new__(WindowsHandler)
    handler.base_path = tmp_path
    handler.logger = logging.getLogger("test")
    handler.interactive = False
    handler.force = False
    handler.python_mode, handler.python_explicit_path = "auto", None
    handler.python_path = None
    handler.venv_path = None
    handler.environment_type = "unknown"
    handler.may_create_env = may_create

    created = {"ran": False}
    def fake_run(cmd, *a, **k):
        if "venv" in [str(c) for c in cmd]:
            created["ran"] = True
        raise RuntimeError("venv creation should not be reached in this test")
    handler.run_command = fake_run

    try:
        handler._setup_python_environment()
    except Exception:
        pass  # creation path raises by design above; we only assert whether it was attempted

    assert created["ran"] is expect_created, why
    if not may_create:
        assert handler.environment_type == "system", (
            "a read-only command should fall back to system Python, not build a venv"
        )
