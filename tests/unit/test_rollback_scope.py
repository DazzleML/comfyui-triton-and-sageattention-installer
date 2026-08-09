"""Rollback and teardown scope tests (Issue #36).

A German Windows user reported that the installer removed their working PyTorch
and Triton. The mechanism was two independent defects:

  A. `subprocess.run(..., text=True)` with no `encoding=` decodes child output
     using the PARENT's locale codepage (cp1252 on German Windows), while native
     tools like nvcc/cl/nmake emit the OEM console codepage (cp850). Byte 0x81
     is "u-umlaut" in cp850 and undefined in cp1252, so one umlaut anywhere in a
     tool's output raised UnicodeDecodeError and failed the install.

  B. `install()` caught that (or any) exception and called
     `cleanup_installation()`, which pip-uninstalled a HARDCODED list --
     torch, torchvision, torchaudio, triton, triton-windows, sageattention --
     regardless of what the run had actually installed. On a crash that happened
     before installing anything, it removed packages the installer never touched.

`self.installed_packages` had recorded the truth the entire time (since the
initial commit) and the cleanup path ignored it.

NOTHING HERE ACTUALLY UNINSTALLS. `pip_uninstall` is mocked throughout and the
tests assert the ARGUMENT LIST -- what *would* be removed -- which is precisely
the property that was wrong.
"""
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

from comfyui_triton_sageattention import ComfyUIInstaller


@pytest.fixture
def installer():
    """A bare installer with pip_uninstall mocked and nothing else real."""
    import logging

    inst = object.__new__(ComfyUIInstaller)
    inst.logger = logging.getLogger("test")
    inst.interactive = True
    inst.installed_packages = []
    inst._restore_on_failure = {}
    inst.handler = Mock()
    inst.handler.pip_uninstall = Mock()
    inst.handler.pip_install = Mock()
    inst.handler.python_path = "python.exe"
    return inst


def uninstalled(installer):
    """Every package name passed to pip_uninstall across all calls."""
    return [pkg for call in installer.handler.pip_uninstall.call_args_list
            for pkg in call.args[0]]


# ---------------------------------------------------------------------------
# Failure rollback -- scope must equal what this run installed
# ---------------------------------------------------------------------------

def test_crash_before_installing_anything_removes_nothing(installer):
    """THE REPORTED BUG. A crash with nothing installed must uninstall nothing.

    Issue #36: the run died on a UnicodeDecodeError before installing anything,
    and the handler removed the user's PyTorch and Triton anyway.
    """
    installer.installed_packages = []
    installer._rollback_failed_install()

    assert installer.handler.pip_uninstall.call_count == 0, (
        f"rollback uninstalled {uninstalled(installer)} after installing nothing"
    )


def test_rollback_never_touches_torch_it_did_not_install(installer):
    """A run that only installed SageAttention must not remove PyTorch."""
    installer.installed_packages = ["sageattention"]
    installer._rollback_failed_install()

    removed = uninstalled(installer)
    assert removed == ["sageattention"], f"expected only sageattention, got {removed}"
    for survivor in ("torch", "torchvision", "torchaudio", "triton", "triton-windows"):
        assert survivor not in removed, (
            f"rollback removed {survivor}, which this run never installed"
        )


def test_rollback_removes_exactly_what_was_recorded(installer):
    """When the run did install torch, rolling it back is correct and expected."""
    installer.installed_packages = ["torch", "torchvision", "torchaudio", "sageattention"]
    installer._rollback_failed_install()

    assert sorted(uninstalled(installer)) == sorted(
        ["sageattention", "torch", "torchaudio", "torchvision"]
    )


def test_rollback_deduplicates(installer):
    """SageAttention can be appended from several code paths in one run."""
    installer.installed_packages = ["sageattention", "sageattention", "torch"]
    installer._rollback_failed_install()

    removed = uninstalled(installer)
    assert sorted(removed) == ["sageattention", "torch"], removed


def test_failure_path_never_uses_the_hardcoded_list(installer):
    """The hardcoded TRITON_PACKAGES must not reach the failure path at all.

    This is the structural version of the bug: any future change that routes
    the teardown list back into rollback trips this.
    """
    installer.installed_packages = ["sageattention"]
    installer._rollback_failed_install()

    removed = set(uninstalled(installer))
    assert not removed.issuperset(set(ComfyUIInstaller.TRITON_PACKAGES)), (
        "failure rollback passed the full teardown list to pip_uninstall"
    )


def test_rollback_survives_pip_failure(installer):
    """A failing uninstall during rollback must not mask the original error."""
    installer.installed_packages = ["sageattention"]
    installer.handler.pip_uninstall = Mock(side_effect=RuntimeError("pip exploded"))
    installer._rollback_failed_install()  # must not raise


# ---------------------------------------------------------------------------
# Explicit --cleanup -- destructive by design, so it must ask
# ---------------------------------------------------------------------------

def test_cleanup_declined_removes_nothing(installer, monkeypatch):
    """Answering no to the teardown prompt must uninstall nothing."""
    monkeypatch.setattr(ComfyUIInstaller, "_confirm", lambda self, q, **k: False)
    installer.base_path = Path("nonexistent-for-test")
    installer.cleanup_installation()

    assert installer.handler.pip_uninstall.call_count == 0, (
        f"declined cleanup still removed {uninstalled(installer)}"
    )


def test_cleanup_non_interactive_declines(installer):
    """--non-interactive must not assume consent for removing PyTorch.

    Failing safe matters more than convenience here: a teardown that does not
    happen is recoverable by re-running; one that does is not.
    """
    installer.interactive = False
    installer.base_path = Path("nonexistent-for-test")
    installer.cleanup_installation()

    assert installer.handler.pip_uninstall.call_count == 0, (
        f"non-interactive cleanup removed {uninstalled(installer)} without consent"
    )


def test_cleanup_default_leaves_pytorch_alone(installer, monkeypatch):
    """A plain --cleanup removes the acceleration stack, NOT PyTorch.

    These were one hardcoded list inherited from the original batch script, so
    `--cleanup` removed torch too. The documented "clean slate reinstall" is
    --cleanup then --install, and an --install can fail for reasons outside the
    user's control (a missing upstream wheel is enough). That sequence could
    leave someone with no PyTorch and no way back, since rollback only undoes
    what the failed run itself installed.
    """
    monkeypatch.setattr(ComfyUIInstaller, "_confirm", lambda self, q, **k: True)
    installer.base_path = Path("nonexistent-for-test")
    installer.cleanup_installation()

    removed = uninstalled(installer)
    assert "sageattention" in removed and "triton-windows" in removed, removed
    for survivor in ("torch", "torchvision", "torchaudio"):
        assert survivor not in removed, (
            f"a plain --cleanup removed {survivor}; PyTorch requires `--cleanup all`"
        )


def test_cleanup_all_removes_pytorch_too(installer, monkeypatch):
    """`--cleanup all` is the explicit opt-in for a full teardown."""
    monkeypatch.setattr(ComfyUIInstaller, "_confirm", lambda self, q, **k: True)
    installer.base_path = Path("nonexistent-for-test")
    installer.cleanup_installation(include_pytorch=True)

    removed = uninstalled(installer)
    assert "torch" in removed and "sageattention" in removed, removed


def test_cleanup_all_still_requires_consent(installer, monkeypatch):
    """The opt-in does not bypass the prompt -- it is the more destructive path."""
    monkeypatch.setattr(ComfyUIInstaller, "_confirm", lambda self, q, **k: False)
    installer.base_path = Path("nonexistent-for-test")
    installer.cleanup_installation(include_pytorch=True)

    assert installer.handler.pip_uninstall.call_count == 0


# ---------------------------------------------------------------------------
# Encoding -- the trigger
# ---------------------------------------------------------------------------

def test_german_console_output_does_not_crash_the_decode():
    """b'Ung\\x81ltig' is 'Ungueltig' in cp850; 0x81 is undefined in cp1252.

    This is the reported crash byte-for-byte. The run path now decodes as UTF-8
    with errors="replace", which cannot raise for any input.
    """
    german_cp850 = "Ungültig".encode("cp850")
    assert b"\x81" in german_cp850, "fixture assumption: u-umlaut is 0x81 in cp850"

    with pytest.raises(UnicodeDecodeError):
        german_cp850.decode("cp1252")          # what the old code did

    assert german_cp850.decode("utf-8", errors="replace")  # what it does now


@pytest.mark.parametrize("raw,expected", [
    (b"Cuda compilation tools, release 13.2, V13.2.51", "13.2"),
    (b"Ung\x81ltig\nCuda compilation tools, release 12.8, V12.8.61", "12.8"),
    (b"\x81\x82\x83 release 13.0, V13.0.1", "13.0"),
])
def test_version_parsing_survives_undecodable_bytes(raw, expected):
    """Replacement characters must not break the ASCII patterns we extract."""
    import re

    text = raw.decode("utf-8", errors="replace")
    match = re.search(r"release (\d+\.\d+)", text)
    assert match and match.group(1) == expected, f"failed on {text!r}"


def test_run_command_specifies_an_encoding():
    """Structural guard: no subprocess.run may rely on the machine's locale.

    The defect was an absent keyword argument, so absence is what to assert.
    """
    import ast

    src = Path(__file__).parent.parent.parent / "comfyui_triton_sageattention.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"):
            continue
        kwargs = {k.arg for k in node.keywords}
        if "text" in kwargs and "encoding" not in kwargs:
            offenders.append(node.lineno)

    assert not offenders, (
        f"subprocess.run(text=True) without encoding= at line(s) {offenders}; "
        f"this decodes with the machine's locale codepage and crashes on "
        f"non-English Windows (Issue #36)"
    )


# ---------------------------------------------------------------------------
# Restoring what the run deliberately removed (the CPU->CUDA switch)
# ---------------------------------------------------------------------------

def installed(installer):
    """Every package spec passed to pip_install across all calls."""
    return [pkg for call in installer.handler.pip_install.call_args_list
            for pkg in call.args[0]]


def test_failed_cpu_to_cuda_switch_restores_torch(installer):
    """The one path where the installer removes working packages on purpose.

    install_pytorch() uninstalls torch/torchvision/torchaudio before reinstalling,
    because pip will not swap a CPU build for a CUDA one. If that reinstall then
    fails, the user is left with NO PyTorch -- rollback cannot help, because
    installed_packages never recorded a package we removed rather than added.
    """
    installer._restore_on_failure = {
        "torch": "2.5.1+cpu",
        "torchvision": "0.20.1+cpu",
        "torchaudio": "2.5.1+cpu",
    }
    installer._rollback_failed_install()

    restored = installed(installer)
    assert "torch==2.5.1+cpu" in restored, f"torch was not restored: {restored}"
    assert "torchvision==0.20.1+cpu" in restored
    assert "torchaudio==2.5.1+cpu" in restored


def test_restore_uses_the_index_the_build_came_from(installer):
    """A +cpu build must be reinstalled from the CPU index, not PyPI."""
    installer._restore_on_failure = {"torch": "2.5.1+cpu"}
    installer._rollback_failed_install()

    args = installer.handler.pip_install.call_args.args
    assert "--index-url" in args[1], f"no index url passed: {args}"
    assert "download.pytorch.org/whl/cpu" in " ".join(args[1])


@pytest.mark.parametrize("version,expected", [
    ("2.5.1+cpu", "https://download.pytorch.org/whl/cpu"),
    ("2.9.1+cu130", "https://download.pytorch.org/whl/cu130"),
    ("2.7.0+cu128", "https://download.pytorch.org/whl/cu128"),
    ("2.5.1", None),                    # plain PyPI build -- no index needed
    ("2.5.1+somethingelse", None),      # unrecognised -- must not guess
    ("", None),
])
def test_index_inferred_from_version(version, expected):
    """Restoring the wrong build would be its own bug, so an unknown shape
    returns None rather than guessing an index."""
    assert ComfyUIInstaller._index_url_for_version(version) == expected


def test_nothing_recorded_means_nothing_restored(installer):
    """A normal failure (no deliberate removal) must not reinstall anything."""
    installer.installed_packages = ["sageattention"]
    installer._restore_on_failure = {}
    installer._rollback_failed_install()

    assert installer.handler.pip_install.call_count == 0


def test_restore_failure_tells_the_user_how_to_recover(installer, capsys):
    """If the reinstall fails, the user must not be left guessing.

    They are now without PyTorch through no action of their own, so the exact
    command to put it back has to be on screen.
    """
    installer._restore_on_failure = {"torch": "2.5.1+cpu"}
    installer.handler.pip_install = Mock(side_effect=RuntimeError("network gone"))
    installer._rollback_failed_install()          # must not raise

    out = capsys.readouterr().out
    assert "no PyTorch" in out
    assert "torch==2.5.1+cpu" in out, "the exact restore command must be shown"
    assert "--backup-restore" in out, "the backup escape hatch must be offered"


def test_rollback_removes_ours_before_restoring_theirs(installer):
    """Ordering matters: uninstall this run's installs, then put theirs back.

    Reversed, the uninstall would strip the torch we just restored.
    """
    order = []
    installer.installed_packages = ["torch", "torchvision", "torchaudio"]
    installer._restore_on_failure = {"torch": "2.5.1+cpu"}
    installer.handler.pip_uninstall = Mock(side_effect=lambda *a, **k: order.append("uninstall"))
    installer.handler.pip_install = Mock(side_effect=lambda *a, **k: order.append("install"))

    installer._rollback_failed_install()
    assert order == ["uninstall", "install"], order


# ---------------------------------------------------------------------------
# Telling the user PyTorch is missing after a failed install
# ---------------------------------------------------------------------------

def make_plan(cuda="130"):
    from comfyui_triton_sageattention import InstallPlan
    plan = InstallPlan()
    plan.cuda_for_wheels = cuda
    return plan


def prepare_report(installer, torch_version=None, cuda="130"):
    """Wire an installer for _report_missing_pytorch(); torch_version=None means absent."""
    from comfyui_triton_sageattention import WindowsHandler
    installer._restore_on_failure = {}
    installer._current_plan = make_plan(cuda)
    installer._get_installed_version = Mock(return_value=torch_version)
    installer.handler.python_path = r"C:\scratch\venv\Scripts\python.exe"
    installer.handler.get_pytorch_install_url = (
        lambda c: WindowsHandler.get_pytorch_install_url(installer.handler, c)
    )
    installer.handler.pytorch_index_cuda = WindowsHandler.pytorch_index_cuda
    installer.handler.PYTORCH_INDEX_ALIASES = WindowsHandler.PYTORCH_INDEX_ALIASES
    return installer


def test_missing_pytorch_is_reported_with_an_install_command(installer, capsys):
    """"Nothing to roll back" alone reads as an all-clear to someone with no torch.

    Reached identically from a fresh install that failed, from `--cleanup all`,
    or from a venv that never had torch. The installer cannot tell those apart --
    two separate invocations mean the packages are already absent by the time it
    plans -- so it reports the end state rather than attributing a cause.
    """
    prepare_report(installer, torch_version=None)
    installer._report_missing_pytorch()

    out = capsys.readouterr().out
    assert "no PyTorch" in out
    assert "pip install" in out, "the user must get an actual command, not just bad news"
    assert "torch>=2.5.0" in out
    assert r"C:\scratch\venv\Scripts\python.exe" in out, "must target THEIR interpreter"
    assert "--backup-restore" in out


def test_missing_pytorch_uses_the_index_for_their_cuda(installer, capsys):
    """The command has to point at the right wheel index or it is useless."""
    prepare_report(installer, torch_version=None, cuda="130")
    installer._report_missing_pytorch()

    out = capsys.readouterr().out
    assert "download.pytorch.org/whl/cu130" in out, out


def test_missing_pytorch_offers_pytorch_org_as_a_fallback(installer, capsys):
    """When OUR index choice is what failed, repeating it back helps nobody.

    The cu132 torchaudio gap fixed in 0.8.13 is exactly that case.
    """
    prepare_report(installer, torch_version=None)
    installer._report_missing_pytorch()

    assert "pytorch.org/get-started" in capsys.readouterr().out


def test_no_cuda_detected_links_the_toolkit_and_warns_about_cpu_builds(installer, capsys):
    """Without CUDA the command installs a CPU-only build, which is useless here.

    SageAttention requires CUDA. The README badge has linked to the toolkit
    since the first release; the installer never did.
    """
    prepare_report(installer, torch_version=None, cuda="cpu")
    installer._report_missing_pytorch()

    out = capsys.readouterr().out
    assert "CPU-only" in out
    assert "developer.nvidia.com/cuda-downloads" in out, out


def test_cuda_present_does_not_show_the_cpu_warning(installer, capsys):
    """The toolkit note must not fire when CUDA was found -- it would be wrong."""
    prepare_report(installer, torch_version=None, cuda="130")
    out = capsys.readouterr().out
    installer._report_missing_pytorch()

    out = capsys.readouterr().out
    assert "developer.nvidia.com" not in out
    assert "CPU-only" not in out


def test_nothing_reported_when_pytorch_is_present(installer, capsys):
    """A failure with torch installed was a failure somewhere else. Stay quiet."""
    prepare_report(installer, torch_version="2.13.0+cu130")
    installer._report_missing_pytorch()

    assert "no PyTorch" not in capsys.readouterr().out


def test_restore_path_is_not_duplicated(installer, capsys):
    """The CPU->CUDA restore already named exact versions -- do not talk over it.

    That guidance is strictly better than a generic torch>=2.5.0, so this method
    stays silent when it has run.
    """
    prepare_report(installer, torch_version=None)
    installer._restore_on_failure = {"torch": "2.5.1+cpu"}
    installer._report_missing_pytorch()

    assert "no PyTorch" not in capsys.readouterr().out
