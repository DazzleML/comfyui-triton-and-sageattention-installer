"""Gap found while verifying v0.8.13 (Issue #36 rollback-scope fix).

install()'s failure handler is `except Exception as e:` (comfyui_triton_sageattention.py
~line 5113). KeyboardInterrupt is a BaseException, not an Exception subclass, so it is
NOT caught there. A real Ctrl-C during --install propagates past this handler entirely:
_rollback_failed_install() and _restore_removed_packages() never run.

This matters for exactly the property v0.8.13 is about. Two consequences, escalating:

  1. A Ctrl-C after some packages installed leaves them in place with no rollback
     message and no attempt to remove them -- surprising, though not destructive.

  2. A Ctrl-C landing during the CPU->CUDA switch (after the old CPU-only PyTorch has
     already been uninstalled and recorded in _restore_on_failure, but before the CUDA
     reinstall completes) leaves the user with NO PyTorch, NO "Restoring packages..."
     message, and NO printed recovery command. This is silently worse than the
     restore-failure path (_restore_removed_packages' own except branch, tested in
     test_rollback_scope.py::test_restore_failure_tells_the_user_how_to_recover),
     which at least prints the exact command to run. Ctrl-C at the wrong moment
     reproduces the spirit of Issue #36 (installer leaves the user with no PyTorch and
     no way back) through a different trigger (SIGINT instead of a pip failure).

Both tests below encode the CORRECT behavior and currently FAIL against the real code
(xfail, strict=True) -- they are the bug report. If a future change makes install()
also catch KeyboardInterrupt (e.g. `except (Exception, KeyboardInterrupt) as e:`, being
careful that whatever runs afterward does not itself hang or swallow a second Ctrl-C),
these should start passing; strict=True turns that into a visible XPASS so the xfail
marker gets removed rather than silently forgotten.

Reproduced independently against the real (non-mocked) subprocess/pip path during the
v0.8.13 checklist run -- see the checklist's Section 1.1 for the human-driven version
(marked MANUAL there: a genuine Ctrl-C is a physical keypress and cannot be scripted
without either poisoning the parent console's own process group on Windows, or relying
on SIGBREAK semantics that do not exercise the same code path).
"""
import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

from comfyui_triton_sageattention import ComfyUIInstaller, ComponentAction, InstallPlan


@pytest.fixture
def installer():
    inst = object.__new__(ComfyUIInstaller)
    inst.logger = logging.getLogger("test")
    inst.interactive = False
    inst.force = False
    inst.experimental = False
    inst.installed_packages = []
    inst._restore_on_failure = {}
    inst.handler = Mock()
    inst.handler.pip_uninstall = Mock()
    inst.handler.pip_install = Mock()
    inst.handler.run_command = Mock()
    inst.handler.python_path = "python.exe"
    inst._executing = False
    return inst


@pytest.mark.xfail(
    strict=True,
    reason="install() catches `except Exception`, which does not include "
           "KeyboardInterrupt -- a real Ctrl-C skips _rollback_failed_install() "
           "entirely. Found during the v0.8.13 checklist run (gap hunt).",
)
def test_keyboard_interrupt_during_install_still_triggers_rollback(installer):
    """A Ctrl-C mid-install must still roll back whatever this run installed.

    Today it does not: KeyboardInterrupt propagates past install()'s
    `except Exception as e:` uncaught, and _rollback_failed_install() never runs.
    """
    installer.installed_packages = ["torch", "torchvision", "torchaudio"]
    fake_plan = InstallPlan(actions=[], warnings=[], sa_wheel_url=None, cuda_for_wheels="128")

    rollback_called = []
    orig_rollback = ComfyUIInstaller._rollback_failed_install

    def spy_rollback(self):
        rollback_called.append(True)
        return orig_rollback(self)

    with patch.object(ComfyUIInstaller, "plan_installation", return_value=fake_plan), \
         patch.object(ComfyUIInstaller, "install_build_tools", side_effect=KeyboardInterrupt), \
         patch.object(ComfyUIInstaller, "_rollback_failed_install", spy_rollback):
        try:
            installer.install()
        except KeyboardInterrupt:
            pass  # this is the bug: it should have been caught and handled instead

    assert rollback_called, (
        "KeyboardInterrupt during install() must still trigger "
        "_rollback_failed_install(), the same as any other failure"
    )


def test_keyboard_interrupt_during_cpu_to_cuda_switch_still_restores(installer, capsys):
    """The worst-case timing: Ctrl-C lands after the old CPU torch is uninstalled
    and recorded for restore, but before the CUDA reinstall finishes.

    Today, _restore_removed_packages() never runs and nothing is printed -- silence,
    which is worse than a failed restore (that path at least prints the manual
    recovery command; see test_rollback_scope.py::test_restore_failure_tells_the_user_how_to_recover).
    """
    installer.handler.pip_install = Mock(side_effect=KeyboardInterrupt)

    torch_action = ComponentAction(
        component="PyTorch", action="UPGRADE", current_version="2.5.1+cpu",
        target_version=">=2.5.0", reason="CPU-only PyTorch found, CUDA available",
    )
    fake_plan = InstallPlan(actions=[torch_action], warnings=[], sa_wheel_url=None,
                             cuda_for_wheels="130")

    versions = {"torch": "2.5.1+cpu", "torchvision": "0.20.1+cpu", "torchaudio": "2.5.1+cpu"}

    with patch.object(ComfyUIInstaller, "plan_installation", return_value=fake_plan), \
         patch.object(ComfyUIInstaller, "install_build_tools", return_value=None), \
         patch.object(ComfyUIInstaller, "detect_and_setup_cuda", return_value="13.0"), \
         patch.object(ComfyUIInstaller, "upgrade_pip_setuptools", return_value=None), \
         patch.object(ComfyUIInstaller, "_get_installed_version", side_effect=versions.get):
        try:
            installer.install()
        except KeyboardInterrupt:
            pass  # this is the bug

    out = capsys.readouterr().out
    assert "torch==2.5.1+cpu" in out, (
        "a Ctrl-C that lands after the CPU torch is removed must still print the "
        "restore attempt or the manual recovery command -- got no mention of the "
        f"recorded version at all. Full output:\n{out}"
    )
