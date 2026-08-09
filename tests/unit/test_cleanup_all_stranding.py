"""Gap found while live-testing the v0.8.14-in-progress `--cleanup all` split
(tester-unbounded run, 2026-08-09).

`--cleanup` used to remove PyTorch unconditionally; it now only does that when
invoked as `--cleanup all` (comfyui_triton_sageattention.py ~line 2415,
`cleanup_installation(include_pytorch=False)`). That closes the *documented*
"clean slate reinstall" workflow's exposure for the DEFAULT case -- confirmed
live: a bare `--cleanup` no longer touches torch, so `--cleanup` + `--install`
can no longer strand anyone via the cleanup step itself.

`--cleanup all` still removes torch/torchvision/torchaudio on request, and
that is by design -- the user asked. The gap is what happens next. Reproduced
live against a real scratch venv with real CPU torch installed:

    python comfyui_triton_sageattention.py --cleanup all --base-path <scratch> --python <venv>
      (answered "y")   -> torch genuinely uninstalled, "Success!"
    HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 \\
      python comfyui_triton_sageattention.py --install --non-interactive --base-path <scratch> --python <venv>
      -> PyTorch install fails (dead proxy)
      -> "Installation failed: ..."
      -> "Rolling back this run's changes..."
      -> "Nothing to roll back - this run installed no packages."
      -> exit 1

That final message is TRUE (this run installed nothing, so there is nothing
of *this run's* to undo) and is exactly what `_rollback_failed_install()`
(~line 2380) is designed to say -- see test_rollback_scope.py's
`test_nothing_recorded_means_nothing_restored`, which asserts the identical
mechanism in isolation and passes today. The gap only appears once you chain
the two commands the way `--cleanup all` followed by `--install` invites: the
message reads as reassuring ("nothing bad happened") to someone who, one
command ago, deliberately removed their own PyTorch and just watched a
supported, documented recovery step fail. They are left with zero PyTorch and
no on-screen recovery command -- the same *symptom* v0.8.13 fixed for the
CPU->CUDA switch (test_restore_failure_tells_the_user_how_to_recover), just
reached through the front door (`--cleanup all`, a fully separate process
invocation) rather than through an in-run uninstall `_rollback_failed_install`
can see coming.

`_rollback_failed_install()` has no way to know the environment had PyTorch
removed by an earlier, unrelated command -- `self.installed_packages` and
`self._restore_on_failure` are both per-run and both correctly empty here.
Any fix would need to check the *plan* (which does know: the Installation
Plan printed at the top of this same run said `PyTorch [INSTALL] not
installed -> ...`) rather than per-run tracking, at the point a failure is
about to be reported as "nothing to roll back".

This is a design gap, not a code defect -- rollback is doing exactly what its
own docstring promises ("Nothing installed means nothing removed"). The test
below encodes the improvement as an aspirational xfail, matching the project's
convention in test_interrupt_handling.py: a live regression that turns to
XPASS (caught by strict=True) if this is ever addressed.
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
    inst.handler.pip_install = Mock(side_effect=RuntimeError("dead proxy"))
    inst.handler.run_command = Mock()
    inst.handler.python_path = "python.exe"
    inst._executing = False
    return inst


def test_rollback_message_is_accurate_but_silent_on_current_absence(installer, capsys):
    """Locks in today's actual behavior (the live-observed repro).

    Not a bug in isolation -- matches test_rollback_scope.py's
    test_nothing_recorded_means_nothing_restored exactly. Kept here as a
    regression pin so a future change to this message is a deliberate,
    visible diff rather than an accidental one, and so this file stands on
    its own without cross-referencing that one to make sense.
    """
    torch_action = ComponentAction(
        component="PyTorch", action="INSTALL", current_version=None,
        target_version=">=2.5.0", reason="fresh install (or removed by a prior --cleanup all)",
    )
    fake_plan = InstallPlan(actions=[torch_action], warnings=[], sa_wheel_url=None,
                             cuda_for_wheels="130")

    with patch.object(ComfyUIInstaller, "plan_installation", return_value=fake_plan), \
         patch.object(ComfyUIInstaller, "install_build_tools", return_value=None), \
         patch.object(ComfyUIInstaller, "detect_and_setup_cuda", return_value="13.0"), \
         patch.object(ComfyUIInstaller, "upgrade_pip_setuptools", return_value=None):
        result = installer.install()

    assert result is False
    out = capsys.readouterr().out
    assert "Nothing to roll back - this run installed no packages." in out


@pytest.mark.xfail(
    strict=True,
    reason="Gap found live-testing the v0.8.14-in-progress --cleanup all split "
           "(tester-unbounded, 2026-08-09): a failed --install after --cleanup all "
           "reports 'Nothing to roll back' with no mention that PyTorch is "
           "CURRENTLY ABSENT from the environment. The plan already knows -- it "
           "printed 'PyTorch [INSTALL] not installed' moments earlier -- but the "
           "failure path never consults it. Someone who ran the documented "
           "--cleanup all / --install sequence is left with zero PyTorch and no "
           "on-screen recovery command, the same symptom Issue #36 fixed for the "
           "CPU->CUDA switch, reached through a different (fully supported) path.",
)
def test_failure_after_full_cleanup_should_flag_pytorch_is_currently_missing(installer, capsys):
    """Desired future behavior: when this run installed/restored nothing AND the
    plan shows PyTorch was not present at plan time, the failure message should
    say so explicitly -- not just that this run has nothing of its own to undo.
    """
    torch_action = ComponentAction(
        component="PyTorch", action="INSTALL", current_version=None,
        target_version=">=2.5.0", reason="fresh install (or removed by a prior --cleanup all)",
    )
    fake_plan = InstallPlan(actions=[torch_action], warnings=[], sa_wheel_url=None,
                             cuda_for_wheels="130")

    with patch.object(ComfyUIInstaller, "plan_installation", return_value=fake_plan), \
         patch.object(ComfyUIInstaller, "install_build_tools", return_value=None), \
         patch.object(ComfyUIInstaller, "detect_and_setup_cuda", return_value="13.0"), \
         patch.object(ComfyUIInstaller, "upgrade_pip_setuptools", return_value=None):
        installer.install()

    out = capsys.readouterr().out
    assert "no PyTorch" in out or "not installed" in out.split("Nothing to roll back")[-1], (
        "a failure with nothing to roll back, on an environment that currently has "
        "no PyTorch, must say so -- not read as purely reassuring. Full output:\n" + out
    )


def test_cleanup_all_warning_text_does_not_mention_install_can_fail(installer, capsys):
    """What --cleanup all actually tells the user before they consent.

    Confirms the exact warning text live-observed at the confirm prompt, and
    documents that it frames the cost as bandwidth/time ("downloading several
    GB again") rather than as a risk of ending up with nothing. Not asserting
    that framing is wrong -- just pinning it, since it is the one warning the
    project controls before the destructive removal in this scenario. See
    docs/parameters.md "Clean Slate Reinstall" and README.md's cleanup example,
    both still describing pre-split bare-`--cleanup` behavior as of this run --
    reported separately, not fixed here (docs are off-limits for this test pass).
    """
    installer.interactive = True
    with patch("builtins.input", return_value="n"):
        installer.cleanup_installation(include_pytorch=True)

    out = capsys.readouterr().out
    assert "This includes PyTorch" in out
    assert "downloading several GB again" in out
    assert "fail" not in out.lower(), (
        "if this ever starts warning about install failure risk, update the "
        "docstring above accordingly -- this assertion is a tripwire, not a "
        "statement that the current wording is sufficient"
    )
