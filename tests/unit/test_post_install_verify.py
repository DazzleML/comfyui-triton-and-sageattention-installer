"""Post-install verification (C-lite-3) tests -- v0.8.15.

The verifier compares the plan's claims to the environment after execution.
It is report-only and must never fail an install. Design and the six-action
check matrix: the plan/execute divergence consultation, final assessment
`2026-08-10__19-37-52__DISCUSS_Rnd4_FINAL_ASSESSMENT_(plan-exec-divergence-enforcement).md`
(project-private maintainer notes; filename kept for traceability).

Historical shapes these checks would have caught:
- Issue #18 downgrade-on-KEEP        -> KEEP check
- 2026-08-10 setuptools 83->84       -> build-tools check
- v0.8.14 phantom Triton UPGRADE     -> planned-change-occurred check

Known blind spot, deliberately not asserted anywhere: same-version churn
(uninstall + reinstall at the same version) ends at post == pre and passes.
"""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root
sys.path.insert(0, str(Path(__file__).parent))                # sibling test modules

from comfyui_triton_sageattention import (
    ComponentAction,
    EnvironmentState,
    InstallPlan,
)


def _state(**kw):
    s = EnvironmentState()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _verify(installer, plan, post_state, sage_attention_failed=False):
    """Run the verifier with a pinned post-state; return (count, output)."""
    installer._detect_current_environment = Mock(return_value=post_state)
    return installer._verify_install_outcome(plan, sage_attention_failed)


def _plan(actions, current=None, build_tools=False):
    return InstallPlan(
        current=current or EnvironmentState(),
        actions=actions,
        build_tools_upgrade_needed=build_tools,
    )


# ---------------------------------------------------------------------------
# KEEP rows
# ---------------------------------------------------------------------------

def test_keep_unchanged_passes(mock_installer, capsys):
    plan = _plan(
        [ComponentAction(component="Triton", action="KEEP",
                         current_version="3.6.0.post26", reason="Compatible")],
        current=_state(triton_version="3.6.0.post26"),
    )
    count = _verify(mock_installer, plan, _state(triton_version="3.6.0.post26"))

    assert count == 0
    out = capsys.readouterr().out
    assert "[OK]" in out and "[DEVIATION]" not in out


def test_keep_changed_is_a_deviation_worded_as_observation(mock_installer, capsys):
    """A KEEP component that moved is reported WITHOUT blaming the executor --
    pip can legitimately drag a KEEP component while resolving another step."""
    plan = _plan(
        [ComponentAction(component="Triton", action="KEEP",
                         current_version="3.6.0.post26", reason="Compatible")],
        current=_state(triton_version="3.6.0.post26"),
    )
    count = _verify(mock_installer, plan, _state(triton_version="3.7.1.post27"))

    assert count == 1
    out = capsys.readouterr().out
    assert "changed outside the plan" in out


# ---------------------------------------------------------------------------
# Planned-change-occurred (the v0.8.14 phantom's check)
# ---------------------------------------------------------------------------

def test_upgrade_that_did_not_happen_is_a_deviation(mock_installer, capsys):
    """THE phantom: plan said UPGRADE, environment identical afterwards.

    Live incident 2026-08-10: plan claimed
    'Triton [UPGRADE] 3.6.0.post26 -> 3.7.1.post27', execution skipped, and
    nothing in the output said so.
    """
    plan = _plan(
        [ComponentAction(component="Triton", action="UPGRADE",
                         current_version="3.6.0.post26",
                         target_version="3.7.1.post27", reason="Update available")],
        current=_state(triton_version="3.6.0.post26"),
    )
    count = _verify(mock_installer, plan, _state(triton_version="3.6.0.post26"))

    assert count == 1
    out = capsys.readouterr().out
    assert "the planned change did not happen" in out


def test_upgrade_that_happened_passes(mock_installer):
    plan = _plan(
        [ComponentAction(component="Triton", action="UPGRADE",
                         current_version="3.6.0.post26",
                         target_version="3.6.0.post27", reason="Update available")],
        current=_state(triton_version="3.6.0.post26"),
    )
    assert _verify(mock_installer, plan, _state(triton_version="3.6.0.post27")) == 0


def test_downgrade_that_did_not_happen_is_a_deviation(mock_installer):
    """DOWNGRADE is destructive by definition -- silence would be worst here.
    Target equality is deliberately not asserted (installed SA wheel versions
    carry local tags that never string-equal the plan's decorated target)."""
    plan = _plan(
        [ComponentAction(component="SageAttention", action="DOWNGRADE",
                         current_version="2.2.0+cu130torch2.9.0andhigher.post4",
                         target_version="1.0.6 - exact",
                         reason="--sage-version 1 requested")],
        current=_state(sageattention_version="2.2.0+cu130torch2.9.0andhigher.post4"),
    )
    post_same = _state(sageattention_version="2.2.0+cu130torch2.9.0andhigher.post4")
    assert _verify(mock_installer, plan, post_same) == 1


# ---------------------------------------------------------------------------
# INSTALL / REINSTALL presence checks
# ---------------------------------------------------------------------------

def test_install_that_left_nothing_is_a_deviation(mock_installer):
    plan = _plan(
        [ComponentAction(component="SageAttention", action="INSTALL",
                         target_version="2.2.0.post6", reason="Not installed")],
    )
    assert _verify(mock_installer, plan, _state(sageattention_version=None)) == 1


def test_reinstall_same_version_back_passes(mock_installer):
    """REINSTALL asserts presence, not change -- same version back is the
    expected outcome of --force."""
    plan = _plan(
        [ComponentAction(component="Triton", action="REINSTALL",
                         current_version="3.6.0.post26",
                         target_version="triton-windows", reason="Force mode enabled")],
        current=_state(triton_version="3.6.0.post26"),
    )
    assert _verify(mock_installer, plan, _state(triton_version="3.6.0.post26")) == 0


def test_reinstall_that_removed_the_package_is_a_deviation(mock_installer):
    """The REINSTALL failure worth catching: uninstall succeeded, install
    failed, package gone."""
    plan = _plan(
        [ComponentAction(component="Triton", action="REINSTALL",
                         current_version="3.6.0.post26",
                         target_version="triton-windows", reason="Force mode enabled")],
        current=_state(triton_version="3.6.0.post26"),
    )
    assert _verify(mock_installer, plan, _state(triton_version=None)) == 1


# ---------------------------------------------------------------------------
# Build tools (the setuptools 83->84 collateral)
# ---------------------------------------------------------------------------

def test_unauthorized_setuptools_change_is_a_deviation(mock_installer, capsys):
    """2026-08-10: a phantom UPGRADE authorized a real setuptools replacement
    the preview never mentioned. With build_tools_upgrade_needed False, any
    pip/setuptools movement is a deviation."""
    plan = _plan(
        [ComponentAction(component="Triton", action="KEEP",
                         current_version="3.6.0.post26", reason="Compatible")],
        current=_state(triton_version="3.6.0.post26", setuptools_version="83.0.0",
                       pip_version="26.2.1"),
        build_tools=False,
    )
    post = _state(triton_version="3.6.0.post26", setuptools_version="84.0.0",
                  pip_version="26.2.1")
    count = _verify(mock_installer, plan, post)

    assert count == 1
    assert "setuptools 83.0.0 -> 84.0.0" in capsys.readouterr().out


def test_authorized_build_tools_change_is_not_checked(mock_installer):
    """When the plan authorized touching build tools, movement is expected."""
    plan = _plan(
        [ComponentAction(component="Triton", action="KEEP",
                         current_version="3.6.0.post26", reason="Compatible")],
        current=_state(triton_version="3.6.0.post26", setuptools_version="83.0.0"),
        build_tools=True,
    )
    post = _state(triton_version="3.6.0.post26", setuptools_version="84.0.0")
    assert _verify(mock_installer, plan, post) == 0


def test_missing_build_tools_baseline_is_skipped(mock_installer):
    """A failed pre-probe (None) means no baseline to hold -- not a deviation."""
    plan = _plan(
        [ComponentAction(component="Triton", action="KEEP",
                         current_version="3.6.0.post26", reason="Compatible")],
        current=_state(triton_version="3.6.0.post26", setuptools_version=None),
        build_tools=False,
    )
    post = _state(triton_version="3.6.0.post26", setuptools_version="84.0.0")
    assert _verify(mock_installer, plan, post) == 0


# ---------------------------------------------------------------------------
# Scope fences
# ---------------------------------------------------------------------------

def test_custom_node_rows_are_skipped(mock_installer, capsys):
    """Rows without an EnvironmentState observable are outside the verifier's
    scope -- skipped silently, neither passed nor failed."""
    plan = _plan(
        [ComponentAction(component="ComfyUI-VideoHelperSuite", action="INSTALL",
                         target_version="latest", reason="Custom node"),
         ComponentAction(component="Triton", action="KEEP",
                         current_version="3.6.0.post26", reason="Compatible")],
        current=_state(triton_version="3.6.0.post26"),
    )
    count = _verify(mock_installer, plan, _state(triton_version="3.6.0.post26"))

    assert count == 0
    assert "VideoHelperSuite" not in capsys.readouterr().out


def test_verification_continues_past_unverifiable_rows(mock_installer, capsys):
    """Killing test for mutant M4 (scope fence removed): an unverifiable row
    must be SKIPPED, not allowed to abort the check loop.

    Without the `if field is None: continue` fence, a custom-node row reaches
    getattr(obj, None) -> TypeError -> the outer fence eats it -> every later
    row's finding is silently dropped. The deviation AFTER the custom-node row
    is what proves the loop kept going.
    """
    plan = _plan(
        [ComponentAction(component="ComfyUI-VideoHelperSuite", action="INSTALL",
                         target_version="latest", reason="Custom node"),
         ComponentAction(component="Triton", action="UPGRADE",
                         current_version="3.6.0.post26",
                         target_version="3.6.0.post27", reason="Update available")],
        current=_state(triton_version="3.6.0.post26"),
    )
    # The Triton UPGRADE did not happen -- a deviation the verifier must still
    # find on the far side of the skipped row.
    count = _verify(mock_installer, plan, _state(triton_version="3.6.0.post26"))

    assert count == 1, (
        "the deviation after the unverifiable row was lost -- the check loop "
        "did not continue past it"
    )
    assert "the planned change did not happen" in capsys.readouterr().out


def test_sa_failure_path_is_fenced(mock_installer):
    """When the SA failure already warned (and the run continued with consent),
    an absent SA is the explained outcome, not a new deviation."""
    plan = _plan(
        [ComponentAction(component="SageAttention", action="INSTALL",
                         target_version="2.2.0.post6", reason="Not installed")],
    )
    count = _verify(mock_installer, plan, _state(sageattention_version=None),
                    sage_attention_failed=True)
    assert count == 0


def test_sa_keep_is_still_checked_when_failure_flag_set(mock_installer):
    """The fence covers only the failed non-KEEP path; a KEEP row that changed
    is still a finding."""
    plan = _plan(
        [ComponentAction(component="SageAttention", action="KEEP",
                         current_version="2.2.0.post6", reason="Compatible")],
        current=_state(sageattention_version="2.2.0.post6"),
    )
    count = _verify(mock_installer, plan, _state(sageattention_version=None),
                    sage_attention_failed=True)
    assert count == 1


# ---------------------------------------------------------------------------
# The fence: verification can never fail an install
# ---------------------------------------------------------------------------

def test_probe_failure_cannot_raise(mock_installer):
    """A verifier that takes down a completed install would be worse than no
    verifier. Probe explosion -> 0 deviations, warning logged, no exception."""
    plan = _plan(
        [ComponentAction(component="Triton", action="KEEP",
                         current_version="3.6.0.post26", reason="Compatible")],
        current=_state(triton_version="3.6.0.post26"),
    )
    mock_installer._detect_current_environment = Mock(
        side_effect=RuntimeError("probe exploded"))

    count = mock_installer._verify_install_outcome(plan)

    assert count == 0
    assert mock_installer.logger.warning.called


def test_install_actually_calls_the_verifier(mock_installer, capsys):
    """WIRING: a verifier that exists but is never called is the failure mode
    that matters most. install() must invoke it on the success path."""
    from unittest.mock import patch
    from comfyui_triton_sageattention import ComfyUIInstaller

    inst = mock_installer
    inst._restore_on_failure = {}
    inst._executing = False
    plan = _plan(
        [ComponentAction(component="Triton", action="KEEP",
                         current_version="3.6.0.post26", reason="Compatible")],
        current=_state(triton_version="3.6.0.post26"),
    )

    called = {"n": 0}

    def fake_verify(p, sage_failed=False):
        called["n"] += 1
        return 0

    with patch.object(ComfyUIInstaller, "plan_installation", return_value=plan), \
         patch.object(ComfyUIInstaller, "install_build_tools", return_value=None), \
         patch.object(ComfyUIInstaller, "detect_and_setup_cuda", return_value="13.0"), \
         patch.object(ComfyUIInstaller, "upgrade_pip_setuptools", return_value=None), \
         patch.object(ComfyUIInstaller, "install_pytorch", return_value=None), \
         patch.object(ComfyUIInstaller, "install_triton", return_value=None), \
         patch.object(ComfyUIInstaller, "setup_python_dev_files", return_value=None), \
         patch.object(ComfyUIInstaller, "clone_and_install_repositories", return_value=None), \
         patch.object(ComfyUIInstaller, "create_run_script", return_value=None), \
         patch.object(ComfyUIInstaller, "_verify_install_outcome", side_effect=fake_verify):
        result = inst.install()

    assert result is True
    assert called["n"] == 1, "install() completed without running post-install verification"


def test_deviations_change_the_final_status_line(mock_installer, capsys):
    """When the verifier reports deviations, install() must not print an
    unqualified "Success!" -- the status line names the deviation count.
    (The zero-deviation wiring test cannot see this branch.)"""
    from unittest.mock import patch
    from comfyui_triton_sageattention import ComfyUIInstaller

    inst = mock_installer
    inst._restore_on_failure = {}
    inst._executing = False
    plan = _plan(
        [ComponentAction(component="Triton", action="UPGRADE",
                         current_version="3.6.0.post26",
                         target_version="3.6.0.post27", reason="Update available")],
        current=_state(triton_version="3.6.0.post26"),
    )

    with patch.object(ComfyUIInstaller, "plan_installation", return_value=plan), \
         patch.object(ComfyUIInstaller, "install_build_tools", return_value=None), \
         patch.object(ComfyUIInstaller, "detect_and_setup_cuda", return_value="13.0"), \
         patch.object(ComfyUIInstaller, "upgrade_pip_setuptools", return_value=None), \
         patch.object(ComfyUIInstaller, "install_pytorch", return_value=None), \
         patch.object(ComfyUIInstaller, "install_triton", return_value=None), \
         patch.object(ComfyUIInstaller, "setup_python_dev_files", return_value=None), \
         patch.object(ComfyUIInstaller, "clone_and_install_repositories", return_value=None), \
         patch.object(ComfyUIInstaller, "create_run_script", return_value=None), \
         patch.object(ComfyUIInstaller, "_verify_install_outcome", return_value=2):
        result = inst.install()

    out = capsys.readouterr().out
    assert result is True, "deviations are report-only; the install still succeeds"
    assert "Completed -- 2 verification deviation(s)" in out
    assert "\nSuccess!" not in out


def test_output_is_ascii_only(mock_installer, capsys):
    """Codepage rule: the block must survive cp437/cp1252 consoles -- the bug
    that started this whole release line was a codepage crash."""
    plan = _plan(
        [ComponentAction(component="Triton", action="UPGRADE",
                         current_version="3.6.0.post26",
                         target_version="3.6.0.post27", reason="Update available")],
        current=_state(triton_version="3.6.0.post26"),
    )
    _verify(mock_installer, plan, _state(triton_version="3.6.0.post26"))

    out = capsys.readouterr().out
    out.encode("ascii")  # raises UnicodeEncodeError on any non-ASCII character


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
