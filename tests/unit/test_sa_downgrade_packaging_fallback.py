"""Regression test for a defect found while running the v0.8.12 checklist
(tests/checklists/v0.8.12__Epic__plan-execute-integrity.md, Section 6.2).

Bug: `_sa_is_downgrade()` (comfyui_triton_sageattention.py ~line 3594) compares
SageAttention versions via `packaging.version.Version`, and falls back to
`_parse_sageattention_major_version()` (major-number-only comparison) when
`packaging` is not importable. That fallback is too coarse: "2.2.0" and "2.1.1"
share major version 2, so the fallback's `cur > tgt` is `2 > 2` = False, and a
real downgrade is misclassified as "not a downgrade".

This is NOT a hypothetical: `packaging` is not declared in requirements.txt
(the project treats it as optional everywhere else it's imported, always
inside a try/except), and this machine's system `py -3.13` genuinely has no
`packaging` installed while `py -3.12` does. Reproduced live via:

    py -3.12 ... --sage-version 2.1.1 --python <venv with SA 2.2.0.post6>
        -> DOWNGRADE (correct)
    py -3.13 ... --sage-version 2.1.1 --python <SAME venv, same target>
        -> KEEP, "Already at target version" (wrong: install silently keeps
           2.2.0.post6, ignoring the explicit request for 2.1.1)

Both the dryrun preview AND the real (non-dryrun) install agree on the wrong
answer -- `clone_and_install_repositories()` correctly consults the plan
(no D1/D3-style divergence), but the plan itself is wrong. The bug is a
fresh discovery, not one of the release's tracked D1-D10 defects, but it is
in the same family as D10 ("read the wrong interpreter"): `_sa_is_downgrade`
runs in whichever Python launched the installer script, not the target venv's
Python, exactly like the wheel-matching bug D10 fixed for `_target_python_tag`.

These tests are EXPECTED TO FAIL until the fallback is fixed (e.g. by parsing
release-segment tuples for comparison instead of bare major version, or by
vendoring a minimal version comparator that doesn't need `packaging` at all).
They exist to pin down and guard the fix, not to assert current (buggy)
behavior.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import builtins

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from comfyui_triton_sageattention import ComfyUIInstaller, parse_sage_version


@pytest.fixture
def mock_installer():
    with patch.object(ComfyUIInstaller, '__init__', return_value=None):
        installer = ComfyUIInstaller.__new__(ComfyUIInstaller)
        installer.base_path = Path("C:/fake/comfyui")
        installer.force = False
        installer.upgrade = False
        installer.interactive = False
        installer.experimental = False
        installer.with_custom_nodes = False
        installer.logger = Mock()
        installer.handler = Mock()
        installer.handler.python_path = Path("C:/fake/python.exe")
        installer.handler.environment_type = "venv"
        installer.installed_packages = []
        return installer


def _block_packaging_import(monkeypatch):
    """Simulate `packaging` being absent from the launching interpreter,
    regardless of whether it's actually installed on the machine running
    the test suite."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "packaging" or name.startswith("packaging."):
            raise ImportError(f"No module named {name!r} (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


class TestSaIsDowngradeFallback:
    """Direct tests of `_sa_is_downgrade()` in isolation."""

    def test_patch_level_downgrade_detected_with_packaging(self, mock_installer):
        """Sanity check: with `packaging` available, 2.2.0 -> 2.1.1 IS a downgrade."""
        result = mock_installer._sa_is_downgrade(
            "2.2.0+cu130torch2.10.0andhigher.post6", "2.1.1"
        )
        assert result is True

    def test_patch_level_downgrade_detected_without_packaging(self, mock_installer, monkeypatch):
        """The bug: with `packaging` unavailable, the same comparison silently
        flips to "not a downgrade" because the fallback only compares major
        version numbers (2 vs 2), losing the 2.2.0 -> 2.1.1 patch-level drop.

        EXPECTED TO FAIL until the fallback compares more than the major version.
        """
        _block_packaging_import(monkeypatch)
        result = mock_installer._sa_is_downgrade(
            "2.2.0+cu130torch2.10.0andhigher.post6", "2.1.1"
        )
        assert result is True, (
            "_sa_is_downgrade fell back to major-version-only comparison and "
            "missed that 2.1.1 < 2.2.0 within the same major version"
        )

    def test_real_downgrade_across_major_versions_still_works_without_packaging(
        self, mock_installer, monkeypatch
    ):
        """The coarse fallback happens to work for major-version-crossing
        downgrades (2.x -> 1.x) since that's what it was written to check --
        confirming the fallback isn't *entirely* broken, just too coarse for
        same-major-version requests."""
        _block_packaging_import(monkeypatch)
        result = mock_installer._sa_is_downgrade(
            "2.2.0+cu130torch2.10.0andhigher.post6", "1.0.6"
        )
        assert result is True


class TestPlanLevelExactVersionRequest:
    """Integration-level: does `_plan_sageattention_action` honor an explicit
    exact --sage-version request that differs only within the same major
    version, when the launching interpreter lacks `packaging`?"""

    def test_exact_patch_downgrade_request_honored_without_packaging(
        self, mock_installer, monkeypatch
    ):
        """Checklist Section 6.2 reproduction: requesting --sage-version 2.1.1
        against an environment with 2.2.0.post6 installed must plan a
        DOWNGRADE, not silently KEEP -- regardless of whether the launching
        interpreter has `packaging` installed. A KEEP here means the tool
        prints "Already at target version" and both dryrun and the real
        install ignore the user's explicit request.

        EXPECTED TO FAIL until _sa_is_downgrade (or its caller) stops
        depending on the launcher's own `packaging` availability for a
        decision that should only depend on the two version strings.
        """
        _block_packaging_import(monkeypatch)

        mock_installer.sage_version_raw = "2.1.1"
        mock_installer.sage_version_major, mock_installer.sage_version_exact = (
            parse_sage_version("2.1.1")
        )

        from comfyui_triton_sageattention import EnvironmentState
        state = EnvironmentState(
            torch_version="2.13.0+cu130",
            torch_cuda="13.0",
            torch_cuda_available=True,
            nvcc_cuda="13.0",
            triton_version="3.7.1.post27",
            sageattention_version="2.2.0+cu130torch2.10.0andhigher.post6",
            python_version="3.12.0",
            environment_type="venv",
        )

        with patch.object(ComfyUIInstaller, "check_compatibility") as mock_compat:
            mock_compat.return_value = {
                "compatible": True,
                "match": {"sage_version": "2.2.0.post6", "wheel_url": "https://example.invalid/fake.whl"},
            }
            action = mock_installer._plan_sageattention_action(
                state, cuda_for_wheels="130", torch_version="2.13.0+cu130", torch_pending=False
            )

        assert action.action == "DOWNGRADE", (
            f"Expected DOWNGRADE for explicit --sage-version 2.1.1 request against "
            f"installed 2.2.0.post6, got {action.action!r} (reason: {action.reason!r}). "
            f"This is the checklist v0.8.12 Section 6.2 defect: the launcher lacking "
            f"`packaging` silently drops a same-major-version downgrade request."
        )
