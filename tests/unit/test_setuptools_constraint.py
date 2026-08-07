"""Unit tests for setuptools constraint discovery (Issue #34).

The installer used to run `pip install --upgrade pip setuptools` unconditionally,
which pushes setuptools past upper bounds declared by installed packages -- most
notably PyTorch 2.12.x, which requires setuptools<82. pip then reports:

    ERROR: pip's dependency resolver does not currently take into account all the
    packages that are installed. ...
    torch 2.12.1+cu130 requires setuptools<82, but you have setuptools 83.0.0

pip exits 0 (so the install continues), but the environment is left violating a
constraint its own packages declare. These tests cover the parsing half of the fix;
the live end-to-end check lives in tests/one-offs/test_setuptools_constraint_live.py
"""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from comfyui_triton_sageattention import ComfyUIInstaller


# (name, raw_requirements, expected_spec)
PARSE_MATRIX = [
    # === The Issue #34 case ===
    ("torch_212_upper_bound", ["setuptools<82"], "setuptools<82"),

    # === No upper bound (torch 2.13 declares only a floor) ===
    ("torch_213_lower_bound", ["setuptools>=77.0.3"], "setuptools>=77.0.3"),

    # === Nothing constrains setuptools -> upgrade freely ===
    ("no_constraints", [], "setuptools"),
    ("empty_input", None, "setuptools"),
    ("bare_requirement", ["setuptools"], "setuptools"),
    ("unrelated_package", ["numpy>=1.0", "filelock"], "setuptools"),

    # === Multiple packages constrain it -> combine (pip accepts comma-joined) ===
    ("two_constraints", ["setuptools<82", "setuptools>=77.0.3"],
     "setuptools<82,>=77.0.3"),
    ("duplicate_constraints", ["setuptools<82", "setuptools<82"], "setuptools<82"),

    # === Extras-conditional requirements only apply if the extra is installed ===
    ("extras_only_ignored", ['setuptools; extra == "dev"'], "setuptools"),
    ("extras_with_spec_ignored", ['setuptools>=70; extra == "build"'], "setuptools"),

    # === Environment markers (not extras) are real constraints -> keep ===
    ("env_marker_kept", ['setuptools>=77.0.3; python_version < "3.12"'],
     "setuptools>=77.0.3"),

    # === Formatting robustness ===
    ("case_insensitive", ["SetupTools<82"], "setuptools<82"),
    ("bracketed_extras_stripped", ["setuptools[core]>=70"], "setuptools>=70"),
    ("whitespace_tolerant", ["  setuptools  <82  "], "setuptools<82"),
]


@pytest.mark.parametrize(
    "name,raw_reqs,expected",
    PARSE_MATRIX,
    ids=[c[0] for c in PARSE_MATRIX],
)
def test_parse_setuptools_constraint(name, raw_reqs, expected):
    """Requirement strings from installed packages combine into one pip spec."""
    result = ComfyUIInstaller._parse_setuptools_constraint(raw_reqs)
    assert result == expected, f"{name}: expected {expected!r}, got {result!r}"


class TestUpgradePipSetuptools:
    """The upgrade path honors (or safely skips) discovered constraints."""

    def _installer(self, raw_reqs):
        inst = ComfyUIInstaller.__new__(ComfyUIInstaller)
        inst.logger = Mock()
        inst.handler = Mock()
        inst._get_installed_setuptools_requirements = Mock(return_value=raw_reqs)
        return inst

    def test_constrained_upgrade_passes_bound_to_pip(self):
        """A torch<82 environment must not be upgraded past that bound."""
        inst = self._installer(["setuptools<82"])
        inst.upgrade_pip_setuptools()
        inst.handler.pip_install.assert_called_once_with(
            ["pip", "setuptools<82"], ["--upgrade"]
        )

    def test_unconstrained_upgrade_is_unbounded(self):
        """With nothing constraining setuptools, behavior is the original upgrade."""
        inst = self._installer([])
        inst.upgrade_pip_setuptools()
        inst.handler.pip_install.assert_called_once_with(
            ["pip", "setuptools"], ["--upgrade"]
        )

    def test_discovery_failure_upgrades_pip_only(self):
        """If we cannot determine constraints, don't touch setuptools at all.

        Upgrading blind is what broke environments in the first place; pip alone
        is safe, and leaving setuptools as-is cannot violate a constraint.
        """
        inst = self._installer(None)
        inst.upgrade_pip_setuptools()
        inst.handler.pip_install.assert_called_once_with(["pip"], ["--upgrade"])
        assert inst.logger.warning.called

    def test_never_upgrades_setuptools_past_a_declared_bound(self):
        """Regression guard for Issue #34: no unbounded 'setuptools' when bounded."""
        inst = self._installer(["setuptools<82"])
        inst.upgrade_pip_setuptools()
        packages = inst.handler.pip_install.call_args[0][0]
        assert "setuptools" not in packages, "bare (unbounded) setuptools was requested"
        assert "setuptools<82" in packages
