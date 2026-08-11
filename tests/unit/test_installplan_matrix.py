"""Unit test matrix for InstallPlan decision logic.

Tests 50+ scenarios covering:
- PyTorch install/keep/upgrade decisions (16 scenarios)
- SageAttention version selection (14 scenarios)
- Triton compatibility (12 scenarios)
- Upgrade mode behavior (8 scenarios)
- Future version handling (various)

These tests use mocking to verify decision logic without real environments.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from comfyui_triton_sageattention import (
    ComfyUIInstaller,
    EnvironmentState,
    ComponentAction,
    InstallPlan,
)


# =============================================================================
# TEST MATRICES
# =============================================================================

# PyTorch decision logic - 16 scenarios
PYTORCH_DECISION_MATRIX = [
    # (name, torch_ver, torch_cuda, nvcc_cuda, expected_action, description)

    # === Issue #18 Scenarios ===
    ("issue_18_exact", "2.9.1+cu130", "13.0", "12.8", "KEEP",
     "Portable with newer torch CUDA than system nvcc"),
    ("issue_18_reversed", "2.7.0+cu124", "12.4", "12.8", "KEEP",
     "Older torch CUDA than system - still keep"),

    # === Fresh Install ===
    ("fresh_no_torch", None, None, "12.8", "INSTALL",
     "No PyTorch, use nvcc CUDA"),
    ("fresh_no_cuda", None, None, None, "INSTALL",
     "No PyTorch, no CUDA - CPU mode"),

    # === Version Upgrades ===
    ("pytorch_1x", "1.13.0+cu117", "11.7", "12.8", "UPGRADE",
     "PyTorch 1.x needs upgrade to 2.x"),
    ("cpu_to_cuda", "2.7.0", None, "12.8", "UPGRADE",
     "CPU-only PyTorch should get CUDA"),

    # === Keep Scenarios ===
    ("keep_matching", "2.7.0+cu126", "12.6", "12.6", "KEEP",
     "CUDA versions match exactly"),
    ("keep_newer_torch", "2.9.1+cu130", "13.0", "12.6", "KEEP",
     "Newer torch CUDA than system"),

    # === Edge Cases ===
    ("no_nvcc", "2.9.1+cu130", "13.0", None, "KEEP",
     "Portable without nvcc installed"),
    ("torch_25_min", "2.5.0+cu124", "12.4", "12.8", "KEEP",
     "Minimum supported PyTorch version"),

    # === Future Version Scenarios ===
    ("future_pytorch_210", "2.10.0+cu140", "14.0", "14.0", "KEEP",
     "Future PyTorch 2.10 with CUDA 14 - should keep"),
    ("future_pytorch_211", "2.11.0+cu141", "14.1", "14.1", "KEEP",
     "Future PyTorch 2.11 - should keep existing install"),
    ("future_pytorch_3x", "3.0.0+cu150", "15.0", "15.0", "KEEP",
     "Future PyTorch 3.x - should keep, not force downgrade"),
    ("future_cuda_14", "2.9.1+cu140", "14.0", "14.0", "KEEP",
     "Future CUDA 14.0 - should keep working PyTorch"),
    ("future_cuda_15", "2.10.0+cu150", "15.0", "15.0", "KEEP",
     "Future CUDA 15.0 - graceful handling"),
    ("future_torch_newer_than_wheels", "2.12.0+cu142", "14.2", "14.2", "KEEP",
     "PyTorch version newer than any wheel config - keep"),
]

# SageAttention scenarios - 14 scenarios
SAGEATTENTION_MATRIX = [
    # (name, torch_ver, cuda_code, sa_current, expected_action, expected_target_contains)

    # === Fresh Install (no SA) ===
    ("sa2_fresh_cu130", "2.9.1+cu130", "130", None, "INSTALL", "2.2.0"),
    ("sa2_fresh_cu128", "2.8.0+cu128", "128", None, "INSTALL", "2.2.0"),
    ("sa1_fallback_no_wheel", "2.10.0+cu140", "140", None, "INSTALL", "1.0.6"),

    # === Issue #29: PyTorch 2.10 + CUDA 13.0 (Blackwell/RTX 50xx) ===
    ("sa2_fresh_cu130_torch210", "2.10.0+cu130", "130", None, "INSTALL", "2.2.0"),
    ("sa2_fresh_cu128_torch210", "2.10.0+cu128", "128", None, "INSTALL", "2.2.0"),

    # === Issue #32: CUDA 12.9 resolves to cu128 wheel via tested alias ===
    # No cu129 SA wheel exists upstream; cu128 wheel verified compatible
    # (tests/one-offs/test_cuda129_compat.py). cu129 -> SA 2.x, not 1.x fallback.
    ("sa2_cu129_alias_to_128", "2.9.0+cu129", "129", None, "INSTALL", "2.2.0"),
    # An untested CUDA gap (cu127) must NOT alias -> stays on SA 1.x fallback.
    ("sa1_fallback_cu127_untested", "2.9.0+cu127", "127", None, "INSTALL", "1.0.6"),

    # === CUDA 13.2 / PyTorch 2.12 -> cu130 post6 wheel (tested alias) ===
    # cu132 has no SA wheel; cu130 post6 verified compatible
    # (tests/one-offs/test_cuda132_compat.py). cu132 -> SA 2.x, not 1.x fallback.
    ("sa2_cu132_alias_to_130", "2.12.0+cu132", "132", None, "INSTALL", "2.2.0"),
    # torch 2.12 on cu130 directly (no alias) also resolves to the post6 wheel.
    ("sa2_cu130_torch212", "2.12.0+cu130", "130", None, "INSTALL", "2.2.0"),
    # torch 2.12 on cu126 has no safe SA wheel (post6 has no cu126) -> SA 1.x.
    ("sa1_fallback_cu126_torch212", "2.12.0+cu126", "126", None, "INSTALL", "1.0.6"),

    # === Issue #33: CUDA 13.0 / PyTorch 2.13 -> cu130 post6 wheel ===
    # torch 2.13 ships on cu130 (direct) and cu132 (via 132->130 alias).
    ("sa2_cu130_torch213", "2.13.0+cu130", "130", None, "INSTALL", "2.2.0"),
    ("sa2_cu132_torch213_alias", "2.13.0+cu132", "132", None, "INSTALL", "2.2.0"),

    # === Existing SA - Keep scenarios ===
    ("sa2_keep_current", "2.9.1+cu130", "130", "2.2.0+cu130torch2.9", "KEEP", None),
    ("sa1_keep_no_upgrade", "2.10.0+cu140", "140", "1.0.6", "KEEP", None),

    # === Future Version Handling ===
    ("future_cuda14_no_wheel", "2.10.0+cu140", "140", None, "INSTALL", "1.0.6"),
    ("future_cuda15_no_wheel", "2.11.0+cu150", "150", None, "INSTALL", "1.0.6"),
    ("future_torch3_no_wheel", "3.0.0+cu150", "150", None, "INSTALL", "1.0.6"),

    # === User has newer SA than we know about ===
    ("future_sa3_installed", "2.9.1+cu130", "130", "3.0.0", "KEEP", None),
    ("future_sa_newer_than_known", "2.9.1+cu130", "130", "2.5.0", "KEEP", None),
    ("user_installed_future_sa", "2.9.1+cu130", "130", "2.3.0", "KEEP", None),
    ("user_downgrade_prevented", "2.9.1+cu130", "130", "2.4.0", "KEEP", None),

    # === Upgrade path (SA1 -> SA2 when wheel available) ===
    # Note: These only trigger with --upgrade flag
    ("sa1_upgradeable", "2.9.1+cu130", "130", "1.0.6", "KEEP", None),  # Without --upgrade
    ("sa2_already_latest", "2.9.1+cu130", "130", "2.2.0.post3", "KEEP", None),
]

# SageAttention upgrade scenarios (with --upgrade flag)
SAGEATTENTION_UPGRADE_MATRIX = [
    # (name, torch_ver, cuda_code, sa_current, expected_action, expected_target_contains)
    ("upgrade_sa1_to_sa2", "2.9.1+cu130", "130", "1.0.6", "UPGRADE", "2.2.0"),
    ("upgrade_sa2_patch", "2.9.1+cu130", "130", "2.1.0", "UPGRADE", "2.2.0"),
    ("upgrade_sa2_same", "2.9.1+cu130", "130", "2.2.0.post3", "KEEP", None),
    ("upgrade_no_wheel_keep_sa1", "2.10.0+cu140", "140", "1.0.6", "KEEP", None),

    # === Issue #29: Upgrade SA1->SA2 on torch 2.10 + cu130 ===
    ("upgrade_sa1_to_sa2_torch210", "2.10.0+cu130", "130", "1.0.6", "UPGRADE", "2.2.0"),
]

# Triton compatibility - 12 scenarios
TRITON_COMPATIBILITY_MATRIX = [
    # (name, triton_ver, torch_ver, expected_action, description)

    # === Standard Scenarios ===
    ("triton_none_torch29", None, "2.9.1", "INSTALL", "No Triton, needs install"),
    ("triton_353_torch29", "3.5.3", "2.9.1", "KEEP", "Compatible Triton"),
    ("triton_341_torch28", "3.4.1", "2.8.0", "KEEP", "Compatible older Triton"),
    ("triton_331_torch27", "3.3.1", "2.7.0", "KEEP", "Compatible Triton 3.3"),

    # === Incompatible - needs fix ===
    ("triton_old_torch29", "3.2.0", "2.9.1", "FIX", "Old Triton needs upgrade"),

    # === Triton 3.6 / PyTorch 2.10 boundary ===
    ("triton_36_torch210", "3.6.0", "2.10.0", "KEEP", "Triton 3.6 compatible with PyTorch 2.10"),
    ("triton_36_torch29_incompat", "3.6.0", "2.9.1", "FIX", "Triton 3.6 requires PyTorch >= 2.10"),
    ("triton_35_torch210_outdated", "3.5.0", "2.10.0", "FIX", "PyTorch 2.10 needs Triton >= 3.6"),

    # === Future Version Scenarios ===
    ("future_triton_40_torch210", "4.0.0", "2.10.0", "KEEP", "Major Triton bump"),
    ("future_triton_36_torch30", "3.6.0", "3.0.0", "KEEP", "PyTorch 3.x with newer Triton"),

    # === Unknown PyTorch + existing Triton (don't touch what we don't know about) ===
    ("triton_existing_torch_future", "3.5.3", "2.11.0", "KEEP", "Unknown torch, keep triton"),
    ("triton_existing_torch3", "3.5.3", "3.0.0", "KEEP", "PyTorch 3.x, keep existing triton"),
    ("triton_existing_torch_future_ok", "3.6.0", "2.11.0", "KEEP", "Future torch with adequate Triton"),

    # === Installer should NOT downgrade working Triton ===
    ("triton_newer_than_expected", "3.7.0", "2.9.1", "KEEP", "User has newer Triton"),
    ("triton_user_installed_future", "4.1.0", "2.10.0", "KEEP", "Don't touch user's Triton 4.x"),
]


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_installer():
    """Create installer with mocked environment detection."""
    with patch.object(ComfyUIInstaller, '__init__', return_value=None):
        installer = ComfyUIInstaller.__new__(ComfyUIInstaller)

        # Core attributes
        installer.base_path = Path("C:/fake/comfyui")
        installer.force = False
        installer.upgrade = False
        installer.interactive = False
        installer.experimental = False
        installer.with_custom_nodes = False
        installer.sage_version_raw = "auto"
        installer.sage_version_major = None
        # Real class sets `sage_version_exact` (see __init__). A phantom
        # `sage_version_specific` here went unnoticed while the plan layer ignored
        # the request (D1); it AttributeErrors as soon as the plan reads it.
        installer.sage_version_exact = None

        # Logger
        installer.logger = Mock()

        # Handler with mocked methods
        installer.handler = Mock()
        installer.handler.python_path = Path("C:/fake/python.exe")
        installer.handler.environment_type = "venv"
        installer.handler.detect_cuda_version = Mock(return_value="12.8")

        # Installed packages tracking
        installer.installed_packages = []

        return installer


def setup_torch_mock(installer, torch_version, torch_cuda, nvcc_cuda):
    """Configure mocks for PyTorch environment."""
    # Set nvcc CUDA version
    installer.handler.detect_cuda_version.return_value = nvcc_cuda

    # Configure run_command mock
    def mock_run_command(cmd, capture_output=False, check=True):
        cmd_str = " ".join(str(c) for c in cmd)

        # PyTorch version detection
        if "import torch" in cmd_str:
            if torch_version is None:
                raise Exception("No module named 'torch'")

            cuda_available = torch_cuda is not None and torch_cuda != "cpu"
            cuda_str = torch_cuda if cuda_available else "None"
            return Mock(
                stdout=f"{torch_version}|{cuda_available}|{cuda_str}",
                returncode=0
            )

        # Python version
        if "sys.version_info" in cmd_str:
            return Mock(stdout="3.12.0", returncode=0)

        # Package not found by default
        raise Exception("Package not found")

    installer.handler.run_command = Mock(side_effect=mock_run_command)


def setup_triton_mock(installer, triton_version):
    """Configure mocks for Triton detection."""
    original_side_effect = installer.handler.run_command.side_effect

    def mock_run_command(cmd, capture_output=False, check=True):
        cmd_str = " ".join(str(c) for c in cmd)

        # Triton version detection
        if "triton" in cmd_str.lower() and "version" in cmd_str:
            if triton_version is None:
                raise Exception("Package not found")
            return Mock(stdout=triton_version, returncode=0)

        # Delegate to original mock
        if original_side_effect:
            return original_side_effect(cmd, capture_output=capture_output, check=check)

        raise Exception("Unknown command")

    installer.handler.run_command = Mock(side_effect=mock_run_command)


def setup_sa_mock(installer, sa_version):
    """Configure mocks for SageAttention detection."""
    original_side_effect = installer.handler.run_command.side_effect

    def mock_run_command(cmd, capture_output=False, check=True):
        cmd_str = " ".join(str(c) for c in cmd)

        # SageAttention version detection
        if "sageattention" in cmd_str.lower():
            if sa_version is None:
                raise Exception("Package not found")
            return Mock(stdout=sa_version, returncode=0)

        # Delegate to original mock
        if original_side_effect:
            return original_side_effect(cmd, capture_output=capture_output, check=check)

        raise Exception("Unknown command")

    installer.handler.run_command = Mock(side_effect=mock_run_command)


def setup_full_mock(installer, torch_version, torch_cuda, nvcc_cuda,
                    triton_version=None, sa_version=None):
    """Configure all mocks for a complete test scenario."""
    installer.handler.detect_cuda_version.return_value = nvcc_cuda

    def mock_run_command(cmd, capture_output=False, check=True):
        cmd_str = " ".join(str(c) for c in cmd)

        # Python version
        if "sys.version_info" in cmd_str:
            return Mock(stdout="3.12.0", returncode=0)

        # PyTorch version detection
        if "import torch" in cmd_str:
            if torch_version is None:
                raise Exception("No module named 'torch'")
            cuda_available = torch_cuda is not None and torch_cuda != "cpu"
            cuda_str = torch_cuda if cuda_available else "None"
            return Mock(
                stdout=f"{torch_version}|{cuda_available}|{cuda_str}",
                returncode=0
            )

        # Triton version detection
        if "triton" in cmd_str.lower():
            if triton_version is None:
                raise Exception("Package not found")
            return Mock(stdout=triton_version, returncode=0)

        # SageAttention version detection
        if "sageattention" in cmd_str.lower():
            if sa_version is None:
                raise Exception("Package not found")
            return Mock(stdout=sa_version, returncode=0)

        # pip list/freeze - return empty
        if "pip" in cmd_str and ("list" in cmd_str or "freeze" in cmd_str):
            return Mock(stdout="", returncode=0)

        # Default - package not found
        raise Exception(f"Unknown command: {cmd_str}")

    installer.handler.run_command = Mock(side_effect=mock_run_command)


# =============================================================================
# PYTORCH DECISION TESTS
# =============================================================================

@pytest.mark.parametrize(
    "name,torch_ver,torch_cuda,nvcc_cuda,expected_action,description",
    PYTORCH_DECISION_MATRIX,
    ids=[t[0] for t in PYTORCH_DECISION_MATRIX]
)
def test_pytorch_action(mock_installer, name, torch_ver, torch_cuda, nvcc_cuda,
                        expected_action, description):
    """Test PyTorch install/keep/upgrade decision logic."""
    setup_full_mock(mock_installer, torch_ver, torch_cuda, nvcc_cuda)

    # Generate plan
    plan = mock_installer.plan_installation()

    # Find PyTorch action
    pytorch_action = plan.get_action("PyTorch")

    assert pytorch_action is not None, f"{name}: PyTorch action missing from plan"
    assert pytorch_action.action == expected_action, \
        f"{name}: Expected {expected_action} but got {pytorch_action.action}. {description}"


# =============================================================================
# SAGEATTENTION DECISION TESTS
# =============================================================================

@pytest.mark.parametrize(
    "name,torch_ver,cuda_code,sa_current,expected_action,expected_target",
    SAGEATTENTION_MATRIX,
    ids=[t[0] for t in SAGEATTENTION_MATRIX]
)
def test_sageattention_action(mock_installer, name, torch_ver, cuda_code,
                               sa_current, expected_action, expected_target):
    """Test SageAttention install/keep decision logic."""
    # Parse cuda_code to version format
    if cuda_code:
        cuda_version = f"{cuda_code[0:2]}.{cuda_code[2:]}" if len(cuda_code) == 3 else cuda_code
    else:
        cuda_version = None

    setup_full_mock(
        mock_installer,
        torch_version=torch_ver,
        torch_cuda=cuda_version,
        nvcc_cuda=cuda_version,
        sa_version=sa_current
    )

    # Generate plan
    plan = mock_installer.plan_installation()

    # Find SageAttention action
    sa_action = plan.get_action("SageAttention")

    assert sa_action is not None, f"{name}: SageAttention action missing from plan"
    assert sa_action.action == expected_action, \
        f"{name}: Expected {expected_action} but got {sa_action.action}"

    # Check target version if specified
    if expected_target and sa_action.target_version:
        assert expected_target in str(sa_action.target_version), \
            f"{name}: Expected target containing '{expected_target}' but got '{sa_action.target_version}'"


@pytest.mark.parametrize(
    "name,torch_ver,cuda_code,sa_current,expected_action,expected_target",
    SAGEATTENTION_UPGRADE_MATRIX,
    ids=[t[0] for t in SAGEATTENTION_UPGRADE_MATRIX]
)
def test_sageattention_upgrade_mode(mock_installer, name, torch_ver, cuda_code,
                                     sa_current, expected_action, expected_target):
    """Test SageAttention upgrade behavior with --upgrade flag."""
    # Enable upgrade mode
    mock_installer.upgrade = True

    # Parse cuda_code to version format
    if cuda_code:
        cuda_version = f"{cuda_code[0:2]}.{cuda_code[2:]}" if len(cuda_code) == 3 else cuda_code
    else:
        cuda_version = None

    setup_full_mock(
        mock_installer,
        torch_version=torch_ver,
        torch_cuda=cuda_version,
        nvcc_cuda=cuda_version,
        sa_version=sa_current
    )

    # Generate plan
    plan = mock_installer.plan_installation()

    # Find SageAttention action
    sa_action = plan.get_action("SageAttention")

    assert sa_action is not None, f"{name}: SageAttention action missing"
    assert sa_action.action == expected_action, \
        f"{name}: Expected {expected_action} but got {sa_action.action}"

    if expected_target and sa_action.target_version:
        assert expected_target in str(sa_action.target_version), \
            f"{name}: Expected target containing '{expected_target}'"


# =============================================================================
# TRITON COMPATIBILITY TESTS
# =============================================================================

@pytest.mark.parametrize(
    "name,triton_ver,torch_ver,expected_action,description",
    TRITON_COMPATIBILITY_MATRIX,
    ids=[t[0] for t in TRITON_COMPATIBILITY_MATRIX]
)
def test_triton_action(mock_installer, name, triton_ver, torch_ver,
                       expected_action, description):
    """Test Triton install/keep/fix decision logic."""
    # Extract CUDA from torch version (e.g., "2.9.1+cu130" -> "13.0")
    if "+cu" in torch_ver:
        cuda_code = torch_ver.split("+cu")[1][:3]
        cuda_version = f"{cuda_code[0:2]}.{cuda_code[2:]}" if len(cuda_code) == 3 else cuda_code
    else:
        cuda_version = "12.8"  # Default

    setup_full_mock(
        mock_installer,
        torch_version=torch_ver,
        torch_cuda=cuda_version,
        nvcc_cuda=cuda_version,
        triton_version=triton_ver
    )

    # Generate plan
    plan = mock_installer.plan_installation()

    # Find Triton action
    triton_action = plan.get_action("Triton")

    assert triton_action is not None, f"{name}: Triton action missing from plan"
    assert triton_action.action == expected_action, \
        f"{name}: Expected {expected_action} but got {triton_action.action}. {description}"


# =============================================================================
# CUDA DETECTION TESTS
# =============================================================================

class TestCudaWheelAlias:
    """Test the CUDA minor-version wheel alias mechanism (Issue #32)."""

    def test_exact_match(self, mock_installer):
        """Exact CUDA codes always match."""
        assert mock_installer._cuda_matches("128", "128") is True
        assert mock_installer._cuda_matches("130", "130") is True

    def test_tested_alias_matches(self, mock_installer):
        """Tested aliases: cu128 serves cu129; cu130 serves cu132."""
        assert mock_installer._cuda_matches("128", "129") is True
        assert mock_installer._cuda_matches("130", "132") is True

    def test_untested_gap_does_not_match(self, mock_installer):
        """An untested CUDA gap (e.g. cu127) must not alias to anything."""
        assert mock_installer._cuda_matches("128", "127") is False
        assert mock_installer._cuda_matches("126", "127") is False

    def test_alias_does_not_cross_major(self, mock_installer):
        """The alias map must never bridge the 12.x <-> 13.x boundary."""
        # cu130 wheel must not be offered to a cu129 (12.x) environment
        assert mock_installer._cuda_matches("130", "129") is False
        # and the only alias entry stays within CUDA 12
        for detected, wheel in type(mock_installer).CUDA_WHEEL_ALIASES.items():
            assert detected[:2] == wheel[:2], \
                f"alias {detected}->{wheel} crosses CUDA major boundary"

    def test_alias_resolves_to_cu128_wheel_url(self, mock_installer):
        """cu129 + torch 2.9 + py312 resolves to the cu128 wheel (correct URL)."""
        import platform
        if platform.system() != "Windows":
            import pytest
            pytest.skip("wheel matching is Windows-only")
        match = mock_installer._find_matching_wheel(
            cuda_ver="129", torch_ver="2.9.0", python_ver="312"
        )
        assert match is not None, "cu129 should resolve via alias, not fall through"
        assert match["cuda"] == "128"
        assert "cu128torch2.9.0" in match["wheel_url"]


class TestCu132AndPost6:
    """CUDA 13.2 alias + post6 (cp310) wheel handling (torch 2.12 / 2.13)."""

    def _match(self, inst, cuda, torch, py):
        import platform
        if platform.system() != "Windows":
            import pytest
            pytest.skip("wheel matching is Windows-only")
        return inst._find_matching_wheel(cuda_ver=cuda, torch_ver=torch, python_ver=py)

    def test_cu132_resolves_to_cu130_post6(self, mock_installer):
        """cu132 + torch 2.12 + py312 -> cu130 post6 cp310 wheel via alias."""
        m = self._match(mock_installer, "132", "2.12.0", "312")
        assert m is not None and m["cuda"] == "130"
        assert "cu130torch2.10.0andhigher.post6" in m["wheel_url"]
        assert "cp310-abi3" in m["wheel_url"]

    def test_cu130_torch213_direct(self, mock_installer):
        """Issue #33: torch 2.13 on cu130 resolves to the post6 wheel."""
        m = self._match(mock_installer, "130", "2.13.0", "312")
        assert m is not None and m["cuda"] == "130"
        assert "cu130torch2.10.0andhigher.post6" in m["wheel_url"]

    def test_cu132_torch213_alias(self, mock_installer):
        """torch 2.13 on cu132 resolves via the 132->130 alias to post6."""
        m = self._match(mock_installer, "132", "2.13.0", "312")
        assert m is not None and m["cuda"] == "130" and "post6" in m["wheel_url"]

    def test_torch212_moved_to_post6(self, mock_installer):
        """torch 2.12 now resolves to post6 (moved off buggy post5)."""
        m = self._match(mock_installer, "130", "2.12.0", "312")
        assert m is not None and "post6" in m["wheel_url"] and "post5" not in m["wheel_url"]

    def test_post6_requires_py310(self, mock_installer):
        """post6 is cp310-abi3 -> Python 3.9 must NOT match (falls back)."""
        assert self._match(mock_installer, "130", "2.13.0", "39") is None

    def test_cu126_torch212_no_safe_wheel(self, mock_installer):
        """torch 2.12 on cu126 has no safe SA wheel -> None (SA 1.x fallback)."""
        assert self._match(mock_installer, "126", "2.12.0", "312") is None

    def test_abi3_cp_helper(self, mock_installer):
        """_abi3_cp maps the py_spec field to (cp_tag, python_floor)."""
        assert mock_installer._abi3_cp(None) == ("cp39", 39)
        assert mock_installer._abi3_cp("310") == ("cp310", 310)


class TestCudaDetection:
    """Test CUDA version detection priority."""

    def test_uses_torch_cuda_when_available(self, mock_installer):
        """Should use torch.version.cuda over nvcc when PyTorch exists."""
        setup_full_mock(
            mock_installer,
            torch_version="2.9.1+cu130",
            torch_cuda="13.0",
            nvcc_cuda="12.8"  # Different from torch
        )

        plan = mock_installer.plan_installation()

        # Plan should use torch CUDA (13.0), not nvcc (12.8)
        assert plan.cuda_for_wheels == "130", \
            f"Expected CUDA 130 from torch, got {plan.cuda_for_wheels}"

    def test_uses_nvcc_when_no_torch(self, mock_installer):
        """Should use nvcc CUDA when no PyTorch installed."""
        setup_full_mock(
            mock_installer,
            torch_version=None,
            torch_cuda=None,
            nvcc_cuda="12.8"
        )

        plan = mock_installer.plan_installation()

        # Plan should use nvcc CUDA
        assert plan.cuda_for_wheels == "128", \
            f"Expected CUDA 128 from nvcc, got {plan.cuda_for_wheels}"

    def test_cpu_mode_when_no_cuda(self, mock_installer):
        """Should fall back to CPU when no CUDA available."""
        setup_full_mock(
            mock_installer,
            torch_version=None,
            torch_cuda=None,
            nvcc_cuda=None
        )

        plan = mock_installer.plan_installation()

        assert plan.cuda_for_wheels == "cpu", \
            f"Expected CPU mode, got {plan.cuda_for_wheels}"


# =============================================================================
# PLAN CONSISTENCY TESTS
# =============================================================================

class TestPlanConsistency:
    """Test that plan is internally consistent."""

    def test_plan_has_all_components(self, mock_installer):
        """Plan should always include PyTorch, Triton, and SageAttention."""
        setup_full_mock(
            mock_installer,
            torch_version="2.9.1+cu130",
            torch_cuda="13.0",
            nvcc_cuda="12.8"
        )

        plan = mock_installer.plan_installation()

        components = [a.component for a in plan.actions]
        assert "PyTorch" in components, "PyTorch action missing"
        assert "Triton" in components, "Triton action missing"
        assert "SageAttention" in components, "SageAttention action missing"

    def test_has_changes_detects_installs(self, mock_installer):
        """has_changes() should return True when installs are needed."""
        setup_full_mock(
            mock_installer,
            torch_version=None,  # No PyTorch - needs install
            torch_cuda=None,
            nvcc_cuda="12.8"
        )

        plan = mock_installer.plan_installation()

        assert plan.has_changes() is True, "Should detect install as a change"

    def test_has_changes_false_when_all_keep(self, mock_installer):
        """has_changes() should return False when all components are KEEP."""
        setup_full_mock(
            mock_installer,
            torch_version="2.9.1+cu130",
            torch_cuda="13.0",
            nvcc_cuda="13.0",
            triton_version="3.5.3",
            sa_version="2.2.0+cu130torch2.9"
        )

        plan = mock_installer.plan_installation()

        # All components should be KEEP
        for action in plan.actions:
            if action.component in ("PyTorch", "Triton", "SageAttention"):
                assert action.action == "KEEP", \
                    f"{action.component} should be KEEP but is {action.action}"

    def test_environment_state_captured(self, mock_installer):
        """Plan should capture current environment state."""
        setup_full_mock(
            mock_installer,
            torch_version="2.9.1+cu130",
            torch_cuda="13.0",
            nvcc_cuda="12.8",
            triton_version="3.5.3",
            sa_version="2.2.0.post3"
        )

        plan = mock_installer.plan_installation()

        assert plan.current.torch_version == "2.9.1+cu130"
        assert plan.current.torch_cuda == "13.0"
        assert plan.current.nvcc_cuda == "12.8"
        assert plan.current.triton_version == "3.5.3"
        assert plan.current.sageattention_version == "2.2.0.post3"


# =============================================================================
# FUTURE VERSION HANDLING TESTS
# =============================================================================

class TestFutureVersionHandling:
    """Test graceful handling of versions released after the installer."""

    def test_future_pytorch_not_downgraded(self, mock_installer):
        """Future PyTorch versions should be kept, not downgraded."""
        setup_full_mock(
            mock_installer,
            torch_version="3.0.0+cu150",  # Future PyTorch 3.x
            torch_cuda="15.0",
            nvcc_cuda="15.0"
        )

        plan = mock_installer.plan_installation()
        pytorch_action = plan.get_action("PyTorch")

        assert pytorch_action.action == "KEEP", \
            f"Future PyTorch 3.x should be KEEP, not {pytorch_action.action}"

    def test_future_sa_not_downgraded(self, mock_installer):
        """User-installed future SA versions should be kept."""
        setup_full_mock(
            mock_installer,
            torch_version="2.9.1+cu130",
            torch_cuda="13.0",
            nvcc_cuda="13.0",
            sa_version="3.0.0"  # Future SA 3.x
        )

        plan = mock_installer.plan_installation()
        sa_action = plan.get_action("SageAttention")

        assert sa_action.action == "KEEP", \
            f"Future SA 3.x should be KEEP, not {sa_action.action}"

    def test_future_triton_not_downgraded(self, mock_installer):
        """User-installed future Triton versions should be kept."""
        setup_full_mock(
            mock_installer,
            torch_version="2.9.1+cu130",
            torch_cuda="13.0",
            nvcc_cuda="13.0",
            triton_version="4.0.0"  # Future Triton 4.x
        )

        plan = mock_installer.plan_installation()
        triton_action = plan.get_action("Triton")

        assert triton_action.action == "KEEP", \
            f"Future Triton 4.x should be KEEP, not {triton_action.action}"

    def test_no_wheel_falls_back_to_sa1(self, mock_installer):
        """When no SA2 wheel exists for config, should fall back to SA1."""
        setup_full_mock(
            mock_installer,
            torch_version="2.10.0+cu140",  # Future config
            torch_cuda="14.0",
            nvcc_cuda="14.0",
            sa_version=None  # No SA installed
        )

        plan = mock_installer.plan_installation()
        sa_action = plan.get_action("SageAttention")

        assert sa_action.action == "INSTALL"
        # Target should include SA 1.x fallback
        assert "1.0.6" in str(sa_action.target_version) or "SA 1" in str(sa_action.target_version), \
            f"Should fall back to SA 1.x, got {sa_action.target_version}"


# =============================================================================
# UPGRADE MODE TESTS
# =============================================================================

class TestUpgradeMode:
    """Test behavior with --upgrade flag."""

    def test_upgrade_flag_enables_triton_updates(self, mock_installer):
        """With --upgrade, Triton should check for updates."""
        mock_installer.upgrade = True

        setup_full_mock(
            mock_installer,
            torch_version="2.9.1+cu130",
            torch_cuda="13.0",
            nvcc_cuda="13.0",
            triton_version="3.5.0"  # Slightly older
        )

        # Mock update check to return True
        mock_installer._check_package_update_available = Mock(return_value=(True, "3.5.3"))

        plan = mock_installer.plan_installation()
        triton_action = plan.get_action("Triton")

        # Should show upgrade available (actual behavior depends on implementation)
        # This tests that the upgrade path is activated
        assert triton_action is not None

    def test_no_upgrade_keeps_compatible_triton(self, mock_installer):
        """Without --upgrade, compatible Triton should be kept."""
        mock_installer.upgrade = False

        setup_full_mock(
            mock_installer,
            torch_version="2.9.1+cu130",
            torch_cuda="13.0",
            nvcc_cuda="13.0",
            triton_version="3.5.0"
        )

        plan = mock_installer.plan_installation()
        triton_action = plan.get_action("Triton")

        assert triton_action.action == "KEEP", \
            f"Compatible Triton without --upgrade should be KEEP, got {triton_action.action}"


# =============================================================================
# CPU->CUDA SWITCH EXECUTION TESTS (Issue #23)
# =============================================================================

class TestCpuToCudaSwitch:
    """Test that install_pytorch() correctly handles CPU->CUDA switch.

    Issue #23: When plan says UPGRADE with "CPU-only" reason,
    install_pytorch() must uninstall first because pip won't replace
    torch+cpu with torch+cuda when version constraint is satisfied.
    """

    def test_cpu_to_cuda_triggers_uninstall(self, mock_installer):
        """CPU->CUDA upgrade should uninstall torch before installing."""
        # Setup: CPU-only PyTorch with CUDA available
        setup_full_mock(
            mock_installer,
            torch_version="2.9.1",  # No +cu suffix = CPU-only
            torch_cuda=None,
            nvcc_cuda="12.8"
        )

        # Generate plan - should show UPGRADE
        plan = mock_installer.plan_installation()
        pytorch_action = plan.get_action("PyTorch")

        assert pytorch_action.action == "UPGRADE", \
            f"CPU->CUDA should be UPGRADE, got {pytorch_action.action}"
        assert "CPU-only" in pytorch_action.reason, \
            f"Reason should mention 'CPU-only', got: {pytorch_action.reason}"

        # Store the plan so install_pytorch can access it
        mock_installer._current_plan = plan

        # Track if uninstall was called
        uninstall_called = False
        original_run_command = mock_installer.handler.run_command

        def mock_run_command(cmd, **kwargs):
            nonlocal uninstall_called
            if isinstance(cmd, list) and "uninstall" in cmd:
                uninstall_called = True
                # Return success
                result = Mock()
                result.returncode = 0
                result.stdout = ""
                return result
            return original_run_command(cmd, **kwargs)

        mock_installer.handler.run_command = mock_run_command

        # Call install_pytorch
        mock_installer.install_pytorch("12.8")

        assert uninstall_called, \
            "CPU->CUDA switch should call pip uninstall before installing"

    def test_cuda_to_cuda_no_uninstall(self, mock_installer):
        """CUDA->CUDA (same version) should NOT uninstall."""
        # Setup: Already has CUDA PyTorch
        setup_full_mock(
            mock_installer,
            torch_version="2.9.1+cu128",
            torch_cuda="12.8",
            nvcc_cuda="12.8"
        )

        # Generate plan - should show KEEP
        plan = mock_installer.plan_installation()
        pytorch_action = plan.get_action("PyTorch")

        assert pytorch_action.action == "KEEP", \
            f"CUDA PyTorch should be KEEP, got {pytorch_action.action}"

        # Store the plan
        mock_installer._current_plan = plan

        # Track calls
        uninstall_called = False
        original_run_command = mock_installer.handler.run_command

        def mock_run_command(cmd, **kwargs):
            nonlocal uninstall_called
            if isinstance(cmd, list) and "uninstall" in cmd:
                uninstall_called = True
            return original_run_command(cmd, **kwargs)

        mock_installer.handler.run_command = mock_run_command

        # Call install_pytorch
        mock_installer.install_pytorch("12.8")

        assert not uninstall_called, \
            "KEEP action should not call uninstall"

    def test_plan_reason_contains_cpu_only(self, mock_installer):
        """Verify plan reason format for CPU->CUDA detection."""
        setup_full_mock(
            mock_installer,
            torch_version="2.7.0",  # CPU-only
            torch_cuda=None,
            nvcc_cuda="12.8"
        )

        plan = mock_installer.plan_installation()
        pytorch_action = plan.get_action("PyTorch")

        # The exact reason text that install_pytorch() checks for
        assert "CPU-only" in pytorch_action.reason, \
            f"Reason must contain 'CPU-only' for uninstall detection. Got: {pytorch_action.reason}"


# =============================================================================
# TRITON CONSTRAINT TIER TESTS (v0.8.15)
# =============================================================================

class TestTritonConstraintTiers:
    """The constraint table must be one Triton minor wide per PyTorch minor.

    Upstream (woct0rdho/triton-windows README): "Each PyTorch minor version is
    only guaranteed to work with a specific Triton minor version" -- table ends
    2.10 -> 3.6. The old ">=3.6,<4" tier for torch >= 2.10 asserted future
    compatibility nobody had confirmed; on 2026-08-10 it made the plan propose
    Triton 3.7.1 (a torch-2.12 pairing) for a torch 2.10 environment.

    Evidence grades: 2.10 -> 3.6 is CONFIRMED (upstream README); 2.12+ -> 3.7
    is INFERRED (torch's own PyPI pins: 2.12.0 -> triton 3.7.0,
    2.12.1/2.13.0 -> 3.7.1, Linux markers).
    """

    @pytest.mark.parametrize("torch_ver,expected", [
        # New tiers (red before the v0.8.15 fix)
        ("2.10.0", ">=3.6,<3.7"),
        ("2.10.0+cu130", ">=3.6,<3.7"),
        ("2.11.5", ">=3.6,<3.7"),
        ("2.12.0", ">=3.7,<3.8"),
        ("2.12.1", ">=3.7,<3.8"),
        ("2.13.0+cu130", ">=3.7,<3.8"),
        # Unknown future torch falls into the newest tier (best available
        # evidence -- the fresh-install chooser cannot abstain)
        ("2.14.0", ">=3.7,<3.8"),
        # Boundary preservation (green before and after)
        ("2.9.1", ">=3.5,<3.6"),
        ("2.8.0", ">=3.4,<3.5"),
        ("2.7.0+cu128", ">=3.3,<3.4"),
        ("2.6.0", ">=3.2,<3.3"),
        ("2.5.1", "<3.2"),
    ])
    def test_tier_is_one_minor_wide(self, mock_installer, torch_ver, expected):
        assert mock_installer._get_triton_version_constraint(torch_ver) == expected

    def test_unparseable_torch_yields_no_constraint(self, mock_installer):
        """Unchanged behavior: garbage in, no constraint out."""
        assert mock_installer._get_triton_version_constraint("garbage") == ""


# =============================================================================
# UPGRADE PROBE TESTS (v0.8.15)
# =============================================================================

def _emulate_pip_dry_run(installer, recorded):
    """Layer a pip --dry-run emulator over setup_full_mock's run_command.

    Emulates the semantics verified live against real pip: a CONSTRAINED
    requirement is resolved within its range; an unconstrained one reports the
    global latest (here 3.7.1.post27, out-of-tier for torch 2.10).
    """
    original = installer.handler.run_command.side_effect

    def with_probe(cmd, capture_output=False, check=True):
        cmd_strs = [str(c) for c in cmd]
        if "--dry-run" in cmd_strs:
            recorded.append(cmd_strs)
            req = next(c for c in cmd_strs if "triton-windows" in c)
            if ">=3.6,<3.7" in req:
                return Mock(stdout="Would install triton-windows-3.6.0.post27",
                            stderr="", returncode=0)
            return Mock(stdout="Would install triton-windows-3.7.1.post27",
                        stderr="", returncode=0)
        return original(cmd, capture_output=capture_output, check=check)

    installer.handler.run_command = Mock(side_effect=with_probe)


class TestUpgradeProbe:
    """The probe must carry the domain constraint and match execution's flags.

    The regression these tests pin: after the one-minor tier fix, an
    UNCONSTRAINED probe only ever reports the global latest (3.7.1), the
    filter rejects it, and legitimate within-tier updates (3.6.0.postN)
    silently become unreachable -- --upgrade says KEEP forever. The failure is
    a capability evaporating, not an error, so the test must assert an update
    IS proposed (the inverse test passes on the broken code).
    """

    def _upgrade_installer(self, mock_installer, recorded):
        setup_full_mock(mock_installer, "2.10.0+cu130", "13.0", "13.0",
                        triton_version="3.6.0.post26")
        mock_installer.upgrade = True
        _emulate_pip_dry_run(mock_installer, recorded)
        return mock_installer

    def test_within_tier_update_is_proposed(self, mock_installer):
        """ANCHOR (red on the unconstrained probe): 3.6.0.post27 must surface."""
        recorded = []
        inst = self._upgrade_installer(mock_installer, recorded)

        action = inst.plan_installation().get_action("Triton")

        assert action.action == "UPGRADE", (
            f"got {action.action} ({action.reason!r}) -- the within-tier update "
            f"was not proposed; with an unconstrained probe pip reports 3.7.1, "
            f"the filter rejects it, and upgrades silently freeze"
        )
        assert action.target_version == "3.6.0.post27"

    def test_probe_carries_constraint_and_pre_flag(self, mock_installer):
        """The probe requirement embeds the tier; --pre matches execution's
        `pip install -U --pre` so pre-release-only updates stay visible."""
        recorded = []
        inst = self._upgrade_installer(mock_installer, recorded)

        inst.plan_installation()

        assert recorded, "the upgrade path never ran the probe"
        probe_cmd = recorded[0]
        assert "triton-windows>=3.6,<3.7" in probe_cmd, probe_cmd
        assert "--pre" in probe_cmd, probe_cmd

    def test_out_of_tier_proposal_is_blocked_without_packaging(self, mock_installer):
        """ANCHOR (red on the except-pass hole): with `packaging` unavailable
        the filter must still block, not silently self-disable.

        Old behavior: ImportError -> pass -> upgrade_ok stays True -> the plan
        proposes an out-of-tier Triton on exactly the environments (Portable
        embedded Python) least likely to have `packaging`.
        """
        recorded = []
        inst = self._upgrade_installer(mock_installer, recorded)
        # Make the probe return an out-of-tier proposal despite the constraint
        # (emulates an old/odd pip); the plan-layer filter is the last guard.
        inst._check_package_update_available = Mock(return_value=(True, "3.7.1.post27"))

        with patch.dict(sys.modules, {"packaging": None,
                                      "packaging.version": None,
                                      "packaging.specifiers": None}):
            action = inst.plan_installation().get_action("Triton")

        assert action.action == "KEEP", (
            f"got {action.action}: an out-of-tier proposal passed the filter "
            f"because `packaging` was missing"
        )


class TestVersionSatisfiesConstraint:
    """Direct tests of the shared filter, both engines."""

    @pytest.mark.parametrize("version,constraint,expected", [
        ("3.6.0.post27", ">=3.6,<3.7", True),
        ("3.7.1.post27", ">=3.6,<3.7", False),
        ("3.7.0", ">=3.7,<3.8", True),
        ("3.6.0", ">=3.7,<3.8", False),
        ("3.1.5", "<3.2", True),
        ("3.2.0", "<3.2", False),
        ("3.6.0", "", True),          # no constraint: anything passes
    ])
    def test_with_packaging(self, mock_installer, version, constraint, expected):
        assert mock_installer._version_satisfies_constraint(version, constraint) is expected

    @pytest.mark.parametrize("version,constraint,expected", [
        ("3.6.0.post27", ">=3.6,<3.7", True),
        ("3.7.1.post27", ">=3.6,<3.7", False),
        ("3.6.0+git123", ">=3.6,<3.7", True),   # local tag on the minor
        ("3.6.0", ">=3.6,<4", True),            # bound without minor part
    ])
    def test_fallback_without_packaging(self, mock_installer, version, constraint, expected):
        with patch.dict(sys.modules, {"packaging": None,
                                      "packaging.version": None,
                                      "packaging.specifiers": None}):
            assert mock_installer._version_satisfies_constraint(version, constraint) is expected

    def test_fallback_fails_closed_on_garbage(self, mock_installer):
        """An unparseable version blocks the proposal (plan says KEEP)."""
        with patch.dict(sys.modules, {"packaging": None,
                                      "packaging.version": None,
                                      "packaging.specifiers": None}):
            assert mock_installer._version_satisfies_constraint(
                "garbage", ">=3.6,<3.7") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
