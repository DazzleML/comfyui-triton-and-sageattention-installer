"""Shared pytest fixtures for unit tests."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def mock_installer():
    """Create installer with mocked environment detection.

    Returns a ComfyUIInstaller instance with all external dependencies mocked,
    allowing us to test decision logic without real subprocess calls.
    """
    # Import here to avoid import errors before path is set
    from comfyui_triton_sageattention import ComfyUIInstaller

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
        installer.sage_version_specific = None

        # Logger
        installer.logger = Mock()
        installer.logger.debug = Mock()
        installer.logger.info = Mock()
        installer.logger.warning = Mock()
        installer.logger.error = Mock()

        # Handler with mocked methods
        installer.handler = Mock()
        installer.handler.python_path = Path("C:/fake/python.exe")
        installer.handler.environment_type = "venv"
        installer.handler.detect_cuda_version = Mock(return_value="12.8")
        installer.handler.run_command = Mock()

        # Installed packages tracking
        installer.installed_packages = []

        return installer


@pytest.fixture
def mock_installer_upgrade(mock_installer):
    """Installer with --upgrade mode enabled."""
    mock_installer.upgrade = True
    return mock_installer


@pytest.fixture
def mock_installer_portable(mock_installer):
    """Installer configured for portable environment."""
    mock_installer.handler.environment_type = "portable"
    return mock_installer


def setup_mock_torch(installer, torch_version, torch_cuda, cuda_available=True):
    """Helper to configure torch mocks for a test scenario.

    Args:
        installer: Mock installer instance
        torch_version: PyTorch version string (e.g., "2.9.1+cu130") or None
        torch_cuda: torch.version.cuda value (e.g., "13.0") or None/"cpu"
        cuda_available: Whether torch.cuda.is_available() returns True
    """
    if torch_version is None:
        # No PyTorch installed - run_command should raise exception
        installer.handler.run_command.side_effect = Exception("No torch installed")
    else:
        # PyTorch installed
        cuda_str = torch_cuda if torch_cuda and torch_cuda != "cpu" else "None"
        cuda_avail = "True" if cuda_available and torch_cuda and torch_cuda != "cpu" else "False"
        output = f"{torch_version}|{cuda_avail}|{cuda_str}"

        installer.handler.run_command.return_value = Mock(
            stdout=output,
            returncode=0
        )
        installer.handler.run_command.side_effect = None


def setup_mock_packages(installer, triton_version=None, sa_version=None):
    """Helper to configure package version mocks.

    Args:
        installer: Mock installer instance
        triton_version: Triton version string or None
        sa_version: SageAttention version string or None
    """
    def mock_get_version(cmd, **kwargs):
        """Mock run_command for package version detection."""
        cmd_str = str(cmd)

        # Triton version check
        if "triton-windows" in cmd_str or "'triton'" in cmd_str:
            if triton_version:
                return Mock(stdout=triton_version, returncode=0)
            else:
                raise Exception("Package not found")

        # SageAttention version check
        if "sageattention" in cmd_str:
            if sa_version:
                return Mock(stdout=sa_version, returncode=0)
            else:
                raise Exception("Package not found")

        # Default - return empty
        return Mock(stdout="", returncode=0)

    # Only set side_effect if not already set for torch
    if installer.handler.run_command.side_effect is None:
        installer.handler.run_command.side_effect = mock_get_version
