"""Unit tests for InstallationTarget and discovery functions."""
import pytest
import tempfile
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from comfyui_triton_sageattention import (
    PythonEnvironment,
    InstallationTarget,
    is_comfyui_directory,
    discover_comfyui_installations,
    select_installation_interactive,
)


class TestPythonEnvironment:
    """Tests for PythonEnvironment dataclass."""

    def test_from_venv_windows(self):
        """Test creating environment from Windows venv path."""
        venv_path = Path("C:/fake/comfyui/.venv")
        env = PythonEnvironment.from_venv(venv_path, "Windows")

        assert env.python_path == Path("C:/fake/comfyui/.venv/Scripts/python.exe")
        assert env.venv_path == venv_path
        assert env.environment_type == "venv"

    def test_from_venv_linux(self):
        """Test creating environment from Linux venv path."""
        venv_path = Path("/home/user/comfyui/venv")
        env = PythonEnvironment.from_venv(venv_path, "Linux")

        assert env.python_path == Path("/home/user/comfyui/venv/bin/python")
        assert env.venv_path == venv_path
        assert env.environment_type == "venv"

    def test_from_portable(self):
        """Test creating environment from portable distribution."""
        base_path = Path("C:/ComfyUI_windows_portable")
        env = PythonEnvironment.from_portable(base_path)

        assert env.python_path == Path("C:/ComfyUI_windows_portable/python_embeded/python.exe")
        assert env.venv_path == Path("C:/ComfyUI_windows_portable/python_embeded")
        assert env.environment_type == "portable"

    def test_system(self):
        """Test creating system Python environment."""
        env = PythonEnvironment.system()

        assert env.python_path == Path(sys.executable)
        assert env.venv_path is None
        assert env.environment_type == "system"

    def test_exists_true(self, temp_base):
        """Test exists() returns True when python executable exists."""
        # Create fake python executable
        venv = temp_base / "venv" / "Scripts"
        venv.mkdir(parents=True)
        python_exe = venv / "python.exe"
        python_exe.write_text("fake")

        env = PythonEnvironment.from_venv(temp_base / "venv", "Windows")
        assert env.exists() is True

    def test_exists_false(self, temp_base):
        """Test exists() returns False when python executable doesn't exist."""
        env = PythonEnvironment.from_venv(temp_base / "nonexistent", "Windows")
        assert env.exists() is False


class TestInstallationTarget:
    """Tests for InstallationTarget dataclass."""

    def test_is_valid_comfyui_with_main_py(self, temp_base):
        """Test is_valid_comfyui returns True when main.py exists."""
        (temp_base / "main.py").write_text("# ComfyUI main")

        target = InstallationTarget(base_path=temp_base)
        assert target.is_valid_comfyui() is True

    def test_is_valid_comfyui_with_subdir_main_py(self, temp_base):
        """Test is_valid_comfyui returns True when ComfyUI/main.py exists."""
        comfyui_dir = temp_base / "ComfyUI"
        comfyui_dir.mkdir()
        (comfyui_dir / "main.py").write_text("# ComfyUI main")

        target = InstallationTarget(base_path=temp_base)
        assert target.is_valid_comfyui() is True

    def test_is_valid_comfyui_desktop_structure(self, temp_base):
        """Test is_valid_comfyui returns True for ComfyUI Desktop data directory."""
        (temp_base / "custom_nodes").mkdir()
        (temp_base / "models").mkdir()

        target = InstallationTarget(base_path=temp_base)
        assert target.is_valid_comfyui() is True

    def test_is_valid_comfyui_false(self, temp_base):
        """Test is_valid_comfyui returns False when no ComfyUI indicators exist."""
        target = InstallationTarget(base_path=temp_base)
        assert target.is_valid_comfyui() is False

    def test_has_python_environment_portable(self, temp_base):
        """Test has_python_environment detects portable distribution."""
        embeded = temp_base / "python_embeded"
        embeded.mkdir()
        (embeded / "python.exe").write_text("fake")

        target = InstallationTarget(base_path=temp_base)
        assert target.has_python_environment() is True

    def test_has_python_environment_dot_venv(self, temp_base):
        """Test has_python_environment detects .venv."""
        (temp_base / ".venv").mkdir()

        target = InstallationTarget(base_path=temp_base)
        assert target.has_python_environment() is True

    def test_has_python_environment_venv(self, temp_base):
        """Test has_python_environment detects venv."""
        (temp_base / "venv").mkdir()

        target = InstallationTarget(base_path=temp_base)
        assert target.has_python_environment() is True

    def test_has_python_environment_false(self, temp_base):
        """Test has_python_environment returns False when no env exists."""
        target = InstallationTarget(base_path=temp_base)
        assert target.has_python_environment() is False

    def test_get_environment_candidates_priority(self, temp_base):
        """Test get_environment_candidates returns in correct priority order."""
        # Create all three environment types
        embeded = temp_base / "python_embeded"
        embeded.mkdir()
        (embeded / "python.exe").write_text("fake")

        dot_venv = temp_base / ".venv" / "Scripts"
        dot_venv.mkdir(parents=True)
        (dot_venv / "python.exe").write_text("fake")

        venv = temp_base / "venv" / "Scripts"
        venv.mkdir(parents=True)
        (venv / "python.exe").write_text("fake")

        target = InstallationTarget(base_path=temp_base)
        candidates = target.get_environment_candidates("Windows")

        # Should be: portable, .venv, venv
        assert len(candidates) == 3
        assert candidates[0].environment_type == "portable"
        assert candidates[1].environment_type == "venv"  # .venv is still "venv" type
        assert candidates[1].venv_path.name == ".venv"
        assert candidates[2].environment_type == "venv"
        assert candidates[2].venv_path.name == "venv"


class TestIsComfyuiDirectory:
    """Tests for is_comfyui_directory function."""

    def test_valid_directory_with_main_py(self, temp_base):
        """Test returns True for traditional ComfyUI with main.py."""
        (temp_base / "main.py").write_text("# ComfyUI")
        assert is_comfyui_directory(temp_base) is True

    def test_valid_directory_with_subdir_main_py(self, temp_base):
        """Test returns True when main.py is in ComfyUI subdirectory."""
        comfyui_dir = temp_base / "ComfyUI"
        comfyui_dir.mkdir()
        (comfyui_dir / "main.py").write_text("# ComfyUI")
        assert is_comfyui_directory(temp_base) is True

    def test_valid_comfyui_desktop_structure(self, temp_base):
        """Test returns True for ComfyUI Desktop user data directory."""
        # ComfyUI Desktop basePath has custom_nodes and models but no main.py
        (temp_base / "custom_nodes").mkdir()
        (temp_base / "models").mkdir()
        assert is_comfyui_directory(temp_base) is True

    def test_invalid_directory(self, temp_base):
        """Test returns False for directory without ComfyUI indicators."""
        assert is_comfyui_directory(temp_base) is False

    def test_invalid_partial_desktop_structure(self, temp_base):
        """Test returns False when only one of custom_nodes/models exists."""
        (temp_base / "custom_nodes").mkdir()
        # Missing models/ - not a valid ComfyUI Desktop structure
        assert is_comfyui_directory(temp_base) is False

    def test_nonexistent_path(self):
        """Test returns False for non-existent path."""
        assert is_comfyui_directory(Path("/nonexistent/path")) is False

    def test_file_not_directory(self, temp_base):
        """Test returns False when path is a file."""
        file_path = temp_base / "somefile.txt"
        file_path.write_text("not a directory")
        assert is_comfyui_directory(file_path) is False


class TestDiscoverComfyuiInstallations:
    """Tests for discover_comfyui_installations function."""

    def test_finds_desktop_config(self, temp_base, monkeypatch):
        """Test discovery finds installation from Desktop config.json."""
        # Create fake ComfyUI installation
        comfyui_path = temp_base / "Documents" / "ComfyUI"
        comfyui_path.mkdir(parents=True)
        (comfyui_path / "main.py").write_text("# ComfyUI")

        # Create fake config
        config_dir = temp_base / "AppData" / "Roaming" / "ComfyUI"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"basePath": str(comfyui_path)}))

        # Patch Path.home() to return our temp directory
        monkeypatch.setattr(Path, "home", lambda: temp_base)

        installations = discover_comfyui_installations()

        # Should find the Desktop installation
        assert len(installations) >= 1
        sources = [source for source, _ in installations]
        assert "ComfyUI Desktop" in sources

    def test_finds_common_locations(self, temp_base, monkeypatch):
        """Test discovery finds installation in common locations."""
        # Create fake ComfyUI in Documents
        comfyui_path = temp_base / "Documents" / "ComfyUI"
        comfyui_path.mkdir(parents=True)
        (comfyui_path / "main.py").write_text("# ComfyUI")

        # Patch Path.home() to return our temp directory
        monkeypatch.setattr(Path, "home", lambda: temp_base)

        installations = discover_comfyui_installations()

        # Should find the Documents installation
        paths = [path for _, path in installations]
        assert comfyui_path in paths

    def test_no_duplicates(self, temp_base, monkeypatch):
        """Test discovery doesn't return duplicates."""
        # Create fake ComfyUI installation
        comfyui_path = temp_base / "Documents" / "ComfyUI"
        comfyui_path.mkdir(parents=True)
        (comfyui_path / "main.py").write_text("# ComfyUI")

        # Create config pointing to same location
        config_dir = temp_base / "AppData" / "Roaming" / "ComfyUI"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"basePath": str(comfyui_path)}))

        monkeypatch.setattr(Path, "home", lambda: temp_base)

        installations = discover_comfyui_installations()

        # Should not have duplicates
        paths = [path.resolve() for _, path in installations]
        assert len(paths) == len(set(paths))

    def test_empty_when_nothing_found(self, temp_base, monkeypatch):
        """Test discovery returns empty list when nothing found."""
        # Empty temp directory, nothing to find
        monkeypatch.setattr(Path, "home", lambda: temp_base)

        installations = discover_comfyui_installations()
        assert installations == []

    def test_invalid_config_skipped(self, temp_base, monkeypatch):
        """Test discovery handles invalid config.json gracefully."""
        # Create invalid config
        config_dir = temp_base / "AppData" / "Roaming" / "ComfyUI"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.json"
        config_file.write_text("not valid json")

        monkeypatch.setattr(Path, "home", lambda: temp_base)

        # Should not crash, just return empty
        installations = discover_comfyui_installations()
        assert isinstance(installations, list)


class TestSelectInstallationInteractive:
    """Tests for select_installation_interactive function."""

    def test_returns_none_on_empty_list(self):
        """Test returns None when no installations provided."""
        result = select_installation_interactive([])
        assert result is None

    def test_returns_none_on_quit(self, monkeypatch):
        """Test returns None when user enters 'q'."""
        monkeypatch.setattr("builtins.input", lambda: "q")

        installations = [("Test", Path("/fake/path"))]
        result = select_installation_interactive(installations)
        assert result is None

    def test_returns_path_on_valid_index(self, monkeypatch):
        """Test returns correct path when user enters valid index."""
        monkeypatch.setattr("builtins.input", lambda: "1")

        installations = [
            ("First", Path("/fake/first")),
            ("Second", Path("/fake/second")),
        ]
        result = select_installation_interactive(installations)
        assert result == Path("/fake/first")

    def test_returns_none_on_invalid_index(self, monkeypatch, capsys):
        """Test returns None when user enters invalid index."""
        monkeypatch.setattr("builtins.input", lambda: "99")

        installations = [("Test", Path("/fake/path"))]
        result = select_installation_interactive(installations)

        assert result is None
        captured = capsys.readouterr()
        assert "Invalid selection" in captured.out

    def test_accepts_valid_path(self, temp_base, monkeypatch):
        """Test accepts user-entered path if it's valid ComfyUI directory."""
        # Create valid ComfyUI directory
        (temp_base / "main.py").write_text("# ComfyUI")

        monkeypatch.setattr("builtins.input", lambda: str(temp_base))

        installations = [("Other", Path("/fake/other"))]
        result = select_installation_interactive(installations)

        assert result == temp_base


# Fixture from conftest.py
@pytest.fixture
def temp_base():
    """Provide temporary directory as Path for test isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
