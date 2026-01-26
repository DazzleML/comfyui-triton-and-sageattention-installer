"""Helper functions for unit tests."""
from pathlib import Path
from typing import Optional
from unittest.mock import Mock


def create_env_structure(base_path: Path, env_name: str, nested: bool = True) -> Path:
    """Create environment directory with test files.

    Args:
        base_path: Parent directory
        env_name: Environment folder name (venv, .venv, python_embeded, etc.)
        nested: Whether to create nested subfolder structure

    Returns:
        Path to created environment directory
    """
    env_path = base_path / env_name
    env_path.mkdir(parents=True, exist_ok=True)
    (env_path / "test_file.txt").write_text("test content")
    if nested:
        (env_path / "subfolder").mkdir()
        (env_path / "subfolder" / "nested.txt").write_text("nested content")
    return env_path


def create_backup_structure(base_path: Path, timestamp: str, env_name: str) -> Path:
    """Create backup directory structure with test file.

    Args:
        base_path: ComfyUI base path
        timestamp: Backup timestamp (e.g., "20260105_143000")
        env_name: Environment folder name

    Returns:
        Path to backup's environment directory
    """
    backup_dir = base_path / ".comfyui_backups" / timestamp / env_name
    backup_dir.mkdir(parents=True)
    (backup_dir / "test.txt").write_text("test")
    return backup_dir


def create_mock_handler(
    venv_path: Optional[Path] = None,
    env_type: str = "venv",
    with_run_command: bool = False
) -> Mock:
    """Create configured mock handler for BackupManager tests.

    Args:
        venv_path: Path to venv directory (None for no venv)
        env_type: Environment type string ("venv", "portable", etc.)
        with_run_command: Whether to configure run_command mock for pip freeze

    Returns:
        Configured Mock handler
    """
    handler = Mock()
    handler.python_path = (venv_path / "Scripts" / "python.exe") if venv_path else Path("/fake/python")
    handler.environment_type = env_type
    handler.venv_path = venv_path
    if with_run_command:
        handler.run_command = Mock(return_value=Mock(stdout="package==1.0.0\n", returncode=0))
    return handler
