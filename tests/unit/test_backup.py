"""Unit tests for BackupManager class."""
import pytest
import tempfile
from unittest.mock import Mock
from pathlib import Path
import sys

# Add project root and tests/unit to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from comfyui_triton_sageattention import BackupManager, BackupInfo
from helpers import create_env_structure, create_backup_structure, create_mock_handler


class TestBackupInfo:
    """Tests for BackupInfo dataclass."""

    @pytest.mark.parametrize("size_bytes,expected", [
        (500 * 1024 * 1024, "500.0 MB"),      # 500 MB
        (3 * 1024 * 1024 * 1024, "3.0 GB"),   # 3 GB
        (10 * 1024 * 1024, "10.0 MB"),        # 10 MB
    ])
    def test_size_human(self, size_bytes, expected):
        """Test human-readable size formatting."""
        info = BackupInfo(
            timestamp="20260105_143000",
            path=Path("/fake/backup"),
            size_bytes=size_bytes,
            env_type="venv",
            index=1
        )
        assert info.size_human == expected


class TestBackupManagerInit:
    """Tests for BackupManager initialization."""

    def test_backup_root_path(self):
        """Test backup_root is correctly set."""
        handler = create_mock_handler()
        manager = BackupManager(
            base_path=Path("/fake/comfyui"),
            handler=handler,
            interactive=True
        )
        assert manager.backup_root == Path("/fake/comfyui/.comfyui_backups")


class TestBackupManagerListBackups:
    """Tests for list_backups method."""

    def test_list_no_backups(self, temp_base):
        """Test listing when no backups exist."""
        handler = create_mock_handler()
        manager = BackupManager(temp_base, handler, interactive=True)
        assert manager.list_backups() == []

    @pytest.mark.parametrize("env_name,expected_type", [
        ("venv", "venv"),
        (".venv", ".venv"),
        ("python_embeded", "python_embeded"),
    ])
    def test_list_single_backup_types(self, temp_base, env_name, expected_type):
        """Test listing single backup of various environment types."""
        create_backup_structure(temp_base, "20260105_143000", env_name)
        handler = create_mock_handler()

        manager = BackupManager(temp_base, handler, interactive=True)
        backups = manager.list_backups()

        assert len(backups) == 1
        assert backups[0].timestamp == "20260105_143000"
        assert backups[0].env_type == expected_type
        assert backups[0].index == 1

    def test_list_multiple_backups_ordered(self, temp_base):
        """Test that backups are ordered newest first."""
        for ts in ["20260103_100000", "20260105_143000", "20260104_120000"]:
            create_backup_structure(temp_base, ts, "venv")

        handler = create_mock_handler()
        manager = BackupManager(temp_base, handler, interactive=True)
        backups = manager.list_backups()

        assert len(backups) == 3
        # Should be ordered newest first
        assert [b.timestamp for b in backups] == [
            "20260105_143000", "20260104_120000", "20260103_100000"
        ]
        assert [b.index for b in backups] == [1, 2, 3]

    def test_list_mixed_backup_types(self, temp_base):
        """Test listing mixed backup types (venv, .venv, portable)."""
        for ts, env in [
            ("20260125_100000", "venv"),
            ("20260125_110000", ".venv"),
            ("20260125_120000", "python_embeded"),
        ]:
            create_backup_structure(temp_base, ts, env)

        handler = create_mock_handler()
        manager = BackupManager(temp_base, handler, interactive=True)
        backups = manager.list_backups()

        assert len(backups) == 3
        # Ordered by timestamp (newest first)
        assert [b.env_type for b in backups] == ["python_embeded", ".venv", "venv"]


class TestBackupManagerGetByIdentifier:
    """Tests for get_backup_by_identifier method."""

    def test_get_by_index(self, temp_base):
        """Test getting backup by index."""
        for ts in ["20260103_100000", "20260105_143000"]:
            create_backup_structure(temp_base, ts, "venv")

        handler = create_mock_handler()
        manager = BackupManager(temp_base, handler, interactive=True)

        # Index 1 should be newest
        backup = manager.get_backup_by_identifier("1")
        assert backup is not None
        assert backup.timestamp == "20260105_143000"

        # Index 2 should be older
        backup = manager.get_backup_by_identifier("2")
        assert backup is not None
        assert backup.timestamp == "20260103_100000"

    def test_get_by_timestamp(self, temp_base):
        """Test getting backup by timestamp."""
        create_backup_structure(temp_base, "20260105_143000", "venv")
        handler = create_mock_handler()

        manager = BackupManager(temp_base, handler, interactive=True)
        backup = manager.get_backup_by_identifier("20260105_143000")

        assert backup is not None
        assert backup.timestamp == "20260105_143000"

    def test_get_invalid_index(self, temp_base):
        """Test getting backup with invalid index."""
        create_backup_structure(temp_base, "20260105_143000", "venv")
        handler = create_mock_handler()

        manager = BackupManager(temp_base, handler, interactive=True)
        backup = manager.get_backup_by_identifier("5")  # Doesn't exist

        assert backup is None

    def test_get_invalid_timestamp(self, temp_base):
        """Test getting backup with non-existent timestamp."""
        handler = create_mock_handler()
        manager = BackupManager(temp_base, handler, interactive=True)

        backup = manager.get_backup_by_identifier("20260199_999999")
        assert backup is None


class TestBackupManagerCleanSafety:
    """Tests for clean method safety features."""

    def test_clean_refuses_noninteractive(self, temp_base, capsys):
        """Test that clean refuses to run in non-interactive mode."""
        backup_dir = create_backup_structure(temp_base, "20260105_143000", "venv")
        handler = create_mock_handler()

        # Non-interactive mode
        manager = BackupManager(temp_base, handler, interactive=False)
        result = manager.clean(indices=[1])

        assert result == 0  # Nothing removed
        assert backup_dir.exists()  # Backup should still exist

        captured = capsys.readouterr()
        assert "Cannot delete backups in non-interactive mode" in captured.out


class TestBackupManagerRestoreSafety:
    """Tests for restore method safety features."""

    def test_restore_refuses_noninteractive(self, temp_base, capsys):
        """Test that restore refuses to run in non-interactive mode."""
        # Create a backup
        backup_dir = temp_base / ".comfyui_backups" / "20260105_143000"
        backup_dir.mkdir(parents=True)
        venv_backup = backup_dir / "venv"
        venv_backup.mkdir()
        (venv_backup / "backed_up.txt").write_text("backup content")

        # Create current venv
        current_venv = temp_base / "venv"
        current_venv.mkdir()
        (current_venv / "current.txt").write_text("current content")

        handler = create_mock_handler(current_venv, "venv")

        # Non-interactive mode
        manager = BackupManager(temp_base, handler, interactive=False)
        result = manager.restore("1")

        assert result is False  # Restore failed
        assert (current_venv / "current.txt").exists()  # Original content preserved

        captured = capsys.readouterr()
        assert "Cannot restore in non-interactive mode" in captured.out


class TestBackupManagerCreate:
    """Tests for create method."""

    @pytest.mark.parametrize("env_name", ["venv", ".venv", "my_custom_env"])
    def test_create_backup_env_types(self, temp_base, env_name):
        """Test creating backup for various environment types."""
        env_path = create_env_structure(temp_base, env_name)
        handler = create_mock_handler(env_path, "venv", with_run_command=True)

        manager = BackupManager(temp_base, handler, interactive=True)
        result = manager.create()

        assert result is not None
        assert result.exists()

        # Check backup contents
        backup_env = result / env_name
        assert backup_env.exists()
        assert (backup_env / "test_file.txt").exists()
        assert (backup_env / "subfolder" / "nested.txt").exists()

        # Check metadata files
        assert (result / "requirements.txt").exists()
        assert (result / "RESTORE.txt").exists()

    def test_create_backup_no_env(self, temp_base, capsys):
        """Test creating backup when no environment exists."""
        handler = create_mock_handler(None, "venv")

        manager = BackupManager(temp_base, handler, interactive=True)
        result = manager.create()

        assert result is None

        captured = capsys.readouterr()
        assert "No environment found" in captured.out

    def test_create_backup_excludes_other_envs(self, temp_base):
        """Test that backup of .venv doesn't create venv folder."""
        env_path = create_env_structure(temp_base, ".venv")
        handler = create_mock_handler(env_path, "venv", with_run_command=True)

        manager = BackupManager(temp_base, handler, interactive=True)
        result = manager.create()

        # .venv should exist in backup
        assert (result / ".venv").exists()
        # venv should NOT exist
        assert not (result / "venv").exists()
