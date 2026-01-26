"""Unit tests for BackupManager class."""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from comfyui_triton_sageattention import BackupManager, BackupInfo


class TestBackupInfo:
    """Tests for BackupInfo dataclass."""

    def test_size_human_megabytes(self):
        """Test human-readable size for MB range."""
        info = BackupInfo(
            timestamp="20260105_143000",
            path=Path("/fake/backup"),
            size_bytes=500 * 1024 * 1024,  # 500 MB
            env_type="venv",
            index=1
        )
        assert info.size_human == "500.0 MB"

    def test_size_human_gigabytes(self):
        """Test human-readable size for GB range."""
        info = BackupInfo(
            timestamp="20260105_143000",
            path=Path("/fake/backup"),
            size_bytes=3 * 1024 * 1024 * 1024,  # 3 GB
            env_type="venv",
            index=1
        )
        assert info.size_human == "3.0 GB"

    def test_size_human_small(self):
        """Test human-readable size for small sizes."""
        info = BackupInfo(
            timestamp="20260105_143000",
            path=Path("/fake/backup"),
            size_bytes=10 * 1024 * 1024,  # 10 MB
            env_type="portable",
            index=2
        )
        assert info.size_human == "10.0 MB"


class TestBackupManagerInit:
    """Tests for BackupManager initialization."""

    def test_backup_root_path(self):
        """Test backup_root is correctly set."""
        mock_handler = Mock()
        mock_handler.python_path = Path("/fake/python")

        manager = BackupManager(
            base_path=Path("/fake/comfyui"),
            handler=mock_handler,
            interactive=True
        )

        assert manager.backup_root == Path("/fake/comfyui/.comfyui_backups")


class TestBackupManagerListBackups:
    """Tests for list_backups method."""

    def test_list_no_backups(self):
        """Test listing when no backups exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")

            manager = BackupManager(base_path, mock_handler, interactive=True)
            backups = manager.list_backups()

            assert backups == []

    def test_list_single_backup(self):
        """Test listing with one backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create backup structure
            backup_dir = base_path / ".comfyui_backups" / "20260105_143000"
            backup_dir.mkdir(parents=True)
            venv_dir = backup_dir / "venv"
            venv_dir.mkdir()
            (venv_dir / "test.txt").write_text("test content")

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")

            manager = BackupManager(base_path, mock_handler, interactive=True)
            backups = manager.list_backups()

            assert len(backups) == 1
            assert backups[0].timestamp == "20260105_143000"
            assert backups[0].env_type == "venv"
            assert backups[0].index == 1

    def test_list_multiple_backups_ordered(self):
        """Test that backups are ordered newest first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create multiple backups (older timestamps)
            for ts in ["20260103_100000", "20260105_143000", "20260104_120000"]:
                backup_dir = base_path / ".comfyui_backups" / ts / "venv"
                backup_dir.mkdir(parents=True)
                (backup_dir / "test.txt").write_text("test")

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")

            manager = BackupManager(base_path, mock_handler, interactive=True)
            backups = manager.list_backups()

            assert len(backups) == 3
            # Should be ordered newest first
            assert backups[0].timestamp == "20260105_143000"
            assert backups[0].index == 1
            assert backups[1].timestamp == "20260104_120000"
            assert backups[1].index == 2
            assert backups[2].timestamp == "20260103_100000"
            assert backups[2].index == 3

    def test_list_portable_backup(self):
        """Test listing backup of portable environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create portable backup
            backup_dir = base_path / ".comfyui_backups" / "20260105_143000"
            backup_dir.mkdir(parents=True)
            portable_dir = backup_dir / "python_embeded"
            portable_dir.mkdir()
            (portable_dir / "test.txt").write_text("test")

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")

            manager = BackupManager(base_path, mock_handler, interactive=True)
            backups = manager.list_backups()

            assert len(backups) == 1
            assert backups[0].env_type == "python_embeded"


class TestBackupManagerGetByIdentifier:
    """Tests for get_backup_by_identifier method."""

    def test_get_by_index(self):
        """Test getting backup by index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create backups
            for ts in ["20260103_100000", "20260105_143000"]:
                backup_dir = base_path / ".comfyui_backups" / ts / "venv"
                backup_dir.mkdir(parents=True)
                (backup_dir / "test.txt").write_text("test")

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")

            manager = BackupManager(base_path, mock_handler, interactive=True)

            # Index 1 should be newest
            backup = manager.get_backup_by_identifier("1")
            assert backup is not None
            assert backup.timestamp == "20260105_143000"

            # Index 2 should be older
            backup = manager.get_backup_by_identifier("2")
            assert backup is not None
            assert backup.timestamp == "20260103_100000"

    def test_get_by_timestamp(self):
        """Test getting backup by timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create backup
            backup_dir = base_path / ".comfyui_backups" / "20260105_143000" / "venv"
            backup_dir.mkdir(parents=True)
            (backup_dir / "test.txt").write_text("test")

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")

            manager = BackupManager(base_path, mock_handler, interactive=True)

            backup = manager.get_backup_by_identifier("20260105_143000")
            assert backup is not None
            assert backup.timestamp == "20260105_143000"

    def test_get_invalid_index(self):
        """Test getting backup with invalid index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create one backup
            backup_dir = base_path / ".comfyui_backups" / "20260105_143000" / "venv"
            backup_dir.mkdir(parents=True)
            (backup_dir / "test.txt").write_text("test")

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")

            manager = BackupManager(base_path, mock_handler, interactive=True)

            # Index 5 doesn't exist
            backup = manager.get_backup_by_identifier("5")
            assert backup is None

    def test_get_invalid_timestamp(self):
        """Test getting backup with non-existent timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")

            manager = BackupManager(base_path, mock_handler, interactive=True)

            backup = manager.get_backup_by_identifier("20260199_999999")
            assert backup is None


class TestBackupManagerCleanSafety:
    """Tests for clean method safety features."""

    def test_clean_refuses_noninteractive(self, capsys):
        """Test that clean refuses to run in non-interactive mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create a backup
            backup_dir = base_path / ".comfyui_backups" / "20260105_143000" / "venv"
            backup_dir.mkdir(parents=True)
            (backup_dir / "test.txt").write_text("test")

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")

            # Non-interactive mode
            manager = BackupManager(base_path, mock_handler, interactive=False)
            result = manager.clean(indices=[1])

            assert result == 0  # Nothing removed
            # Backup should still exist
            assert backup_dir.exists()

            captured = capsys.readouterr()
            assert "Cannot delete backups in non-interactive mode" in captured.out


class TestBackupManagerRestoreSafety:
    """Tests for restore method safety features."""

    def test_restore_refuses_noninteractive(self, capsys):
        """Test that restore refuses to run in non-interactive mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create a backup
            backup_dir = base_path / ".comfyui_backups" / "20260105_143000"
            backup_dir.mkdir(parents=True)
            venv_backup = backup_dir / "venv"
            venv_backup.mkdir()
            (venv_backup / "backed_up.txt").write_text("backup content")

            # Create current venv
            current_venv = base_path / "venv"
            current_venv.mkdir()
            (current_venv / "current.txt").write_text("current content")

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")
            mock_handler.environment_type = "venv"

            # Non-interactive mode
            manager = BackupManager(base_path, mock_handler, interactive=False)
            result = manager.restore("1")

            assert result is False  # Restore failed
            # Current venv should still have its original content
            assert (current_venv / "current.txt").exists()

            captured = capsys.readouterr()
            assert "Cannot restore in non-interactive mode" in captured.out


class TestBackupManagerCreate:
    """Tests for create method."""

    def test_create_backup_venv(self):
        """Test creating backup of venv environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create venv to backup
            venv_dir = base_path / "venv"
            venv_dir.mkdir()
            (venv_dir / "test_file.txt").write_text("test content")
            (venv_dir / "subfolder").mkdir()
            (venv_dir / "subfolder" / "nested.txt").write_text("nested content")

            mock_handler = Mock()
            mock_handler.python_path = venv_dir / "Scripts" / "python.exe"
            mock_handler.environment_type = "venv"
            mock_handler.venv_path = venv_dir  # Handler detected venv
            mock_handler.run_command = Mock(return_value=Mock(stdout="package==1.0.0\n", returncode=0))

            manager = BackupManager(base_path, mock_handler, interactive=True)
            result = manager.create()

            assert result is not None
            assert result.exists()

            # Check backup contents
            backup_venv = result / "venv"
            assert backup_venv.exists()
            assert (backup_venv / "test_file.txt").exists()
            assert (backup_venv / "subfolder" / "nested.txt").exists()

            # Check pip freeze was saved
            assert (result / "requirements.txt").exists()

            # Check restore instructions
            assert (result / "RESTORE.txt").exists()

    def test_create_backup_dot_venv(self):
        """Test creating backup of .venv environment (uv, poetry, modern tooling)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create .venv to backup (used by uv, poetry, etc.)
            dot_venv_dir = base_path / ".venv"
            dot_venv_dir.mkdir()
            (dot_venv_dir / "test_file.txt").write_text("test content")
            (dot_venv_dir / "subfolder").mkdir()
            (dot_venv_dir / "subfolder" / "nested.txt").write_text("nested content")

            mock_handler = Mock()
            mock_handler.python_path = dot_venv_dir / "Scripts" / "python.exe"
            mock_handler.environment_type = "venv"
            mock_handler.venv_path = dot_venv_dir  # Handler detected .venv
            mock_handler.run_command = Mock(return_value=Mock(stdout="package==1.0.0\n", returncode=0))

            manager = BackupManager(base_path, mock_handler, interactive=True)
            result = manager.create()

            assert result is not None
            assert result.exists()

            # Check backup contents - should be under ".venv" not "venv"
            backup_dot_venv = result / ".venv"
            assert backup_dot_venv.exists()
            assert (backup_dot_venv / "test_file.txt").exists()
            assert (backup_dot_venv / "subfolder" / "nested.txt").exists()

            # Verify "venv" folder was NOT created
            backup_venv = result / "venv"
            assert not backup_venv.exists()

            # Check pip freeze was saved
            assert (result / "requirements.txt").exists()

    def test_create_backup_custom_python_path(self):
        """Test creating backup with custom --python path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create custom venv at arbitrary location
            custom_venv = base_path / "my_custom_env"
            custom_venv.mkdir()
            (custom_venv / "test_file.txt").write_text("test content")

            mock_handler = Mock()
            mock_handler.python_path = custom_venv / "Scripts" / "python.exe"
            mock_handler.environment_type = "venv"
            mock_handler.venv_path = custom_venv  # Handler detected custom path
            mock_handler.run_command = Mock(return_value=Mock(stdout="package==1.0.0\n", returncode=0))

            manager = BackupManager(base_path, mock_handler, interactive=True)
            result = manager.create()

            assert result is not None
            assert result.exists()

            # Check backup contents - should be under custom name
            backup_custom = result / "my_custom_env"
            assert backup_custom.exists()
            assert (backup_custom / "test_file.txt").exists()

    def test_create_backup_no_env(self, capsys):
        """Test creating backup when no environment exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")
            mock_handler.environment_type = "venv"
            mock_handler.venv_path = None  # No venv detected

            manager = BackupManager(base_path, mock_handler, interactive=True)
            result = manager.create()

            assert result is None

            captured = capsys.readouterr()
            assert "No environment found" in captured.out


class TestBackupManagerListDotVenv:
    """Tests for list_backups with .venv environments."""

    def test_list_dot_venv_backup(self):
        """Test listing backup of .venv environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create .venv backup
            backup_dir = base_path / ".comfyui_backups" / "20260125_120000"
            backup_dir.mkdir(parents=True)
            dot_venv_dir = backup_dir / ".venv"
            dot_venv_dir.mkdir()
            (dot_venv_dir / "test.txt").write_text("test")

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")

            manager = BackupManager(base_path, mock_handler, interactive=True)
            backups = manager.list_backups()

            assert len(backups) == 1
            assert backups[0].env_type == ".venv"
            assert backups[0].timestamp == "20260125_120000"

    def test_list_mixed_backup_types(self):
        """Test listing mixed backup types (venv, .venv, portable)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create different backup types
            for ts, env_name in [
                ("20260125_100000", "venv"),
                ("20260125_110000", ".venv"),
                ("20260125_120000", "python_embeded"),
            ]:
                backup_dir = base_path / ".comfyui_backups" / ts / env_name
                backup_dir.mkdir(parents=True)
                (backup_dir / "test.txt").write_text("test")

            mock_handler = Mock()
            mock_handler.python_path = Path("/fake/python")

            manager = BackupManager(base_path, mock_handler, interactive=True)
            backups = manager.list_backups()

            assert len(backups) == 3
            # Ordered by timestamp (newest first)
            assert backups[0].env_type == "python_embeded"
            assert backups[1].env_type == ".venv"
            assert backups[2].env_type == "venv"
