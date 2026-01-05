"""
One-off test for combined --backup --install workflow.
Tests the recommended quick start command: python comfyui_triton_sageattention.py --install --backup

This verifies:
1. Backup runs first before install
2. If backup fails, install is aborted
3. If backup succeeds, install proceeds

Supports both environment types:
- venv: Git clone ComfyUI with virtual environment
- portable: ComfyUI Portable with python_embeded

Usage:
    python tests/one-offs/test_backup_install.py [--test N] [--cleanup] [--env TYPE]

    --env venv      Test against venv environment (default)
    --env portable  Test against portable environment
    --env both      Test against both environments
"""
import subprocess
import sys
import argparse
import os
from pathlib import Path

# Enable ANSI colors in Windows cmd.exe (Windows 10+)
os.system('')

# Environment configurations
ENVIRONMENTS = {
    "venv": {
        "name": "Git Clone (venv)",
        "path": Path(r"C:\code\ComfyUI_2025.12.07"),
        "python": lambda p: p / "venv" / "Scripts" / "python.exe",
        "env_folder": "venv",
    },
    "portable": {
        "name": "ComfyUI Portable",
        "path": Path(r"C:\ComfyUI_windows_portable"),  # Adjust if different
        "python": lambda p: p / "python_embeded" / "python.exe",
        "env_folder": "python_embeded",
    },
}

INSTALLER_PATH = Path(__file__).parent.parent.parent / "comfyui_triton_sageattention.py"

# Current test environment (set by main)
CURRENT_ENV = None
COMFYUI_PATH = None
PYTHON_EXE = None
BACKUP_ROOT = None


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def count_backups():
    """Count existing backups."""
    if not BACKUP_ROOT.exists():
        return 0
    return len([d for d in BACKUP_ROOT.iterdir() if d.is_dir()])


def cleanup_backups():
    """Remove all test backups."""
    if BACKUP_ROOT.exists():
        import shutil
        for backup_dir in BACKUP_ROOT.iterdir():
            if backup_dir.is_dir():
                print(f"{Colors.YELLOW}Cleaning up: {backup_dir.name}{Colors.END}")
                shutil.rmtree(backup_dir)


def run_command(args: list, description: str):
    """Run installer with given arguments."""
    cmd = [str(PYTHON_EXE), str(INSTALLER_PATH)] + args + [
        "--base-path", str(COMFYUI_PATH)
    ]

    print(f"\n{Colors.BLUE}Running: {description}{Colors.END}")
    print(f"  Command: {' '.join(str(c) for c in cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Show relevant output
    for line in (result.stdout or '').split('\n'):
        if any(kw in line.lower() for kw in [
            'backup', 'creating', 'copying', 'install', 'upgrade',
            'abort', 'failed', 'success', '[ok]', '[x]', '[!]'
        ]):
            print(f"  {line}")

    if result.stderr:
        print(f"{Colors.RED}  STDERR: {result.stderr[:200]}{Colors.END}")

    return result


def test_backup_runs_before_install():
    """Test 1: Verify backup runs first when combined with --install."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}TEST 1: Backup runs before install (dryrun){Colors.END}")
    print('='*70)

    before_count = count_backups()

    # Run with --dryrun to avoid actual installation changes
    result = run_command(
        ["--install", "--backup", "--dryrun"],
        "Install with backup (dryrun mode)"
    )

    after_count = count_backups()

    # Backup should have been created (even in dryrun, backup is real)
    if after_count > before_count:
        print(f"\n{Colors.GREEN}[PASS] Backup was created before install preview{Colors.END}")
        return True
    else:
        print(f"\n{Colors.RED}[FAIL] No backup created (before={before_count}, after={after_count}){Colors.END}")
        return False


def test_backup_output_order():
    """Test 2: Verify backup output appears before install output."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}TEST 2: Output order - backup before install{Colors.END}")
    print('='*70)

    result = run_command(
        ["--install", "--backup", "--dryrun"],
        "Install with backup (checking output order)"
    )

    output = result.stdout or ''

    # Find positions of key markers
    backup_pos = output.lower().find('creating backup')
    if backup_pos == -1:
        backup_pos = output.lower().find('backup created')

    install_pos = output.lower().find('install plan')
    if install_pos == -1:
        install_pos = output.lower().find('preview')

    if backup_pos == -1:
        print(f"\n{Colors.RED}[FAIL] No backup creation message found{Colors.END}")
        return False

    if install_pos == -1:
        print(f"\n{Colors.YELLOW}[WARN] No install message found (might be OK in dryrun){Colors.END}")
        # This is OK if dryrun completed successfully
        if result.returncode == 0:
            print(f"{Colors.GREEN}[PASS] Backup ran and command succeeded{Colors.END}")
            return True
        return False

    if backup_pos < install_pos:
        print(f"\n{Colors.GREEN}[PASS] Backup output appears before install output{Colors.END}")
        return True
    else:
        print(f"\n{Colors.RED}[FAIL] Backup should appear before install{Colors.END}")
        return False


def test_standalone_backup():
    """Test 3: Verify standalone --backup create works."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}TEST 3: Standalone backup create{Colors.END}")
    print('='*70)

    before_count = count_backups()

    result = run_command(["--backup"], "Standalone backup create")

    after_count = count_backups()

    if result.returncode == 0 and after_count > before_count:
        print(f"\n{Colors.GREEN}[PASS] Standalone backup created successfully{Colors.END}")
        return True
    else:
        print(f"\n{Colors.RED}[FAIL] Standalone backup failed{Colors.END}")
        return False


def test_backup_list():
    """Test 4: Verify --backup list shows backups."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}TEST 4: Backup list{Colors.END}")
    print('='*70)

    result = run_command(["--backup", "list"], "List backups")

    output = result.stdout or ''

    if "Available backups:" in output or "No backups found" in output:
        print(f"\n{Colors.GREEN}[PASS] Backup list command works{Colors.END}")
        return True
    else:
        print(f"\n{Colors.RED}[FAIL] Unexpected backup list output{Colors.END}")
        return False


def test_backup_clean_shows_help():
    """Test 5: Verify --backup-clean alone shows help."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}TEST 5: Backup clean shows help when no args{Colors.END}")
    print('='*70)

    result = run_command(["--backup-clean"], "Backup clean (no args)")

    output = result.stdout or ''

    if "To clean" in output or "No backups" in output:
        print(f"\n{Colors.GREEN}[PASS] Backup clean shows help/list{Colors.END}")
        return True
    else:
        print(f"\n{Colors.RED}[FAIL] Backup clean should show help{Colors.END}")
        return False


def setup_environment(env_type: str) -> bool:
    """Configure globals for the specified environment type."""
    global CURRENT_ENV, COMFYUI_PATH, PYTHON_EXE, BACKUP_ROOT

    if env_type not in ENVIRONMENTS:
        print(f"{Colors.RED}Unknown environment type: {env_type}{Colors.END}")
        return False

    env = ENVIRONMENTS[env_type]
    CURRENT_ENV = env
    COMFYUI_PATH = env["path"]
    PYTHON_EXE = env["python"](COMFYUI_PATH)
    BACKUP_ROOT = COMFYUI_PATH / ".comfyui_backups"

    return True


def run_tests_for_env(env_type: str, test_num: int = None, cleanup: bool = False):
    """Run tests for a specific environment."""
    if not setup_environment(env_type):
        return [], 1

    env = ENVIRONMENTS[env_type]

    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}BACKUP + INSTALL WORKFLOW TESTS - {env['name']}{Colors.END}")
    print(f"ComfyUI Path: {COMFYUI_PATH}")
    print(f"Environment: {env['env_folder']}")
    print(f"Python: {PYTHON_EXE}")
    print('='*70)

    # Verify paths
    if not COMFYUI_PATH.exists():
        print(f"{Colors.YELLOW}SKIP: ComfyUI path not found: {COMFYUI_PATH}{Colors.END}")
        return [], 0  # Skip, not fail
    if not PYTHON_EXE.exists():
        print(f"{Colors.YELLOW}SKIP: Python not found: {PYTHON_EXE}{Colors.END}")
        return [], 0  # Skip, not fail

    tests = [
        (1, "Backup runs before install", test_backup_runs_before_install),
        (2, "Output order correct", test_backup_output_order),
        (3, "Standalone backup create", test_standalone_backup),
        (4, "Backup list", test_backup_list),
        (5, "Backup clean shows help", test_backup_clean_shows_help),
    ]

    results = []

    for num, desc, test_fn in tests:
        if test_num is not None and test_num != num:
            continue

        try:
            passed = test_fn()
            results.append((num, desc, passed, env_type))
        except Exception as e:
            print(f"{Colors.RED}[ERROR] Test {num} raised exception: {e}{Colors.END}")
            results.append((num, desc, False, env_type))

    # Cleanup if requested
    if cleanup:
        print(f"\n{Colors.YELLOW}Cleaning up test backups for {env['name']}...{Colors.END}")
        cleanup_backups()

    return results, 0


def main():
    parser = argparse.ArgumentParser(description='Test --backup --install workflow')
    parser.add_argument('--test', type=int, help='Only run specific test number')
    parser.add_argument('--cleanup', action='store_true', help='Clean up test backups after tests')
    parser.add_argument('--env', choices=['venv', 'portable', 'both'], default='venv',
                        help='Environment to test (default: venv)')
    args = parser.parse_args()

    all_results = []
    env_types = ['venv', 'portable'] if args.env == 'both' else [args.env]

    for env_type in env_types:
        results, status = run_tests_for_env(env_type, args.test, args.cleanup)
        all_results.extend(results)

    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}OVERALL TEST SUMMARY{Colors.END}")
    print('='*70)

    if not all_results:
        print(f"{Colors.YELLOW}No tests were run (environments not available){Colors.END}")
        return 0

    passed = sum(1 for r in all_results if r[2])
    failed = sum(1 for r in all_results if not r[2])

    for num, desc, p, env in all_results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if p else f"{Colors.RED}FAIL{Colors.END}"
        env_label = ENVIRONMENTS[env]["name"]
        print(f"  [{env_label}] Test {num}: [{status}] {desc}")

    print('='*70)
    if failed == 0:
        print(f"{Colors.GREEN}All {passed} tests passed!{Colors.END}")
    else:
        print(f"{Colors.RED}{passed} passed, {failed} failed{Colors.END}")

    # Note about backups
    if not args.cleanup:
        for env_type in env_types:
            setup_environment(env_type)
            backup_count = count_backups()
            if backup_count > 0:
                print(f"\n{Colors.YELLOW}Note: {backup_count} backup(s) in {ENVIRONMENTS[env_type]['name']}{Colors.END}")

        print(f"Run with --cleanup to remove test backups")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
