"""
One-off test script for validating --upgrade flag functionality.
Uses existing ComfyUI installation at C:\\code\\ComfyUI_2025.12.07

Tests the following scenarios:
1. Upgrade from SA 1.0.6 to SA 2.x (primary use case)
2. Upgrade with explicit --sage-version
3. --upgrade without existing installation (should install)
4. Upgrade within SA2 (2.1.x to 2.2.x)

Usage:
    python tests/one-offs/test_upgrade.py [--test N]

    --test N    Only run test number N
"""
import subprocess
import sys
import argparse
import os
from pathlib import Path

# Enable ANSI colors in Windows cmd.exe (Windows 10+)
os.system('')  # Enables VT100 escape codes

# Configuration
COMFYUI_PATH = Path(r"C:\code\ComfyUI_2025.12.07")
INSTALLER_PATH = Path(__file__).parent.parent.parent / "comfyui_triton_sageattention.py"
PYTHON_EXE = COMFYUI_PATH / "venv" / "Scripts" / "python.exe"


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def get_installed_version():
    """Get currently installed sageattention version."""
    result = subprocess.run(
        [str(PYTHON_EXE), "-m", "pip", "show", "sageattention"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':', 1)[1].strip()
    return None


def uninstall_sageattention():
    """Uninstall sageattention."""
    print(f"{Colors.YELLOW}Uninstalling sageattention...{Colors.END}")
    subprocess.run(
        [str(PYTHON_EXE), "-m", "pip", "uninstall", "-y", "sageattention"],
        capture_output=True, text=True
    )


def install_specific_version(version: str):
    """Install a specific sageattention version from PyPI."""
    print(f"{Colors.YELLOW}Installing sageattention=={version}...{Colors.END}")
    result = subprocess.run(
        [str(PYTHON_EXE), "-m", "pip", "install", f"sageattention=={version}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"{Colors.RED}Failed to install: {result.stderr}{Colors.END}")
        return False
    return True


def install_via_installer(sage_version: str):
    """Install a specific sageattention version using the installer.

    Useful for SA2 versions that require pre-built wheels.
    """
    print(f"{Colors.YELLOW}Installing SA {sage_version} via installer...{Colors.END}")
    cmd = [str(PYTHON_EXE), str(INSTALLER_PATH), "--upgrade",
           "--sage-version", sage_version,
           "--base-path", str(COMFYUI_PATH),
           "--non-interactive"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{Colors.RED}Failed to install: {result.stderr}{Colors.END}")
        return False
    return True


def run_upgrade(extra_args: list = None):
    """Run the installer with --upgrade flag."""
    args = extra_args or []
    cmd = [str(PYTHON_EXE), str(INSTALLER_PATH), "--upgrade"] + args + [
        "--base-path", str(COMFYUI_PATH),
        "--non-interactive"
    ]

    print(f"Running: {' '.join(str(c) for c in cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print key output lines
    if result.stdout:
        for line in result.stdout.split('\n'):
            if any(kw in line.lower() for kw in [
                'upgrading', 'current version', 'removing', 'trying',
                'successfully', 'error', 'failed', 'strategy', 'installed'
            ]):
                print(line)

    if result.stderr:
        print(f"{Colors.RED}STDERR: {result.stderr}{Colors.END}")

    return result.returncode == 0


def run_test(test_num: int, description: str, setup_fn, upgrade_args: list,
             expected_version_contains: str):
    """
    Run a single test case.

    Args:
        test_num: Test number for display
        description: Human-readable test description
        setup_fn: Function to call before upgrade (sets up initial state)
        upgrade_args: Additional arguments for --upgrade
        expected_version_contains: String that should be in final version

    Returns:
        bool: True if test passed
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}TEST {test_num}: {description}{Colors.END}")
    print('='*70)

    # Setup
    if setup_fn:
        setup_fn()

    before_version = get_installed_version()
    print(f"Before: {before_version or 'Not installed'}")

    # Run upgrade
    success = run_upgrade(upgrade_args)

    # Check result
    after_version = get_installed_version()
    print(f"After: {after_version or 'Not installed'}")

    passed = False
    if after_version and expected_version_contains in after_version:
        print(f"\n{Colors.GREEN}[PASS] Got version '{after_version}' (expected to contain '{expected_version_contains}'){Colors.END}")
        passed = True
    else:
        print(f"\n{Colors.RED}[FAIL] Expected version containing '{expected_version_contains}' but got '{after_version}'{Colors.END}")

    return passed


def main():
    parser = argparse.ArgumentParser(description='Test --upgrade flag functionality')
    parser.add_argument('--test', type=int, help='Only run specific test number')
    args = parser.parse_args()

    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}--UPGRADE FLAG TEST SUITE{Colors.END}")
    print(f"ComfyUI Path: {COMFYUI_PATH}")
    print(f"Installer: {INSTALLER_PATH}")
    print(f"Python: {PYTHON_EXE}")
    print('='*70)

    # Verify paths exist
    if not COMFYUI_PATH.exists():
        print(f"{Colors.RED}ERROR: ComfyUI path not found: {COMFYUI_PATH}{Colors.END}")
        sys.exit(1)
    if not PYTHON_EXE.exists():
        print(f"{Colors.RED}ERROR: Python executable not found: {PYTHON_EXE}{Colors.END}")
        sys.exit(1)
    if not INSTALLER_PATH.exists():
        print(f"{Colors.RED}ERROR: Installer not found: {INSTALLER_PATH}{Colors.END}")
        sys.exit(1)

    tests = [
        # (test_num, description, setup_fn, upgrade_args, expected_version_contains)

        # Test 1: Primary use case - upgrade from SA 1.0.6 to SA 2.x
        (1, "Upgrade from SA 1.0.6 to SA 2.x (primary use case)",
         lambda: (uninstall_sageattention(), install_specific_version("1.0.6")),
         [],  # Just --upgrade, no other args
         "2.2.0"),  # Should get latest SA2

        # Test 2: Upgrade with explicit --sage-version
        (2, "Upgrade with --sage-version 2.1.1",
         lambda: uninstall_sageattention(),
         ["--sage-version", "2.1.1"],
         "2.1.1"),

        # Test 3: Upgrade when nothing installed (should just install)
        (3, "Upgrade with no existing installation",
         lambda: uninstall_sageattention(),
         [],
         "2.2.0"),  # Should install latest SA2

        # Test 4: Upgrade within SA2 (2.1.x to 2.2.x)
        (4, "Upgrade from SA 2.1.1 to SA 2.2.x",
         lambda: (uninstall_sageattention(), install_via_installer("2.1.1")),
         [],  # Just --upgrade, should go to latest
         "2.2.0"),  # Should upgrade to 2.2.0.post3
    ]

    results = []

    for test_num, description, setup_fn, upgrade_args, expected in tests:
        if args.test is not None and args.test != test_num:
            continue

        passed = run_test(test_num, description, setup_fn, upgrade_args, expected)
        results.append((test_num, description, passed))

    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.END}")
    print('='*70)

    passed_count = sum(1 for _, _, p in results if p)
    total_count = len(results)

    for test_num, description, passed in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if passed else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  Test {test_num}: [{status}] {description}")

    print('='*70)
    if passed_count == total_count:
        print(f"{Colors.GREEN}All {total_count} tests passed!{Colors.END}")
    else:
        print(f"{Colors.RED}{passed_count}/{total_count} tests passed{Colors.END}")
        sys.exit(1)

    # Restore to stable version
    print(f"\n{Colors.YELLOW}Restoring to SA 2.2.0.post3...{Colors.END}")
    uninstall_sageattention()
    run_upgrade([])


if __name__ == "__main__":
    main()
