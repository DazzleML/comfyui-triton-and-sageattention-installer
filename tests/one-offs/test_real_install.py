"""
One-off test script for validating wheel_configs changes.
Uses existing ComfyUI installation at C:\\code\\ComfyUI_2025.12.07

Tests the following scenarios from the test matrix:
1. Default install -> SA 2.2.0.post3
2. --sage-version 2 -> SA 2.2.0.post3
3. --sage-version 1 -> SA 1.0.6
4. --experimental flag behavior
5. --sage-version 2.1.1 -> SA 2.1.1

Usage:
    python tests/one-offs/test_real_install.py [--skip-cleanup] [--test N]

    --skip-cleanup    Don't uninstall sageattention between tests
    --test N          Only run test number N
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
    """Uninstall sageattention to start fresh."""
    print(f"{Colors.YELLOW}Uninstalling sageattention...{Colors.END}")
    subprocess.run(
        [str(PYTHON_EXE), "-m", "pip", "uninstall", "-y", "sageattention"],
        capture_output=True, text=True
    )


def run_test(test_num: int, description: str, args: list, expected_version_contains: str = None):
    """
    Run installer with given args and check result.

    Args:
        test_num: Test number for display
        description: Human-readable test description
        args: List of arguments to pass to installer
        expected_version_contains: String that should be in the installed version

    Returns:
        bool: True if test passed
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}TEST {test_num}: {description}{Colors.END}")
    print(f"Args: {' '.join(args)}")
    print('='*70)

    cmd = [str(PYTHON_EXE), str(INSTALLER_PATH)] + args + [
        "--base-path", str(COMFYUI_PATH),
        "--non-interactive"
    ]

    print(f"Running: {' '.join(str(c) for c in cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print output
    if result.stdout:
        # Filter to show key lines
        for line in result.stdout.split('\n'):
            if any(kw in line.lower() for kw in ['trying', 'successfully', 'error', 'failed', 'warning', 'detected', 'experimental']):
                print(line)

    if result.stderr:
        print(f"{Colors.RED}STDERR: {result.stderr}{Colors.END}")

    # Check result
    passed = True
    if expected_version_contains:
        installed = get_installed_version()
        if installed and expected_version_contains in installed:
            print(f"\n{Colors.GREEN}[PASS] Got version '{installed}' (expected to contain '{expected_version_contains}'){Colors.END}")
        else:
            print(f"\n{Colors.RED}[FAIL] Expected version containing '{expected_version_contains}' but got '{installed}'{Colors.END}")
            passed = False
    elif result.returncode == 0:
        print(f"\n{Colors.GREEN}[PASS] Command succeeded{Colors.END}")
    else:
        print(f"\n{Colors.RED}[FAIL] Command failed with return code {result.returncode}{Colors.END}")
        passed = False

    return passed


def main():
    parser = argparse.ArgumentParser(description='Test wheel_configs implementation')
    parser.add_argument('--skip-cleanup', action='store_true',
                        help='Skip uninstalling sageattention between tests')
    parser.add_argument('--test', type=int,
                        help='Only run specific test number')
    args = parser.parse_args()

    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}WHEEL_CONFIGS TEST SUITE{Colors.END}")
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

    # Show current environment
    print("\nDetecting current environment...")
    subprocess.run([str(PYTHON_EXE), str(INSTALLER_PATH), "--base-path", str(COMFYUI_PATH)])

    tests = [
        # (test_num, description, args, expected_version_contains)
        (1, "Default install -> SA 2.2.0.post3",
         ["--install"], "2.2.0"),

        (2, "--sage-version 2 -> latest SA2 (2.2.0.post3)",
         ["--install", "--sage-version", "2"], "2.2.0"),

        (3, "--sage-version 1 -> SA 1.0.6",
         ["--install", "--sage-version", "1"], "1.0.6"),

        (4, "--sage-version 2.1.1 -> specific version",
         ["--install", "--sage-version", "2.1.1"], "2.1.1"),

        (5, "--experimental flag with default install",
         ["--install", "--experimental"], "2.2.0"),  # Could be post3 or post4
    ]

    results = []

    for test_num, description, test_args, expected in tests:
        if args.test is not None and args.test != test_num:
            continue

        if not args.skip_cleanup:
            uninstall_sageattention()

        passed = run_test(test_num, description, test_args, expected)
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


if __name__ == "__main__":
    main()
