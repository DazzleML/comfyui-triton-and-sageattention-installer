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


def get_pytorch_version():
    """Get installed PyTorch version (major.minor)."""
    result = subprocess.run(
        [str(PYTHON_EXE), "-c", "import torch; print(torch.__version__.split('+')[0])"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        version = result.stdout.strip()
        parts = version.split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
    return None


def is_sa_version_compatible(sa_version: str, torch_version: str) -> bool:
    """Check if SA version has wheels for the given PyTorch version.

    Based on the wheel matrix:
    - SA 2.2.0.postX: PyTorch >= 2.7 (ABI3 wheels)
    - SA 2.1.1: PyTorch 2.5-2.8 only (no 2.9+ support)
    - SA 1.0.6: Any PyTorch (from PyPI, no wheel needed)
    """
    if sa_version.startswith("1."):
        return True  # SA1 is always available from PyPI

    torch_parts = torch_version.split(".")
    torch_minor = int(torch_parts[1]) if len(torch_parts) >= 2 else 0

    if sa_version == "2.1.1":
        # SA 2.1.1 only has wheels for PyTorch 2.5-2.8
        return torch_minor <= 8

    if sa_version.startswith("2.2.0"):
        # SA 2.2.0+ has ABI3 wheels for PyTorch >= 2.7
        return torch_minor >= 7

    # Default: assume compatible
    return True


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
    """Run the installer with --upgrade flag.

    Returns:
        Tuple of (success: bool, output: str) - success status and full stdout
    """
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
                'successfully', 'error', 'failed', 'strategy', 'installed',
                'not available', 'alternatives', 'suggestions', 'requires pytorch'
            ]):
                print(line)

    if result.stderr:
        print(f"{Colors.RED}STDERR: {result.stderr}{Colors.END}")

    return result.returncode == 0, result.stdout


def run_test(test_num: int, description: str, setup_fn, upgrade_args: list,
             expected_version_contains: str, expect_failure: bool = False,
             expected_error_contains: list = None):
    """
    Run a single test case.

    Args:
        test_num: Test number for display
        description: Human-readable test description
        setup_fn: Function to call before upgrade (sets up initial state)
        upgrade_args: Additional arguments for --upgrade
        expected_version_contains: String that should be in final version (if success expected)
        expect_failure: If True, expect the install to fail with helpful error message
        expected_error_contains: List of strings that should appear in error output

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
    success, output = run_upgrade(upgrade_args)

    # Check result
    after_version = get_installed_version()
    print(f"After: {after_version or 'Not installed'}")

    passed = False

    if expect_failure:
        # We expect SA installation to fail (SA not installed), but with helpful error message
        # Note: The installer may still return success code because it's designed to continue without SA
        sa_not_installed = (after_version is None)

        if sa_not_installed:
            # Check that expected error messages are present
            missing_msgs = []
            if expected_error_contains:
                for msg in expected_error_contains:
                    if msg.lower() not in output.lower():
                        missing_msgs.append(msg)

            if not missing_msgs:
                print(f"\n{Colors.GREEN}[PASS] SA installation failed as expected with helpful error message{Colors.END}")
                passed = True
            else:
                print(f"\n{Colors.RED}[FAIL] SA not installed but missing expected messages: {missing_msgs}{Colors.END}")
        else:
            print(f"\n{Colors.RED}[FAIL] Expected SA to not be installed but got '{after_version}'{Colors.END}")
    else:
        # Normal case: expect success
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

    # Get current PyTorch version for compatibility checks
    torch_ver = get_pytorch_version()
    print(f"Detected PyTorch version: {torch_ver}")

    # Build test matrix with compatibility awareness
    # Each test: (test_num, description, setup_fn, upgrade_args, expected_version_contains,
    #             requires_sa_for_setup, test_error_behavior)
    # test_error_behavior: None = normal test, "error_message" = test error handling on incompatible env
    all_tests = [
        # Test 1: Primary use case - upgrade from SA 1.0.6 to SA 2.x
        (1, "Upgrade from SA 1.0.6 to SA 2.x (primary use case)",
         lambda: (uninstall_sageattention(), install_specific_version("1.0.6")),
         [],  # Just --upgrade, no other args
         "2.2.0",  # Should get latest SA2
         None,     # No specific SA version required for setup
         None),    # Normal test behavior

        # Test 2: Request --sage-version 2.1.1
        # On PyTorch <= 2.8: succeeds, installs 2.1.1
        # On PyTorch 2.9+: fails with helpful error message
        (2, "Request --sage-version 2.1.1 (error handling test on PyTorch 2.9+)",
         lambda: uninstall_sageattention(),
         ["--sage-version", "2.1.1"],
         "2.1.1",
         None,             # No setup dependency
         "error_message"), # Test error message behavior if incompatible

        # Test 3: Upgrade when nothing installed (should just install)
        (3, "Upgrade with no existing installation",
         lambda: uninstall_sageattention(),
         [],
         "2.2.0",  # Should install latest SA2
         None,
         None),

        # Test 4: Upgrade within SA2 (2.1.x to 2.2.x)
        # Note: SA 2.1.1 only has wheels for PyTorch <= 2.8, so can't test setup on 2.9+
        (4, "Upgrade from SA 2.1.1 to SA 2.2.x",
         lambda: (uninstall_sageattention(), install_via_installer("2.1.1")),
         [],  # Just --upgrade, should go to latest
         "2.2.0",  # Should upgrade to 2.2.0.post3
         "2.1.1",  # Requires SA 2.1.1 for setup (skip if not compatible)
         None),
    ]

    results = []

    for test_num, description, setup_fn, upgrade_args, expected, requires_sa_for_setup, test_error_behavior in all_tests:
        if args.test is not None and args.test != test_num:
            continue

        # Check if this test requires an SA version for SETUP that's not compatible
        # (e.g., Test 4 needs to install SA 2.1.1 first, which isn't possible on PyTorch 2.9+)
        if requires_sa_for_setup and torch_ver and not is_sa_version_compatible(requires_sa_for_setup, torch_ver):
            print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
            print(f"{Colors.YELLOW}TEST {test_num}: {description}{Colors.END}")
            print('='*70)
            print(f"{Colors.YELLOW}[SKIP] Test setup requires SA {requires_sa_for_setup} which isn't compatible with PyTorch {torch_ver}{Colors.END}")
            print(f"       SA 2.1.1 requires PyTorch <= 2.8 (you have {torch_ver})")
            results.append((test_num, description, None))  # None = skipped
            continue

        # Check if this test should verify error message behavior on incompatible env
        if test_error_behavior == "error_message":
            # Determine if SA version is compatible with current PyTorch
            requested_sa = None
            for i, arg in enumerate(upgrade_args):
                if arg == "--sage-version" and i + 1 < len(upgrade_args):
                    requested_sa = upgrade_args[i + 1]
                    break

            if requested_sa and torch_ver and not is_sa_version_compatible(requested_sa, torch_ver):
                # RUN THE TEST to verify error message - don't skip!
                description_mod = f"{description}"
                passed = run_test(
                    test_num, description_mod, setup_fn, upgrade_args, expected,
                    expect_failure=True,
                    expected_error_contains=[
                        "not available",
                        "alternatives",
                        "requires pytorch",
                    ]
                )
                results.append((test_num, description, passed))
                continue
            # else: SA is compatible, run as normal success test

        passed = run_test(test_num, description, setup_fn, upgrade_args, expected)
        results.append((test_num, description, passed))

    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.END}")
    print('='*70)

    passed_count = sum(1 for _, _, p in results if p is True)
    skipped_count = sum(1 for _, _, p in results if p is None)
    failed_count = sum(1 for _, _, p in results if p is False)
    total_count = len(results)

    for test_num, description, passed in results:
        if passed is None:
            status = f"{Colors.YELLOW}SKIP{Colors.END}"
        elif passed:
            status = f"{Colors.GREEN}PASS{Colors.END}"
        else:
            status = f"{Colors.RED}FAIL{Colors.END}"
        print(f"  Test {test_num}: [{status}] {description}")

    print('='*70)
    if failed_count == 0:
        if skipped_count > 0:
            print(f"{Colors.GREEN}{passed_count} passed, {skipped_count} skipped (incompatible SA versions){Colors.END}")
        else:
            print(f"{Colors.GREEN}All {total_count} tests passed!{Colors.END}")
    else:
        print(f"{Colors.RED}{passed_count} passed, {failed_count} failed, {skipped_count} skipped{Colors.END}")
        sys.exit(1)

    # Restore to stable version
    print(f"\n{Colors.YELLOW}Restoring to SA 2.2.0.post3...{Colors.END}")
    uninstall_sageattention()
    run_upgrade([])


if __name__ == "__main__":
    main()
