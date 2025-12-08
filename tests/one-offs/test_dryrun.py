"""
One-off test script for validating --dryrun flag functionality.
Uses existing ComfyUI installation at C:\\code\\ComfyUI_2025.12.07

Tests the following scenarios:
1. --dryrun with --install displays preview
2. --dryrun with --upgrade displays preview
3. --dryrun without --install/--upgrade shows error
4. --dryrun does not modify anything (no pip install, no git clone)
5. Output contains expected sections (Current Environment, Proposed Changes, Wheel Details)

Usage:
    python tests/one-offs/test_dryrun.py [--test N]

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


def run_installer(*extra_args):
    """Run installer with given arguments and return result."""
    cmd = [str(PYTHON_EXE), str(INSTALLER_PATH),
           "--base-path", str(COMFYUI_PATH)] + list(extra_args)

    print(f"Running: {' '.join(str(c) for c in cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def run_test(test_num: int, description: str, test_fn):
    """
    Run a single test case.

    Args:
        test_num: Test number for display
        description: Human-readable test description
        test_fn: Function that returns (passed: bool, details: str)

    Returns:
        bool: True if test passed
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}TEST {test_num}: {description}{Colors.END}")
    print('='*70)

    passed, details = test_fn()

    if passed:
        print(f"\n{Colors.GREEN}[PASS] {details}{Colors.END}")
    else:
        print(f"\n{Colors.RED}[FAIL] {details}{Colors.END}")

    return passed


def test_1_dryrun_with_install():
    """Test that --dryrun --install displays preview."""
    result = run_installer("--install", "--dryrun")

    print(f"Return code: {result.returncode}")
    print(f"\nOutput:\n{result.stdout}")

    if result.stderr:
        print(f"{Colors.YELLOW}STDERR: {result.stderr}{Colors.END}")

    if result.returncode != 0:
        return False, f"Non-zero return code: {result.returncode}"

    # Check for expected output sections
    if "DRY RUN" not in result.stdout:
        return False, "Missing 'DRY RUN' header"

    if "Current Environment" not in result.stdout:
        return False, "Missing 'Current Environment' section"

    if "Proposed Changes" not in result.stdout:
        return False, "Missing 'Proposed Changes' section"

    if "Wheel Details" not in result.stdout:
        return False, "Missing 'Wheel Details' section"

    # Should have either "run without --dryrun" (if changes needed) or "Nothing to do" (if up to date)
    if "run without --dryrun" not in result.stdout and "Nothing to do" not in result.stdout:
        return False, "Missing final status message (expected 'run without --dryrun' or 'Nothing to do')"

    return True, "--dryrun --install produces correct preview output"


def test_2_dryrun_with_upgrade():
    """Test that --dryrun --upgrade displays preview."""
    result = run_installer("--upgrade", "--dryrun")

    if result.returncode != 0:
        return False, f"Non-zero return code: {result.returncode}"

    if "DRY RUN" not in result.stdout:
        return False, "Missing 'DRY RUN' header"

    # With --upgrade, should show [UPGRADE] actions
    # (may show [KEEP] if already latest, but structure should be there)
    if "Proposed Changes" not in result.stdout:
        return False, "Missing 'Proposed Changes' section"

    return True, "--dryrun --upgrade produces correct preview output"


def test_3_dryrun_without_action():
    """Test that --dryrun without --install/--upgrade shows error."""
    result = run_installer("--dryrun")

    # Should fail with error message
    if result.returncode == 0:
        return False, "Expected non-zero return code but got 0"

    if "requires --install or --upgrade" not in result.stderr:
        return False, f"Expected error message about requiring --install/--upgrade, got: {result.stderr}"

    return True, "--dryrun without action correctly shows error"


def test_4_dryrun_no_modifications():
    """Test that --dryrun doesn't actually install anything."""
    result = run_installer("--install", "--dryrun")

    if result.returncode != 0:
        return False, f"Non-zero return code: {result.returncode}"

    # Check that no actual installation commands were run
    bad_indicators = [
        "pip install",
        "Successfully installed",
        "git clone",
        "Cloning into",
        "Installing collected packages",
        "Step 1:",  # Installation step indicators
        "Step 2:",
        "Step 3:",
    ]

    found_bad = []
    for indicator in bad_indicators:
        if indicator in result.stdout:
            found_bad.append(indicator)

    if found_bad:
        return False, f"Found installation activity: {', '.join(found_bad)}"

    return True, "No actual installation activity detected (dry run confirmed)"


def test_5_contains_action_tags():
    """Test that output contains action tags like [INSTALL], [KEEP], [UPGRADE]."""
    result = run_installer("--install", "--dryrun")

    if result.returncode != 0:
        return False, f"Non-zero return code: {result.returncode}"

    # Should have action tags in output
    has_install = "[INSTALL]" in result.stdout
    has_keep = "[KEEP]" in result.stdout
    has_upgrade = "[UPGRADE]" in result.stdout

    if not (has_install or has_keep or has_upgrade):
        return False, "Missing action tags ([INSTALL], [KEEP], or [UPGRADE])"

    found_tags = []
    if has_install:
        found_tags.append("[INSTALL]")
    if has_keep:
        found_tags.append("[KEEP]")
    if has_upgrade:
        found_tags.append("[UPGRADE]")

    return True, f"Found action tags: {', '.join(found_tags)}"


def test_6_shows_compatibility_status():
    """Test that dryrun shows SA 2.x compatibility."""
    result = run_installer("--install", "--dryrun")

    if result.returncode != 0:
        return False, f"Non-zero return code: {result.returncode}"

    # Should mention which SA version will be installed
    has_sa2 = "SA 2" in result.stdout or "2.2.0" in result.stdout or "2.1.1" in result.stdout
    has_sa1 = "SA 1" in result.stdout or "1.0.6" in result.stdout

    if not (has_sa2 or has_sa1):
        return False, "Missing SageAttention version information"

    if "Wheel Details" not in result.stdout:
        return False, "Missing 'Wheel Details' section"

    return True, "Shows SageAttention version and wheel details"


def main():
    parser = argparse.ArgumentParser(description='Test --dryrun flag functionality')
    parser.add_argument('--test', type=int, help='Only run specific test number')
    args = parser.parse_args()

    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}--DRYRUN FLAG TEST SUITE{Colors.END}")
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
        (1, "--dryrun --install displays preview", test_1_dryrun_with_install),
        (2, "--dryrun --upgrade displays preview", test_2_dryrun_with_upgrade),
        (3, "--dryrun without action shows error", test_3_dryrun_without_action),
        (4, "--dryrun does not modify anything", test_4_dryrun_no_modifications),
        (5, "Output contains action tags", test_5_contains_action_tags),
        (6, "Shows compatibility and wheel info", test_6_shows_compatibility_status),
    ]

    results = []

    for test_num, description, test_fn in tests:
        if args.test is not None and args.test != test_num:
            continue

        passed = run_test(test_num, description, test_fn)
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
