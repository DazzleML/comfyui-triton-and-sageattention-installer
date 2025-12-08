"""
One-off test script for validating --show-installed flag functionality.
Uses existing ComfyUI installation at C:\\code\\ComfyUI_2025.12.07

Tests the following scenarios:
1. --show-installed displays formatted table
2. --show-installed works without other flags (standalone)
3. Output contains expected components (SageAttention, Triton, PyTorch, CUDA, Python)

Usage:
    python tests/one-offs/test_show_installed.py [--test N]

    --test N    Only run test number N
"""
import subprocess
import sys
import argparse
import os
import re
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


def run_show_installed():
    """Run --show-installed and return output."""
    cmd = [str(PYTHON_EXE), str(INSTALLER_PATH), "--show-installed",
           "--base-path", str(COMFYUI_PATH)]

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


def test_1_displays_table():
    """Test that --show-installed displays a formatted table."""
    result = run_show_installed()

    print(f"Return code: {result.returncode}")
    print(f"\nOutput:\n{result.stdout}")

    if result.stderr:
        print(f"{Colors.YELLOW}STDERR: {result.stderr}{Colors.END}")

    # Check return code
    if result.returncode != 0:
        return False, f"Non-zero return code: {result.returncode}"

    # Check for table structure
    if "Current Installation" not in result.stdout:
        return False, "Missing 'Current Installation' header"

    if "Component" not in result.stdout:
        return False, "Missing 'Component' column header"

    if "Version" not in result.stdout:
        return False, "Missing 'Version' column header"

    if "Status" not in result.stdout:
        return False, "Missing 'Status' column header"

    return True, "Table structure is correct"


def test_2_contains_all_components():
    """Test that output contains all expected components."""
    result = run_show_installed()

    if result.returncode != 0:
        return False, f"Non-zero return code: {result.returncode}"

    expected_components = ["SageAttention", "Triton", "PyTorch", "CUDA", "Python"]
    missing = []

    for component in expected_components:
        if component not in result.stdout:
            missing.append(component)

    if missing:
        return False, f"Missing components: {', '.join(missing)}"

    return True, f"All components present: {', '.join(expected_components)}"


def test_3_shows_version_numbers():
    """Test that version numbers are displayed (not just dashes)."""
    result = run_show_installed()

    if result.returncode != 0:
        return False, f"Non-zero return code: {result.returncode}"

    # Look for version patterns (digits with dots)
    version_pattern = r'\d+\.\d+'
    versions_found = re.findall(version_pattern, result.stdout)

    print(f"Versions found: {versions_found}")

    # We should have at least PyTorch and Python versions
    if len(versions_found) < 2:
        return False, f"Expected at least 2 version numbers, found {len(versions_found)}"

    return True, f"Found {len(versions_found)} version numbers"


def test_4_standalone_operation():
    """Test that --show-installed works without --install or --upgrade."""
    # Just verify it doesn't try to install anything
    result = run_show_installed()

    if result.returncode != 0:
        return False, f"Non-zero return code: {result.returncode}"

    # Should NOT contain installation messages
    bad_indicators = ["Installing", "Downloading", "Upgrading", "Step 1:", "Step 2:"]
    found_bad = []

    for indicator in bad_indicators:
        if indicator in result.stdout:
            found_bad.append(indicator)

    if found_bad:
        return False, f"Unexpected installation activity: {', '.join(found_bad)}"

    return True, "No installation activity detected (standalone operation confirmed)"


def main():
    parser = argparse.ArgumentParser(description='Test --show-installed flag functionality')
    parser.add_argument('--test', type=int, help='Only run specific test number')
    args = parser.parse_args()

    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}--SHOW-INSTALLED FLAG TEST SUITE{Colors.END}")
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
        (1, "Display formatted table", test_1_displays_table),
        (2, "Contains all expected components", test_2_contains_all_components),
        (3, "Shows version numbers", test_3_shows_version_numbers),
        (4, "Standalone operation (no install activity)", test_4_standalone_operation),
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
