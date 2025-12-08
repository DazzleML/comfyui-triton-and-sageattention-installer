"""
One-off test script for validating --with-custom-nodes flag functionality.
Uses existing ComfyUI installation at C:\\code\\ComfyUI_2025.12.07

Tests the following scenarios:
1. Install without --with-custom-nodes (should NOT install custom nodes)
2. Install with --with-custom-nodes (should install VideoHelperSuite and DazzleNodes)

Usage:
    python tests/one-offs/test_custom_nodes.py [--test N]

    --test N    Only run test number N
"""
import subprocess
import sys
import argparse
import os
import shutil
from pathlib import Path

# Enable ANSI colors in Windows cmd.exe (Windows 10+)
os.system('')  # Enables VT100 escape codes

# Configuration
COMFYUI_PATH = Path(r"C:\code\ComfyUI_2025.12.07")
INSTALLER_PATH = Path(__file__).parent.parent.parent / "comfyui_triton_sageattention.py"
PYTHON_EXE = COMFYUI_PATH / "venv" / "Scripts" / "python.exe"
CUSTOM_NODES_PATH = COMFYUI_PATH / "ComfyUI" / "custom_nodes"


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def remove_readonly(func, path, excinfo):
    """Error handler for shutil.rmtree to handle read-only files (e.g., .git)."""
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_custom_nodes():
    """Remove test custom nodes if they exist."""
    nodes_to_remove = ["ComfyUI-VideoHelperSuite", "DazzleNodes"]
    for node in nodes_to_remove:
        node_path = CUSTOM_NODES_PATH / node
        if node_path.exists():
            print(f"{Colors.YELLOW}Removing {node}...{Colors.END}")
            shutil.rmtree(node_path, onerror=remove_readonly)


def check_custom_nodes_exist():
    """Check which custom nodes are installed."""
    nodes = {
        "VideoHelperSuite": CUSTOM_NODES_PATH / "ComfyUI-VideoHelperSuite",
        "DazzleNodes": CUSTOM_NODES_PATH / "DazzleNodes",
    }
    result = {}
    for name, path in nodes.items():
        result[name] = path.exists()
    return result


def run_install(with_custom_nodes: bool):
    """Run the installer."""
    cmd = [str(PYTHON_EXE), str(INSTALLER_PATH), "--install",
           "--base-path", str(COMFYUI_PATH),
           "--non-interactive"]

    if with_custom_nodes:
        cmd.append("--with-custom-nodes")

    print(f"Running: {' '.join(str(c) for c in cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print key output lines
    if result.stdout:
        for line in result.stdout.split('\n'):
            if any(kw in line.lower() for kw in [
                'custom nodes', 'videohelper', 'dazzle', 'cloning',
                'successfully', 'error', 'failed', 'installing'
            ]):
                print(line)

    if result.stderr:
        # Filter out common warnings
        stderr_lines = [l for l in result.stderr.split('\n')
                       if l and 'WARNING' not in l]
        if stderr_lines:
            print(f"{Colors.RED}STDERR: {chr(10).join(stderr_lines)}{Colors.END}")

    return result.returncode == 0


def run_test(test_num: int, description: str, setup_fn, with_custom_nodes: bool,
             expected_nodes: dict):
    """
    Run a single test case.

    Args:
        test_num: Test number for display
        description: Human-readable test description
        setup_fn: Function to call before install (sets up initial state)
        with_custom_nodes: Whether to use --with-custom-nodes flag
        expected_nodes: Dict of {node_name: should_exist}

    Returns:
        bool: True if test passed
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}TEST {test_num}: {description}{Colors.END}")
    print('='*70)

    # Setup
    if setup_fn:
        setup_fn()

    before_nodes = check_custom_nodes_exist()
    print(f"Before: {before_nodes}")

    # Run install
    success = run_install(with_custom_nodes)

    # Check result
    after_nodes = check_custom_nodes_exist()
    print(f"After: {after_nodes}")

    passed = True
    for node_name, should_exist in expected_nodes.items():
        actual = after_nodes.get(node_name, False)
        if actual != should_exist:
            print(f"\n{Colors.RED}[FAIL] {node_name}: expected {'installed' if should_exist else 'not installed'}, "
                  f"got {'installed' if actual else 'not installed'}{Colors.END}")
            passed = False

    if passed:
        print(f"\n{Colors.GREEN}[PASS] All node states match expected{Colors.END}")

    return passed


def main():
    parser = argparse.ArgumentParser(description='Test --with-custom-nodes flag functionality')
    parser.add_argument('--test', type=int, help='Only run specific test number')
    parser.add_argument('--skip-cleanup', action='store_true', help='Skip initial cleanup')
    args = parser.parse_args()

    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}--WITH-CUSTOM-NODES FLAG TEST SUITE{Colors.END}")
    print(f"ComfyUI Path: {COMFYUI_PATH}")
    print(f"Custom Nodes Path: {CUSTOM_NODES_PATH}")
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
        # (test_num, description, setup_fn, with_custom_nodes, expected_nodes)

        # Test 1: Install WITHOUT --with-custom-nodes (should NOT install custom nodes)
        (1, "Install without --with-custom-nodes (minimal install)",
         remove_custom_nodes,
         False,  # No --with-custom-nodes
         {"VideoHelperSuite": False, "DazzleNodes": False}),

        # Test 2: Install WITH --with-custom-nodes (should install both)
        (2, "Install with --with-custom-nodes",
         remove_custom_nodes,
         True,  # With --with-custom-nodes
         {"VideoHelperSuite": True, "DazzleNodes": True}),
    ]

    results = []

    for test_num, description, setup_fn, with_custom_nodes, expected in tests:
        if args.test is not None and args.test != test_num:
            continue

        passed = run_test(test_num, description, setup_fn, with_custom_nodes, expected)
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

    # Cleanup - remove test nodes
    if not args.skip_cleanup:
        print(f"\n{Colors.YELLOW}Cleaning up test nodes...{Colors.END}")
        remove_custom_nodes()


if __name__ == "__main__":
    main()
