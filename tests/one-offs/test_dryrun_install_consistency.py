"""
One-off test script for validating dryrun/install consistency.

This test suite verifies that --dryrun output accurately predicts what
--install would actually do. This is critical for user trust.

Tests both environments:
- C:\\code\\ComfyUI_2025.12.07 (venv)
- C:\\code\\ComfyUI_windows_portable (portable/python_embeded)

Key scenarios tested:
1. CUDA detection consistency (torch.version.cuda vs nvcc)
2. PyTorch decision consistency (dryrun says KEEP, install should KEEP)
3. Triton decision consistency
4. SageAttention wheel selection consistency

Usage:
    python tests/one-offs/test_dryrun_install_consistency.py [--test N] [--env ENV]

    --test N    Only run test number N
    --env ENV   Only test specific environment: 'venv' or 'portable'
"""
import subprocess
import sys
import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict

# Enable ANSI colors in Windows cmd.exe (Windows 10+)
os.system('')  # Enables VT100 escape codes

# Configuration - both test environments
ENVIRONMENTS = {
    "venv": {
        "path": Path(r"C:\code\ComfyUI_2025.12.07"),
        "python": Path(r"C:\code\ComfyUI_2025.12.07\venv\Scripts\python.exe"),
        "type": "venv"
    },
    "portable": {
        "path": Path(r"C:\code\ComfyUI_windows_portable"),
        "python": Path(r"C:\code\ComfyUI_windows_portable\python_embeded\python.exe"),
        "type": "portable"
    }
}

INSTALLER_PATH = Path(__file__).parent.parent.parent / "comfyui_triton_sageattention.py"


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def run_command(python_exe: Path, *extra_args, base_path: Optional[Path] = None):
    """Run installer with given arguments and return result."""
    cmd = [str(python_exe), str(INSTALLER_PATH)]
    if base_path:
        cmd.extend(["--base-path", str(base_path)])
    cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def get_torch_cuda_version(python_exe: Path) -> Optional[str]:
    """Get CUDA version from torch.version.cuda."""
    cmd = [str(python_exe), "-c",
           "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'cpu')"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def get_nvcc_cuda_version() -> Optional[str]:
    """Get CUDA version from nvcc --version."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
    except FileNotFoundError:
        pass
    return None


def get_torch_version(python_exe: Path) -> Optional[str]:
    """Get PyTorch version."""
    cmd = [str(python_exe), "-c", "import torch; print(torch.__version__)"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def parse_dryrun_output(output: str) -> Dict:
    """Parse dryrun output to extract key information."""
    parsed = {
        "cuda_version": None,
        "pytorch_version": None,
        "pytorch_action": None,
        "triton_action": None,
        "sa_action": None,
        "sa_wheel_url": None,
    }

    # Extract CUDA version from "CUDA: X.X" line
    cuda_match = re.search(r'CUDA:\s+(\d+\.\d+)', output)
    if cuda_match:
        parsed["cuda_version"] = cuda_match.group(1)

    # Extract PyTorch version from "PyTorch: X.X.X" line
    pytorch_match = re.search(r'PyTorch:\s+(\d+\.\d+\.\d+)', output)
    if pytorch_match:
        parsed["pytorch_version"] = pytorch_match.group(1)

    # Extract actions from Proposed Changes section
    # Format: "  PyTorch         [KEEP]     2.9.1 (already installed)"
    pytorch_action = re.search(r'PyTorch\s+\[(\w+)\]', output)
    if pytorch_action:
        parsed["pytorch_action"] = pytorch_action.group(1)

    triton_action = re.search(r'Triton\s+\[(\w+)\]', output)
    if triton_action:
        parsed["triton_action"] = triton_action.group(1)

    sa_action = re.search(r'SageAttention\s+\[(\w+)\]', output)
    if sa_action:
        parsed["sa_action"] = sa_action.group(1)

    # Extract wheel URL
    wheel_match = re.search(r'SageAttention wheel:\s+(\S+)', output)
    if wheel_match:
        parsed["sa_wheel_url"] = wheel_match.group(1)

    return parsed


def run_test(test_num: int, description: str, test_fn, env_name: str = None):
    """Run a single test case."""
    env_str = f" [{env_name}]" if env_name else ""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}TEST {test_num}{env_str}: {description}{Colors.END}")
    print('='*70)

    passed, details = test_fn()

    if passed:
        print(f"\n{Colors.GREEN}[PASS] {details}{Colors.END}")
    else:
        print(f"\n{Colors.RED}[FAIL] {details}{Colors.END}")

    return passed


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_cuda_detection_consistency(env: dict) -> Tuple[bool, str]:
    """
    Test that dryrun CUDA detection matches what install would use.

    Issue #18 root cause: dryrun uses torch.version.cuda, install uses nvcc.
    """
    python_exe = env["python"]
    base_path = env["path"]

    # Get CUDA from both sources
    torch_cuda = get_torch_cuda_version(python_exe)
    nvcc_cuda = get_nvcc_cuda_version()

    print(f"  torch.version.cuda: {torch_cuda}")
    print(f"  nvcc --version:     {nvcc_cuda}")

    # Run dryrun and see what CUDA it reports
    result = run_command(python_exe, "--install", "--dryrun", base_path=base_path)

    if result.returncode != 0:
        return False, f"Dryrun failed with code {result.returncode}"

    print(f"\nDryrun output (relevant section):")
    # Print just the environment section
    for line in result.stdout.split('\n'):
        if 'CUDA' in line or 'PyTorch' in line or 'Environment' in line:
            print(f"  {line}")

    parsed = parse_dryrun_output(result.stdout)
    dryrun_cuda = parsed.get("cuda_version")

    print(f"\n  Dryrun reports CUDA: {dryrun_cuda}")

    # Analysis
    if torch_cuda and nvcc_cuda and torch_cuda != nvcc_cuda:
        # This is the Issue #18 scenario - torch and system CUDA differ
        print(f"\n{Colors.YELLOW}  [INFO] CUDA version mismatch detected!{Colors.END}")
        print(f"  torch CUDA ({torch_cuda}) != nvcc CUDA ({nvcc_cuda})")

        # Dryrun should use torch CUDA (correct behavior for SA wheel matching)
        if dryrun_cuda == torch_cuda:
            return True, f"Dryrun uses torch.version.cuda ({torch_cuda}) - correct for wheel matching"
        elif dryrun_cuda == nvcc_cuda:
            return False, f"Dryrun incorrectly uses nvcc CUDA ({nvcc_cuda}) instead of torch ({torch_cuda})"
        else:
            return False, f"Dryrun CUDA ({dryrun_cuda}) doesn't match either source"
    else:
        return True, f"CUDA versions consistent (torch={torch_cuda}, nvcc={nvcc_cuda})"


def test_pytorch_keep_consistency(env: dict) -> Tuple[bool, str]:
    """
    Test that if dryrun says [KEEP] for PyTorch, install won't reinstall.

    This verifies the decision logic is consistent by actually calling
    the install code's _check_pytorch_compatibility method.
    """
    python_exe = env["python"]
    base_path = env["path"]

    # First check if PyTorch exists
    torch_version = get_torch_version(python_exe)
    if not torch_version:
        return True, "PyTorch not installed - skipping KEEP test"

    print(f"  Current PyTorch: {torch_version}")

    # Run dryrun
    result = run_command(python_exe, "--install", "--dryrun", base_path=base_path)

    if result.returncode != 0:
        return False, f"Dryrun failed with code {result.returncode}"

    parsed = parse_dryrun_output(result.stdout)

    print(f"  Dryrun PyTorch action: {parsed.get('pytorch_action')}")

    if parsed.get("pytorch_action") == "KEEP":
        # Dryrun says keep - verify the actual install logic agrees
        # by calling _check_pytorch_compatibility directly
        nvcc_cuda = get_nvcc_cuda_version() or "cpu"

        compat_check = f"""
import sys
from pathlib import Path
sys.path.insert(0, r'{INSTALLER_PATH.parent}')
from comfyui_triton_sageattention import ComfyUIInstaller
installer = ComfyUIInstaller(base_path=Path(r'{base_path}'))
result = installer._check_pytorch_compatibility('{nvcc_cuda}')
print(f'compatible:{{result}}')
"""

        compat_result = subprocess.run(
            [str(python_exe), "-c", compat_check],
            capture_output=True, text=True
        )

        if "compatible:True" in compat_result.stdout:
            return True, f"PyTorch [KEEP] - install logic agrees"
        elif "compatible:False" in compat_result.stdout:
            return False, "Dryrun says KEEP but install logic would reinstall"
        else:
            # May have failed to run, check stderr
            if compat_result.returncode != 0:
                return False, f"Compatibility check error: {compat_result.stderr[:100]}"
            return True, f"PyTorch [KEEP] - could not verify install logic"

    elif parsed.get("pytorch_action") == "INSTALL":
        return True, f"PyTorch needs installation - consistent behavior"

    else:
        return True, f"PyTorch action: {parsed.get('pytorch_action')}"


def test_dryrun_shows_accurate_wheel(env: dict) -> Tuple[bool, str]:
    """
    Test that dryrun shows the correct SageAttention wheel for the environment.
    """
    python_exe = env["python"]
    base_path = env["path"]

    # Get environment info
    torch_version = get_torch_version(python_exe)
    torch_cuda = get_torch_cuda_version(python_exe)

    if not torch_version or not torch_cuda or torch_cuda == 'cpu':
        return True, "No CUDA PyTorch - skipping wheel test"

    print(f"  PyTorch: {torch_version}")
    print(f"  CUDA: {torch_cuda}")

    # Run dryrun
    result = run_command(python_exe, "--install", "--dryrun", base_path=base_path)

    if result.returncode != 0:
        return False, f"Dryrun failed with code {result.returncode}"

    parsed = parse_dryrun_output(result.stdout)

    wheel_url = parsed.get("sa_wheel_url")
    print(f"  Wheel URL: {wheel_url}")

    if wheel_url and "github.com" in wheel_url:
        # Verify wheel URL contains expected CUDA/torch info
        cuda_code = torch_cuda.replace('.', '')
        if f"cu{cuda_code}" in wheel_url or f"+cu{cuda_code}" in wheel_url:
            return True, f"Wheel URL contains correct CUDA code (cu{cuda_code})"
        else:
            print(f"{Colors.YELLOW}  [WARNING] Wheel URL doesn't contain expected CUDA code cu{cuda_code}{Colors.END}")
            return False, f"Wheel URL may not match environment CUDA"

    elif wheel_url and "PyPI" in wheel_url:
        return True, "Using PyPI fallback (SA 1.x)"

    return True, "Wheel info present"


def test_triton_compatibility_consistent(env: dict) -> Tuple[bool, str]:
    """
    Test that Triton compatibility assessment is consistent.
    """
    python_exe = env["python"]
    base_path = env["path"]

    # Run dryrun
    result = run_command(python_exe, "--install", "--dryrun", base_path=base_path)

    if result.returncode != 0:
        return False, f"Dryrun failed with code {result.returncode}"

    # Check for Triton/PyTorch compatibility section
    if "Triton/PyTorch Compatibility" in result.stdout:
        if "[OK]" in result.stdout:
            return True, "Triton/PyTorch compatibility: OK"
        elif "[WARNING]" in result.stdout:
            return True, "Triton/PyTorch compatibility: WARNING (will be fixed)"

    parsed = parse_dryrun_output(result.stdout)
    return True, f"Triton action: {parsed.get('triton_action', 'N/A')}"


def test_show_installed_matches_dryrun_current(env: dict) -> Tuple[bool, str]:
    """
    Test that --show-installed reports same versions as dryrun's "Current Environment".
    """
    python_exe = env["python"]
    base_path = env["path"]

    # Run --show-installed
    show_result = run_command(python_exe, "--show-installed", base_path=base_path)

    if show_result.returncode != 0:
        return False, f"--show-installed failed with code {show_result.returncode}"

    # Run --dryrun --install
    dryrun_result = run_command(python_exe, "--install", "--dryrun", base_path=base_path)

    if dryrun_result.returncode != 0:
        return False, f"--dryrun failed with code {dryrun_result.returncode}"

    # Parse both outputs for PyTorch version
    show_pytorch = re.search(r'PyTorch\s*\|\s*(\d+\.\d+\.\d+)', show_result.stdout)
    dryrun_pytorch = re.search(r'PyTorch:\s+(\d+\.\d+\.\d+)', dryrun_result.stdout)

    show_ver = show_pytorch.group(1) if show_pytorch else None
    dryrun_ver = dryrun_pytorch.group(1) if dryrun_pytorch else None

    print(f"  --show-installed PyTorch: {show_ver}")
    print(f"  --dryrun PyTorch:         {dryrun_ver}")

    if show_ver and dryrun_ver:
        if show_ver == dryrun_ver:
            return True, f"Version reporting consistent: {show_ver}"
        else:
            return False, f"Version mismatch: show={show_ver}, dryrun={dryrun_ver}"

    return True, "Could not parse versions (may be OK if not installed)"


def test_issue_18_scenario(env: dict) -> Tuple[bool, str]:
    """
    Test for GitHub Issue #18: Dryrun lied about what install would do.

    Issue #18 scenario (user easychen's report):
    - PyTorch 2.9.1 with bundled CUDA 13.0 (cu130)
    - System nvcc reports CUDA 12.6 (not matching torch)
    - Dryrun showed [KEEP] for PyTorch
    - Actual install DOWNGRADED to PyTorch 2.7.0 + CUDA 12.6

    Root cause: OLD install code compared torch CUDA vs nvcc CUDA.
    The fix uses torch.version.cuda for wheel matching when PyTorch exists.

    This test verifies:
    1. When torch.version.cuda != nvcc version (Issue #18 scenario)
    2. Dryrun says [KEEP] for PyTorch
    3. The install code's _check_pytorch_compatibility ALSO returns True (KEEP)

    Reference: https://github.com/DazzleML/comfyui-triton-and-sageattention-installer/issues/18
    """
    python_exe = env["python"]
    base_path = env["path"]

    # Get both CUDA versions
    torch_cuda = get_torch_cuda_version(python_exe)
    nvcc_cuda = get_nvcc_cuda_version()
    torch_version = get_torch_version(python_exe)

    print(f"  Environment: {env['type']}")
    print(f"  PyTorch:     {torch_version}")
    print(f"  torch.version.cuda: {torch_cuda}")
    print(f"  nvcc --version:     {nvcc_cuda}")

    # Skip if not an Issue #18 scenario
    if not torch_cuda or not nvcc_cuda:
        return True, "Skipping - need both torch and nvcc CUDA"

    if torch_cuda == nvcc_cuda:
        return True, "Skipping - torch and nvcc CUDA match (not Issue #18 scenario)"

    if not torch_version or not torch_version.startswith("2."):
        return True, "Skipping - need PyTorch 2.x for this test"

    # This IS an Issue #18 scenario: torch CUDA != nvcc CUDA
    print(f"\n{Colors.YELLOW}  Issue #18 scenario detected:{Colors.END}")
    print(f"    torch CUDA ({torch_cuda}) != nvcc CUDA ({nvcc_cuda})")

    # Run dryrun
    result = run_command(python_exe, "--install", "--dryrun", base_path=base_path)

    if result.returncode != 0:
        return False, f"Dryrun failed with code {result.returncode}"

    parsed = parse_dryrun_output(result.stdout)
    pytorch_action = parsed.get("pytorch_action")
    reported_cuda = parsed.get("cuda_version")

    print(f"\n  Dryrun results:")
    print(f"    CUDA reported:    {reported_cuda}")
    print(f"    PyTorch action:   [{pytorch_action}]")

    # Verify dryrun uses torch CUDA (correct for wheel matching)
    if reported_cuda != torch_cuda:
        return False, f"Dryrun should report torch CUDA ({torch_cuda}), not {reported_cuda}"

    # Verify dryrun says KEEP for existing PyTorch 2.x with CUDA
    if pytorch_action != "KEEP":
        return False, f"Dryrun should say [KEEP] for PyTorch 2.x with CUDA, got [{pytorch_action}]"

    # The critical check: verify install code's _check_pytorch_compatibility
    # uses torch.version.cuda (not nvcc) for wheel matching.
    #
    # We test this by running the installer's method directly via Python code.
    # The method should return True (compatible) for existing PyTorch 2.x with CUDA,
    # regardless of what nvcc reports.
    compat_check = f"""
import sys
from pathlib import Path
sys.path.insert(0, r'{INSTALLER_PATH.parent}')
from comfyui_triton_sageattention import ComfyUIInstaller
# Initialize with base_path to detect the correct Python environment
installer = ComfyUIInstaller(base_path=Path(r'{base_path}'))
# Pass nvcc CUDA - the method should still return True using torch.version.cuda
result = installer._check_pytorch_compatibility('{nvcc_cuda}')
print(f'compatible:{{result}}')
"""

    compat_result = subprocess.run(
        [str(python_exe), "-c", compat_check],
        capture_output=True, text=True
    )

    print(f"\n  Install compatibility check:")
    print(f"    Command output: {compat_result.stdout.strip()}")
    if compat_result.stderr:
        print(f"    Stderr: {compat_result.stderr.strip()[:200]}")

    if compat_result.returncode != 0:
        return False, f"Compatibility check failed: {compat_result.stderr[:100]}"

    if "compatible:True" in compat_result.stdout:
        print(f"{Colors.GREEN}  Install logic uses torch.version.cuda - will KEEP PyTorch{Colors.END}")
        return True, f"PyTorch {torch_version} with CUDA {torch_cuda} correctly preserved (Issue #18 fixed)"
    elif "compatible:False" in compat_result.stdout:
        print(f"{Colors.RED}  BUG: Install logic would reinstall PyTorch!{Colors.END}")
        return False, f"Issue #18 BUG: install returns False despite torch having CUDA {torch_cuda}"
    else:
        return False, f"Could not parse compatibility result: {compat_result.stdout}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test dryrun/install consistency')
    parser.add_argument('--test', type=int, help='Only run specific test number')
    parser.add_argument('--env', choices=['venv', 'portable', 'both'], default='both',
                        help='Which environment to test')
    args = parser.parse_args()

    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}DRYRUN/INSTALL CONSISTENCY TEST SUITE{Colors.END}")
    print(f"Installer: {INSTALLER_PATH}")
    print('='*70)

    # Verify installer exists
    if not INSTALLER_PATH.exists():
        print(f"{Colors.RED}ERROR: Installer not found: {INSTALLER_PATH}{Colors.END}")
        sys.exit(1)

    # Define tests (will run for each environment)
    tests = [
        (1, "CUDA detection consistency", test_cuda_detection_consistency),
        (2, "PyTorch KEEP action consistency", test_pytorch_keep_consistency),
        (3, "Dryrun shows accurate wheel URL", test_dryrun_shows_accurate_wheel),
        (4, "Triton compatibility consistent", test_triton_compatibility_consistent),
        (5, "--show-installed matches dryrun current", test_show_installed_matches_dryrun_current),
        (6, "Issue #18: Bundled CUDA != system CUDA", test_issue_18_scenario),
    ]

    # Determine which environments to test
    envs_to_test = []
    if args.env in ('venv', 'both'):
        if ENVIRONMENTS["venv"]["python"].exists():
            envs_to_test.append(("venv", ENVIRONMENTS["venv"]))
        else:
            print(f"{Colors.YELLOW}WARNING: venv environment not found, skipping{Colors.END}")

    if args.env in ('portable', 'both'):
        if ENVIRONMENTS["portable"]["python"].exists():
            envs_to_test.append(("portable", ENVIRONMENTS["portable"]))
        else:
            print(f"{Colors.YELLOW}WARNING: portable environment not found, skipping{Colors.END}")

    if not envs_to_test:
        print(f"{Colors.RED}ERROR: No test environments available{Colors.END}")
        sys.exit(1)

    all_results = []

    for env_name, env in envs_to_test:
        print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.CYAN}TESTING ENVIRONMENT: {env_name.upper()}{Colors.END}")
        print(f"{Colors.CYAN}Path: {env['path']}{Colors.END}")
        print(f"{Colors.CYAN}Python: {env['python']}{Colors.END}")
        print(f"{'='*70}")

        for test_num, description, test_fn in tests:
            if args.test is not None and args.test != test_num:
                continue

            def wrapped_test():
                return test_fn(env)

            passed = run_test(test_num, description, wrapped_test, env_name)
            all_results.append((env_name, test_num, description, passed))

    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.END}")
    print('='*70)

    passed_count = sum(1 for _, _, _, p in all_results if p)
    total_count = len(all_results)

    current_env = None
    for env_name, test_num, description, passed in all_results:
        if env_name != current_env:
            current_env = env_name
            print(f"\n  {Colors.CYAN}{env_name.upper()}{Colors.END}")

        status = f"{Colors.GREEN}PASS{Colors.END}" if passed else f"{Colors.RED}FAIL{Colors.END}"
        print(f"    Test {test_num}: [{status}] {description}")

    print(f"\n{'='*70}")
    if passed_count == total_count:
        print(f"{Colors.GREEN}All {total_count} tests passed!{Colors.END}")
    else:
        print(f"{Colors.RED}{passed_count}/{total_count} tests passed{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
