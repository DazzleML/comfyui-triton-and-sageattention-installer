#!/usr/bin/env python
"""
Test Driver Script
==================

Unified test runner for comfyui-triton-sageattention-installer.

Handles two types of tests:
1. Unit tests (./unit/) - pytest-based mocked tests, safe to run anywhere
2. One-off tests (./one-offs/) - real-environment integration tests

IMPORTANT: One-off tests require ComfyUI installations to test against.
These paths are configurable via environment variables or test_config.json.

Configuration:
--------------
Option 1: Environment variables
    set COMFYUI_VENV_PATH=C:\\path\\to\\ComfyUI_with_venv
    set COMFYUI_PORTABLE_PATH=C:\\path\\to\\ComfyUI_windows_portable

Option 2: Create tests/test_config.json
    {
        "environments": {
            "venv": {
                "path": "C:/path/to/ComfyUI_with_venv",
                "python": "C:/path/to/ComfyUI_with_venv/venv/Scripts/python.exe"
            },
            "portable": {
                "path": "C:/path/to/ComfyUI_windows_portable",
                "python": "C:/path/to/ComfyUI_windows_portable/python_embeded/python.exe"
            }
        }
    }

Usage:
    python tests/run_tests.py              # Run all tests (with guards)
    python tests/run_tests.py --unit       # Run only unit tests (always safe)
    python tests/run_tests.py --oneoff     # Run only one-off tests
    python tests/run_tests.py --quick      # Run unit tests only (alias)
    python tests/run_tests.py --env venv   # One-offs: test only venv environment
    python tests/run_tests.py --check      # Check environment configuration only
    python tests/run_tests.py -v           # Verbose output
"""
import subprocess
import sys
import argparse
import json
from pathlib import Path
import os

# Enable ANSI colors in Windows cmd.exe
os.system('')

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

TESTS_DIR = Path(__file__).parent
PROJECT_DIR = TESTS_DIR.parent
CONFIG_FILE = TESTS_DIR / "test_config.json"

# Default paths (developer's environment - will be validated before use)
DEFAULT_ENVIRONMENTS = {
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


def load_config():
    """Load test configuration from file or environment variables."""
    environments = {}

    # Try loading from config file first
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
            for name, env in config.get("environments", {}).items():
                environments[name] = {
                    "path": Path(env["path"]),
                    "python": Path(env["python"]),
                    "type": env.get("type", name)
                }
            if environments:
                return environments
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{YELLOW}Warning: Could not parse {CONFIG_FILE}: {e}{RESET}")

    # Try environment variables
    venv_path = os.environ.get("COMFYUI_VENV_PATH")
    portable_path = os.environ.get("COMFYUI_PORTABLE_PATH")

    if venv_path:
        venv_path = Path(venv_path)
        environments["venv"] = {
            "path": venv_path,
            "python": venv_path / "venv" / "Scripts" / "python.exe",
            "type": "venv"
        }

    if portable_path:
        portable_path = Path(portable_path)
        environments["portable"] = {
            "path": portable_path,
            "python": portable_path / "python_embeded" / "python.exe",
            "type": "portable"
        }

    if environments:
        return environments

    # Fall back to defaults
    return DEFAULT_ENVIRONMENTS


def validate_environments(environments: dict) -> dict:
    """Validate which environments actually exist and are usable."""
    valid = {}
    invalid = {}

    for name, env in environments.items():
        path = env["path"]
        python = env["python"]

        if not path.exists():
            invalid[name] = f"Path does not exist: {path}"
        elif not python.exists():
            invalid[name] = f"Python not found: {python}"
        else:
            valid[name] = env

    return valid, invalid


def print_environment_status(valid: dict, invalid: dict):
    """Print environment configuration status."""
    print(f"\n{BOLD}Environment Configuration{RESET}")
    print("=" * 50)

    if valid:
        print(f"\n{GREEN}Available environments:{RESET}")
        for name, env in valid.items():
            print(f"  {name}:")
            print(f"    Path:   {env['path']}")
            print(f"    Python: {env['python']}")

    if invalid:
        print(f"\n{YELLOW}Unavailable environments:{RESET}")
        for name, reason in invalid.items():
            print(f"  {name}: {reason}")

    if not valid:
        print(f"\n{RED}No valid test environments found!{RESET}")
        print(f"\n{BOLD}To configure test environments:{RESET}")
        print(f"\nOption 1: Set environment variables:")
        print(f"  set COMFYUI_VENV_PATH=C:\\path\\to\\ComfyUI_with_venv")
        print(f"  set COMFYUI_PORTABLE_PATH=C:\\path\\to\\ComfyUI_portable")
        print(f"\nOption 2: Create {CONFIG_FILE}")
        print(f"  See script header for JSON format")
        print(f"\nOption 3: Run only unit tests (no environment needed):")
        print(f"  python tests/run_tests.py --unit")


def find_python_with_pytest(environments: dict):
    """Find a Python interpreter with pytest available."""
    # First try valid environments
    for name, env in environments.items():
        python = env["python"]
        if python.exists():
            try:
                result = subprocess.run(
                    [str(python), "-c", "import pytest"],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return python
            except Exception:
                pass

    # Fall back to current Python
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import pytest"],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            return Path(sys.executable)
    except Exception:
        pass

    return None


def run_unit_tests(python: Path, verbose: bool = False) -> bool:
    """Run pytest unit tests (safe - uses mocking, no real environments)."""
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{CYAN}RUNNING UNIT TESTS (pytest - mocked, safe){RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

    unit_dir = TESTS_DIR / "unit"
    if not unit_dir.exists():
        print(f"{YELLOW}No unit tests found in {unit_dir}{RESET}")
        return True

    if python is None:
        print(f"{RED}No Python with pytest found. Install pytest:{RESET}")
        print(f"  pip install pytest")
        return False

    cmd = [str(python), "-m", "pytest", str(unit_dir)]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "-q"])

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))

    if result.returncode == 0:
        print(f"\n{GREEN}Unit tests: PASSED{RESET}")
        return True
    else:
        print(f"\n{RED}Unit tests: FAILED{RESET}")
        return False


def run_oneoff_tests(python: Path, valid_envs: dict, env_filter: str = None,
                     verbose: bool = False) -> bool:
    """Run one-off integration tests against real environments."""
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{CYAN}RUNNING ONE-OFF TESTS (integration - real environments){RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

    if not valid_envs:
        print(f"{RED}No valid test environments configured!{RESET}")
        print(f"One-off tests require real ComfyUI installations.")
        print(f"Run with --check to see configuration options.")
        return False

    # Filter environments if specified
    if env_filter:
        if env_filter not in valid_envs:
            print(f"{RED}Environment '{env_filter}' not available.{RESET}")
            print(f"Available: {', '.join(valid_envs.keys())}")
            return False
        test_envs = {env_filter: valid_envs[env_filter]}
    else:
        test_envs = valid_envs

    print(f"Testing against: {', '.join(test_envs.keys())}")

    oneoff_dir = TESTS_DIR / "one-offs"
    if not oneoff_dir.exists():
        print(f"{YELLOW}No one-off tests found in {oneoff_dir}{RESET}")
        return True

    # Find test scripts
    test_scripts = sorted(oneoff_dir.glob("test_*.py"))
    if not test_scripts:
        print(f"{YELLOW}No test_*.py files found in {oneoff_dir}{RESET}")
        return True

    all_passed = True

    for script in test_scripts:
        print(f"\n{BOLD}Running: {script.name}{RESET}")
        print("-" * 50)

        cmd = [str(python), str(script)]
        if env_filter:
            cmd.extend(["--env", env_filter])

        result = subprocess.run(cmd, cwd=str(PROJECT_DIR))

        if result.returncode != 0:
            all_passed = False
            print(f"{RED}FAILED: {script.name}{RESET}")
        else:
            print(f"{GREEN}PASSED: {script.name}{RESET}")

    if all_passed:
        print(f"\n{GREEN}One-off tests: ALL PASSED{RESET}")
    else:
        print(f"\n{RED}One-off tests: SOME FAILED{RESET}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Test driver for comfyui-triton-sageattention-installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py --unit     # Safe: run only mocked unit tests
  python tests/run_tests.py --check    # Check environment configuration
  python tests/run_tests.py --oneoff   # Run integration tests (requires config)
"""
    )
    parser.add_argument("--unit", action="store_true",
                        help="Run only unit tests (safe, no real environments)")
    parser.add_argument("--oneoff", action="store_true",
                        help="Run only one-off integration tests")
    parser.add_argument("--quick", action="store_true",
                        help="Run only unit tests (alias for --unit)")
    parser.add_argument("--check", action="store_true",
                        help="Check environment configuration and exit")
    parser.add_argument("--env", choices=["venv", "portable"],
                        help="One-off tests: test only specific environment")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--python", type=Path,
                        help="Python interpreter to use")

    args = parser.parse_args()

    # Load and validate environments
    environments = load_config()
    valid_envs, invalid_envs = validate_environments(environments)

    # Check-only mode
    if args.check:
        print_environment_status(valid_envs, invalid_envs)
        sys.exit(0 if valid_envs else 1)

    # Find Python
    python = args.python if args.python else find_python_with_pytest(valid_envs)

    print(f"{BOLD}Test Driver{RESET}")
    print(f"Python: {python or 'Not found'}")
    print(f"Project: {PROJECT_DIR}")

    # Determine what to run
    run_unit = True
    run_oneoff = True

    if args.unit or args.quick:
        run_oneoff = False
    if args.oneoff:
        run_unit = False

    # Warn if trying to run one-off tests without valid environments
    if run_oneoff and not valid_envs:
        print(f"\n{YELLOW}Warning: No valid test environments for one-off tests.{RESET}")
        print(f"Run with --check to see configuration options.")
        if run_unit:
            print(f"Continuing with unit tests only...")
            run_oneoff = False
        else:
            print_environment_status(valid_envs, invalid_envs)
            sys.exit(1)

    results = []

    # Run tests
    if run_unit:
        results.append(("Unit Tests", run_unit_tests(python, args.verbose)))

    if run_oneoff:
        results.append(("One-off Tests",
                       run_oneoff_tests(python, valid_envs, args.env, args.verbose)))

    # Summary
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}FINAL SUMMARY{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    all_passed = True
    for name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {name}: [{status}]")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n{GREEN}{BOLD}All tests passed!{RESET}")
        sys.exit(0)
    else:
        print(f"\n{RED}{BOLD}Some tests failed!{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
