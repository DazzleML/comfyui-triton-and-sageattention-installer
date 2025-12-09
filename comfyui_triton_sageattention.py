#!/usr/bin/env python3
"""
Cross-Platform ComfyUI with Triton and SageAttention Installer

A Python-based installer that replicates the functionality of the Windows batch scripts
while providing cross-platform support for Linux, macOS, and Windows.

Includes all functionality from:
- (Step 1) Remove Triton Dependency Packages.bat
- (Step 2) Install Triton Dependency Packages.bat  
- run_nvidia_gpu.bat
"""

import argparse
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Version information
__version__ = "0.6.6"


def parse_sage_version(version_str: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse --sage-version argument into (major, exact) tuple.

    Args:
        version_str: User input like "auto", "1", "2", "1.0.6", "2.1.1"

    Returns:
        Tuple of (major_version: int|None, exact_version: str|None)
        - (None, None) for auto mode
        - (1, None) for "1"
        - (2, None) for "2"
        - (1, "1.0.6") for exact 1.x version
        - (2, "2.1.1") for exact 2.x version

    Raises:
        ValueError: If version string is invalid
    """
    version_str = version_str.strip().lower()

    if version_str == "auto":
        return (None, None)

    if version_str == "1":
        return (1, None)

    if version_str == "2":
        return (2, None)

    # Check for exact version (e.g., "1.0.6", "2.1.1")
    match = re.match(r'^([12])\.(\d+)\.(\d+)$', version_str)
    if match:
        major = int(match.group(1))
        return (major, version_str)

    raise ValueError(
        f"Invalid --sage-version: '{version_str}'. "
        f"Use 'auto', '1', '2', or exact version like '1.0.6' or '2.1.1'"
    )


class ComfyUIInstallerError(Exception):
    """Base exception for installer errors."""
    pass


class PlatformHandler(ABC):
    """Abstract base class for platform-specific installation handlers."""
    
    def __init__(self, base_path: Path, logger: logging.Logger, interactive: bool = True, force: bool = False):
        self.base_path = base_path
        self.logger = logger
        self.interactive = interactive
        self.force = force
        self.python_path = None
        self.venv_path = None
        self._setup_python_environment()
        
    @abstractmethod
    def install_build_tools(self) -> bool:
        """Install platform-specific build tools."""
        pass
    
    @abstractmethod
    def detect_cuda_version(self) -> Optional[str]:
        """Detect installed CUDA version."""
        pass
    
    @abstractmethod
    def get_pytorch_install_url(self, cuda_version: str) -> str:
        """Get platform-specific PyTorch installation URL."""
        pass
    
    @abstractmethod
    def _setup_python_environment(self):
        """Set up platform-specific Python environment."""
        pass
    
    @abstractmethod
    def create_run_script(self, use_sage: bool = True, fast_mode: bool = True) -> Path:
        """Create platform-specific run script."""
        pass
    
    def run_command(self, cmd: List[str], check: bool = True, capture_output: bool = False, 
                   shell: bool = False) -> subprocess.CompletedProcess:
        """Run a command with proper error handling."""
        self.logger.info(f"Running command: {' '.join(cmd) if not shell else cmd[0]}")
        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    check=check,
                    capture_output=True,
                    text=True,
                    shell=shell,
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
                )
            else:
                # For non-captured output, let it stream to console
                result = subprocess.run(
                    cmd,
                    check=check,
                    text=True,
                    shell=shell,
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
                )
            if capture_output and result.stdout:
                self.logger.debug(f"Command output: {result.stdout}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(cmd) if not shell else cmd[0]}")
            self.logger.error(f"Error: {e}")
            if hasattr(e, 'stdout') and e.stdout:
                self.logger.error(f"Stdout: {e.stdout}")
            if hasattr(e, 'stderr') and e.stderr:
                self.logger.error(f"Stderr: {e.stderr}")
            raise ComfyUIInstallerError(f"Command failed: {e}")
    
    def pip_install(self, packages: List[str], extra_args: List[str] = None) -> None:
        """Install packages using pip."""
        cmd = [str(self.python_path), "-m", "pip", "install"] + (extra_args or []) + packages
        self.run_command(cmd)
    
    def pip_uninstall(self, packages: List[str]) -> None:
        """Uninstall packages using pip."""
        cmd = [str(self.python_path), "-m", "pip", "uninstall", "-y"] + packages
        try:
            self.run_command(cmd)
        except ComfyUIInstallerError:
            self.logger.warning(f"Some packages could not be uninstalled: {packages}")


class WindowsHandler(PlatformHandler):
    """Windows-specific installation handler."""
    
    BUILD_TOOLS_CONFIG = {
        "installer_id": "Microsoft.VisualStudio.2022.BuildTools",
        "components": [
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "Microsoft.VisualStudio.Component.Windows10SDK.20348"
        ]
    }
    
    def _setup_python_environment(self):
        """Setup Windows Python environment (uses python_embeded structure)."""
        # Check if we're in a ComfyUI distribution with python_embeded
        embeded_path = self.base_path / "python_embeded" / "python.exe"
        if embeded_path.exists():
            self.python_path = embeded_path
            self.venv_path = self.base_path / "python_embeded"
            self.logger.info(f"Using existing python_embeded: {self.python_path}")
        else:
            # Check for existing virtual environment first
            venv_path = self.base_path / "venv"
            venv_python = venv_path / "Scripts" / "python.exe"  # Windows venv structure
            
            if venv_python.exists() and self._validate_python_environment(venv_python):
                self.python_path = venv_python
                self.venv_path = venv_path
                self.logger.info(f"Using existing virtual environment: {self.python_path}")
            else:
                # Create new virtual environment
                self.logger.info("Creating new virtual environment...")
                try:
                    self.run_command([sys.executable, "-m", "venv", str(venv_path)])
                    self.python_path = venv_python
                    self.venv_path = venv_path
                    self.logger.info(f"Created virtual environment: {self.python_path}")
                except ComfyUIInstallerError:
                    # Fallback to system Python with warning
                    self.python_path = Path(sys.executable)
                    self.venv_path = None
                    self.logger.warning("Could not create virtual environment, using system Python")
    
    def _validate_python_environment(self, python_path: Path) -> bool:
        """Validate that a Python environment is functional."""
        try:
            result = self.run_command([str(python_path), "--version"], capture_output=True)
            # Check if it's a reasonable Python version
            if "Python 3." in result.stdout:
                return True
        except (ComfyUIInstallerError, FileNotFoundError):
            pass
        return False
    
    def install_build_tools(self) -> bool:
        """Install Visual Studio Build Tools using winget."""
        # Check if build tools are already installed
        if not self.force and self._check_existing_build_tools():
            self.logger.info("Visual Studio Build Tools already installed")
            return True
        
        if self.force and self._check_existing_build_tools():
            print("WARNING: Visual Studio Build Tools already installed but --force specified")
            print("This may reinstall or modify existing build tools")
            if self.interactive:
                response = input("Continue with forced installation? (y/N): ")
                if response.lower() != 'y':
                    self.logger.info("Skipping build tools installation")
                    return True
        
        try:
            # Check if winget is available
            self.run_command(["winget", "--version"], capture_output=True)
            
            # Build winget install command
            cmd = [
                "winget", "install",
                "--id", self.BUILD_TOOLS_CONFIG["installer_id"],
                "-e", "--source", "winget",
                "--override", self._build_override_string()
            ]
            
            self.run_command(cmd)
            
            # Verify installation succeeded
            if self._check_existing_build_tools():
                return True
            else:
                self.logger.warning("Build tools installation may have failed")
                return False
                
        except (ComfyUIInstallerError, FileNotFoundError):
            self.logger.warning("Failed to install Visual Studio Build Tools automatically")
            self.logger.info("Please install Visual Studio Build Tools manually:")
            self.logger.info("https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022")
            return False
    
    def _check_existing_build_tools(self) -> bool:
        """Check if Visual Studio Build Tools are already installed."""
        # Method 1: Check for cl.exe (MSVC compiler)
        try:
            result = self.run_command(["cl"], capture_output=True, check=False)
            if "Microsoft (R) C/C++ Optimizing Compiler" in result.stderr:
                self.logger.debug("Found cl.exe (MSVC compiler)")
                return True
        except (ComfyUIInstallerError, FileNotFoundError):
            pass
        
        # Method 2: Check for nmake.exe
        try:
            result = self.run_command(["nmake", "/?"], capture_output=True, check=False)
            if "Microsoft (R) Program Maintenance Utility" in result.stdout:
                self.logger.debug("Found nmake.exe")
                return True
        except (ComfyUIInstallerError, FileNotFoundError):
            pass
        
        # Method 3: Check Visual Studio installation paths
        vs_paths = [
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/BuildTools"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools"),
            Path("C:/Program Files/Microsoft Visual Studio/2019/BuildTools"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Community"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Community"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Professional"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Professional"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Enterprise"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise"),
        ]
        
        for vs_path in vs_paths:
            if vs_path.exists():
                # Check for specific build tools
                vc_tools = vs_path / "VC" / "Tools" / "MSVC"
                if vc_tools.exists() and any(vc_tools.iterdir()):
                    self.logger.debug(f"Found Visual Studio installation at {vs_path}")
                    return True
        
        # Method 4: Check Windows SDK
        try:
            import winreg
            sdk_key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Microsoft SDKs\Windows\v10.0"
            )
            install_path, _ = winreg.QueryValueEx(sdk_key, "InstallationFolder")
            if Path(install_path).exists():
                self.logger.debug(f"Found Windows SDK at {install_path}")
                return True
        except (ImportError, OSError, FileNotFoundError):
            pass
        
        # Method 5: Check for vswhere utility
        try:
            result = self.run_command([
                "C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe",
                "-latest", "-products", "*", "-requires", 
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
            ], capture_output=True, check=False)
            
            if result.returncode == 0 and result.stdout.strip():
                self.logger.debug("Found Visual Studio via vswhere")
                return True
        except (ComfyUIInstallerError, FileNotFoundError):
            pass
        
        self.logger.debug("No existing Visual Studio Build Tools found")
        return False
    
    def _build_override_string(self) -> str:
        """Build the override string for Visual Studio installation."""
        components = " ".join(f"--add {comp}" for comp in self.BUILD_TOOLS_CONFIG["components"])
        return f"--quiet --wait --norestart {components}"
    
    def detect_cuda_version(self) -> Optional[str]:
        """Detect CUDA version using nvcc."""
        try:
            result = self.run_command(["nvcc", "--version"], capture_output=True)
            
            # Parse version from nvcc output
            version_match = re.search(r'release (\d+\.\d+)', result.stdout)
            if version_match:
                return version_match.group(1)
        except (ComfyUIInstallerError, FileNotFoundError):
            self.logger.warning("CUDA not found or nvcc not in PATH")
        
        return None
    
    def get_pytorch_install_url(self, cuda_version: str) -> str:
        """Get PyTorch installation URL for Windows."""
        if cuda_version == "cpu":
            return "https://download.pytorch.org/whl/cpu"
        cuda_tag = cuda_version.replace(".", "")
        return f"https://download.pytorch.org/whl/cu{cuda_tag}"
    
    def create_run_script(self, use_sage: bool = True, fast_mode: bool = True) -> Path:
        """Create Windows batch script to run ComfyUI."""
        script_path = self.base_path / "run_nvidia_gpu.bat"
        
        # Build command arguments
        args = ["ComfyUI\\main.py", "--windows-standalone-build"]
        if use_sage:
            args.append("--use-sage-attention")
        if fast_mode:
            args.append("--fast")
        
        # Create batch script content (matches original exactly)
        script_content = f'"{self.python_path}" -s {" ".join(args)}\npause\n'
        
        script_path.write_text(script_content, encoding='utf-8')
        self.logger.info(f"Created run script: {script_path}")
        return script_path


class LinuxHandler(PlatformHandler):
    """Linux-specific installation handler."""
    
    BUILD_TOOLS_PACKAGES = {
        "apt": ["build-essential", "python3-dev", "python3-venv", "git", "curl", "wget"],
        "yum": ["gcc", "gcc-c++", "python3-devel", "python3-venv", "git", "curl", "wget"],
        "dnf": ["gcc", "gcc-c++", "python3-devel", "python3-venv", "git", "curl", "wget"],
        "pacman": ["base-devel", "python", "python-venv", "git", "curl", "wget"],
        "zypper": ["gcc", "gcc-c++", "python3-devel", "python3-venv", "git", "curl", "wget"]
    }
    
    def _setup_python_environment(self):
        """Setup Linux Python virtual environment."""
        self.venv_path = self.base_path / "venv"
        venv_python = self.venv_path / "bin" / "python"
        
        # Check if virtual environment already exists and is valid
        if venv_python.exists() and self._validate_python_environment(venv_python):
            self.python_path = venv_python
            self.logger.info(f"Using existing virtual environment: {self.python_path}")
        else:
            # Create or recreate virtual environment
            if self.venv_path.exists():
                self.logger.info("Existing venv appears invalid, recreating...")
                # Only remove if we're confident it's broken
                if self.interactive:
                    response = input("Existing venv found but appears broken. Recreate? (y/N): ")
                    if response.lower() != 'y':
                        # Try to use it anyway
                        self.python_path = venv_python
                        self.logger.warning("Using potentially invalid virtual environment")
                        return
                    else:
                        shutil.rmtree(self.venv_path)
                else:
                    # Non-interactive mode: recreate automatically
                    self.logger.info("Non-interactive mode: recreating invalid venv")
                    shutil.rmtree(self.venv_path)
            
            self.logger.info("Creating Python virtual environment...")
            try:
                # Try python3 -m venv first
                self.run_command([sys.executable, "-m", "venv", str(self.venv_path)])
            except ComfyUIInstallerError:
                # Fallback to virtualenv if available
                try:
                    self.run_command(["virtualenv", str(self.venv_path)])
                except (ComfyUIInstallerError, FileNotFoundError):
                    raise ComfyUIInstallerError("Could not create virtual environment. Install python3-venv package.")
            
            self.python_path = venv_python
            
        if not self.python_path.exists():
            raise ComfyUIInstallerError(f"Python interpreter not found at {self.python_path}")
        
        self.logger.info(f"Using Python virtual environment: {self.python_path}")
    
    def _validate_python_environment(self, python_path: Path) -> bool:
        """Validate that a Python environment is functional."""
        try:
            result = self.run_command([str(python_path), "--version"], capture_output=True)
            # Check if it's a reasonable Python version
            if "Python 3." in result.stdout:
                return True
        except (ComfyUIInstallerError, FileNotFoundError):
            pass
        return False
    
    def install_build_tools(self) -> bool:
        """Install build tools using the system package manager."""
        # Check if essential build tools are already installed
        if not self.force and self._check_existing_build_tools():
            self.logger.info("Build tools already installed")
            return True
        
        if self.force and self._check_existing_build_tools():
            print("WARNING: Build tools already installed but --force specified")
            print("This may reinstall or upgrade existing build tools")
            if self.interactive:
                response = input("Continue with forced installation? (y/N): ")
                if response.lower() != 'y':
                    self.logger.info("Skipping build tools installation")
                    return True
        
        package_manager = self._detect_package_manager()
        if not package_manager:
            self.logger.error("Could not detect package manager")
            self._manual_install_instructions()
            return False
        
        packages = self.BUILD_TOOLS_PACKAGES.get(package_manager, [])
        if not packages:
            self.logger.error(f"Unknown package manager: {package_manager}")
            return False
        
        # Filter out already installed packages
        packages_to_install = self._filter_installed_packages(packages, package_manager)
        
        if not packages_to_install:
            self.logger.info("All required build tools already installed")
            return True
        
        try:
            if package_manager == "apt":
                self.run_command(["sudo", "apt", "update"])
                self.run_command(["sudo", "apt", "install", "-y"] + packages_to_install)
            elif package_manager in ["yum", "dnf"]:
                self.run_command(["sudo", package_manager, "install", "-y"] + packages_to_install)
            elif package_manager == "pacman":
                self.run_command(["sudo", "pacman", "-Sy", "--noconfirm"] + packages_to_install)
            elif package_manager == "zypper":
                self.run_command(["sudo", "zypper", "install", "-y"] + packages_to_install)
            
            return True
        except ComfyUIInstallerError:
            self.logger.error(f"Failed to install build tools with {package_manager}")
            self._manual_install_instructions()
            return False
    
    def _check_existing_build_tools(self) -> bool:
        """Check if essential build tools are already installed."""
        essential_tools = ["gcc", "g++", "make", "git", "curl"]
        
        for tool in essential_tools:
            try:
                result = self.run_command([tool, "--version"], capture_output=True, check=False)
                if result.returncode != 0:
                    self.logger.debug(f"Build tool not found: {tool}")
                    return False
            except (ComfyUIInstallerError, FileNotFoundError):
                self.logger.debug(f"Build tool not found: {tool}")
                return False
        
        # Check for python3-dev headers
        python_h_paths = [
            f"/usr/include/python{sys.version_info.major}.{sys.version_info.minor}",
            f"/usr/local/include/python{sys.version_info.major}.{sys.version_info.minor}",
        ]
        
        python_dev_found = any(
            (Path(path) / "Python.h").exists() 
            for path in python_h_paths
        )
        
        if not python_dev_found:
            self.logger.debug("Python development headers not found")
            return False
        
        self.logger.debug("All essential build tools found")
        return True
    
    def _filter_installed_packages(self, packages: List[str], package_manager: str) -> List[str]:
        """Filter out already installed packages."""
        packages_to_install = []
        
        for package in packages:
            try:
                if package_manager == "apt":
                    result = self.run_command([
                        "dpkg", "-l", package
                    ], capture_output=True, check=False)
                    if result.returncode != 0:
                        packages_to_install.append(package)
                elif package_manager in ["yum", "dnf"]:
                    result = self.run_command([
                        package_manager, "list", "installed", package
                    ], capture_output=True, check=False)
                    if result.returncode != 0:
                        packages_to_install.append(package)
                elif package_manager == "pacman":
                    result = self.run_command([
                        "pacman", "-Qi", package
                    ], capture_output=True, check=False)
                    if result.returncode != 0:
                        packages_to_install.append(package)
                else:
                    # For unknown package managers, install everything
                    packages_to_install.append(package)
            except (ComfyUIInstallerError, FileNotFoundError):
                packages_to_install.append(package)
        
        return packages_to_install
    
    def _detect_package_manager(self) -> Optional[str]:
        """Detect the system package manager."""
        managers = ["apt", "yum", "dnf", "pacman", "zypper"]
        for manager in managers:
            try:
                subprocess.run([manager, "--version"], 
                             capture_output=True, check=True)
                return manager
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        return None
    
    def _manual_install_instructions(self):
        """Provide manual installation instructions."""
        self.logger.info("Manual installation required. Install these packages:")
        self.logger.info("- build-essential / gcc gcc-c++ / base-devel")
        self.logger.info("- python3-dev / python3-devel")
        self.logger.info("- python3-venv")
        self.logger.info("- git, curl, wget")
    
    def detect_cuda_version(self) -> Optional[str]:
        """Detect CUDA version on Linux."""
        # Method 1: Try nvcc first
        try:
            result = self.run_command(["nvcc", "--version"], capture_output=True)
            version_match = re.search(r'release (\d+\.\d+)', result.stdout)
            if version_match:
                cuda_version = version_match.group(1)
                self.logger.info(f"Found CUDA via nvcc: {cuda_version}")
                return cuda_version
        except (ComfyUIInstallerError, FileNotFoundError):
            self.logger.debug("nvcc not found")
        
        # Method 2: Try nvidia-smi
        try:
            result = self.run_command(["nvidia-smi"], capture_output=True)
            version_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
            if version_match:
                cuda_version = version_match.group(1)
                self.logger.info(f"Found CUDA via nvidia-smi: {cuda_version}")
                return cuda_version
        except (ComfyUIInstallerError, FileNotFoundError):
            self.logger.debug("nvidia-smi not found")
        
        # Method 3: Check for CUDA installation paths
        cuda_paths = [
            "/usr/local/cuda/version.txt",
            "/usr/local/cuda/version.json",
            "/opt/cuda/version.txt"
        ]
        
        for cuda_path in cuda_paths:
            try:
                if Path(cuda_path).exists():
                    content = Path(cuda_path).read_text()
                    version_match = re.search(r'(\d+\.\d+)', content)
                    if version_match:
                        cuda_version = version_match.group(1)
                        self.logger.info(f"Found CUDA via {cuda_path}: {cuda_version}")
                        return cuda_version
            except Exception:
                continue
        
        self.logger.warning("CUDA not detected on Linux system")
        return None
    
    def get_pytorch_install_url(self, cuda_version: str) -> str:
        """Get PyTorch installation URL for Linux."""
        if cuda_version == "cpu":
            return "https://download.pytorch.org/whl/cpu"
        cuda_tag = cuda_version.replace(".", "")
        return f"https://download.pytorch.org/whl/cu{cuda_tag}"
    
    def create_run_script(self, use_sage: bool = True, fast_mode: bool = True) -> Path:
        """Create Linux shell script to run ComfyUI."""
        script_path = self.base_path / "run_comfyui.sh"
        
        # Build command arguments
        args = ["ComfyUI/main.py"]
        if use_sage:
            args.append("--use-sage-attention")
        if fast_mode:
            args.append("--fast")
        
        # Create shell script content
        script_content = f'#!/bin/bash\n"{self.python_path}" -s {" ".join(args)}\necho "Press Enter to continue..."\nread\n'
        
        script_path.write_text(script_content, encoding='utf-8')
        script_path.chmod(0o755)  # Make executable
        self.logger.info(f"Created run script: {script_path}")
        return script_path


class MacOSHandler(PlatformHandler):
    """macOS-specific installation handler."""
    
    def _setup_python_environment(self):
        """Setup macOS Python virtual environment."""
        self.venv_path = self.base_path / "venv"
        venv_python = self.venv_path / "bin" / "python"
        
        # Check if virtual environment already exists and is valid
        if venv_python.exists() and self._validate_python_environment(venv_python):
            self.python_path = venv_python
            self.logger.info(f"Using existing virtual environment: {self.python_path}")
        else:
            # Create or recreate virtual environment
            if self.venv_path.exists():
                self.logger.info("Existing venv appears invalid, recreating...")
                # Only remove if we're confident it's broken
                if self.interactive:
                    response = input("Existing venv found but appears broken. Recreate? (y/N): ")
                    if response.lower() != 'y':
                        # Try to use it anyway
                        self.python_path = venv_python
                        self.logger.warning("Using potentially invalid virtual environment")
                        return
                    else:
                        shutil.rmtree(self.venv_path)
                else:
                    # Non-interactive mode: recreate automatically
                    self.logger.info("Non-interactive mode: recreating invalid venv")
                    shutil.rmtree(self.venv_path)
            
            self.logger.info("Creating Python virtual environment...")
            try:
                self.run_command([sys.executable, "-m", "venv", str(self.venv_path)])
            except ComfyUIInstallerError:
                raise ComfyUIInstallerError("Could not create virtual environment. Ensure Python 3.8+ is installed.")
            
            self.python_path = venv_python
        
        if not self.python_path.exists():
            raise ComfyUIInstallerError(f"Python interpreter not found at {self.python_path}")
        
        self.logger.info(f"Using Python virtual environment: {self.python_path}")
    
    def _validate_python_environment(self, python_path: Path) -> bool:
        """Validate that a Python environment is functional."""
        try:
            result = self.run_command([str(python_path), "--version"], capture_output=True)
            # Check if it's a reasonable Python version
            if "Python 3." in result.stdout:
                return True
        except (ComfyUIInstallerError, FileNotFoundError):
            pass
        return False
    
    def install_build_tools(self) -> bool:
        """Install Xcode Command Line Tools and Homebrew packages."""
        # Check if build tools are already installed
        if not self.force and self._check_existing_build_tools():
            self.logger.info("Build tools already installed")
            return True
        
        if self.force and self._check_existing_build_tools():
            print("WARNING: Build tools already installed but --force specified")
            print("This may reinstall or upgrade existing build tools")
            if self.interactive:
                response = input("Continue with forced installation? (y/N): ")
                if response.lower() != 'y':
                    self.logger.info("Skipping build tools installation")
                    return True
        
        # Install Xcode Command Line Tools if not present
        if not self._check_xcode_tools():
            try:
                self.logger.info("Installing Xcode Command Line Tools...")
                self.run_command(["xcode-select", "--install"])
                self.logger.info("Xcode Command Line Tools installation started. Please follow the prompts.")
                # Note: This is interactive and may require user action
            except ComfyUIInstallerError:
                self.logger.warning("Could not install Xcode Command Line Tools automatically")
                self.logger.info("Please install manually: xcode-select --install")
                return False
        
        # Check for Homebrew and install required packages
        homebrew_packages = ["git", "curl", "wget"]
        try:
            self.run_command(["brew", "--version"], capture_output=True)
            self.logger.info("Homebrew found, checking packages...")
            
            # Check which packages need installation
            packages_to_install = []
            for package in homebrew_packages:
                try:
                    self.run_command(["brew", "list", package], capture_output=True)
                    self.logger.debug(f"Homebrew package already installed: {package}")
                except ComfyUIInstallerError:
                    packages_to_install.append(package)
            
            if packages_to_install:
                self.logger.info(f"Installing Homebrew packages: {packages_to_install}")
                self.run_command(["brew", "install"] + packages_to_install)
            else:
                self.logger.info("All required Homebrew packages already installed")
            
            return True
        except (ComfyUIInstallerError, FileNotFoundError):
            self.logger.warning("Homebrew not found. Please install Homebrew:")
            self.logger.info("https://brew.sh/")
            self.logger.info("Then run: brew install git curl wget")
            # Don't return False - git might be available from Xcode tools
            return self._check_essential_tools()
    
    def _check_existing_build_tools(self) -> bool:
        """Check if build tools are already installed."""
        return self._check_xcode_tools() and self._check_essential_tools()
    
    def _check_xcode_tools(self) -> bool:
        """Check if Xcode Command Line Tools are installed."""
        try:
            # Method 1: Check xcode-select path
            result = self.run_command(["xcode-select", "--print-path"], capture_output=True)
            xcode_path = Path(result.stdout.strip())
            if xcode_path.exists():
                self.logger.debug(f"Found Xcode tools at: {xcode_path}")
                return True
        except ComfyUIInstallerError:
            pass
        
        # Method 2: Check for clang compiler
        try:
            result = self.run_command(["clang", "--version"], capture_output=True)
            if "clang version" in result.stdout:
                self.logger.debug("Found clang compiler")
                return True
        except (ComfyUIInstallerError, FileNotFoundError):
            pass
        
        # Method 3: Check for make utility
        try:
            self.run_command(["make", "--version"], capture_output=True)
            self.logger.debug("Found make utility")
            return True
        except (ComfyUIInstallerError, FileNotFoundError):
            pass
        
        self.logger.debug("Xcode Command Line Tools not found")
        return False
    
    def _check_essential_tools(self) -> bool:
        """Check if essential command line tools are available."""
        essential_tools = ["git", "curl"]
        
        for tool in essential_tools:
            try:
                self.run_command([tool, "--version"], capture_output=True, check=False)
                self.logger.debug(f"Found essential tool: {tool}")
            except (ComfyUIInstallerError, FileNotFoundError):
                self.logger.debug(f"Essential tool not found: {tool}")
                return False
        
        return True
    
    def detect_cuda_version(self) -> Optional[str]:
        """CUDA detection for macOS."""
        # Check if we're on Apple Silicon
        if platform.processor() == "arm" or "arm64" in platform.machine().lower():
            self.logger.info("Apple Silicon Mac detected - CUDA not supported, using Metal/CPU backend")
            return "cpu"
        
        # For Intel Macs, try standard CUDA detection
        try:
            result = self.run_command(["nvcc", "--version"], capture_output=True)
            version_match = re.search(r'release (\d+\.\d+)', result.stdout)
            if version_match:
                cuda_version = version_match.group(1)
                self.logger.info(f"Found CUDA on Intel Mac: {cuda_version}")
                return cuda_version
        except (ComfyUIInstallerError, FileNotFoundError):
            self.logger.info("CUDA not found on Intel Mac, using CPU backend")
        
        return "cpu"
    
    def get_pytorch_install_url(self, cuda_version: str) -> str:
        """Get PyTorch installation URL for macOS."""
        # macOS typically uses CPU or Metal backend
        return "https://download.pytorch.org/whl/cpu"
    
    def create_run_script(self, use_sage: bool = True, fast_mode: bool = True) -> Path:
        """Create macOS shell script to run ComfyUI."""
        script_path = self.base_path / "run_comfyui.sh"
        
        # Build command arguments (SageAttention may not work on macOS without CUDA)
        args = ["ComfyUI/main.py"]
        if use_sage and self.detect_cuda_version() != "cpu":
            args.append("--use-sage-attention")
        if fast_mode:
            args.append("--fast")
        
        # Create shell script content
        script_content = f'#!/bin/bash\n"{self.python_path}" -s {" ".join(args)}\necho "Press Enter to continue..."\nread\n'
        
        script_path.write_text(script_content, encoding='utf-8')
        script_path.chmod(0o755)  # Make executable
        self.logger.info(f"Created run script: {script_path}")
        return script_path


class ComfyUIInstaller:
    """Main installer class that orchestrates the installation process."""
    
    REPOSITORIES = {
        "sageattention": "https://github.com/thu-ml/SageAttention",
    }

    # Node preset configurations for --with-custom-nodes
    # Future: Add more presets like "video", "wan", "flux", etc.
    NODE_PRESETS = {
        "default": [
            {
                "name": "ComfyUI-VideoHelperSuite",
                "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
                "description": "Video encoding/decoding utilities",
            },
            {
                "name": "DazzleNodes",
                "url": "https://github.com/DazzleNodes/DazzleNodes.git",
                "description": "DazzleML node collection",
            },
        ],
        # "video": [
        #     # For users who need flow2-wan-video (can conflict with some workflows)
        #     {"name": "flow2-wan-video", "url": "https://github.com/Flow-Two/flow2-wan-video.git"},
        #     {"name": "ComfyUI-VideoHelperSuite", "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"},
        # ],
    }
    
    INCLUDE_LIBS_URL = "https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip"
    
    # Packages to track for cleanup (matches batch script exactly)
    TRITON_PACKAGES = [
        "triton-windows", "triton", "sageattention", 
        "torch", "torchvision", "torchaudio"
    ]
    
    def __init__(self, base_path: Optional[Path] = None, verbose: bool = False, interactive: bool = True, force: bool = False, sage_version: str = "auto", experimental: bool = False, upgrade: bool = False, with_custom_nodes: bool = False):
        self.base_path = base_path or Path.cwd()
        self.interactive = interactive
        self.force = force
        self.experimental = experimental
        self.upgrade = upgrade
        self.with_custom_nodes = with_custom_nodes
        # Parse sage_version into (major, exact) tuple
        self.sage_version_raw = sage_version
        self.sage_version_major, self.sage_version_exact = parse_sage_version(sage_version)
        self.setup_logging(verbose)
        self.handler = self._create_platform_handler()
        self.installed_packages = []
        self.created_directories = []
        
    def setup_logging(self, verbose: bool):
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.base_path / 'comfyui_install.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_platform_handler(self) -> PlatformHandler:
        """Create appropriate platform handler."""
        system = platform.system()
        if system == "Windows":
            return WindowsHandler(self.base_path, self.logger, self.interactive, self.force)
        elif system == "Linux":
            return LinuxHandler(self.base_path, self.logger, self.interactive, self.force)
        elif system == "Darwin":
            return MacOSHandler(self.base_path, self.logger, self.interactive, self.force)
        else:
            raise ComfyUIInstallerError(f"Unsupported platform: {system}")
    
    def cleanup_installation(self):
        """Remove installed packages and created directories (matches Step 1 batch script)."""
        self.logger.info("Starting cleanup process...")
        
        print("Uninstalling Triton dependency...")
        
        # Uninstall packages (exact match to batch script) - but only ComfyUI-specific ones
        try:
            self.handler.pip_uninstall(self.TRITON_PACKAGES)
        except Exception as e:
            self.logger.warning(f"Some packages could not be uninstalled: {e}")
        
        print("Removing SageAttention build files...")
        
        # Remove directories (matches batch script) - but preserve user's venv
        directories_to_remove = [
            "SageAttention",
            # Custom nodes are user-managed (opt-in with --with-custom-nodes)
            # Users can manually remove them if needed
        ]
        
        # Only remove Python dev directories on Windows
        if platform.system() == "Windows":
            directories_to_remove.extend([
                "python_embeded/libs", 
                "python_embeded/include"
            ])
        
        for dir_name in directories_to_remove:
            directory = self.base_path / dir_name
            if directory.exists():
                try:
                    shutil.rmtree(directory)
                    self.logger.info(f"Removed directory: {directory}")
                except Exception as e:
                    self.logger.warning(f"Could not remove {directory}: {e}")
        
        # Remove downloaded files (matches batch script)
        files_to_remove = [
            "python_3.12.7_include_libs.zip",
            "run_nvidia_gpu.bat",
            "run_comfyui.sh"
        ]
        
        for file_name in files_to_remove:
            file_path = self.base_path / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                    self.logger.info(f"Removed file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Could not remove {file_path}: {e}")
        
        # Ask user about virtual environment removal (instead of blindly deleting)
        venv_path = self.base_path / "venv"
        if venv_path.exists():
            if self.interactive:
                response = input(f"Remove virtual environment at {venv_path}? This will delete ALL packages in it. (y/N): ")
                should_remove = response.lower() == 'y'
            else:
                # Non-interactive mode: don't remove venv by default (safer)
                should_remove = False
                self.logger.info("Non-interactive mode: preserving virtual environment")
            
            if should_remove:
                try:
                    shutil.rmtree(venv_path)
                    self.logger.info(f"Removed virtual environment: {venv_path}")
                except Exception as e:
                    self.logger.warning(f"Could not remove {venv_path}: {e}")
            else:
                self.logger.info("Virtual environment preserved. You may want to manually clean ComfyUI packages.")
        
        print("Success!")
    
    def install_build_tools(self):
        """Install platform-specific build tools."""
        print("Installing build tools...")
        if not self.handler.install_build_tools():
            raise ComfyUIInstallerError("Failed to install build tools")
    
    def detect_and_setup_cuda(self) -> str:
        """Detect CUDA version and return it."""
        print("Finding installed CUDA...")
        cuda_version = self.handler.detect_cuda_version()
        
        if cuda_version and cuda_version != "cpu":
            print(f"CUDA version: {cuda_version}")
        else:
            print("CUDA not detected or not supported, using CPU backend")
            cuda_version = "cpu"
        
        return cuda_version
    
    def upgrade_pip_setuptools(self):
        """Upgrade pip and setuptools."""
        print("Upgrading pip and setuptools...")
        self.handler.pip_install(["pip", "setuptools"], ["--upgrade"])
    
    def install_pytorch(self, cuda_version: str):
        """Install PyTorch with appropriate CUDA support."""
        # Check if compatible PyTorch is already installed
        if not self.force and self._check_pytorch_compatibility(cuda_version):
            print("Compatible PyTorch already installed")
            return
        
        if self.force and self._check_pytorch_compatibility(cuda_version):
            print("WARNING: Compatible PyTorch already installed but --force specified")
            print("This will reinstall PyTorch and may break existing installations")
            if self.interactive:
                response = input("Continue with forced PyTorch installation? (y/N): ")
                if response.lower() != 'y':
                    print("Skipping PyTorch installation")
                    return
        
        print("Installing PyTorch...")
        
        if cuda_version != "cpu":
            index_url = self.handler.get_pytorch_install_url(cuda_version)
            extra_args = ["--index-url", index_url]
            packages = ["torch==2.7.0", "torchvision", "torchaudio"]
        else:
            extra_args = []
            packages = ["torch", "torchvision", "torchaudio"]
        
        self.handler.pip_install(packages, extra_args)
        self.installed_packages.extend(["torch", "torchvision", "torchaudio"])
    
    def _check_pytorch_compatibility(self, cuda_version: str) -> bool:
        """Check if existing PyTorch installation is compatible."""
        try:
            # Test if torch is importable and get version info
            result = self.handler.run_command([
                str(self.handler.python_path), "-c",
                "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.version.cuda if torch.cuda.is_available() else \"None\"}')"
            ], capture_output=True)
            
            version_info = result.stdout.strip().split('|')
            torch_version, cuda_available, torch_cuda_version = version_info
            
            self.logger.debug(f"Found PyTorch {torch_version}, CUDA available: {cuda_available}, CUDA version: {torch_cuda_version}")
            
            # Check version compatibility
            if not torch_version.startswith("2."):
                self.logger.info("PyTorch version is not 2.x, upgrading...")
                return False
            
            # Check CUDA compatibility
            if cuda_version == "cpu":
                # For CPU-only, any PyTorch 2.x is fine
                self.logger.info(f"PyTorch {torch_version} compatible with CPU backend")
                return True
            else:
                # For CUDA, check if CUDA is available and version matches
                if cuda_available == "False":
                    self.logger.info("Existing PyTorch is CPU-only but CUDA is available, upgrading...")
                    return False
                
                # Check CUDA version compatibility (allow minor version differences)
                if torch_cuda_version != "None":
                    torch_cuda_major = torch_cuda_version.split('.')[0]
                    system_cuda_major = cuda_version.split('.')[0]
                    
                    if torch_cuda_major == system_cuda_major:
                        self.logger.info(f"PyTorch {torch_version} with CUDA {torch_cuda_version} is compatible")
                        return True
                    else:
                        self.logger.info(f"PyTorch CUDA version ({torch_cuda_version}) doesn't match system CUDA ({cuda_version}), upgrading...")
                        return False
            
        except (ComfyUIInstallerError, Exception) as e:
            self.logger.debug(f"Could not check PyTorch compatibility: {e}")
            return False
        
        return False
    
    def install_triton(self):
        """Install or fix Triton with version constraint based on PyTorch compatibility.

        Triton/PyTorch version compatibility (triton-windows):
        - Triton 3.5.x requires PyTorch >= 2.9
        - Triton 3.4.x requires PyTorch >= 2.8
        - Triton 3.3.x requires PyTorch >= 2.7
        - Triton 3.2.x requires PyTorch >= 2.6

        If Triton is already installed but incompatible with PyTorch, it will be
        uninstalled and reinstalled with the correct version constraint.

        See: https://github.com/woct0rdho/triton-windows/issues/158
        """
        # Determine base package name
        if platform.system() == "Windows":
            base_package = "triton-windows"
        else:
            base_package = "triton"

        # Get PyTorch version and determine compatible Triton constraint
        torch_ver = self._get_torch_version()
        constraint = self._get_triton_version_constraint(torch_ver)

        # Check if Triton is already installed
        current_triton = self._get_installed_triton_version()

        if current_triton:
            is_compat, compat_msg = self._check_triton_pytorch_compatibility(current_triton, torch_ver)

            if is_compat:
                if not self.force:
                    print(f"Triton {current_triton} is compatible with PyTorch {torch_ver} - skipping")
                    return
                else:
                    print(f"Triton {current_triton} is compatible, but --force specified - reinstalling")
            else:
                # Incompatible - need to fix
                print(f"Fixing Triton compatibility: {compat_msg}")
                print(f"  Removing incompatible Triton {current_triton}...")
                try:
                    self.handler.pip_uninstall([base_package])
                except Exception as e:
                    self.logger.warning(f"Could not uninstall {base_package}: {e}")
        else:
            print("Installing Triton...")

        # Install with constraint
        if constraint:
            package = f"{base_package}{constraint}"
            print(f"  PyTorch {torch_ver} detected - using Triton constraint: {constraint}")
        else:
            package = base_package
            print(f"  Could not determine PyTorch version - installing latest Triton")

        self.handler.pip_install([package], ["-U", "--pre"])
        self.installed_packages.append(base_package)
    
    def setup_python_dev_files(self):
        """Download and extract Python development files (Windows only)."""
        if platform.system() != "Windows":
            return
        
        # Check if development files already exist
        # For ComfyUI portable, use python_embeded; for venv, use venv path
        if (self.base_path / "python_embeded").exists():
            python_dir = self.base_path / "python_embeded"
        else:
            python_dir = self.handler.venv_path
            
        include_dir = python_dir / "include"
        libs_dir = python_dir / "libs"
        
        if not self.force and self._check_python_dev_files(include_dir, libs_dir):
            print("Python development files already present")
            return
        
        if self.force and self._check_python_dev_files(include_dir, libs_dir):
            print("WARNING: Python development files already present but --force specified")
            print("This will redownload and overwrite existing files")
            if self.interactive:
                response = input("Continue with forced download? (y/N): ")
                if response.lower() != 'y':
                    print("Skipping Python development files download")
                    return
        
        print("Downloading Python include/libs from URL...")
        
        # Download the zip file
        zip_name = "python_3.12.7_include_libs.zip"
        zip_path = self.base_path / zip_name
        
        # Check if zip already downloaded
        if not zip_path.exists():
            urllib.request.urlretrieve(self.INCLUDE_LIBS_URL, zip_path)
        else:
            self.logger.info("Using existing downloaded zip file")
        
        print("Extracting Python include/libs...")
        
        # Extract to python_embeded directory (matches batch script)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(python_dir)
        
        self.logger.info("Python development files extracted")
    
    def _check_python_dev_files(self, include_dir: Path, libs_dir: Path) -> bool:
        """Check if Python development files are already extracted."""
        # Check for essential header files
        essential_headers = ["Python.h", "pyconfig.h", "object.h"]
        for header in essential_headers:
            if not (include_dir / header).exists():
                return False
        
        # Check for essential library files
        if not libs_dir.exists():
            return False
        
        # Look for some .lib files (exact files may vary)
        lib_files = list(libs_dir.glob("*.lib"))
        if len(lib_files) < 5:  # Should have multiple .lib files
            return False
        
        self.logger.debug("Python development files appear complete")
        return True

    def _get_system_info_string(self) -> str:
        """Get formatted system info for error messages."""
        try:
            torch_ver = self._get_torch_version()
            cuda_ver = self._get_cuda_version_from_torch()
            python_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            return f"PyTorch: {torch_ver}, CUDA: {cuda_ver}, Python: {python_ver}"
        except Exception:
            return "Could not detect versions"

    def _get_installed_sageattention_version(self) -> Optional[str]:
        """Get currently installed sageattention version, or None if not installed.

        Returns:
            Version string (e.g., "1.0.6", "2.2.0+cu128torch2.7.1.post3") or None.
        """
        try:
            result = self.handler.run_command([
                str(self.handler.python_path), "-c",
                "from importlib.metadata import version; print(version('sageattention'))"
            ], capture_output=True)
            return result.stdout.strip() if result.stdout else None
        except Exception:
            return None

    def _get_installed_triton_version(self) -> Optional[str]:
        """Get currently installed triton/triton-windows version, or None if not installed.

        Returns:
            Version string (e.g., "3.5.1.post22") or None.
        """
        # Try triton-windows first (Windows), then triton (Linux/Mac)
        for package in ["triton-windows", "triton"]:
            try:
                result = self.handler.run_command([
                    str(self.handler.python_path), "-c",
                    f"from importlib.metadata import version; print(version('{package}'))"
                ], capture_output=True)
                if result.stdout and result.stdout.strip():
                    return result.stdout.strip()
            except Exception:
                continue
        return None

    def _get_triton_version_constraint(self, torch_ver: str) -> str:
        """Get compatible triton-windows version constraint for PyTorch version.

        Based on https://github.com/woct0rdho/triton-windows compatibility:
        - Triton 3.5.x requires PyTorch >= 2.9
        - Triton 3.4.x requires PyTorch >= 2.8
        - Triton 3.3.x requires PyTorch >= 2.7
        - Triton 3.2.x requires PyTorch >= 2.6

        Args:
            torch_ver: PyTorch version string (e.g., "2.7.0", "2.7.0+cu128")

        Returns:
            Version constraint string (e.g., ">=3.3,<3.4") or empty string if can't parse.
        """
        try:
            parts = torch_ver.split(".")
            major = int(parts[0])
            # Handle "+cu128" suffix in minor version
            minor = int(parts[1].split("+")[0])

            if (major, minor) >= (2, 9):
                return ">=3.5,<4"
            elif (major, minor) >= (2, 8):
                return ">=3.4,<3.5"
            elif (major, minor) >= (2, 7):
                return ">=3.3,<3.4"
            elif (major, minor) >= (2, 6):
                return ">=3.2,<3.3"
            else:
                return "<3.2"
        except (ValueError, IndexError):
            return ""  # No constraint if can't parse

    def _check_triton_pytorch_compatibility(self, triton_ver: str, torch_ver: str) -> Tuple[bool, str]:
        """Check if installed Triton version is compatible with PyTorch.

        Args:
            triton_ver: Triton version string (e.g., "3.5.1.post22")
            torch_ver: PyTorch version string (e.g., "2.7.0")

        Returns:
            Tuple of (is_compatible, message).
        """
        try:
            # Parse Triton major.minor
            triton_parts = triton_ver.split(".")
            triton_major = int(triton_parts[0])
            triton_minor = int(triton_parts[1])

            # Parse PyTorch major.minor
            torch_parts = torch_ver.split(".")
            torch_major = int(torch_parts[0])
            torch_minor = int(torch_parts[1].split("+")[0])

            # Check compatibility based on the matrix
            if (triton_major, triton_minor) >= (3, 5):
                required_torch = (2, 9)
                required_str = "2.9"
            elif (triton_major, triton_minor) >= (3, 4):
                required_torch = (2, 8)
                required_str = "2.8"
            elif (triton_major, triton_minor) >= (3, 3):
                required_torch = (2, 7)
                required_str = "2.7"
            elif (triton_major, triton_minor) >= (3, 2):
                required_torch = (2, 6)
                required_str = "2.6"
            else:
                # Older Triton, assume compatible
                return True, "Compatible (legacy Triton)"

            if (torch_major, torch_minor) >= required_torch:
                return True, f"Compatible (Triton {triton_major}.{triton_minor}.x works with PyTorch >= {required_str})"
            else:
                return False, f"INCOMPATIBLE: Triton {triton_major}.{triton_minor}.x requires PyTorch >= {required_str} (you have {torch_major}.{torch_minor})"
        except (ValueError, IndexError):
            return True, "Unknown (could not parse versions)"

    def _check_package_update_available(self, package: str) -> Tuple[bool, Optional[str]]:
        """Check if a package has an update available using pip --dry-run.

        Args:
            package: Package name (e.g., "triton-windows")

        Returns:
            Tuple of (has_update, new_version or None)
        """
        try:
            # pip install --dry-run --upgrade shows what would be installed
            result = self.handler.run_command([
                str(self.handler.python_path), "-m", "pip", "install",
                "--dry-run", "--upgrade", package
            ], capture_output=True)

            output = result.stdout + result.stderr if result.stderr else result.stdout

            # Look for "Would install" in output
            # Example: "Would install triton-windows-3.5.2"
            if "Would install" in output:
                # Extract version from output
                import re
                match = re.search(rf'{re.escape(package)}-(\d+\.\d+[^\s,]*)', output, re.IGNORECASE)
                if match:
                    return True, match.group(1)
                return True, None  # Update available but couldn't parse version

            # "Requirement already satisfied" means no update
            if "Requirement already satisfied" in output:
                return False, None

            return False, None
        except Exception:
            return False, None

    def _format_cuda_version(self, cuda_code: str) -> str:
        """Format CUDA version code to human-readable format.

        Args:
            cuda_code: Version like "128" or "126" or "cpu"

        Returns:
            Formatted version like "12.8" or "12.6" or "N/A (CPU)"
        """
        if not cuda_code or cuda_code == "cpu":
            return "N/A (CPU)"
        # Handle 3-digit codes like "128" -> "12.8"
        if len(cuda_code) == 3:
            return f"{cuda_code[0:2]}.{cuda_code[2]}"
        # Handle 2-digit codes like "90" -> "9.0"
        if len(cuda_code) == 2:
            return f"{cuda_code[0]}.{cuda_code[1]}"
        return cuda_code

    def _get_wheel_configs(self) -> List[Tuple]:
        """Get the list of known wheel configurations.

        This is the SINGLE SOURCE OF TRUTH for all wheel configurations.
        Used by: _find_matching_wheel(), _try_install_sageattention_v2(), _build_wheel_url()

        Returns:
            List of tuples: (sage_ver, cuda, torch_pattern, py_spec, tag, is_abi3, is_experimental, torch_filename_ver)

            - sage_ver: SageAttention version (e.g., "2.2.0.post3")
            - cuda: CUDA version code (e.g., "128" for 12.8)
            - torch_pattern: PyTorch pattern for matching (e.g., "2.7" matches 2.7.x, "2.7.0" matches exact)
            - py_spec: Python version spec (None for ABI3, or "312" for exact match)
            - tag: GitHub release tag (e.g., "v2.2.0-windows.post3")
            - is_abi3: Whether this is an ABI3 wheel (Python 3.9+ compatible)
            - is_experimental: Whether this is an experimental/prerelease wheel
            - torch_filename_ver: Exact torch version used in wheel filename (for ABI3 wheels)
        """
        return [
            # === SA 2.2.0.post3 (ABI3 - Python 3.9+) - STABLE, PRIMARY ===
            # torch_filename_ver is the exact version in the wheel filename
            ("2.2.0.post3", "130", "2.9", None, "v2.2.0-windows.post3", True, False, "2.9.0"),
            ("2.2.0.post3", "128", "2.9", None, "v2.2.0-windows.post3", True, False, "2.9.0"),
            ("2.2.0.post3", "128", "2.8", None, "v2.2.0-windows.post3", True, False, "2.8.0"),
            ("2.2.0.post3", "128", "2.7", None, "v2.2.0-windows.post3", True, False, "2.7.1"),
            ("2.2.0.post3", "126", "2.6", None, "v2.2.0-windows.post3", True, False, "2.6.0"),
            ("2.2.0.post3", "124", "2.5", None, "v2.2.0-windows.post3", True, False, "2.5.1"),

            # === SA 2.2.0.post4 (ABI3) - EXPERIMENTAL (torch.compile support) ===
            ("2.2.0.post4", "130", "2.9", None, "v2.2.0-windows.post4", True, True, "2.9.0andhigher"),
            ("2.2.0.post4", "128", "2.9", None, "v2.2.0-windows.post4", True, True, "2.9.0andhigher"),

            # === SA 2.1.1 (per-Python) - for --sage-version 2.1.1 requests ===
            # Non-ABI3 wheels: torch_filename_ver is same as torch_pattern (exact match)
            ("2.1.1", "128", "2.8.0", "313", "v2.1.1-windows", False, False, None),
            ("2.1.1", "128", "2.8.0", "312", "v2.1.1-windows", False, False, None),
            ("2.1.1", "128", "2.8.0", "311", "v2.1.1-windows", False, False, None),
            ("2.1.1", "128", "2.8.0", "310", "v2.1.1-windows", False, False, None),
            ("2.1.1", "128", "2.7.1", "312", "v2.1.1-windows", False, False, None),
            ("2.1.1", "128", "2.7.0", "312", "v2.1.1-windows", False, False, None),
            ("2.1.1", "128", "2.7.0", "311", "v2.1.1-windows", False, False, None),
            ("2.1.1", "126", "2.6.0", "312", "v2.1.1-windows", False, False, None),
            ("2.1.1", "126", "2.6.0", "311", "v2.1.1-windows", False, False, None),
            ("2.1.1", "124", "2.5.1", "312", "v2.1.1-windows", False, False, None),
            ("2.1.1", "124", "2.5.1", "311", "v2.1.1-windows", False, False, None),

            # === Legacy 2.0.1 (per-Python) - backward compat ===
            ("2.0.1", "126", "2.5.0", "312", "v2.0.1-windows", False, False, None),
            ("2.0.1", "121", "2.4.0", "312", "v2.0.1-windows", False, False, None),
            ("2.0.1", "118", "2.4.0", "311", "v2.0.1-windows", False, False, None),
        ]

    def _find_matching_wheel(self, cuda_ver: str, torch_ver: str, python_ver: str,
                             exact_version: Optional[str] = None,
                             include_experimental: bool = False) -> Optional[Dict]:
        """Find matching wheel configuration without installing.

        Args:
            cuda_ver: CUDA version code (e.g., "128")
            torch_ver: PyTorch version (e.g., "2.7.0")
            python_ver: Python version code (e.g., "312")
            exact_version: If set, only match this SA version
            include_experimental: Include experimental wheels

        Returns:
            Dict with wheel info if found, None otherwise.
        """
        if platform.system() != "Windows":
            return None

        wheel_configs = self._get_wheel_configs()

        # Extract torch major.minor for pattern matching
        torch_parts = torch_ver.split(".")
        torch_major_minor = f"{torch_parts[0]}.{torch_parts[1]}" if len(torch_parts) >= 2 else torch_ver
        py_int = int(python_ver)

        for sage_ver, cuda_whl, torch_pattern, py_spec, tag, is_abi3, is_experimental, torch_filename_ver in wheel_configs:
            # Skip experimental unless requested
            if is_experimental and not include_experimental:
                continue

            # Skip if exact version requested and this isn't it
            if exact_version and not sage_ver.startswith(exact_version):
                continue

            # CUDA must match exactly
            if cuda_whl != cuda_ver:
                continue

            # PyTorch matching
            if "." in torch_pattern and torch_pattern.count(".") == 2:
                if torch_ver != torch_pattern:
                    continue
            else:
                if torch_major_minor != torch_pattern:
                    continue

            # Python matching
            if is_abi3:
                if py_int < 39:
                    continue
            else:
                if py_spec != python_ver:
                    continue

            # Found a match - build wheel URL
            wheel_url = self._build_wheel_url(sage_ver, cuda_whl, torch_pattern,
                                              torch_ver, py_spec, tag, is_abi3,
                                              torch_filename_ver)

            return {
                'sage_version': sage_ver,
                'cuda': cuda_whl,
                'torch_pattern': torch_pattern,
                'wheel_url': wheel_url,
                'is_experimental': is_experimental,
                'is_abi3': is_abi3,
                'torch_filename_ver': torch_filename_ver
            }

        return None

    def check_compatibility(self, torch_ver: Optional[str] = None,
                            cuda_ver: Optional[str] = None) -> Dict[str, Any]:
        """Check if current environment is compatible with SA 2.x.

        Args:
            torch_ver: Pre-collected PyTorch version (avoids re-running subprocess)
            cuda_ver: Pre-collected CUDA version code (avoids re-running subprocess)

        Returns:
            Dict with compatibility info.
        """
        try:
            # Use provided values or fetch them
            if torch_ver is None:
                torch_ver = self._get_torch_version()
            if cuda_ver is None:
                cuda_ver = self._get_cuda_version_from_torch()
            python_ver = f"{sys.version_info.major}{sys.version_info.minor}"

            match = self._find_matching_wheel(cuda_ver, torch_ver, python_ver,
                                              include_experimental=self.experimental)

            if match:
                return {
                    'compatible': True,
                    'match': match,
                    'cuda_ver': cuda_ver,
                    'torch_ver': torch_ver,
                    'python_ver': python_ver,
                    'fallback': None,
                    'message': f"Supported (SA {match['sage_version']})"
                }
            else:
                return {
                    'compatible': False,
                    'match': None,
                    'cuda_ver': cuda_ver,
                    'torch_ver': torch_ver,
                    'python_ver': python_ver,
                    'fallback': 'SA 1.0.6',
                    'message': f"No matching wheel for CUDA {self._format_cuda_version(cuda_ver)} + PyTorch {torch_ver}"
                }
        except Exception as e:
            return {
                'compatible': False,
                'match': None,
                'cuda_ver': None,
                'torch_ver': None,
                'python_ver': None,
                'fallback': 'SA 1.0.6',
                'message': f"Could not determine compatibility: {e}"
            }

    def get_environment_info(self) -> Tuple[Dict[str, Dict[str, str]], str, str]:
        """Collect current environment information.

        Collects all data in one pass to minimize subprocess calls.

        Returns:
            Tuple of:
            - Dict with component info, each containing 'version' and 'status'
            - torch_version (raw, for use by check_compatibility)
            - cuda_code (raw, for use by check_compatibility)
        """
        sa_version = self._get_installed_sageattention_version()
        triton_version = self._get_installed_triton_version()
        torch_version = self._get_torch_version()
        cuda_code = self._get_cuda_version_from_torch()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        info = {
            "SageAttention": {
                "version": sa_version or "-",
                "status": "Installed" if sa_version else "Not installed"
            },
            "Triton": {
                "version": triton_version or "-",
                "status": "Installed" if triton_version else "Not installed"
            },
            "PyTorch": {
                "version": torch_version or "-",
                "status": "Installed" if torch_version else "Not installed"
            },
            "CUDA": {
                "version": self._format_cuda_version(cuda_code),
                "status": "Detected" if cuda_code and cuda_code != "cpu" else "CPU only"
            },
            "Python": {
                "version": python_version,
                "status": "Active"
            }
        }
        return info, torch_version, cuda_code

    def show_installed(self) -> None:
        """Display current installation status as a formatted table.

        Collects all data first (which may generate INFO logs), then prints
        the clean formatted output so it's easy to copy-paste for support.
        """
        # Collect all data first (this generates INFO logs from subprocess calls)
        # Pass torch_ver and cuda_ver to check_compatibility to avoid re-fetching
        info, torch_ver, cuda_ver = self.get_environment_info()
        compat = self.check_compatibility(torch_ver=torch_ver, cuda_ver=cuda_ver)
        script_name = Path(sys.argv[0]).name

        # Now print clean output (all INFO logs are above this)
        print()  # Blank line to separate from any INFO logs
        print("=" * 62)
        print("Current Installation")
        print("=" * 62)
        print(f"| {'Component':<15} | {'Version':<28} | {'Status':<9} |")
        print("|" + "-" * 17 + "|" + "-" * 30 + "|" + "-" * 11 + "|")
        for component, data in info.items():
            version = data["version"]
            status = data["status"]
            print(f"| {component:<15} | {version:<28} | {status:<9} |")
        print("=" * 62)
        print(f"{script_name} version: {__version__}")

        # Compatibility status
        print()
        if compat['compatible']:
            print(f"SA 2.x Compatibility: [OK] {compat['message']}")
        else:
            print(f"SA 2.x Compatibility: [NO] {compat['message']}")
            if compat['fallback']:
                print(f"                      Fallback: {compat['fallback']} (~2.1x speedup)")
            print("                      Request support: https://github.com/DazzleML/comfyui-triton-and-sageattention-installer/issues")

        # Triton/PyTorch compatibility check
        triton_ver = info["Triton"]["version"]
        if triton_ver != "-":
            is_compat, compat_msg = self._check_triton_pytorch_compatibility(triton_ver, torch_ver)
            if is_compat:
                print(f"Triton/PyTorch:       [OK] {compat_msg}")
            else:
                print(f"Triton/PyTorch:       [WARNING] {compat_msg}")
                constraint = self._get_triton_version_constraint(torch_ver)
                triton_pkg = "triton-windows" if sys.platform == "win32" else "triton"
                print(f"                      Fix: pip install \"{triton_pkg}{constraint}\"")
                print("                      This affects torch.compile (inductor backend)")
        else:
            print(f"Triton/PyTorch:       [-] Triton not installed")

    def preview_changes(self) -> None:
        """Preview what install/upgrade would do without making changes."""
        # Collect all data first (this generates INFO logs from subprocess calls)
        info, torch_ver, cuda_ver = self.get_environment_info()
        compat = self.check_compatibility(torch_ver=torch_ver, cuda_ver=cuda_ver)

        current_sa = info["SageAttention"]["version"]
        current_triton = info["Triton"]["version"]
        current_torch = info["PyTorch"]["version"]
        current_cuda = info["CUDA"]["version"]
        current_python = info["Python"]["version"]

        # Pre-check Triton update (generates INFO log, must be before formatted output)
        triton_package = "triton-windows" if sys.platform == "win32" else "triton"
        triton_has_update, triton_new_version = (False, None)
        if current_triton != "-" and self.upgrade:
            triton_has_update, triton_new_version = self._check_package_update_available(triton_package)

        # Get Triton version constraint for PyTorch compatibility
        triton_constraint = self._get_triton_version_constraint(torch_ver)

        # Check Triton/PyTorch compatibility
        triton_is_compat = True
        triton_compat_msg = ""
        if current_triton != "-":
            triton_is_compat, triton_compat_msg = self._check_triton_pytorch_compatibility(
                current_triton, torch_ver
            )

        # Determine target versions
        if compat['compatible']:
            target_sa = compat['match']['sage_version']
            target_sa_type = "SA 2.x (~3x speedup)"
            wheel_url = compat['match']['wheel_url']
        else:
            target_sa = "1.0.6"
            target_sa_type = "SA 1.x (~2.1x speedup)"
            wheel_url = "PyPI (pip install sageattention==1.0.6)"

        # Now print clean output (all INFO logs are above this)
        print()  # Blank line to separate from any INFO logs
        print("=" * 70)
        print("DRY RUN - Preview of Changes (no changes will be made)")
        print("=" * 70)

        print()
        print("Current Environment:")
        print(f"  Python:        {current_python}")
        print(f"  PyTorch:       {current_torch}")
        print(f"  CUDA:          {current_cuda}")
        print(f"  Triton:        {current_triton}")
        print(f"  SageAttention: {current_sa}")

        print()
        print("-" * 70)
        print("Proposed Changes:")
        print("-" * 70)

        # Show what would be installed/upgraded
        changes = []

        # PyTorch (only if not installed or wrong CUDA)
        if current_torch == "-":
            changes.append(("PyTorch", "[INSTALL]", f"PyTorch with CUDA (auto-detected)"))
        else:
            changes.append(("PyTorch", "[KEEP]", f"{current_torch} (already installed)"))

        # Triton - check compatibility first, then update availability
        triton_constraint_desc = f" [constraint: {triton_constraint}]" if triton_constraint else ""
        if current_triton == "-":
            changes.append(("Triton", "[INSTALL]", f"{triton_package}{triton_constraint_desc}"))
        elif not triton_is_compat:
            # Incompatible Triton - will be fixed (uninstall + reinstall with constraint)
            changes.append(("Triton", "[FIX]", f"{current_triton} -> compatible version{triton_constraint_desc}"))
        else:
            # Compatible Triton - check for updates if upgrading
            if self.upgrade:
                if triton_has_update:
                    if triton_new_version:
                        changes.append(("Triton", "[UPGRADE]", f"{current_triton} -> {triton_new_version}{triton_constraint_desc}"))
                    else:
                        changes.append(("Triton", "[UPGRADE]", f"{current_triton} -> (newer version available){triton_constraint_desc}"))
                else:
                    changes.append(("Triton", "[KEEP]", f"{current_triton} (compatible, up to date)"))
            else:
                changes.append(("Triton", "[KEEP]", f"{current_triton} (compatible)"))

        # SageAttention
        if current_sa == "-":
            changes.append(("SageAttention", "[INSTALL]", f"{target_sa} - {target_sa_type}"))
        elif self.upgrade:
            # Check if installed version matches target
            if self._sageattention_version_matches(current_sa, target_sa):
                changes.append(("SageAttention", "[KEEP]", f"{current_sa} (already at target version)"))
            else:
                current_major = self._parse_sageattention_major_version(current_sa)
                target_major = 2 if compat['compatible'] else 1
                if current_major == target_major:
                    changes.append(("SageAttention", "[UPGRADE]", f"{target_sa} - {target_sa_type}"))
                else:
                    changes.append(("SageAttention", "[UPGRADE]", f"{target_sa} - {target_sa_type} (version change!)"))
        else:
            changes.append(("SageAttention", "[KEEP]", f"{current_sa} (already installed)"))

        # Custom nodes
        if self.with_custom_nodes:
            changes.append(("VideoHelperSuite", "[INSTALL]", "Custom node for video encoding"))
            changes.append(("DazzleNodes", "[INSTALL]", "Custom node collection"))

        for component, action, detail in changes:
            print(f"  {component:<15} {action:<10} {detail}")

        # Check if there are any actual changes to make
        has_changes = any(action in ("[INSTALL]", "[UPGRADE]", "[FIX]") for _, action, _ in changes)

        print()
        print("-" * 70)
        print("Wheel Details:")
        print("-" * 70)
        if compat['compatible']:
            print(f"  SageAttention wheel: {wheel_url}")
            if compat['match'].get('is_experimental'):
                print(f"  [NOTE] This is an experimental wheel")
        else:
            print(f"  No SA 2.x wheel available for:")
            print(f"    CUDA {current_cuda} + PyTorch {current_torch} + Python {current_python}")
            print(f"  Will install: {wheel_url}")

        # Show Triton/PyTorch compatibility status
        if current_triton != "-":
            is_triton_compat, triton_compat_msg = self._check_triton_pytorch_compatibility(
                current_triton, torch_ver
            )
            print()
            print("-" * 70)
            print("Triton/PyTorch Compatibility:")
            print("-" * 70)
            if is_triton_compat:
                print(f"  [OK] {triton_compat_msg}")
            else:
                print(f"  [WARNING] {triton_compat_msg}")
                print(f"  Fix: pip install \"{triton_package}{triton_constraint}\"")
                print(f"  This affects torch.compile (inductor backend)")

        print()
        print("=" * 70)
        if has_changes:
            print("To execute these changes, run without --dryrun")
        else:
            print("Nothing to do - all components are already up to date")
        print("=" * 70)

    def _parse_sageattention_major_version(self, version_str: str) -> Optional[int]:
        """Extract major version (1 or 2) from sageattention version string.

        Args:
            version_str: Version like "1.0.6", "2.2.0+cu128torch2.7.1.post3"

        Returns:
            1, 2, or None if cannot parse.
        """
        if not version_str:
            return None
        # Version starts with major.minor.patch, possibly with +suffix
        match = re.match(r'^(\d+)\.', version_str)
        if match:
            return int(match.group(1))
        return None

    def _sageattention_version_matches(self, installed: str, target: str) -> bool:
        """Check if installed SA version matches the target wheel version.

        Args:
            installed: Installed version like "2.2.0+cu128torch2.7.1.post3" or "1.0.6"
            target: Target wheel version like "2.2.0.post3" or "1.0.6"

        Returns:
            True if versions match (considering post suffix and local version).
        """
        if not installed or not target or installed == "-":
            return False

        # Normalize target: "2.2.0.post3" -> base="2.2.0", post="post3"
        target_match = re.match(r'^(\d+\.\d+\.\d+)(?:\.?(post\d+))?$', target)
        if not target_match:
            return installed == target  # Fallback to exact match

        target_base = target_match.group(1)  # "2.2.0"
        target_post = target_match.group(2)  # "post3" or None

        # Installed version formats:
        # - "2.2.0+cu128torch2.7.1.post3" (SA 2.x with local version)
        # - "1.0.6" (SA 1.x from PyPI)
        # Extract base version and any post suffix from installed
        installed_match = re.match(r'^(\d+\.\d+\.\d+)(?:\+.*?(post\d+))?$', installed)
        if not installed_match:
            # Try without + suffix (like "1.0.6")
            installed_match = re.match(r'^(\d+\.\d+\.\d+)(?:\.?(post\d+))?$', installed)
            if not installed_match:
                return False

        installed_base = installed_match.group(1)  # "2.2.0"
        installed_post = installed_match.group(2)  # "post3" or None

        # Compare base versions and post suffixes
        if installed_base != target_base:
            return False

        # Both must have same post suffix (or both None)
        return installed_post == target_post

    def _build_wheel_url(self, sage_ver: str, cuda: str, torch_pattern: str,
                         torch_ver: str, py_spec: Optional[str], tag: str, is_abi3: bool,
                         torch_filename_ver: Optional[str] = None) -> str:
        """Build the wheel URL for a SageAttention wheel.

        Args:
            sage_ver: SageAttention version (e.g., "2.2.0.post3")
            cuda: CUDA version code (e.g., "128" for 12.8)
            torch_pattern: PyTorch pattern from config (e.g., "2.7" or "2.7.0")
            torch_ver: Actual PyTorch version installed (e.g., "2.7.0")
            py_spec: Python version spec (None for ABI3, or "312" for exact)
            tag: GitHub release tag (e.g., "v2.2.0-windows.post3")
            is_abi3: Whether this is an ABI3 wheel
            torch_filename_ver: Exact torch version for wheel filename (from _get_wheel_configs)

        Returns:
            Full URL to the wheel file.
        """
        base_url = f"https://github.com/woct0rdho/SageAttention/releases/download/{tag}"

        if is_abi3:
            # ABI3 wheels have format: sageattention-2.2.0+cu128torch2.7.1.post3-cp39-abi3-win_amd64.whl
            # Note: sage version in filename is base (2.2.0), .postX is appended to torch version
            sage_base = sage_ver.split(".post")[0] if ".post" in sage_ver else sage_ver

            # Get post suffix from sage_ver (e.g., "2.2.0.post3" -> ".post3")
            post_suffix = ""
            if ".post" in sage_ver:
                post_idx = sage_ver.find(".post")
                post_suffix = sage_ver[post_idx:]

            # Use torch_filename_ver from config, or fallback
            if torch_filename_ver:
                torch_filename = torch_filename_ver + post_suffix
            else:
                torch_filename = torch_pattern + ".0" + post_suffix

            wheel_name = f"sageattention-{sage_base}+cu{cuda}torch{torch_filename}-cp39-abi3-win_amd64.whl"
        else:
            # Regular wheels: sageattention-2.1.1+cu128torch2.7.0-cp312-cp312-win_amd64.whl
            wheel_name = f"sageattention-{sage_ver}+cu{cuda}torch{torch_pattern}-cp{py_spec}-cp{py_spec}-win_amd64.whl"

        return f"{base_url}/{wheel_name}"

    def _try_install_sageattention_v2(self, exact_version: Optional[str] = None) -> bool:
        """Attempt to install SageAttention 2.x from pre-built wheel.

        Uses _get_wheel_configs() as the single source of truth for wheel configurations.

        Args:
            exact_version: If specified, only try this exact version (e.g., "2.1.1")

        Returns:
            True if installation succeeded, False otherwise.
        """
        if platform.system() != "Windows":
            self.logger.info("SageAttention 2.x pre-built wheels only available on Windows")
            return False

        try:
            torch_ver = self._get_torch_version()
            cuda_ver = self._get_cuda_version_from_torch()
            python_ver = f"{sys.version_info.major}{sys.version_info.minor}"

            print(f"  Detected: PyTorch {torch_ver}, CUDA {cuda_ver}, Python {sys.version_info.major}.{sys.version_info.minor}")

            if exact_version:
                print(f"  Looking for exact version: {exact_version}")
            else:
                print("  Checking for compatible pre-built wheel...")

            # Use centralized wheel configurations
            wheel_configs = self._get_wheel_configs()

            # Extract user's torch major.minor for pattern matching
            torch_parts = torch_ver.split(".")
            torch_major_minor = f"{torch_parts[0]}.{torch_parts[1]}" if len(torch_parts) >= 2 else torch_ver
            py_int = int(python_ver)  # e.g., "312" -> 312

            for sage_ver, cuda_whl, torch_pattern, py_spec, tag, is_abi3, is_experimental, torch_filename_ver in wheel_configs:
                # Skip experimental unless --experimental flag is set
                if is_experimental and not self.experimental:
                    continue

                # Skip if exact version requested and this isn't it
                if exact_version and not sage_ver.startswith(exact_version):
                    continue

                # CUDA must match exactly
                if cuda_whl != cuda_ver:
                    continue

                # PyTorch matching: pattern "2.7" matches "2.7.0", "2.7.1", etc.
                if "." in torch_pattern and torch_pattern.count(".") == 2:
                    # Exact version like "2.7.0" - must match exactly
                    if torch_ver != torch_pattern:
                        continue
                else:
                    # Pattern like "2.7" - match major.minor
                    if torch_major_minor != torch_pattern:
                        continue

                # Python matching
                if is_abi3:
                    # ABI3 wheels work with Python 3.9+
                    if py_int < 39:
                        continue
                else:
                    # Exact Python version required
                    if py_spec != python_ver:
                        continue

                # Build the wheel URL based on wheel type
                wheel_url = self._build_wheel_url(sage_ver, cuda_whl, torch_pattern, torch_ver,
                                                  py_spec, tag, is_abi3, torch_filename_ver)

                try:
                    self.logger.info(f"Trying pre-built wheel: {wheel_url}")
                    self.handler.pip_install([wheel_url])
                    self.installed_packages.append("sageattention")

                    print()
                    print("  " + "-" * 50)
                    print(f"  [OK] Installed SageAttention {sage_ver} (pre-built wheel)")
                    print("       -> ~3x faster than FlashAttention2")
                    print("  " + "-" * 50)
                    self.logger.info(f"Successfully installed SageAttention {sage_ver} from wheel")
                    return True

                except Exception as e:
                    self.logger.debug(f"Wheel not compatible: {e}")
                    continue

        except Exception as e:
            self.logger.debug(f"Could not check for pre-built wheels: {e}")

        return False

    def _install_sageattention_v1(self, exact_version: Optional[str] = None, is_fallback: bool = False):
        """Install SageAttention 1.x from PyPI.

        Args:
            exact_version: If specified, install this exact version (e.g., "1.0.6")
            is_fallback: If True, show fallback message. If False, show direct install message.
        """
        version = exact_version or "1.0.6"  # Default to 1.0.6 if no exact version

        try:
            self.handler.pip_install([f"sageattention=={version}"])
            self.installed_packages.append("sageattention")

            print()
            print("  " + "-" * 50)

            if is_fallback:
                print("  [i] No matching SageAttention 2.x wheel found")
                print(f"  [OK] Installed SageAttention {version} (Triton-based)")
                print("       -> ~2.1x faster than FlashAttention2")
                print()
                print("       For SA2 (~3x speedup), see:")
                print("       https://github.com/woct0rdho/SageAttention/releases")
            else:
                print(f"  [OK] Installed SageAttention {version} (Triton-based)")
                print("       -> ~2.1x faster than FlashAttention2")

            print("  " + "-" * 50)
            self.logger.info(f"Successfully installed SageAttention {version} from PyPI")

        except Exception as e:
            self.logger.error(f"Failed to install SageAttention {version}: {e}")
            print(f"  [X] Failed to install SageAttention {version}: {e}")
            raise

    def _fail_sageattention_v2(self):
        """Print failure message when SA2 explicitly requested but unavailable."""
        print()
        print("  " + "-" * 50)
        print("  [X] SageAttention 2.x installation failed")
        print()
        print(f"      No pre-built wheel available for your configuration:")
        print(f"        {self._get_system_info_string()}")
        print()
        print("      Options:")
        print("        1. Use --sage-version auto to fall back to SA 1.0.6")
        print("        2. Manually install from:")
        print("           https://github.com/woct0rdho/SageAttention/releases")
        print("        3. Open an issue to request this configuration")
        print("  " + "-" * 50)

        self.logger.error("SageAttention 2.x installation failed - no compatible wheel")
        raise ComfyUIInstallerError("SageAttention 2.x not available for this configuration")

    def clone_and_install_repositories(self):
        """Clone and install required repositories."""
        sage_failed = False

        # ─────────────────────────────────────────────────────────────
        # Install SageAttention
        # ─────────────────────────────────────────────────────────────
        print()
        print("=" * 60)
        if self.upgrade:
            print("Upgrading SageAttention")
        else:
            print("Installing SageAttention")
        print("=" * 60)
        print(f"  Requested: --sage-version {self.sage_version_raw}")

        # Handle upgrade mode: detect current version and remove before reinstall
        if self.upgrade:
            current_version = self._get_installed_sageattention_version()
            if current_version:
                current_major = self._parse_sageattention_major_version(current_version)
                print(f"  Current version: {current_version} (SA{current_major})")
                print("  Removing existing installation...")
                try:
                    self.handler.pip_uninstall(["sageattention"])
                except Exception as e:
                    self.logger.debug(f"Could not uninstall sageattention: {e}")
            else:
                print("  No existing SageAttention found, proceeding with install...")

        # Determine install strategy based on parsed version
        major = self.sage_version_major      # None, 1, or 2
        exact = self.sage_version_exact      # None or "X.Y.Z"

        try:
            if major is None:
                # Auto mode: try SA2, fallback to SA1
                print("  Strategy: Try SA2, fallback to SA1")
                if not self._try_install_sageattention_v2():
                    self._install_sageattention_v1(is_fallback=True)

            elif major == 1:
                # SA1 requested (either "1" or "1.0.6")
                if exact:
                    print(f"  Strategy: Install exact version {exact}")
                else:
                    print("  Strategy: Install latest SA1")
                self._install_sageattention_v1(exact_version=exact, is_fallback=False)

            elif major == 2:
                # SA2 requested (either "2" or "2.1.1")
                if exact:
                    print(f"  Strategy: Install exact version {exact}")
                else:
                    print("  Strategy: Install any compatible SA2")

                if not self._try_install_sageattention_v2(exact_version=exact):
                    self._fail_sageattention_v2()

        except ComfyUIInstallerError:
            sage_failed = True
            # Don't re-raise here, continue with other installations

        # ─────────────────────────────────────────────────────────────
        # Setup ComfyUI custom nodes (opt-in with --with-custom-nodes)
        # ─────────────────────────────────────────────────────────────
        if self.with_custom_nodes:
            print()
            print("=" * 60)
            print("Installing Custom Nodes")
            print("=" * 60)

            comfyui_nodes = self.base_path / "ComfyUI" / "custom_nodes"
            comfyui_nodes.mkdir(parents=True, exist_ok=True)

            for node_config in self.NODE_PRESETS["default"]:
                node_name = node_config["name"]
                node_url = node_config["url"]
                node_dir = comfyui_nodes / node_name

                if self._update_or_clone_repo(node_dir, node_url, node_name):
                    # Install requirements if present
                    requirements_file = node_dir / "requirements.txt"
                    if requirements_file.exists():
                        try:
                            self.handler.pip_install(["-r", str(requirements_file)])
                        except Exception as e:
                            self.logger.warning(f"Failed to install {node_name} requirements: {e}")
        
        # If SageAttention failed, raise at the end so we still install other components
        if sage_failed:
            raise ComfyUIInstallerError("Failed to install SageAttention")
    
    def _update_or_clone_repo(self, repo_dir: Path, repo_url: str, repo_name: str) -> bool:
        """Update existing repository or clone if it doesn't exist."""
        if repo_dir.exists() and (repo_dir / ".git").exists():
            if self.force:
                print(f"WARNING: {repo_name} repository exists but --force specified")
                print(f"This will delete existing repository and re-clone fresh copy")
                if self.interactive:
                    response = input(f"Delete and re-clone {repo_name}? (y/N): ")
                    if response.lower() != 'y':
                        print(f"Using existing {repo_name} repository")
                        return True
                # Force mode: delete and re-clone
                shutil.rmtree(repo_dir)
            else:
                try:
                    print(f"Updating existing {repo_name} repository...")
                    # Check if repo is clean (no uncommitted changes)
                    result = self.handler.run_command([
                        "git", "-C", str(repo_dir), "status", "--porcelain"
                    ], capture_output=True, check=False)
                    
                    if result.stdout.strip():
                        self.logger.warning(f"{repo_name} repository has uncommitted changes, skipping update")
                        return True
                    
                    # Update the repository
                    self.handler.run_command([
                        "git", "-C", str(repo_dir), "pull", "origin", "main"
                    ])
                    self.logger.info(f"Updated {repo_name} repository")
                    return True
                    
                except ComfyUIInstallerError:
                    self.logger.warning(f"Failed to update {repo_name}, will re-clone")
                    shutil.rmtree(repo_dir)
        
        # Clone repository
        print(f"Cloning {repo_name} repository...")
        try:
            self.handler.run_command([
                "git", "clone", repo_url, str(repo_dir)
            ])
            self.created_directories.append(repo_dir)
            self.logger.info(f"Cloned {repo_name} repository")
            return True
        except ComfyUIInstallerError as e:
            self.logger.error(f"Failed to clone {repo_name}: {e}")
            return False
    
    def _get_torch_version(self) -> str:
        """Get installed PyTorch version."""
        try:
            result = self.handler.run_command([
                str(self.handler.python_path), "-c",
                "import torch; print(torch.__version__.split('+')[0])"
            ], capture_output=True)
            return result.stdout.strip()
        except Exception:
            return "2.7.0"  # Default to latest
    
    def _get_cuda_version_from_torch(self) -> str:
        """Get CUDA version from PyTorch."""
        try:
            result = self.handler.run_command([
                str(self.handler.python_path), "-c",
                "import torch; print(torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu')"
            ], capture_output=True)
            return result.stdout.strip()
        except Exception:
            return "128"  # Default to CUDA 12.8
    
    def create_run_script(self, cuda_version: str):
        """Create platform-appropriate run script (matches run_nvidia_gpu.bat functionality)."""
        use_sage = cuda_version != "cpu"  # Only use SageAttention if CUDA is available
        script_path = self.handler.create_run_script(use_sage=use_sage, fast_mode=True)
        return script_path
    
    def run_comfyui(self):
        """Run ComfyUI directly (equivalent to running the batch script)."""
        print("Starting ComfyUI...")
        
        cuda_version = self.detect_and_setup_cuda()
        use_sage = cuda_version != "cpu"
        
        # Build arguments matching the batch script
        args = [str(self.handler.python_path), "-s", "ComfyUI/main.py"]
        
        if platform.system() == "Windows":
            args.append("--windows-standalone-build")
        if use_sage:
            args.append("--use-sage-attention")
        args.append("--fast")
        
        try:
            # Run ComfyUI
            self.handler.run_command(args)
        except KeyboardInterrupt:
            print("\nComfyUI stopped by user.")
        except Exception as e:
            print(f"Error running ComfyUI: {e}")
        
        # Pause equivalent (cross-platform)
        input("Press Enter to continue...")
    
    def install(self):
        """Run the complete installation process (matches Step 2 batch script)."""
        if self.force:
            print("FORCE MODE ENABLED")
            print("WARNING: --force will bypass all existing installation checks")
            print("This may:")
            print("   - Reinstall already working components")
            print("   - Overwrite existing configurations") 
            print("   - Break working installations")
            print("   - Delete and re-clone repositories with uncommitted changes")
            print("   - Reinstall build tools and development packages")
            print()
            if self.interactive:
                response = input("Are you sure you want to continue with force mode? (y/N): ")
                if response.lower() != 'y':
                    print("Installation cancelled.")
                    return False
            else:
                print("Non-interactive force mode: proceeding with installation...")
            print()

        if self.experimental:
            print("EXPERIMENTAL MODE ENABLED")
            print("WARNING: You have enabled experimental/prerelease versions.")
            print("These versions may:")
            print("   - Cause black or noisy outputs")
            print("   - Have compatibility issues with some workflows")
            print("   - Be less stable than release versions")
            print()
            print("Use --sage-version auto (without --experimental) to revert to stable.")
            print()
            if self.interactive:
                response = input("Continue with experimental mode? (y/N): ")
                if response.lower() != 'y':
                    print("Installation cancelled.")
                    return False
            else:
                print("Non-interactive experimental mode: proceeding...")
            print()

        sage_attention_failed = False
        
        try:
            print("Starting ComfyUI installation...")
            
            # Step 1: Install build tools (matches batch script flow)
            self.install_build_tools()
            
            # Step 2: Detect CUDA
            cuda_version = self.detect_and_setup_cuda()
            
            # Step 3: Upgrade pip/setuptools
            self.upgrade_pip_setuptools()
            
            # Step 4: Install PyTorch
            self.install_pytorch(cuda_version)
            
            # Step 5: Install Triton
            self.install_triton()
            
            # Step 6: Setup Python dev files (Windows only)
            self.setup_python_dev_files()
            
            # Step 7: Clone and install repositories
            try:
                self.clone_and_install_repositories()
            except ComfyUIInstallerError as e:
                if "Failed to install SageAttention" in str(e):
                    sage_attention_failed = True
                    print("\nWARNING: SageAttention installation failed!")
                    print("This is a known issue on Windows with CUDA compilation.")
                    print("The rest of ComfyUI will still work, but without SageAttention acceleration.")
                    
                    if self.interactive:
                        response = input("\nContinue installation without SageAttention? (Y/n): ")
                        if response.lower() == 'n':
                            raise
                    else:
                        print("Non-interactive mode: continuing without SageAttention...")
                else:
                    raise
            
            # Step 8: Create run script
            self.create_run_script(cuda_version)
            
            if sage_attention_failed:
                print("\nWARNING: Installation completed with warnings!")
                print("WARNING: SageAttention could not be installed due to compilation issues.")
                print("WARNING: You can try installing it manually later or use ComfyUI without it.")
            else:
                print("\nSuccess!")
            
            print()
            self.logger.info("Installation completed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Installation failed: {e}")
            self.logger.info("Running cleanup...")
            self.cleanup_installation()
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-platform ComfyUI with Triton and SageAttention installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --install                    # Install everything (Step 2)
  %(prog)s --cleanup                    # Clean up previous installation (Step 1)
  %(prog)s --run                        # Run ComfyUI (equivalent to run_nvidia_gpu.bat)
  %(prog)s --install --verbose          # Install with verbose output
  %(prog)s --install --force            # Force reinstall all components (original script behavior)
  %(prog)s --install --base-path /opt/comfyui  # Install to specific directory
  %(prog)s --install --non-interactive --force  # Automated forced install (CI/Docker)
        """
    )
    
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run the installation process (equivalent to Step 2 batch script)"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true", 
        help="Clean up previous installation (equivalent to Step 1 batch script)"
    )
    
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run ComfyUI (equivalent to run_nvidia_gpu.bat)"
    )
    
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path.cwd(),
        help="Base installation directory (default: current directory)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force installation/reinstallation of all components (bypasses existing installation checks)"
    )
    
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (no user prompts, safer defaults)"
    )

    parser.add_argument(
        "--sage-version",
        default="auto",
        metavar="VERSION",
        help="SageAttention version to install: "
             "'auto' (try 2.x, fallback to 1.x - default), "
             "'1' (any 1.x, ~2.1x speedup), "
             "'2' (any 2.x, ~3x speedup). "
             "Advanced: use exact version like '2.1.1'"
    )

    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Allow installation of experimental/prerelease SageAttention versions. "
             "Use with caution - these versions may cause issues."
    )

    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade existing SageAttention installation to latest compatible version. "
             "Removes current installation before reinstalling. "
             "Use with --sage-version for explicit target, or --experimental for prerelease versions."
    )

    parser.add_argument(
        "--with-custom-nodes",
        action="store_true",
        help="Install recommended custom nodes (VideoHelperSuite, DazzleNodes). "
             "Omit this flag for minimal Triton/SageAttention-only installation."
    )

    parser.add_argument(
        "--show-installed",
        action="store_true",
        help="Display current installation status (SageAttention, Triton, PyTorch, CUDA, Python)"
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Preview what would be installed/upgraded without making changes"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    args = parser.parse_args()

    # Check for mutually exclusive options
    if args.install and args.upgrade:
        parser.error("--install and --upgrade are mutually exclusive. Use --upgrade to upgrade existing installation.")

    # --dryrun requires --install or --upgrade
    if args.dryrun and not (args.install or args.upgrade):
        parser.error("--dryrun requires --install or --upgrade")

    if not (args.install or args.cleanup or args.run or args.upgrade or args.show_installed or args.dryrun):
        parser.print_help()
        return 1

    # Create installer instance
    installer = ComfyUIInstaller(
        base_path=args.base_path,
        verbose=args.verbose,
        interactive=not args.non_interactive,
        force=args.force,
        sage_version=args.sage_version,
        experimental=args.experimental,
        upgrade=args.upgrade,
        with_custom_nodes=args.with_custom_nodes
    )

    # Handle --show-installed (standalone action, exit after displaying)
    if args.show_installed:
        installer.show_installed()
        return 0

    # Handle --dryrun (preview changes without executing)
    if args.dryrun:
        installer.preview_changes()
        return 0

    success = True

    if args.cleanup:
        installer.cleanup_installation()

    if args.install:
        success = installer.install()
        if not success:
            return 1

    if args.upgrade:
        success = installer.install()  # install() handles upgrade logic internally
        if not success:
            return 1

    if args.run:
        installer.run_comfyui()

    return 0


if __name__ == "__main__":
    sys.exit(main())