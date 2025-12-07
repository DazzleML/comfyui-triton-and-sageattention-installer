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
from typing import Dict, List, Optional, Tuple, Union

# Version information
__version__ = "0.5.0"


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
        "flow2_wan_video": "https://github.com/Flow-Two/flow2-wan-video.git",
        "videohelper_suite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
    }
    
    INCLUDE_LIBS_URL = "https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip"
    
    # Packages to track for cleanup (matches batch script exactly)
    TRITON_PACKAGES = [
        "triton-windows", "triton", "sageattention", 
        "torch", "torchvision", "torchaudio"
    ]
    
    def __init__(self, base_path: Optional[Path] = None, verbose: bool = False, interactive: bool = True, force: bool = False, sage_version: str = "auto"):
        self.base_path = base_path or Path.cwd()
        self.interactive = interactive
        self.force = force
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
            "ComfyUI/custom_nodes/flow2-wan-video",
            "ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite"
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
        """Install Triton."""
        print("Installing Triton...")
        
        if platform.system() == "Windows":
            package = "triton-windows"
        else:
            package = "triton"
        
        self.handler.pip_install([package], ["-U", "--pre"])
        self.installed_packages.append(package)
    
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

    def _try_install_sageattention_v2(self, exact_version: Optional[str] = None) -> bool:
        """Attempt to install SageAttention 2.x from pre-built wheel.

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

            # Known working wheel combinations from woct0rdho
            wheel_configs = [
                ("2.1.1", "128", "270", "312"),  # CUDA 12.8, PyTorch 2.7.0, Python 3.12
                ("2.1.1", "126", "260", "312"),  # CUDA 12.6, PyTorch 2.6.0, Python 3.12
                ("2.0.1", "126", "250", "312"),  # CUDA 12.6, PyTorch 2.5.0, Python 3.12
                ("2.0.1", "121", "240", "312"),  # CUDA 12.1, PyTorch 2.4.0, Python 3.12
                ("2.0.1", "118", "240", "311"),  # CUDA 11.8, PyTorch 2.4.0, Python 3.11
            ]

            for sage_ver, cuda_whl, torch_whl, py_whl in wheel_configs:
                # Skip if exact version requested and this isn't it
                if exact_version and sage_ver != exact_version:
                    continue

                if py_whl != python_ver:
                    continue  # Skip incompatible Python versions

                wheel_url = f"https://github.com/woct0rdho/SageAttention/releases/download/v{sage_ver}-windows/sageattention-{sage_ver}+cu{cuda_whl}torch{torch_whl[0]}.{torch_whl[1]}.{torch_whl[2]}-cp{py_whl}-cp{py_whl}-win_amd64.whl"

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
        print("Installing SageAttention")
        print("=" * 60)
        print(f"  Requested: --sage-version {self.sage_version_raw}")

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
        # Setup ComfyUI custom nodes directory
        # ─────────────────────────────────────────────────────────────
        comfyui_nodes = self.base_path / "ComfyUI" / "custom_nodes"
        comfyui_nodes.mkdir(parents=True, exist_ok=True)
        
        # Clone flow2-wan-video
        flow2_dir = comfyui_nodes / "flow2-wan-video"
        if self._update_or_clone_repo(flow2_dir, self.REPOSITORIES["flow2_wan_video"], "flow2-wan-video"):
            # Install flow2-wan-video requirements
            requirements_file = flow2_dir / "requirements.txt"
            if requirements_file.exists():
                try:
                    self.handler.pip_install(["-r", str(requirements_file)])
                except Exception as e:
                    self.logger.warning(f"Failed to install flow2-wan-video requirements: {e}")
        
        # Clone VideoHelperSuite
        video_dir = comfyui_nodes / "ComfyUI-VideoHelperSuite"
        if self._update_or_clone_repo(video_dir, self.REPOSITORIES["videohelper_suite"], "ComfyUI-VideoHelperSuite"):
            # Install VideoHelperSuite requirements
            requirements_file = video_dir / "requirements.txt"
            if requirements_file.exists():
                try:
                    self.handler.pip_install(["-r", str(requirements_file)])
                except Exception as e:
                    self.logger.warning(f"Failed to install ComfyUI-VideoHelperSuite requirements: {e}")
        
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
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    args = parser.parse_args()
    
    if not (args.install or args.cleanup or args.run):
        parser.print_help()
        return 1
    
    # Create installer instance
    installer = ComfyUIInstaller(
        base_path=args.base_path,
        verbose=args.verbose,
        interactive=not args.non_interactive,
        force=args.force,
        sage_version=args.sage_version
    )
    
    success = True
    
    if args.cleanup:
        installer.cleanup_installation()
    
    if args.install:
        success = installer.install()
        if not success:
            return 1
    
    if args.run:
        installer.run_comfyui()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())