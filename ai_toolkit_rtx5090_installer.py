#!/usr/bin/env python3
"""
Ostris AI Toolkit RTX 5090 Compatibility Installer

This installer helps setup the Ostris AI Toolkit for RTX 5090 GPUs by:
1. Installing PyTorch with sm_120 support (CUDA 12.8 nightly)
2. Configuring the environment for Blackwell architecture
3. Disabling incompatible features (quantization, flash attention) when necessary
4. Providing WSL2 setup guidance for Windows users
5. Creating optimized configuration files for training

Based on the successful ComfyUI Triton installer pattern.
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Version information
__version__ = "1.0.0"


class AIToolkitInstallerError(Exception):
    """Base exception for installer errors."""
    pass


class AIToolkitInstaller:
    """Main installer class for Ostris AI Toolkit RTX 5090 compatibility."""

    def __init__(self, base_path: Optional[Path] = None, verbose: bool = False,
                 interactive: bool = True, force: bool = False):
        self.base_path = base_path or Path.cwd()
        self.interactive = interactive
        self.force = force
        self.setup_logging(verbose)
        self.venv_path = self.base_path / "venv"
        self.python_path = self._get_python_path()
        self.cuda_version = None
        self.gpu_info = None
        self.is_rtx5090 = False
        self.is_blackwell = False

    def setup_logging(self, verbose: bool):
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.base_path / 'ai_toolkit_install.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_python_path(self) -> Path:
        """Get the Python interpreter path."""
        if platform.system() == "Windows":
            venv_python = self.venv_path / "Scripts" / "python.exe"
        else:
            venv_python = self.venv_path / "bin" / "python"

        if venv_python.exists():
            return venv_python
        return Path(sys.executable)

    def run_command(self, cmd: List[str], check: bool = True, capture_output: bool = False,
                   shell: bool = False, env: dict = None) -> subprocess.CompletedProcess:
        """Run a command with proper error handling."""
        self.logger.info(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

        if env is None:
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

        try:
            if capture_output:
                result = subprocess.run(
                    cmd, check=check, capture_output=True, text=True, shell=shell, env=env
                )
            else:
                result = subprocess.run(
                    cmd, check=check, text=True, shell=shell, env=env
                )

            if capture_output and result.stdout:
                self.logger.debug(f"Output: {result.stdout}")
            return result

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
            self.logger.error(f"Error: {e}")
            if hasattr(e, 'stdout') and e.stdout:
                self.logger.error(f"Stdout: {e.stdout}")
            if hasattr(e, 'stderr') and e.stderr:
                self.logger.error(f"Stderr: {e.stderr}")
            raise AIToolkitInstallerError(f"Command failed: {e}")

    def detect_gpu_and_cuda(self) -> Dict:
        """Detect GPU information and CUDA compatibility."""
        print("\n[DETECTING] GPU and CUDA environment...")

        info = {
            "gpu_name": None,
            "cuda_available": False,
            "cuda_version": None,
            "driver_version": None,
            "compute_capability": None,
            "is_rtx5090": False,
            "is_blackwell": False,
            "pytorch_version": None,
            "pytorch_cuda": None,
            "pytorch_arch_list": None
        }

        # Check nvidia-smi
        try:
            result = self.run_command(
                ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                 "--format=csv,noheader"],
                capture_output=True
            )
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                info["gpu_name"] = parts[0]
                info["driver_version"] = parts[1]
                info["compute_capability"] = parts[2]

                # Check for RTX 5090 / Blackwell
                if "5090" in info["gpu_name"] or "5080" in info["gpu_name"] or "5070" in info["gpu_name"]:
                    info["is_rtx5090"] = True
                    info["is_blackwell"] = True
                elif info["compute_capability"].startswith("12."):
                    info["is_blackwell"] = True

                self.logger.info(f"GPU: {info['gpu_name']}")
                self.logger.info(f"Compute Capability: {info['compute_capability']}")

        except (AIToolkitInstallerError, FileNotFoundError):
            self.logger.warning("nvidia-smi not found or failed")

        # Check CUDA version
        try:
            result = self.run_command(["nvcc", "--version"], capture_output=True)
            version_match = re.search(r'release (\d+\.\d+)', result.stdout)
            if version_match:
                info["cuda_version"] = version_match.group(1)
                self.logger.info(f"CUDA Version: {info['cuda_version']}")
        except (AIToolkitInstallerError, FileNotFoundError):
            self.logger.warning("nvcc not found")

        # Check current PyTorch
        try:
            result = self.run_command([
                str(self.python_path), "-c",
                "import torch; import json; "
                "print(json.dumps({"
                "'version': torch.__version__, "
                "'cuda_available': torch.cuda.is_available(), "
                "'cuda_version': torch.version.cuda if torch.cuda.is_available() else None, "
                "'arch_list': torch.cuda.get_arch_list() if torch.cuda.is_available() else []"
                "}))"
            ], capture_output=True)

            pytorch_info = json.loads(result.stdout)
            info["pytorch_version"] = pytorch_info["version"]
            info["cuda_available"] = pytorch_info["cuda_available"]
            info["pytorch_cuda"] = pytorch_info["cuda_version"]
            info["pytorch_arch_list"] = pytorch_info["arch_list"]

            self.logger.info(f"PyTorch: {info['pytorch_version']}")
            self.logger.info(f"PyTorch CUDA: {info['pytorch_cuda']}")
            self.logger.info(f"Supported Architectures: {info['pytorch_arch_list']}")

        except (AIToolkitInstallerError, json.JSONDecodeError, Exception):
            self.logger.warning("PyTorch not installed or not importable")

        self.gpu_info = info
        self.is_rtx5090 = info["is_rtx5090"]
        self.is_blackwell = info["is_blackwell"]
        self.cuda_version = info["cuda_version"] or info["pytorch_cuda"]

        return info

    def check_sm120_support(self) -> bool:
        """Check if current PyTorch installation supports sm_120."""
        if not self.gpu_info or not self.gpu_info["pytorch_arch_list"]:
            return False

        arch_list = self.gpu_info["pytorch_arch_list"]
        return any("sm_120" in arch or "12.0" in arch for arch in arch_list)

    def install_pytorch_nightly(self):
        """Install PyTorch nightly with CUDA 12.8 for sm_120 support."""
        print("\n[INSTALLING] PyTorch with sm_120 support...")

        # Check if already has sm_120 support
        if not self.force and self.check_sm120_support():
            print("[OK] PyTorch already supports sm_120")
            return

        # Uninstall existing PyTorch
        print("Removing existing PyTorch installation...")
        try:
            self.run_command([
                str(self.python_path), "-m", "pip", "uninstall", "-y",
                "torch", "torchvision", "torchaudio"
            ])
        except AIToolkitInstallerError:
            self.logger.warning("Some PyTorch packages may not have been installed")

        # Install nightly with CUDA 12.8
        print("Installing PyTorch nightly with CUDA 12.8...")

        if platform.system() == "Windows":
            # Windows nightly build
            self.run_command([
                str(self.python_path), "-m", "pip", "install",
                "--pre", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/nightly/cu128"
            ])
        else:
            # Linux/WSL nightly build
            self.run_command([
                str(self.python_path), "-m", "pip", "install",
                "--pre", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/nightly/cu128"
            ])

        print("[OK] PyTorch nightly installed")

        # Re-detect to verify
        self.detect_gpu_and_cuda()
        if self.check_sm120_support():
            print("[OK] sm_120 support confirmed!")
        else:
            print("[WARNING] sm_120 support not detected - may need WSL2 or different build")

    def disable_incompatible_features(self):
        """Disable features that don't work with sm_120 yet."""
        print("\n[CONFIGURING] RTX 5090 compatibility...")

        # Create a compatibility config
        compat_config = {
            "quantization": {
                "enabled": False,
                "reason": "bitsandbytes doesn't support sm_120 yet"
            },
            "flash_attention": {
                "enabled": False,
                "reason": "Flash Attention needs sm_120 compilation"
            },
            "xformers": {
                "enabled": False,
                "reason": "xFormers may not support sm_120"
            },
            "optimizer": "adamw",  # Not adamw8bit
            "precision": "bf16",
            "recommendations": [
                "Use standard attention (sdpa)",
                "Use bf16 precision",
                "Use regular AdamW optimizer",
                "Consider WSL2 for full compatibility"
            ]
        }

        # Save compatibility report
        report_path = self.base_path / "rtx5090_compatibility.json"
        with open(report_path, 'w') as f:
            json.dump(compat_config, f, indent=2)

        print(f"[OK] Compatibility configuration saved to {report_path}")

        # Uninstall incompatible packages
        print("\nRemoving incompatible packages...")
        packages_to_remove = ["bitsandbytes", "flash-attn", "xformers"]

        for package in packages_to_remove:
            try:
                self.run_command([
                    str(self.python_path), "-m", "pip", "uninstall", "-y", package
                ])
                print(f"  [OK] Removed {package}")
            except AIToolkitInstallerError:
                self.logger.debug(f"{package} was not installed")

        return compat_config

    def create_optimized_config(self):
        """Create an optimized training configuration for RTX 5090."""
        print("\n[CREATING] Optimized training configuration...")

        config = {
            "job": "extension",
            "config": {
                "name": "rtx5090_optimized_lora",
                "process": [{
                    "type": "sd_trainer",
                    "training_folder": "output",
                    "device": "cuda:0",
                    "network": {
                        "type": "lora",
                        "linear": 16,
                        "linear_alpha": 16
                    },
                    "save": {
                        "dtype": "float16",
                        "save_every": 250,
                        "max_step_saves_to_keep": 4
                    },
                    "datasets": [{
                        "folder_path": str(self.base_path / "datasets" / "your_dataset"),
                        "caption_ext": "txt",
                        "caption_dropout_rate": 0.05,
                        "shuffle_tokens": False,
                        "cache_latents_to_disk": True,
                        "resolution": [512, 768, 1024]
                    }],
                    "train": {
                        "batch_size": 1,
                        "cache_text_embeddings": True,
                        "steps": 2000,
                        "gradient_accumulation": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw",  # Not adamw8bit for RTX 5090
                        "lr": 1e-4,
                        "dtype": "bf16"
                    },
                    "model": {
                        "name_or_path": "Qwen/Qwen-Image",
                        "arch": "qwen_image",
                        "quantize": False,  # Disabled for RTX 5090
                        "quantize_te": False,  # Disabled for RTX 5090
                        "low_vram": False  # RTX 5090 has plenty of VRAM
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": 250,
                        "width": 1024,
                        "height": 1024,
                        "prompts": [
                            "a photo of a person",
                            "a portrait in natural lighting"
                        ],
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 3,
                        "sample_steps": 25
                    }
                }],
                "meta": {
                    "name": "RTX 5090 Optimized Config",
                    "version": "1.0",
                    "gpu": "RTX 5090",
                    "compatibility_mode": True
                }
            }
        }

        # Save config
        config_path = self.base_path / "config" / "rtx5090_qwen_lora.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"[OK] Optimized config saved to {config_path}")

        return config_path

    def install_dependencies(self):
        """Install AI Toolkit dependencies with RTX 5090 compatibility."""
        print("\n[INSTALLING] AI Toolkit dependencies...")

        requirements_path = self.base_path / "requirements.txt"
        if not requirements_path.exists():
            print("[WARNING] requirements.txt not found")
            return

        # Read requirements and filter out incompatible packages
        with open(requirements_path, 'r') as f:
            requirements = f.readlines()

        # Filter out problematic packages for RTX 5090
        skip_packages = ["bitsandbytes", "flash-attn", "xformers"]
        filtered_requirements = []

        for req in requirements:
            req = req.strip()
            if req and not req.startswith("#"):
                if not any(skip in req.lower() for skip in skip_packages):
                    filtered_requirements.append(req)
                else:
                    self.logger.info(f"Skipping incompatible package: {req}")

        # Create temporary requirements file
        temp_req_path = self.base_path / "requirements_rtx5090.txt"
        with open(temp_req_path, 'w') as f:
            f.write('\n'.join(filtered_requirements))

        # Install filtered requirements
        self.run_command([
            str(self.python_path), "-m", "pip", "install", "-r", str(temp_req_path)
        ])

        print("[OK] Dependencies installed")

        # Clean up
        temp_req_path.unlink()

    def create_wsl_setup_script(self):
        """Create a WSL2 setup script for Windows users."""
        if platform.system() != "Windows":
            return

        print("\n[CREATING] WSL2 setup guide...")

        wsl_script = '''#!/bin/bash
# WSL2 Setup Script for Ostris AI Toolkit with RTX 5090

echo "=== WSL2 Setup for RTX 5090 ==="
echo "This script will help you setup the AI Toolkit in WSL2"
echo ""

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
echo "Installing Python and development tools..."
sudo apt install -y python3.11 python3.11-venv python3-pip git curl wget build-essential

# Install CUDA toolkit for WSL
echo "Installing CUDA toolkit for WSL..."
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8

# Clone AI Toolkit
echo "Cloning Ostris AI Toolkit..."
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit

# Create virtual environment
echo "Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch nightly with CUDA 12.8
echo "Installing PyTorch nightly with sm_120 support..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install dependencies
echo "Installing AI Toolkit dependencies..."
pip install -r requirements.txt

# Test GPU detection
echo "Testing GPU detection..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Arch List: {torch.cuda.get_arch_list()}')"

echo ""
echo "=== Setup Complete ==="
echo "To activate the environment in future sessions:"
echo "  cd ai-toolkit"
echo "  source venv/bin/activate"
echo ""
echo "To start training:"
echo "  python run.py --config config/rtx5090_qwen_lora.yaml"
'''

        wsl_script_path = self.base_path / "setup_wsl2.sh"
        with open(wsl_script_path, 'w', newline='\n') as f:  # Unix line endings
            f.write(wsl_script)

        print(f"[OK] WSL2 setup script created: {wsl_script_path}")
        print("\nTo use WSL2 (recommended for RTX 5090):")
        print("1. Install WSL2: wsl --install")
        print("2. Copy ai-toolkit folder to WSL: cp -r . /mnt/c/path/in/wsl/")
        print("3. Run setup script: bash setup_wsl2.sh")

    def test_installation(self):
        """Test the installation for RTX 5090 compatibility."""
        print("\n[TESTING] Installation...")

        tests = []

        # Test 1: PyTorch import
        try:
            result = self.run_command([
                str(self.python_path), "-c",
                "import torch; print(f'PyTorch {torch.__version__} imported successfully')"
            ], capture_output=True)
            tests.append(("PyTorch Import", True, result.stdout.strip()))
        except AIToolkitInstallerError as e:
            tests.append(("PyTorch Import", False, str(e)))

        # Test 2: CUDA availability
        try:
            result = self.run_command([
                str(self.python_path), "-c",
                "import torch; "
                "assert torch.cuda.is_available(), 'CUDA not available'; "
                "print(f'CUDA available: {torch.cuda.get_device_name(0)}')"
            ], capture_output=True)
            tests.append(("CUDA Availability", True, result.stdout.strip()))
        except AIToolkitInstallerError as e:
            tests.append(("CUDA Availability", False, "CUDA not available"))

        # Test 3: sm_120 support
        sm120_supported = self.check_sm120_support()
        if sm120_supported:
            tests.append(("sm_120 Support", True, "Blackwell architecture supported"))
        else:
            tests.append(("sm_120 Support", False, "sm_120 not in arch list - consider WSL2"))

        # Test 4: Basic tensor operation
        try:
            result = self.run_command([
                str(self.python_path), "-c",
                "import torch; "
                "x = torch.randn(100, 100).cuda(); "
                "y = torch.matmul(x, x); "
                "print(f'Tensor operation successful: {y.shape}')"
            ], capture_output=True)
            tests.append(("GPU Tensor Operations", True, result.stdout.strip()))
        except AIToolkitInstallerError as e:
            tests.append(("GPU Tensor Operations", False, "Failed - may need sm_120 support"))

        # Print test results
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)

        for test_name, passed, message in tests:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{status} - {test_name}: {message}")

        all_passed = all(passed for _, passed, _ in tests)

        if all_passed:
            print("\n[SUCCESS] All tests passed! Your RTX 5090 is ready for training.")
        else:
            print("\n[WARNING] Some tests failed. Consider using WSL2 for better compatibility.")

        return all_passed

    def run_installation(self, mode: str = "auto"):
        """Run the complete installation process."""
        print("="*60)
        print("Ostris AI Toolkit RTX 5090 Compatibility Installer")
        print(f"Version: {__version__}")
        print("="*60)

        # Detect environment
        self.detect_gpu_and_cuda()

        if not self.is_blackwell:
            print("\n[OK] Non-Blackwell GPU detected. Standard installation should work.")
            if not self.force:
                return

        print(f"\n[DETECTED] RTX 50-series/Blackwell GPU!")
        print("This installer will configure your environment for sm_120 support.")

        if mode == "auto":
            # Automatic installation
            if platform.system() == "Windows":
                print("\n[WARNING] Windows detected. WSL2 is recommended for best compatibility.")
                if self.interactive:
                    response = input("Continue with Windows native installation? (y/N): ")
                    if response.lower() != 'y':
                        print("\nCreating WSL2 setup script instead...")
                        self.create_wsl_setup_script()
                        return

            # Install PyTorch nightly
            self.install_pytorch_nightly()

            # Disable incompatible features
            self.disable_incompatible_features()

            # Install dependencies
            self.install_dependencies()

            # Create optimized config
            self.create_optimized_config()

            # Create WSL script for reference
            if platform.system() == "Windows":
                self.create_wsl_setup_script()

            # Test installation
            self.test_installation()

        elif mode == "diagnose":
            # Just diagnose and report
            print("\n" + "="*50)
            print("DIAGNOSTIC REPORT")
            print("="*50)

            for key, value in self.gpu_info.items():
                print(f"{key}: {value}")

            if self.check_sm120_support():
                print("\n[OK] sm_120 support detected")
            else:
                print("\n[ERROR] sm_120 support NOT detected")
                print("\nRecommendations:")
                print("1. Use WSL2 with Ubuntu 22.04/24.04")
                print("2. Install PyTorch nightly with CUDA 12.8")
                print("3. Disable quantization and Flash Attention")

        print("\n" + "="*60)
        print("Installation complete!")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ostris AI Toolkit RTX 5090 Compatibility Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     # Run automatic installation
  %(prog)s --diagnose          # Diagnose GPU and environment only
  %(prog)s --force             # Force reinstall even if not RTX 5090
  %(prog)s --non-interactive   # Run without prompts
  %(prog)s --verbose           # Show detailed logging
        """
    )

    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Only diagnose GPU and environment without installing"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force installation even if GPU is not Blackwell"
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without user prompts"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path.cwd(),
        help="Base installation directory (default: current directory)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    args = parser.parse_args()

    # Create installer instance
    installer = AIToolkitInstaller(
        base_path=args.base_path,
        verbose=args.verbose,
        interactive=not args.non_interactive,
        force=args.force
    )

    # Run installation or diagnosis
    mode = "diagnose" if args.diagnose else "auto"

    try:
        installer.run_installation(mode=mode)
        return 0
    except Exception as e:
        print(f"\n[ERROR] {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())