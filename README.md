# ComfyUI Triton and SageAttention Installer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Windows](https://img.shields.io/badge/platform-Windows-blue.svg)](https://www.microsoft.com/windows)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Installs](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/djdarcy/77f23ace7465637447db0a6c79cf46ba/raw/installs.json)](https://dazzleml.github.io/comfyui-triton-and-sageattention-installer/stats/)
[![GitHub Discussions](https://img.shields.io/github/discussions/DazzleML/comfyui-triton-and-sageattention-installer)](https://github.com/DazzleML/comfyui-triton-and-sageattention-installer/discussions)

Cross-platform installer for Triton and SageAttention on ComfyUI. Simplifies GPU-accelerated inference setup for Windows users with automated dependency management and RTX 5090 support.

## Overview

Installing SageAttention on Windows has been notoriously difficult due to compilation issues, missing dependencies, and platform-specific challenges. This installer automates the entire process, making advanced GPU optimizations accessible to all ComfyUI users.

## Features

- **One-command installation** - No manual compilation or complex setup required
- **Windows-first design** - Specifically addresses Windows compilation challenges  
- **Automatic CUDA detection** - Detects your GPU and CUDA version automatically
- **Pre-built wheel support** - Uses pre-compiled wheels to avoid build issues
- **Visual Studio automation** - Handles Build Tools installation if needed
- **Python header management** - Automatically sets up development headers
- **RTX 5090 ready** - Full support for Blackwell architecture with CUDA 12.8
- **Graceful fallbacks** - Continues installation even if some components fail
- **Detailed logging** - Comprehensive logs for troubleshooting

## Prerequisites

- Windows 10/11 (stubs for Linux/macOS experimental)
- NVIDIA GPU with CUDA support
- Existing ComfyUI installation
- Python 3.8+ (or ComfyUI portable's embedded Python)

## Installation

### Quick Start

```bash
python comfyui_triton_sageattention.py --install --with-custom-nodes --backup
```

**Note:** Run from your ComfyUI directory by doing `cd C:\path\to\ComfyUI`, or use `--base-path C:\path\to\ComfyUI` to specify the location. If you're unsure where ComfyUI is installed, use `--install discover` to automatically find it.

By default `comfyui_triton_sageattention.py` installs Triton and SageAttention 2 (falls back to 1 if unavailable). Add `--with-custom-nodes` to also install VideoHelperSuite and DazzleNodes. It's recommended to add `--backup` to automatically save your environment first, in case you opt to restore later.

### Find ComfyUI Installations

Not sure where ComfyUI is installed? The installer can find it for you:

```bash
# List all found ComfyUI installations
python comfyui_triton_sageattention.py --show-installed locations

# Install with interactive discovery (finds and lets you select)
python comfyui_triton_sageattention.py --install discover
```

### Check Current Installation

See what's already installed before or after running the installer:

```bash
python comfyui_triton_sageattention.py --show-installed
```

### Preview Changes (Dry Run)

Preview what would be installed or upgraded without making any changes:

```bash
# Preview install
python comfyui_triton_sageattention.py --install --dryrun  #or --upgrade
```

### Additional Installation Options

```bash
# Verbose mode for debugging
python comfyui_triton_sageattention.py --install --verbose

# Custom ComfyUI location
python comfyui_triton_sageattention.py --install --base-path C:\path\to\comfyui

# Cleanup previous installation
python comfyui_triton_sageattention.py --cleanup
```

### ComfyUI Portable Support

For [ComfyUI Portable](https://docs.comfy.org/installation/comfyui_portable_windows) distributions (the pre-packaged version with `python_embeded` folder):

```bash
# Run from your ComfyUI Portable directory
cd C:\ComfyUI_windows_portable
python path\to\comfyui_triton_sageattention.py --install
# Or to run from any folder without "cd" simply add: --base-path C:\ComfyUI_windows_portable

# Check what the installer detected
python comfyui_triton_sageattention.py --show-installed --base-path C:\ComfyUI_windows_portable
```

The installer automatically detects the `python_embeded` folder and uses that Python environment instead of creating a new virtual environment. All packages are installed into the portable distribution's embedded Python.

### Python Environment Selection

Control which Python environment the installer uses with the `--python` flag:

```bash
# Default --install auto-detects (portable > .venv > venv > system)
# To override, use --python with one of the options:
python comfyui_triton_sageattention.py --install --python {auto, system, venv, portable, .\path\...}
# --python also supports relative / absolute paths: .\path\to\venv, C:\Python312\python.exe
```

**Note:** To distinguish keywords from folder names, paths must include a separator (`/`, `\`, or start with `./`). For example, `--python venv` uses the keyword (venv at base-path), while `--python .\venv` uses the folder in the current working directory.

For full parameter documentation, see [docs/parameters.md](docs/parameters.md).

### SageAttention Version Control

Control which SageAttention version gets installed (default: SA 2.2.0.post3):

```bash
# Explicitly install SageAttention 1 (~2.1x speedup, more compatible) or 2 (~3x speedup)
python comfyui_triton_sageattention.py --install --sage-version 1  # or 2

# Opt-in to experimental/prerelease versions (use with caution)
python comfyui_triton_sageattention.py --install --experimental

# Upgrade existing installation to latest version
python comfyui_triton_sageattention.py --upgrade
```

For a full list of supported PyTorch/CUDA/Python combinations, see [docs/supported_wheels.md](docs/supported_wheels.md).

### Environment Backup

Before making changes, it's helpful to create a backup of your environment:

```bash
# Create a backup before upgrading
python comfyui_triton_sageattention.py --backup

# List available backups
python comfyui_triton_sageattention.py --backup list

# Restore from the most recent backup (by index)
python comfyui_triton_sageattention.py --backup-restore 1

# Clean up old backups (keep only the latest)
python comfyui_triton_sageattention.py --backup-clean --keep-latest 1
```

Backups include:
- Full copy of your `venv` or `python_embeded` folder
- `requirements.txt` (pip freeze) for reference
- `RESTORE.txt` with manual restore instructions

## Usage

After installation, run ComfyUI with SageAttention enabled:

```bash
# Windows
run_nvidia_gpu.bat

# Or manually
python ComfyUI\main.py --use-sage-attention
```

## What Gets Installed

1. **PyTorch** with CUDA support matching your GPU
2. **Triton-Windows** - OpenAI's Triton for Windows
3. **SageAttention** - Efficient attention mechanism
4. **Python Development Files** - Required headers and libs

### With `--with-custom-nodes`

5. **[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** - Video encoding/decoding utilities
6. **[DazzleNodes](https://github.com/DazzleNodes/DazzleNodes)** - DazzleML node collection

## Troubleshooting

### SageAttention Versions

The installer attempts to install **SageAttention 2.x** (faster, CUDA + Triton) when your PyTorch/CUDA/Python combination matches a pre-configured wheel. If no match is found, it falls back to **SageAttention 1.0.6** (Triton-only).

**Check your installed version:**
```bash
python -m pip show sageattention
```

- Version `1.0.6` = SageAttention 1 (~2.1x speedup vs FlashAttention2)
- Version `2.x.x` = SageAttention 2 (~3x speedup vs FlashAttention2)

For detailed information about version differences and troubleshooting, see [Discussion #8: Understanding SageAttention Versions](https://github.com/DazzleML/comfyui-triton-and-sageattention-installer/discussions/8).

### Common Issues

**"No module named 'torch'" during build**
- The installer uses pre-built wheels to avoid this
- Falls back to compatible versions automatically

**Compilation errors**
- Python development headers are downloaded automatically
- Visual Studio Build Tools are installed if needed

**CUDA version mismatch**
- Installer auto-detects your CUDA version
- Update NVIDIA drivers if issues persist

**Getting SageAttention 1.0.6 instead of 2.x**
- Your PyTorch/CUDA/Python combo may not have a pre-configured wheel
- You can manually install from [woct0rdho's releases](https://github.com/woct0rdho/SageAttention/releases)
- Please open an issue so we can add your configuration

### Logs
Check `comfyui_install.log` for detailed information.

## Contributing

Contributions are welcome! Feel free to submit a pull request.

Like the project?

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/djdarcy)


### Development Setup

```bash
git clone https://github.com/djdarcy/comfyui-triton-and-sageattention-installer.git
cd comfyui-triton-and-sageattention-installer
python comfyui_triton_sageattention.py --install --verbose
```

## How It Works

1. Detects platform and creates appropriate handler
2. Installs Visual Studio Build Tools (Windows)
3. Detects CUDA version for compatibility
4. Installs PyTorch with matching CUDA support
5. Installs Triton (triton-windows on Windows)
6. Downloads Python development headers
7. Installs SageAttention (pre-built wheels preferred)
8. Clones required repositories
9. Creates launch scripts

## Additional Tools

### AI Toolkit RTX 5090 Support

This repository also includes an installer for [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit) users with RTX 5090 GPUs. The AI Toolkit installer (`ai_toolkit_rtx5090_installer.py`) addresses Blackwell architecture compatibility issues for LoRA training:

```bash
# From your AI Toolkit directory
python ai_toolkit_rtx5090_installer.py --non-interactive
```

See [docs/ai_toolkit_rtx5090_setup.md](docs/ai_toolkit_rtx5090_setup.md) for details.

## Acknowledgments

- [woct0rdho](https://github.com/woct0rdho) - Windows wheels and triton-windows
- [thu-ml](https://github.com/thu-ml/SageAttention) - SageAttention project
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The amazing UI
- [Ostris](https://github.com/ostris/ai-toolkit) - AI Toolkit for LoRA training
- ComfyUI community - Testing and feedback


## License

comfyui_triton_sageattention.py, Copyright (C) 2025 Dustin Darcy

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
