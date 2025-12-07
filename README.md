# ComfyUI Triton and SageAttention Installer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Windows](https://img.shields.io/badge/platform-Windows-blue.svg)](https://www.microsoft.com/windows)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
python comfyui_triton_sageattention.py --install
```
By default `comfyui_triton_sageattention.py` tries to install SageAttention 2, and falls back to 1 if unavailable.

### Installation Options

```bash
# Verbose mode for debugging
python comfyui_triton_sageattention.py --install --verbose

# Custom ComfyUI location
python comfyui_triton_sageattention.py --install --base-path C:\path\to\comfyui

# Cleanup previous installation
python comfyui_triton_sageattention.py --cleanup
```

### SageAttention Version Control

Control which SageAttention version gets installed:

```bash
# Explicitly install SageAttention 1 (~2.1x speedup, more compatible) or 2 (~3x speedup) & fails if unavailable
python comfyui_triton_sageattention.py --install --sage-version 1  # or 2

# Advanced: install specific version
python comfyui_triton_sageattention.py --install --sage-version 2.1.1
```

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
5. **Additional Custom Nodes**:
   - SageAttention repository
   - flow2-wan-video
   - ComfyUI-VideoHelperSuite

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

[!["Buy Me A Coffee"](https://camo.githubusercontent.com/0b448aabee402aaf7b3b256ae471e7dc66bcf174fad7d6bb52b27138b2364e47/68747470733a2f2f7777772e6275796d6561636f666665652e636f6d2f6173736574732f696d672f637573746f6d5f696d616765732f6f72616e67655f696d672e706e67)](https://www.buymeacoffee.com/djdarcy)


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
