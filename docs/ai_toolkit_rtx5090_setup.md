# AI Toolkit RTX 5090 Installer

## Overview

This installer helps users of [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit) get their RTX 5090 GPUs working properly for LoRA training on Windows. The RTX 5090's new Blackwell architecture (sm_120) requires specific PyTorch builds and configuration adjustments.

## Problem Solved

RTX 5090 users encounter "no kernel image is available for execution on the device" errors because:
- Stable PyTorch releases don't yet support sm_120 (Blackwell architecture)
- Several optimization libraries (bitsandbytes, Flash Attention, xFormers) lack sm_120 support
- Default configurations use 8-bit quantization which isn't compatible

## Usage

### Quick Start

From your AI Toolkit directory:
```bash
python ai_toolkit_rtx5090_installer.py --non-interactive
```

### Interactive Mode
```bash
python ai_toolkit_rtx5090_installer.py
```

### Options
- `--diagnose`: Check GPU and show what would be changed
- `--non-interactive`: Run without prompts
- `--force`: Force reinstall even if already configured
- `--verbose`: Show detailed output

## What It Does

1. **Detects RTX 5090/Blackwell GPUs** and verifies CUDA 12.8+
2. **Installs PyTorch nightly** with CUDA 12.8 for sm_120 support
3. **Removes incompatible packages** (bitsandbytes, flash-attn, xformers)
4. **Creates optimized config** with quantization disabled
5. **Tests installation** to ensure GPU operations work

## Configuration Changes

The installer creates `config/rtx5090_qwen_lora.yaml` with:
- Optimizer: `adamw` (not `adamw8bit`)
- Quantization: Disabled
- Precision: BF16
- Memory optimization: Disabled (32GB VRAM available)

## Verification

After installation, verify with:
```bash
python -c "import torch; print(f'sm_120 support: {\"sm_120\" in torch.cuda.get_arch_list()}')"
```

Expected output: `sm_120 support: True`

## Compatibility

- **OS**: Windows 10/11 (WSL2 recommended for best compatibility)
- **GPU**: RTX 5090 or other Blackwell architecture GPUs
- **CUDA**: 12.8 or newer
- **Python**: 3.10-3.12

## Files Created

- `rtx5090_compatibility.json` - Documents configuration changes
- `config/rtx5090_qwen_lora.yaml` - Optimized training config
- `setup_wsl2.sh` - WSL2 setup script (if requested)
- `ai_toolkit_install.log` - Installation log

## Known Limitations

Until official support arrives:
- No 8-bit quantization (bitsandbytes incompatible)
- No Flash Attention optimization
- No xFormers memory optimizations

These will be re-enabled as libraries add Blackwell support.

## Troubleshooting

### "Still getting CUDA errors"
Ensure you're using the venv Python:
```bash
venv\Scripts\python.exe run.py --config config/rtx5090_qwen_lora.yaml
```

### "Import errors after installation"
The installer removes incompatible packages. Update your configs to not use:
- `optimizer: adamw8bit` → use `adamw`
- `quantize: true` → use `false`

## Related Tools

This installer follows the same pattern as the ComfyUI Triton installer in this repository, providing automated setup for GPU compatibility issues.