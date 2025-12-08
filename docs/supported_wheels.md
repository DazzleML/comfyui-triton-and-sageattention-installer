# Supported SageAttention Wheels

Pre-built wheels are available for the following configurations. The installer automatically selects the best match for your system.

## SageAttention 2.2.0.post3 (Stable, Primary)

Uses ABI3 wheels compatible with Python 3.9+.

| CUDA | PyTorch | Notes |
|------|---------|-------|
| 13.0 | 2.9.x | RTX 5090 / Blackwell |
| 12.8 | 2.9.x | |
| 12.8 | 2.8.x | |
| 12.8 | 2.7.x | |
| 12.6 | 2.6.x | |
| 12.4 | 2.5.x | |

## SageAttention 2.2.0.post4 (Experimental)

Requires `--experimental` flag. Uses ABI3 wheels compatible with Python 3.9+.

| CUDA | PyTorch | Notes |
|------|---------|-------|
| 13.0 | 2.9.x | RTX 5090 / Blackwell |
| 12.8 | 2.9.x | |

## SageAttention 2.1.1

Per-Python wheels (requires exact Python version match).

| CUDA | PyTorch | Python |
|------|---------|--------|
| 12.8 | 2.7.x | 3.12 |
| 12.8 | 2.7.x | 3.11 |
| 12.6 | 2.5.x | 3.12 |
| 12.6 | 2.5.x | 3.11 |
| 12.4 | 2.4.x | 3.12 |
| 12.4 | 2.4.x | 3.11 |

## SageAttention 2.0.1 (Legacy)

| CUDA | PyTorch | Python |
|------|---------|--------|
| 12.4 | 2.4.x | 3.12 |
| 12.1 | 2.4.x | 3.12 |
| 11.8 | 2.4.x | 3.12 |

## SageAttention 1.0.6 (Fallback)

Installed from PyPI when no pre-built SA2 wheel matches. Works with any Python 3.8+ and CUDA 11.8+.

---

## Version Comparison

| Version | Speedup vs FA2 | Wheel Type | Python Support |
|---------|----------------|------------|----------------|
| 2.2.0.post3 | ~3x | ABI3 | 3.9+ |
| 2.1.1 | ~3x | Per-Python | 3.11, 3.12 |
| 2.0.1 | ~3x | Per-Python | 3.12 |
| 1.0.6 | ~2.1x | PyPI | 3.8+ |

## Adding New Configurations

If your configuration isn't listed, please [open an issue](https://github.com/DazzleML/comfyui-triton-and-sageattention-installer/issues) with:
- PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- CUDA version (`nvcc --version`)
- Python version (`python --version`)

Wheels are sourced from [woct0rdho's SageAttention releases](https://github.com/woct0rdho/SageAttention/releases).
