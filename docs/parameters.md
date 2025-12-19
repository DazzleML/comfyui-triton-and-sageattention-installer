# Command Line Parameters

Complete reference for all `comfyui_triton_sageattention.py` command line options.

## Quick Reference

```bash
python comfyui_triton_sageattention.py [ACTION] [OPTIONS]
```

## Actions

| Flag | Description |
|------|-------------|
| `--install` | Install Triton and SageAttention |
| `--upgrade` | Upgrade existing SageAttention installation |
| `--cleanup` | Remove Triton/SageAttention packages |
| `--run` | Run ComfyUI with SageAttention enabled |
| `--show-installed` | Display current installation status |
| `--dryrun` | Preview changes without executing (requires `--install` or `--upgrade`) |

## Environment Options

### `--base-path PATH`

Base installation directory. Default: current working directory.

```bash
python comfyui_triton_sageattention.py --install --base-path C:\ComfyUI
```

### `--python MODE`

Control which Python environment to use. Default: `auto`.

| Value | Behavior |
|-------|----------|
| `auto` | Auto-detect: portable > venv > system (default) |
| `system` | Use system Python directly, no venv |
| `portable` | Require ComfyUI Portable's `python_embeded` (Windows only, error if not found) |
| `venv` | Use/create venv at `--base-path`, skip portable even if exists |
| `<path>` | Explicit path to Python executable or environment folder |

**Keyword vs Path Disambiguation:**

To distinguish keywords from folder names, paths must include a separator (`/`, `\`) or start with `./` or `.\`:

- `--python venv` → keyword: use venv at `{base-path}/venv`
- `--python .\venv` → path: use `{cwd}/venv`
- `--python ./venv2` → path: use `{cwd}/venv2`
- `--python C:\envs\myenv` → path: use specified folder

**Examples:**

```bash
# Auto-detect (default behavior)
python comfyui_triton_sageattention.py --install --python auto

# Force system Python (useful for CI/testing)
python comfyui_triton_sageattention.py --install --python system

# Force venv even when portable distribution exists
python comfyui_triton_sageattention.py --install --python venv

# Require portable distribution (error if not found)
python comfyui_triton_sageattention.py --install --python portable

# Use a specific venv folder
python comfyui_triton_sageattention.py --install --python .\venv2
python comfyui_triton_sageattention.py --install --python C:\my-project\custom-venv

# Use a specific Python executable
python comfyui_triton_sageattention.py --install --python C:\Python312\python.exe
```

## Installation Options

### `--sage-version VERSION`

Control which SageAttention version to install. Default: `auto`.

| Value | Behavior |
|-------|----------|
| `auto` | Try SageAttention 2.x, fall back to 1.x if unavailable |
| `1` | Install any SageAttention 1.x (~2.1x speedup) |
| `2` | Install any SageAttention 2.x (~3x speedup) |
| `X.Y.Z` | Install exact version (e.g., `2.1.1`, `1.0.6`) |

```bash
python comfyui_triton_sageattention.py --install --sage-version 2
python comfyui_triton_sageattention.py --install --sage-version 1.0.6
```

### `--experimental`

Allow installation of experimental/prerelease SageAttention versions. Use with caution.

```bash
python comfyui_triton_sageattention.py --install --experimental
```

### `--with-custom-nodes`

Install recommended custom nodes (VideoHelperSuite, DazzleNodes). Omit for minimal installation.

```bash
python comfyui_triton_sageattention.py --install --with-custom-nodes
```

## Behavior Options

### `--force`

Force reinstallation of all components, bypassing existing installation checks.

```bash
python comfyui_triton_sageattention.py --install --force
```

### `--non-interactive`

Run without user prompts. Uses safer defaults for automated environments (CI/Docker).

```bash
python comfyui_triton_sageattention.py --install --non-interactive
```

### `--verbose` / `-v`

Enable verbose logging for debugging.

```bash
python comfyui_triton_sageattention.py --install --verbose
```

## Information Options

### `--version`

Display installer version and exit.

```bash
python comfyui_triton_sageattention.py --version
```

### `--help` / `-h`

Display help message and exit.

```bash
python comfyui_triton_sageattention.py --help
```

## Common Workflows

### Standard Installation

```bash
python comfyui_triton_sageattention.py --install
```

### Preview Before Installing

```bash
python comfyui_triton_sageattention.py --install --dryrun
```

### Check Current Status

```bash
python comfyui_triton_sageattention.py --show-installed
```

### Upgrade SageAttention

```bash
python comfyui_triton_sageattention.py --upgrade
```

### Clean Slate Reinstall

```bash
python comfyui_triton_sageattention.py --cleanup
python comfyui_triton_sageattention.py --install
```

### CI/Docker Automation

```bash
python comfyui_triton_sageattention.py --install --non-interactive --force
```

### ComfyUI Portable

```bash
cd C:\ComfyUI_windows_portable
python path\to\comfyui_triton_sageattention.py --install
```

Or from any location:

```bash
python comfyui_triton_sageattention.py --install --base-path C:\ComfyUI_windows_portable
```
