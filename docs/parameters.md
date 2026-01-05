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

## Backup Options

Environment backup feature for safe upgrades with full restore capability.

### `--backup [create|list]`

Backup management. Without argument defaults to `create`.

| Value | Behavior |
|-------|----------|
| `create` | Create a timestamped backup of current environment (default) |
| `list` | List all available backups with sizes and indices |

```bash
# Create a backup
python comfyui_triton_sageattention.py --backup
python comfyui_triton_sageattention.py --backup create

# List available backups
python comfyui_triton_sageattention.py --backup list
```

**Output example:**
```
Available backups:
  [1] 20260105_143000  (3.2 GB)  venv
  [2] 20260104_092315  (3.1 GB)  venv

To restore: --backup-restore <index>  (e.g., --backup-restore 1)
To clean:   --backup-clean [indices]  (e.g., --backup-clean 2 3)
```

### `--backup-restore INDEX_OR_TIMESTAMP`

Restore environment from a backup. Accepts either:
- Index number from `--backup list` (e.g., `1` for most recent)
- Full timestamp (e.g., `20260105_143000`)

```bash
# Restore most recent backup
python comfyui_triton_sageattention.py --backup-restore 1

# Restore specific backup by timestamp
python comfyui_triton_sageattention.py --backup-restore 20260105_143000
```

**Safety**: Always requires interactive confirmation. Will refuse to run in `--non-interactive` mode.

### `--backup-clean [INDEX... | all]`

Remove specific backups by index, or all backups with explicit `all` keyword.

```bash
# Show available backups and cleanup syntax (no action taken)
python comfyui_triton_sageattention.py --backup-clean

# Remove specific backups by index
python comfyui_triton_sageattention.py --backup-clean 2 3

# Remove ALL backups (requires explicit 'all' and confirmation)
python comfyui_triton_sageattention.py --backup-clean all
```

**Safety**: Always requires interactive confirmation. Will refuse to run in `--non-interactive` mode.

### `--keep-latest N`

When cleaning backups, preserve the N most recent. Use with `--backup-clean`.

```bash
# Clean all but keep the latest 2 backups
python comfyui_triton_sageattention.py --backup-clean all --keep-latest 2
```

### Combined with Install/Upgrade

When `--backup` is combined with `--install` or `--upgrade`, the backup runs first:

```bash
# Backup first, then install (recommended!)
python comfyui_triton_sageattention.py --install --backup

# Backup first, then upgrade
python comfyui_triton_sageattention.py --upgrade --backup
```

If the backup fails, the install/upgrade is aborted to protect your environment.

### Backup Contents

Each backup directory (`.comfyui_backups/{timestamp}/`) contains:

| File | Description |
|------|-------------|
| `venv/` or `python_embeded/` | Full copy of environment folder |
| `requirements.txt` | pip freeze output at backup time |
| `RESTORE.txt` | Instructions for manual restore |

### Backup Workflow Example

```bash
# Single command: backup and upgrade
python comfyui_triton_sageattention.py --upgrade --backup

# Or step by step:
# 1. Before upgrading, create a backup
python comfyui_triton_sageattention.py --backup

# 2. Perform the upgrade
python comfyui_triton_sageattention.py --upgrade

# 3. If something breaks, restore
python comfyui_triton_sageattention.py --backup-restore 1

# 4. After confirming upgrade works, clean old backups
python comfyui_triton_sageattention.py --backup-clean all --keep-latest 1
```
