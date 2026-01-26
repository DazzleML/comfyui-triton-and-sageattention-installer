# Changelog

All notable changes to the ComfyUI Triton and SageAttention installer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.4] - 2026-01-25

### Changed (Tests)
- Refactored `test_backup.py` to reduce duplication (-45% lines)
- Added `helpers.py` with reusable test utilities
- Added `temp_base` fixture to `conftest.py`
- Used `@pytest.mark.parametrize` for env type tests (venv, .venv, custom)
- Test count: 82 → 83 (parameterization covers more combinations)

## [0.7.3] - 2026-01-25

### Added
- `.venv` detection in auto mode (supports uv, poetry, and modern Python tooling)
- Auto-detection priority is now: `portable > .venv > venv > system`
- Quick Start note in README about running from ComfyUI directory

### Fixed
- BackupManager now uses detected `venv_path` instead of hardcoded `venv` folder
  - Fixes backup failures when using `.venv` or custom `--python` paths
- `list_backups()` now correctly identifies `.venv` backup types

### Changed
- Updated all documentation to reflect new `.venv` detection priority

### Added (Tests)
- 4 new unit tests for `.venv` backup support (82 total)

## [0.7.2] - 2026-01-05

### Fixed
- `--backup-clean` help text now correctly states `'all'` keyword is required (not implicit)

### Changed
- Moved `--upgrade` to appear right after `--install` in help output for discoverability
- Added `--upgrade --base-path` example to help epilog

## [0.7.1](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/a924cfc...v0.7.1) - 2026-01-05

### Fixed
- **CPU→CUDA PyTorch upgrade now works correctly** (Issue #23)
  - `install_pytorch()` now uses InstallPlan as single source of truth
  - Uninstalls CPU-only torch before installing CUDA version
  - Fixes pip "Requirement already satisfied" ignoring CPU→CUDA switch

### Changed
- Installation methods now respect the InstallPlan instead of making independent decisions
- Improved architecture alignment between dryrun preview and actual installation

### Added
- 3 new unit tests for CPU→CUDA switch behavior (78 total)
- Documentation for Issue #22 GPU detection integration points

## [0.7.0](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/1672dd0...a924cfc) - 2026-01-05

### Added
- **Environment backup feature** for safe upgrades with full restore capability
- `BackupManager` class for modular backup management (preparation for future PyPI package)
- `--backup [create|list]` flag for backup creation and listing
- `--backup-restore <INDEX_OR_TIMESTAMP>` flag for environment restoration
- `--backup-clean [INDEX...]` flag for removing specific backups
- `--keep-latest N` flag to preserve recent backups during cleanup
- Index-based backup selection (e.g., `--backup-restore 1` for most recent)
- Automatic `requirements.txt` (pip freeze) saved with each backup
- `RESTORE.txt` instructions included in each backup
- Mandatory confirmation for all destructive operations (restore, clean)
- Unit test suite for BackupManager (16 new tests, total 75 unit tests)

### Changed
- Destructive backup operations (clean, restore) require interactive confirmation
- Non-interactive mode refuses destructive operations to prevent accidental data loss
- Updated Quick Start documentation to recommend backup before install

## [0.6.12](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/8825c97...1672dd0) - 2026-01-05

### Added
- Pre-installation validation for exact SageAttention version availability
- Detailed error messaging when requested SA version is incompatible with environment
- Alternative version suggestions when exact version unavailable

### Fixed
- Issue #19: Replaced misleading "compilation issues" error with specific diagnostics

### Changed
- Enhanced upgrade test suite to verify error message behavior for incompatible versions

## [0.6.11](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/923ed15...8825c97) - 2026-01-05

### Added
- Comprehensive test suite with 59 unit test scenarios validating InstallPlan decision logic
- Test driver (run_tests.py) with environment guards for automated testing
- Bidirectional Triton/PyTorch compatibility check

### Fixed
- Triton compatibility check now validates both directions (PyTorch new enough for Triton AND Triton new enough for PyTorch)

### Changed
- One-off tests now skip incompatible SA version scenarios (SA 2.1.1 only has wheels for PyTorch <= 2.8)

## [0.6.10](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/v0.6.8...923ed15) - 2026-01-05

### Added
- InstallPlan architecture for unified dryrun/install decision logic
- Version comparison helper methods (_compare_versions, _detect_downgrades, _confirm_downgrade)
- Test suite for dryrun/install consistency (12 tests)

### Fixed
- **Critical**: Fixed Issue #18 where `--dryrun` showed [KEEP] for PyTorch but `--install` would reinstall, destroying working ComfyUI environments
- Unified PyTorch decision logic between dryrun and install paths (both now use torch.version.cuda)
- Portable distributions with bundled CUDA now correctly preserved

### Changed
- Changed _get_torch_version() default from "2.7.0" to "" (empty string)
- Removed hardcoded torch==2.7.0, letting pip resolve latest compatible version
- Both preview_changes() and install now call _check_pytorch_compatibility() for consistency

## [0.6.9](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/23422e4...67c3575) - 2025-12-19

### Added
- Unified `--python` flag for explicit Python environment selection
- Support for five environment selection modes: auto, system, portable, venv, path
- Path vs keyword distinction (e.g., 'venv' = keyword, './venv' = path)
- Comprehensive CLI parameter documentation in docs/parameters.md

### Changed
- Default behavior remains auto-detection: portable > venv > system
- Updated README with --python flag examples

## [0.6.8](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/3773611...23422e4) - 2025-12-19 (v0.6.7 in commit)

### Added
- Environment type tracking (portable, venv, system) via environment_type property
- _get_target_python_version() to query target Python instead of invoking interpreter
- ComfyUI Portable Support section in README with examples

### Fixed
- Issue #15: Python version detection now works correctly with ComfyUI Portable distributions

### Changed
- show_installed() and preview_changes() now display detected environment type
- WindowsHandler logs "Detected ComfyUI Portable distribution" when python_embeded folder found

## [0.6.6](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/320ca8d...3773611) - 2025-12-09

### Changed
- `--upgrade` and `--install` flags now automatically fix Triton compatibility issues
- Improved automatic handling of Triton/PyTorch version mismatches

## [0.6.5](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/6f9fae7...320ca8d) - 2025-12-09

### Added
- Triton/PyTorch version compatibility checking
- Validation to prevent installation of incompatible Triton versions
- Warning messages when version conflicts detected

## [0.6.4](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/8cb1f69...6f9fae7) - 2025-12-08

### Added
- `--dryrun` flag to preview installation changes without executing them

### Fixed
- Dryrun output now shows accurate status using pip check

### Changed
- Improved output formatting for cleaner copy-paste operations

## [0.6.3](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/8705998...da06fdd) - 2025-12-08

### Added
- SageAttention 2.x compatibility display showing available pre-built wheels

### Changed
- Enhanced dryrun output with more detailed package information

## [0.6.2](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/d648a90...8705998) - 2025-12-08

### Added
- `--show-installed` flag to display current environment status
- Detailed package version reporting for installed components

## [0.6.1](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/513e10e...d648a90) - 2025-12-08

### Added
- `--with-custom-nodes` flag to opt-in to custom node installation

### Changed
- Custom node installation is now opt-in rather than automatic
- Reduces installation footprint for users who don't need custom nodes

## [0.6.0](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/0fdbb2b...513e10e) - 2025-12-07

### Added
- `--upgrade` flag to update existing SageAttention installations
- Support for upgrading from SA 1.x to SA 2.x
- Upgrade documentation in README

### Changed
- Installation logic now handles version upgrades intelligently
- Better handling of existing installations during upgrade process

## [0.5.9](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/e0f222c...e4da3aa) - 2025-12-07

### Added
- `--experimental` flag to opt-in to experimental/prerelease versions (Issue #9)
- Warning message and user confirmation for experimental mode
- Expanded wheel configurations for SA 2.2.0.post3 and SA 2.2.0.post4
- Support for CUDA 12.4, 12.6, 12.8, 13.0
- Support for PyTorch 2.5, 2.6, 2.7, 2.8, 2.9
- ABI3 wheels for SA 2.2.0.post3 (Python 3.9+)
- Helper methods for wheel URL construction
- Comprehensive docs/supported_wheels.md listing all configurations
- Integration test suite (tests/one-offs/test_real_install.py)

### Changed
- Experimental versions now filtered by default
- PyTorch pattern matching now supports major.minor patterns (2.7 matches 2.7.x)

## [0.5.8](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/586630a...e0f222c) - 2025-12-07

### Changed
- Revised SageAttention installation instructions in documentation

## [0.5.7](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/2918be0...586630a) - 2025-12-07

### Added
- `--sage-version` flag for explicit SageAttention version control (Issue #7)
- Support for version arguments: auto, 1, 2, or exact versions (e.g., 2.1.1)
- parse_sage_version() helper for version string parsing
- Clearer logging for SageAttention installation process (Issue #6)
- Helper methods for SA install logic with detailed status messages

### Changed
- Refactored clone_and_install_repositories() to use version strategy system
- Installation now shows clear strategy messages (auto/exact/fallback)
- Updated README with SageAttention Version Control section

### Testing
- Verified on fresh ComfyUI install with PyTorch 2.7.0+cu128, CUDA 12.8, Python 3.12
- Confirmed working with RTX 5090 and SageAttention 2.1.1

## [0.5.6](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/6a6665e...2918be0) - 2025-12-07

### Added
- SageAttention version documentation in troubleshooting section
- Guidance for version selection and compatibility

## [0.5.5](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/e09d43a...6a6665e) - 2025-09-17

### Added
- AI Toolkit RTX 5090 installer for Blackwell architecture support
- Support for latest NVIDIA GPU architecture
- Installation instructions for RTX 5090 users

## [0.5.0](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/72a1a7c...e09d43a) - 2025-07-22

### Added
- Version support to installer with `__version__` variable
- `--version` CLI argument to display installer version
- Version tracking in code for better release management

## [0.4.0](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/f9af2f9...72a1a7c) - 2025-07-22

### Fixed
- CI/CD configuration and project metadata corrections

### Changed
- Updated project metadata for better package management
- Improved continuous integration workflows

## [0.3.0](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/35a6573...f9af2f9) - 2025-07-22

### Changed
- Minor README touchups for clarity and formatting

## [0.2.0](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/compare/1d64962...35a6573) - 2025-07-22

### Added
- Initial commit of ComfyUI Triton and SageAttention installer
- Automated installation of Triton for CUDA environments
- SageAttention installation with pre-built wheel support
- Windows-specific wheel detection and installation
- Fallback to PyPI for SageAttention 1.0.6
- Source compilation support for SageAttention (when wheels unavailable)
- ComfyUI custom nodes directory setup
- Interactive and non-interactive installation modes
- Verbose logging option
- Base path configuration for custom installation locations
- Python development files verification
- CUDA version detection
- PyTorch version compatibility checking

## [0.1.0](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/releases/tag/v0.1.0) - 2025-07-22

### Added
- Claude AI integration files for development assistance
- Initial repository setup with RepoKit
- Project structure and basic documentation
- Repository initialization with git
