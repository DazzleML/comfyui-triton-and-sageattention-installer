# Contributing to ComfyUI Triton and SageAttention Installer

Thanks for your interest n contributing! This installer helps the ComfyUI community overcome Windows compilation challenges, and additional contributions can help even more users.

## Code of Conduct

Please be respectful and constructive. We're all here to help make ComfyUI better for everyone.

## How Can I Contribute?

### Reporting Bugs

Before submitting a bug report:
1. Check the [existing issues](https://github.com/djdarcy/comfyui-triton-and-sageattention-installer/issues)
2. Run with `--verbose` flag: `python comfyui_triton_sageattention.py --install --verbose`
3. Check `comfyui_install.log` for detailed error information

When reporting bugs, please include:
- Your system specs (Windows version, GPU model, CUDA version)
- Python version (`python --version`)
- ComfyUI installation type (standalone or git clone)
- Complete error message from the console
- Relevant sections from `comfyui_install.log`

### Suggesting Enhancements

We welcome suggestions for:
- Support for additional platforms (Linux distros, macOS)
- New pre-built wheel sources
- Better error handling and recovery
- Documentation improvements
- Additional GPU architecture support

### Testing

You can help by testing the installer on different systems:
- Different Windows versions (10, 11)
- Various NVIDIA GPUs (especially newer models)
- Different CUDA versions
- ComfyUI portable vs git installations

### Pull Requests

1. **Fork and clone** the repository
2. **Create a branch** for your feature: `git checkout -b feature/amazing-feature`
3. **Test your changes** thoroughly on Windows (and other platforms if applicable)
4. **Document** any new command-line options or behaviors
5. **Submit a PR** with a clear description of changes

#### Code Style
- Follow PEP 8 guidelines
- Use descriptive variable names
- Add comments for complex logic
- Maintain cross-platform compatibility where possible

#### Testing Your Changes
```bash
# Test basic functionality
python comfyui_triton_sageattention.py --help
python comfyui_triton_sageattention.py --version

# Test in dry-run mode if adding new features
python comfyui_triton_sageattention.py --install --verbose

# Verify syntax
python -m py_compile comfyui_triton_sageattention.py
```

### Areas Where Help is Needed

1. **Linux/macOS Support**: Expand platform handlers for better cross-platform support
2. **Pre-built Wheels**: Find and maintain sources for pre-built wheels
3. **Error Recovery**: Improve graceful handling of partial installations
4. **Documentation**: Create guides for specific GPU models or edge cases
5. **Translations**: Help make the installer accessible to non-English speakers

## Questions?

Feel free to open an issue for questions or discussions. We'll try to help out where we can!
