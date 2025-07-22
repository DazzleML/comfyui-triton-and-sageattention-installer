from setuptools import setup, find_packages

setup(
    name="comfyui-triton-sageattention-installer",
    version="0.1.1",
    description="Cross-platform installer for Triton and SageAttention on ComfyUI. Simplifies GPU-accelerated inference setup for Windows users with automated dependency management and RTX 5090 support.",
    author="Dustin",
    author_email="6962246+djdarcy@users.noreply.github.com",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
)
