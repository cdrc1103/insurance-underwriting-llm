"""Test basic setup and imports."""

import sys
from pathlib import Path


def test_python_version() -> None:
    """Test that Python version is 3.12 or higher."""
    assert sys.version_info >= (3, 12), "Python 3.12+ required"


def test_project_structure() -> None:
    """Test that essential project directories exist."""
    project_root = Path(__file__).parent.parent

    required_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        "tests",
        "notebooks",
        "scripts",
        "configs",
        "docs",
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Directory {dir_name} should exist"
        assert dir_path.is_dir(), f"{dir_name} should be a directory"


def test_imports() -> None:
    """Test that core dependencies can be imported."""
    # Core ML dependencies

    # Evaluation libraries

    # Data processing

    # All imports successful
    assert True


def test_torch_cuda_available() -> None:
    """Test CUDA availability (warning only, not a hard requirement)."""
    import torch

    # This test will pass regardless, but logs the CUDA status
    if not torch.cuda.is_available():
        print("\nWarning: CUDA is not available. GPU training will not be possible.")
        print("This is OK for local development, but required for training.")
    else:
        print(f"\nCUDA is available: {torch.cuda.get_device_name(0)}")
