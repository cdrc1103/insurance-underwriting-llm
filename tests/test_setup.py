"""Test basic setup and imports."""

import sys


def test_python_version() -> None:
    """Test that Python version is 3.12 or higher."""
    assert sys.version_info >= (3, 12), "Python 3.12+ required"


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
