"""GPU memory management utilities for CUDA inference."""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def cleanup_cuda_memory() -> None:
    """
    Perform comprehensive CUDA memory cleanup.

    Clears CUDA cache and triggers garbage collection to free
    unreferenced tensors. Safe to call on CPU-only systems.

    This should be called:
    - After each batch in batched inference
    - After processing large individual examples
    - Before memory-intensive operations

    Raises:
        None - silently handles non-CUDA systems
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_gpu_memory_stats(device: torch.device | None = None) -> dict[str, float]:
    """
    Get current GPU memory statistics.

    Args:
        device: CUDA device to query (defaults to current device)

    Returns:
        Dictionary with memory stats in MB. Returns all zeros if CUDA unavailable:
            - allocated_mb: Currently allocated memory (0.0 if no CUDA)
            - reserved_mb: Total reserved memory by PyTorch (0.0 if no CUDA)
            - free_mb: Free memory within reserved blocks (0.0 if no CUDA)
    """
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "free_mb": 0.0,
        }

    if device is None:
        device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    free = reserved - allocated

    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "free_mb": round(free, 2),
    }


def log_memory_usage(
    prefix: str = "", device: torch.device | None = None, level: int = logging.INFO
) -> None:
    """
    Log current GPU memory usage.

    Args:
        prefix: String to prepend to log message
        device: CUDA device to query
        level: Logging level (default: INFO)
    """
    if not torch.cuda.is_available():
        logger.debug(f"{prefix}CUDA not available, skipping GPU memory logging")
        return

    gpu_stats = get_gpu_memory_stats(device)
    msg = (
        f"{prefix}GPU: {gpu_stats['allocated_mb']:.2f}MB allocated, "
        f"{gpu_stats['reserved_mb']:.2f}MB reserved, "
        f"{gpu_stats['free_mb']:.2f}MB free"
    )

    logger.log(level, msg)


def check_memory_threshold(
    threshold_mb: float = 1024.0, device: torch.device | None = None
) -> bool:
    """
    Check if allocated GPU memory exceeds threshold.

    Args:
        threshold_mb: Memory threshold in MB
        device: CUDA device to check

    Returns:
        True if memory usage exceeds threshold, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    stats = get_gpu_memory_stats(device)
    return stats["allocated_mb"] > threshold_mb
