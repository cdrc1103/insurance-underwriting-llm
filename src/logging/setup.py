"""Logging configuration utilities.

This module provides centralized logging setup for consistent formatting
and output across all scripts.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_dir: Path,
    log_prefix: str,
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
    console_format: str = "%(message)s",
    file_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> Path:
    """
    Set up logging to both console and file with timestamped log file.

    Args:
        log_dir: Directory to save log files
        log_prefix: Prefix for log filename (e.g., 'inference', 'geval_evaluation')
        console_level: Logging level for console output
        file_level: Logging level for file output
        console_format: Format string for console handler
        file_format: Format string for file handler

    Returns:
        Path to the created log file

    Raises:
        OSError: If log directory cannot be created
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_prefix}_{timestamp}.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(min(console_level, file_level))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(console_format)
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(file_format)
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_file
