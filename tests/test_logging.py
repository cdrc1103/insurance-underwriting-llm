"""Tests for logging configuration utilities."""

import logging
from pathlib import Path

from src.logging import setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_creates_log_file(self, tmp_path: Path) -> None:
        """Test that setup_logging creates a log file with correct prefix."""
        log_dir = tmp_path / "logs"
        log_prefix = "test_logging"

        log_file = setup_logging(log_dir=log_dir, log_prefix=log_prefix)

        assert log_file.exists()
        assert log_file.parent == log_dir
        assert log_file.name.startswith(log_prefix)
        assert log_file.suffix == ".log"

    def test_setup_logging_configures_handlers(self, tmp_path: Path) -> None:
        """Test that setup_logging configures both console and file handlers."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir, log_prefix="test")

        root_logger = logging.getLogger()

        # Should have exactly 2 handlers (console and file)
        assert len(root_logger.handlers) == 2

        # Check handler types
        handler_types = {type(h).__name__ for h in root_logger.handlers}
        assert "StreamHandler" in handler_types
        assert "FileHandler" in handler_types

    def test_setup_logging_writes_to_file(self, tmp_path: Path) -> None:
        """Test that logging actually writes to the file."""
        log_dir = tmp_path / "logs"
        log_file = setup_logging(log_dir=log_dir, log_prefix="test")

        test_message = "Test log message"
        logger = logging.getLogger(__name__)
        logger.info(test_message)

        # Read the log file and verify the message was written
        log_content = log_file.read_text()
        assert test_message in log_content

    def test_setup_logging_custom_formats(self, tmp_path: Path) -> None:
        """Test that custom format strings are applied correctly."""
        log_dir = tmp_path / "logs"
        custom_console_format = "CUSTOM: %(message)s"
        custom_file_format = "FILE: %(message)s"

        setup_logging(
            log_dir=log_dir,
            log_prefix="test",
            console_format=custom_console_format,
            file_format=custom_file_format,
        )

        root_logger = logging.getLogger()

        # Verify formats are applied
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                assert handler.formatter._fmt == custom_file_format
            elif isinstance(handler, logging.StreamHandler):
                assert handler.formatter._fmt == custom_console_format

    def test_setup_logging_creates_directory(self, tmp_path: Path) -> None:
        """Test that setup_logging creates the log directory if it doesn't exist."""
        log_dir = tmp_path / "nonexistent" / "logs"
        assert not log_dir.exists()

        setup_logging(log_dir=log_dir, log_prefix="test")

        assert log_dir.exists()
        assert log_dir.is_dir()
