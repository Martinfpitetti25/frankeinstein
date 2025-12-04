"""
Centralized logging configuration with rotation for Frankeinstein AI Robot

Features:
- Rotating file handler (max 10MB per file, 5 backup files)
- Console output with color-coded levels
- Structured format with timestamps
- Per-module log levels
- Automatic log directory creation
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Determine project root (3 levels up from this file: utils -> src -> robot_ai)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)

# Log file paths
MAIN_LOG_FILE = LOGS_DIR / "robot_ai.log"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"

# Global logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log level (can be overridden per module)
DEFAULT_LOG_LEVEL = logging.INFO

# Module-specific log levels (can be configured here)
MODULE_LOG_LEVELS = {
    "services.camera_service": logging.DEBUG,  # More verbose for camera debugging
    "services.audio_service": logging.INFO,
    "services.servo_service": logging.INFO,
    "services.chat_service": logging.INFO,
    "services.groq_service": logging.INFO,
    "main": logging.INFO,
}

# Store configured loggers to avoid duplicate handlers
_configured_loggers = set()


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance with rotation
    
    Args:
        name: Logger name (typically __name__ from calling module)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if name in _configured_loggers:
        return logger
    
    _configured_loggers.add(name)
    
    # Set log level (module-specific or default)
    log_level = MODULE_LOG_LEVELS.get(name, DEFAULT_LOG_LEVEL)
    logger.setLevel(log_level)
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Rotating file handler for all logs (DEBUG and above)
    # Max 10MB per file, keep 5 backup files
    file_handler = RotatingFileHandler(
        MAIN_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Separate rotating file handler for errors only (ERROR and above)
    error_handler = RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # Prevent propagation to root logger (avoid duplicate logs)
    logger.propagate = False
    
    return logger


def set_log_level(logger_name: str, level: int):
    """
    Change log level for a specific logger at runtime
    
    Args:
        logger_name: Name of the logger to modify
        level: New log level (e.g., logging.DEBUG, logging.INFO)
    
    Example:
        >>> set_log_level("services.camera_service", logging.DEBUG)
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Update module configuration
    MODULE_LOG_LEVELS[logger_name] = level


def get_log_files_info():
    """
    Get information about current log files
    
    Returns:
        dict: Dictionary with log file paths and sizes
    """
    info = {}
    
    for log_file in [MAIN_LOG_FILE, ERROR_LOG_FILE]:
        if log_file.exists():
            size_mb = log_file.stat().st_size / (1024 * 1024)
            info[log_file.name] = {
                "path": str(log_file),
                "size_mb": round(size_mb, 2)
            }
        else:
            info[log_file.name] = {"path": str(log_file), "size_mb": 0}
    
    return info
