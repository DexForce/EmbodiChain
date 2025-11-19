# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

# Create a custom logger
logger = logging.getLogger(__name__)

# Set the default log level
logger.setLevel(logging.INFO)


def decorate_str_color(msg: str, color: str):
    """Decorate a string with a specific color."""
    color_map = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "orange": "\033[33m",
        "white": "\033[97m",
    }
    return f"{color_map.get(color, '')}{msg}\033[0m" if color else msg


def set_log_level(level: str):
    """Set the logging level."""
    level = level.upper()
    assert level in ["DEBUG", "INFO", "WARNING", "ERROR"], "Invalid log level"
    logger.setLevel(getattr(logging, level))


def format_message(level: str, message: str):
    """Format the log message with a consistent prefix."""
    return f"[EmbodiChain {level}]: {message}"


def log_info(message, color=None):
    """Log an info message."""
    logger.info(decorate_str_color(format_message("INFO", message), color))


def log_debug(message, color="blue"):
    """Log a debug message."""
    logger.debug(decorate_str_color(format_message("DEBUG", message), color))


def log_warning(message):
    """Log a warning message."""
    logger.warning(decorate_str_color(format_message("WARNING", message), "purple"))


def log_error(message, error_type=RuntimeError):
    """Log an error message."""
    raise error_type(decorate_str_color(format_message("ERROR", message), "red"))
