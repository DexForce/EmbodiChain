# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import logging
import time
from typing import NoReturn

__all__ = [
    "decorate_str_color",
    "format_message",
    "log_debug",
    "log_error",
    "log_info",
    "log_warning",
    "logger",
    "set_log_level",
]

_LOG_FORMAT = "%(asctime)s.%(msecs)03d UTC │ %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_RESET_COLOR = "\033[0m"
_COLOR_CODES = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "purple": "\033[95m",
    "cyan": "\033[96m",
    "orange": "\033[33m",
    "white": "\033[97m",
}
_DEFAULT_LEVEL_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
}


class _UTCFormatter(logging.Formatter):
    """Format logging timestamps in UTC."""

    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        """Format a record, optionally omitting the standard log prefix."""
        if getattr(record, "embodichain_plain", False):
            return record.getMessage()
        return super().format(record)


_DEFAULT_FORMATTER = _UTCFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
_DEFAULT_HANDLER = logging.StreamHandler()
_DEFAULT_HANDLER.setFormatter(_DEFAULT_FORMATTER)
logging.basicConfig(level=logging.INFO, handlers=[_DEFAULT_HANDLER])

# Create a custom logger
logger = logging.getLogger(__name__)

# Set the default log level
logger.setLevel(logging.INFO)


def decorate_str_color(msg: str, color: str | None) -> str:
    """Decorate a string with an ANSI color.

    Args:
        msg: Text to decorate.
        color: Supported color name, or ``None`` to disable coloring.

    Returns:
        The decorated text, including an ANSI reset sequence when colored.
    """
    return f"{_COLOR_CODES.get(color, '')}{msg}{_RESET_COLOR}" if color else msg


def set_log_level(level: str) -> None:
    """Set the EmbodiChain logging level.

    Args:
        level: One of ``DEBUG``, ``INFO``, ``WARNING``, or ``ERROR``.
    """
    level = level.upper()
    assert level in ["DEBUG", "INFO", "WARNING", "ERROR"], "Invalid log level"
    logger.setLevel(getattr(logging, level))


def format_message(
    level: str,
    message: object,
    color: str | None = None,
) -> str:
    """Format a log message using aligned, optionally colored columns.

    Args:
        level: Logging level displayed in the first message column.
        message: Log message payload.
        color: Supported color name for the level, or ``None`` for no color.

    Returns:
        A formatted message containing the level, component, and payload.
    """
    decorated_level = decorate_str_color(f"{level:<7}", color)
    return f"{decorated_level} │ EmbodiChain │ {message}"


def log_info(
    message: object,
    color: str | None = _DEFAULT_LEVEL_COLORS["INFO"],
    *,
    prefix: bool = True,
) -> None:
    """Log an info message.

    Args:
        message: Log message payload.
        color: Level color override, or ``None`` to disable coloring.
        prefix: Whether to include the timestamp, level, and component columns.
    """
    if not prefix:
        logger.info(message, extra={"embodichain_plain": True})
        return
    logger.info(format_message("INFO", message, color))


def log_debug(
    message: object,
    color: str | None = _DEFAULT_LEVEL_COLORS["DEBUG"],
) -> None:
    """Log a debug message.

    Args:
        message: Log message payload.
        color: Level color override, or ``None`` to disable coloring.
    """
    logger.debug(format_message("DEBUG", message, color))


def log_warning(
    message: object,
    color: str | None = _DEFAULT_LEVEL_COLORS["WARNING"],
) -> None:
    """Log a warning message.

    Args:
        message: Log message payload.
        color: Level and message color override, or ``None`` to disable coloring.
    """
    logger.warning(
        format_message("WARNING", decorate_str_color(str(message), color), color)
    )


def log_error(
    message: object,
    error_type: type[Exception] = RuntimeError,
    color: str | None = _DEFAULT_LEVEL_COLORS["ERROR"],
) -> NoReturn:
    """Raise an exception with an error-formatted message.

    Args:
        message: Error message payload.
        error_type: Exception class to raise.
        color: Level and message color override, or ``None`` to disable coloring.

    Raises:
        Exception: An instance of ``error_type`` containing the formatted message.
    """
    raise error_type(
        format_message("ERROR", decorate_str_color(str(message), color), color)
    )
