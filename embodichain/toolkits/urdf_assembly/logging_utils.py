# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import logging
from typing import Optional

__all__ = ["URDFAssemblyLogger"]


class URDFColorFormatter(logging.Formatter):
    r"""Color formatter for URDF assembly logging"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    # Symbol colors
    BRACKET_COLOR = "\033[94m"  # Bright blue for []
    PAREN_COLOR = "\033[95m"  # Magenta for ()
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)

        # Apply symbol coloring first
        message = self._colorize_symbols(message, color)

        return f"{color}{message}{self.RESET}"

    def _colorize_symbols(self, message, base_color):
        r"""Add colors to brackets and parentheses while preserving base color"""
        import re

        # Color square brackets and their content, then restore base color
        message = re.sub(
            r"\[([^\]]+)\]",
            f"{self.BRACKET_COLOR}[\\1]{self.RESET}{base_color}",
            message,
        )

        # Color parentheses and their content, then restore base color
        message = re.sub(
            r"\(([^)]+)\)", f"{self.PAREN_COLOR}(\\1){self.RESET}{base_color}", message
        )

        return message


class URDFAssemblyLogger:
    r"""URDF Assembly module-specific logger manager"""

    _loggers = {}  # Cache for created loggers
    _initialized = False

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        r"""Get or create a URDF assembly-specific logger

        Args:
            name: Logger name, defaults to calling module name

        Returns:
            Configured logger instance
        """
        if name is None:
            # Get caller's module name
            import inspect

            frame = inspect.currentframe().f_back
            module_name = frame.f_globals.get("__name__", "unknown")
            if module_name == "__main__":
                name = "urdf_assembly.main"
            else:
                name = f'urdf_assembly.{module_name.split(".")[-1]}'
        else:
            # Ensure using urdf_assembly prefix
            if not name.startswith("urdf_assembly."):
                name = f"urdf_assembly.{name}"

        # Return cached logger or create new one
        if name not in cls._loggers:
            logger = logging.getLogger(name)

            # Avoid duplicate handlers
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = URDFColorFormatter("[%(levelname)s] %(name)s: %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                logger.propagate = False  # Don't propagate to root logger

            cls._loggers[name] = logger

        return cls._loggers[name]

    @classmethod
    def set_level(cls, level):
        r"""Set log level for all URDF assembly loggers"""
        for logger in cls._loggers.values():
            logger.setLevel(level)

    @classmethod
    def disable_other_loggers(cls):
        r"""Disable output from other non-URDF loggers"""
        logging.getLogger().setLevel(logging.CRITICAL)


# Remove original setup_logger function, replace with URDF-specific initialization
def setup_urdf_logging():
    """Initialize URDF assembly logging system"""
    # Optional: disable other logger outputs
    URDFAssemblyLogger.disable_other_loggers()
    return URDFAssemblyLogger.get_logger("urdf_assembly.main")
