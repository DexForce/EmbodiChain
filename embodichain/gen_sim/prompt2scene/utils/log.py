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
from typing import Any

__all__ = ["log_api_request_start", "log_info", "log_warning"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EmbodiChain %(levelname)s]: %(message)s",
    datefmt="%H:%M:%S",
)

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)


def _format_message(level: str, message: str) -> str:
    _ = level
    return f"Prompt2Scene: {message}"


def log_info(message: str) -> None:
    """Log an info message using the EmbodiChain log prefix."""
    _LOGGER.info(_format_message("INFO", message))


def log_warning(message: str) -> None:
    """Log a warning message using the EmbodiChain log prefix."""
    _LOGGER.warning(_format_message("WARNING", message))


def log_api_request_start(
    *,
    step: str,
    request: str,
    attempt: int | None = None,
    **details: Any,
) -> None:
    """Log the start of an API request with a stable key order."""
    fields = [f"step={step}", f"request={request}"]
    if attempt is not None:
        fields.append(f"attempt={attempt}")
    for key, value in details.items():
        fields.append(f"{key}={value}")
    log_info("api request start " + " ".join(fields))
