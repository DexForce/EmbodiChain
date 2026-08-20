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
from collections.abc import Callable

import pytest

from embodichain.utils import logger as logger_module

_RESET_COLOR = "\033[0m"
_LEVEL_CASES = (
    (logger_module.log_debug, "debug", "DEBUG", "\033[96m", False),
    (logger_module.log_info, "info", "INFO", "\033[92m", False),
    (logger_module.log_warning, "warning", "WARNING", "\033[93m", True),
)


def test_default_formatter_uses_utc_datetime_and_aligned_layout():
    record = logging.LogRecord(
        name="embodichain.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="INFO    │ EmbodiChain │ Simulation initialized",
        args=(),
        exc_info=None,
    )
    record.created = 0.123
    record.msecs = 123.0

    formatted = logger_module._DEFAULT_FORMATTER.format(record)

    assert formatted == (
        "1970-01-01 00:00:00.123 UTC "
        "│ INFO    │ EmbodiChain │ Simulation initialized"
    )


def test_default_formatter_can_omit_prefix():
    message = "╭─ Environment initialized\n╰─ Ready"
    record = logging.LogRecord(
        name="embodichain.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.embodichain_plain = True

    assert logger_module._DEFAULT_FORMATTER.format(record) == message


@pytest.mark.parametrize(
    ("log_function", "logger_method", "level", "color_code", "colors_message"),
    _LEVEL_CASES,
)
def test_log_methods_use_default_level_colors(
    monkeypatch: pytest.MonkeyPatch,
    log_function: Callable[..., None],
    logger_method: str,
    level: str,
    color_code: str,
    colors_message: bool,
):
    messages: list[str] = []
    monkeypatch.setattr(logger_module.logger, logger_method, messages.append)

    log_function("Test message")

    message = (
        f"{color_code}Test message{_RESET_COLOR}" if colors_message else "Test message"
    )
    assert messages == [
        f"{color_code}{level:<7}{_RESET_COLOR} │ EmbodiChain │ {message}"
    ]


@pytest.mark.parametrize(
    ("log_function", "logger_method", "level", "colors_message"),
    tuple((*case[:3], case[4]) for case in _LEVEL_CASES),
)
def test_log_methods_allow_custom_level_color(
    monkeypatch: pytest.MonkeyPatch,
    log_function: Callable[..., None],
    logger_method: str,
    level: str,
    colors_message: bool,
):
    messages: list[str] = []
    monkeypatch.setattr(logger_module.logger, logger_method, messages.append)

    log_function("Test message", color="purple")

    message = (
        f"\033[95mTest message{_RESET_COLOR}" if colors_message else "Test message"
    )
    assert messages == [f"\033[95m{level:<7}{_RESET_COLOR} │ EmbodiChain │ {message}"]


def test_log_color_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    messages: list[str] = []
    monkeypatch.setattr(logger_module.logger, "info", messages.append)

    logger_module.log_info("Test message", color=None)

    assert messages == ["INFO    │ EmbodiChain │ Test message"]


def test_log_info_can_omit_prefix(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[object, dict[str, object]]] = []

    def capture(message: object, **kwargs: object) -> None:
        calls.append((message, kwargs))

    monkeypatch.setattr(logger_module.logger, "info", capture)

    logger_module.log_info("Environment initialized", prefix=False)

    assert calls == [
        ("Environment initialized", {"extra": {"embodichain_plain": True}})
    ]


def test_log_error_uses_default_color_and_preserves_error_type():
    with pytest.raises(ValueError) as error:
        logger_module.log_error("Test message", ValueError)

    assert str(error.value) == (
        f"\033[91mERROR  {_RESET_COLOR} │ EmbodiChain │ "
        f"\033[91mTest message{_RESET_COLOR}"
    )


def test_log_error_allows_custom_level_color():
    with pytest.raises(RuntimeError) as error:
        logger_module.log_error("Test message", color="purple")

    assert str(error.value) == (
        f"\033[95mERROR  {_RESET_COLOR} │ EmbodiChain │ "
        f"\033[95mTest message{_RESET_COLOR}"
    )
