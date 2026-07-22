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

__all__ = ["format_attempt_error", "format_result_missing_error"]


def format_attempt_error(stage_name: str, attempt_count: int, exc: Exception) -> str:
    """Format a retryable stage failure message."""
    return f"{stage_name} attempt {attempt_count} failed: {exc}"


def format_result_missing_error(
    stage_name: str,
    result_name: str,
    *,
    attempt_count: int,
    last_error: str | None,
    errors: list[str],
) -> str:
    """Format a missing-final-result error message."""
    return (
        f"{stage_name} failed to produce a {result_name} after "
        f"{attempt_count} attempts. Last error: {last_error}. "
        f"All retryable errors: {errors}"
    )
