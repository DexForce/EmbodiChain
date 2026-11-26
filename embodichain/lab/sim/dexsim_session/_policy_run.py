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

"""Run a registered policy on the owning thread of an empty DexSim World."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dexsim.runtime import SessionContext

__all__: list[str] = []


def run(
    context: SessionContext,
    *,
    run_id: str,
    entrypoint: str,
    files: dict[str, str],
    options: dict[str, Any],
    output: str,
) -> dict[str, Any]:
    """Execute an operator-registered backend within the World's owner-thread request."""
    builder = context.builder
    if (
        builder is None
        or builder.is_finalized
        or builder.has_pending_changes
        or context.state.get("embodichain_policy_run") is not None
    ):
        raise ValueError("A policy run requires an unused, empty World")
    module, _, name = entrypoint.partition(":")
    execute = getattr(import_module(module), name)
    context.state["embodichain_policy_run"] = run_id
    return execute(
        context,
        files={key: Path(value) for key, value in files.items()},
        options=options,
        output=Path(output),
    )
