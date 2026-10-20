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

"""Bind instruction text to selected scene assets without changing task logic."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embodichain.lab.task_program import TaskProgramCfg

__all__: list[str] = []


def _render_asset_instruction(env: Any, instruction: str | None) -> str | None:
    """Replace literal ``{scene_uid}`` tokens once, without format expressions."""
    selections = getattr(env, "_asset_selections", {})
    if instruction is None or not selections:
        return instruction
    labels = {"{" + uid + "}": entry["label"] for uid, entry in selections.items()}
    pattern = "|".join(re.escape(token) for token in labels)
    return re.sub(pattern, lambda match: labels[match.group()], instruction)


def _bind_asset_program_instructions(
    env: Any, program: TaskProgramCfg
) -> TaskProgramCfg:
    """Return a language-bound copy; preserve the reusable program template."""
    if not getattr(env, "_asset_selections", {}):
        return program
    bound = deepcopy(program)

    def visit(node: Any) -> None:
        if hasattr(node, "instruction"):
            node.instruction = _render_asset_instruction(env, node.instruction)
        for field in ("items", "branches"):
            for child in getattr(node, field, ()):
                visit(child)
        for field in ("body", "steps"):
            child = getattr(node, field, None)
            if child is not None:
                visit(child)

    bound.instruction = _render_asset_instruction(env, bound.instruction)
    visit(bound.program)
    return bound
