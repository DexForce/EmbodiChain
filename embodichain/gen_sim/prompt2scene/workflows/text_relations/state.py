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

from pathlib import Path
from typing import Any

from embodichain.gen_sim.prompt2scene.workflows.request import Prompt2SceneInput
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.schema import (
    SceneIntakeSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.text_relations.schema import (
    TextRelationSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.attempt_state import AttemptState

__all__ = ["TextRelationsState"]


class TextRelationsState(AttemptState):
    """LangGraph state for explicit text spatial-relation extraction."""

    request: Prompt2SceneInput
    scene_intake: SceneIntakeSpec
    output_root: Path
    messages: list[Any]
    raw_model_output: dict[str, Any] | None
    text_relations: TextRelationSpec | None
