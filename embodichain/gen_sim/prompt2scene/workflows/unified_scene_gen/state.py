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

from embodichain.gen_sim.prompt2scene.workflows.attempt_state import AttemptState

__all__ = ["UnifiedSceneGenState"]


class UnifiedSceneGenState(AttemptState):
    """LangGraph state for downstream unified-scene generation."""

    output_root: Path
    unified_scene_result_path: Path | None
    gravity_settle_mode: str
    llm: Any | None
    unified_scene: dict[str, Any] | None
    input_kind: str | None
    table_result: dict[str, Any] | None
    image_object_results: list[dict[str, Any]]
    image_objects_layout_result: dict[str, Any] | None
    table_fit_result: dict[str, Any] | None
    generation_status: str | None
