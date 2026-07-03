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

import traceback
from pathlib import Path
from typing import Any

from embodichain.gen_sim.prompt2scene.utils.log import log_info, log_warning
from embodichain.gen_sim.prompt2scene.agent_tools.tools.text_object_settle import (
    settle_text_objects_to_ground,
)

__all__ = ["generate_text_clutter_layout"]


def generate_text_clutter_layout(
    *,
    object_results: list[dict[str, Any]],
    spatial_relations: list[dict[str, Any]],
    table_constraints: list[dict[str, Any]],
    output_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Settle and spatially arrange generated text-scene objects."""
    if not object_results:
        return {
            "status": "skipped",
            "reason": "no_text_objects",
        }

    try:
        log_info(f"text clutter layout started count={len(object_results)}")
        result = settle_text_objects_to_ground(
            objects=object_results,
            spatial_relations=spatial_relations,
            table_constraints=table_constraints,
            output_dir=output_dir,
            output_root=output_root,
        )
        log_info(f"text clutter layout completed status={result.get('status')}")
        return result
    except Exception as exc:
        log_warning(f"text clutter layout failed error={exc}")
        return {
            "status": "failed",
            "reason": traceback.format_exc(),
        }
