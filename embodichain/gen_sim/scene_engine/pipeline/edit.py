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

import json
from pathlib import Path

from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
    SceneExportImporter,
)
from embodichain.gen_sim.scene_engine.pipeline.editing.scene_edit_understanding import (
    understand_scene_edit,
)
from embodichain.utils.logger import log_info


def edit_scene(
    *,
    output_root: str | Path,
    edit_prompt: str,
) -> None:
    """Apply one text edit instruction to an existing Scene Engine output."""

    # Initialize the VLM client that will interpret the edit instruction.
    vlm_client = OpenAICompatibleVLM.from_dotenv()
    scene_importer = SceneExportImporter(output_root=output_root)
    # Validate scene_export, write scene.json, and return Scene; failures raise before editing.
    scene, scene_graph = scene_importer.import_scene_and_graph()
    # Only for debug.
    print(
        json.dumps(
            {"scene": scene.to_dict(), "scene_graph": scene_graph.to_dict()},
            indent=2,
            ensure_ascii=False,
        )
    )

    # 1. Edit Understanding
    # Will return an already checked scene edit plan.
    log_info("Starting Edit Understanding")
    scene_edit_plan = understand_scene_edit(
        scene=scene,
        scene_graph=scene_graph,
        edit_prompt=edit_prompt,
        vlm_client=vlm_client,
    )
    log_info("Completed Edit Understanding")

    # 2. Prepare Objects.
    log_info("Preparing Objects if necessary")
    # scene = prepare_objects(
    #     scene=scene,
    #     output_root=output_root,
    # )
    log_info("Completed Preparing Objects")

    # 3. Layout Editing
    log_info("Starting Layout Editing")
    # scene = edit_layout(
    #     scene=scene,
    #     edit_plan=edit_plan,
    #     scene_graph=updated_scene_graph,
    #     output_root=output_root,
    # )
    log_info("Completed Layout Editing")

    # 4. Scene Export
    # Re export the scene to the same output format,
    # and delete some temporary files or folders.
    log_info("Starting Scene Export")
    log_info("Completed Scene Export")

    raise NotImplementedError("Scene editing is not implemented yet.")
