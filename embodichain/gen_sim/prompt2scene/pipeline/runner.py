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

from dataclasses import dataclass
from pathlib import Path

from embodichain.gen_sim.prompt2scene.llms import OpenAICompatibleLLMCfg
from embodichain.gen_sim.prompt2scene.workflows.request import (
    InputKind,
    Prompt2SceneInput,
)
from embodichain.gen_sim.prompt2scene.workflows.paths import (
    IMAGE_SEGMENTS_STEP,
    IMAGE_SPATIAL_RELATIONS_STEP,
    SCENE_EDIT_STEP,
    SCENE_INTAKE_STEP,
    UNIFIED_SCENE_STEP,
    PipelinePaths,
)
from embodichain.gen_sim.prompt2scene.workflows.artifact_writer import (
    write_step_result,
)
from embodichain.gen_sim.prompt2scene.workflows.unified_scene.graph import (
    run_unified_scene,
)
from embodichain.gen_sim.prompt2scene.workflows.unified_scene_gen.graph import (
    run_unified_scene_gen,
)
from embodichain.gen_sim.prompt2scene.workflows.gym_export import (
    export_gym_config,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_edit import run_scene_edit
from embodichain.gen_sim.prompt2scene.workflows.scene_edit.schema import (
    SceneEditRequest,
)
from embodichain.gen_sim.prompt2scene.utils.io import write_json
from embodichain.gen_sim.prompt2scene.utils import log
from embodichain.gen_sim.prompt2scene.workflows.image_relations import (
    run_image_relations,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_intake import run_scene_intake

__all__ = [
    "IMAGE_SEGMENTS_DIRNAME",
    "IMAGE_SPATIAL_RELATIONS_DIRNAME",
    "INPUT_MANIFEST_FILENAME",
    "SCENE_EDIT_DIRNAME",
    "SCENE_INTAKE_DIRNAME",
    "STEP_RESULT_FILENAME",
    "UNIFIED_SCENE_DIRNAME",
    "Prompt2SceneRunResult",
    "run_prompt2scene",
]

INPUT_MANIFEST_FILENAME = "input_manifest.json"
SCENE_INTAKE_DIRNAME = SCENE_INTAKE_STEP
SCENE_EDIT_DIRNAME = SCENE_EDIT_STEP
IMAGE_SEGMENTS_DIRNAME = IMAGE_SEGMENTS_STEP
IMAGE_SPATIAL_RELATIONS_DIRNAME = IMAGE_SPATIAL_RELATIONS_STEP
UNIFIED_SCENE_DIRNAME = UNIFIED_SCENE_STEP


@dataclass(frozen=True)
class Prompt2SceneRunResult:
    """Result returned by the prompt2scene runner.

    Args:
        output_root: Directory where prompt2scene outputs were written.
        manifest_path: Path to the serialized input manifest.
        scene_intake_path: Path to the serialized scene intake output.
        image_segments_path: Path to serialized image segment alignment output.
        image_spatial_relations_path: Path to serialized image spatial relations.
        unified_scene_path: Path to serialized unified scene output.
        gym_config_path: Path to the exported gym config.
        scene_edit_path: Path to serialized scene edit output.
    """

    output_root: Path
    manifest_path: Path
    scene_intake_path: Path | None = None
    image_segments_path: Path | None = None
    image_spatial_relations_path: Path | None = None
    unified_scene_path: Path | None = None
    gym_config_path: Path | None = None
    scene_edit_path: Path | None = None


def run_prompt2scene(
    request: Prompt2SceneInput,
    llm_cfg: OpenAICompatibleLLMCfg | None = None,
) -> Prompt2SceneRunResult:
    """Run the prompt2scene pipeline.

    This runner creates the output directory, writes the parsed input manifest,
    and runs fixed VLM-based scene intake when an LLM config is provided.

    Args:
        request: Parsed prompt2scene input.
        llm_cfg: Optional LLM config used by later pipeline stages.

    Returns:
        Paths created by the runner.
    """
    log.log_info(
        "run start "
        f"input_kind={request.input_kind.value} output_root={request.output_root}"
    )
    request.output_root.mkdir(parents=True, exist_ok=True)
    paths = PipelinePaths(request.output_root)
    manifest_path = request.output_root / INPUT_MANIFEST_FILENAME
    manifest = request.to_manifest()
    if llm_cfg is not None:
        manifest["llm"] = llm_cfg.to_manifest()
    write_json(manifest_path, manifest)

    scene_intake_path = None
    image_segments_path = None
    image_spatial_relations_path = None
    unified_scene_path = None
    gym_config_path = None
    scene_edit_path = None
    if request.input_kind == InputKind.EDIT:
        log.log_info("step start scene_edit")
        run_scene_edit(
            SceneEditRequest(
                output_root=request.output_root,
                prompt=request.prompt or "",
                gravity_settle_mode=request.gravity_settle_mode,
            ),
            llm_cfg=llm_cfg,
        )
        scene_edit_path = paths.step_result(SCENE_EDIT_STEP)
        log.log_info(
            f"step end scene_edit status=pending_implementation output={scene_edit_path}"
        )
    elif llm_cfg is not None:
        log.log_info("step start scene_intake")
        scene_intake = run_scene_intake(request, llm_cfg=llm_cfg)
        scene_intake_path = write_step_result(
            request.output_root,
            SCENE_INTAKE_STEP,
            scene_intake.to_manifest(),
        )
        log.log_info(f"step end scene_intake status=ok output={scene_intake_path}")
        if request.input_kind != InputKind.IMAGE:
            raise ValueError(
                f"Unsupported prompt2scene input_kind: {request.input_kind.value!r}."
            )
        log.log_info("step start image_relations")
        image_relations = run_image_relations(
            request,
            scene_intake=scene_intake,
            llm_cfg=llm_cfg,
            output_root=request.output_root,
        )
        image_segments_path = paths.step_result(
            IMAGE_SEGMENTS_STEP,
        )
        if not image_segments_path.is_file():
            write_step_result(
                request.output_root,
                IMAGE_SEGMENTS_STEP,
                image_relations.to_segmentation_manifest(),
            )
        image_spatial_relations_path = paths.step_result(
            IMAGE_SPATIAL_RELATIONS_STEP,
        )
        if not image_spatial_relations_path.is_file():
            write_step_result(
                request.output_root,
                IMAGE_SPATIAL_RELATIONS_STEP,
                image_relations.to_spatial_manifest(),
            )
        log.log_info(
            "step end image_relations "
            f"status=ok output={image_spatial_relations_path}"
        )
        log.log_info("step start unified_scene")
        unified_scene = run_unified_scene(
            request,
            scene_intake=scene_intake,
            image_relations=image_relations,
            output_root=request.output_root,
        )
        unified_scene_path = paths.step_result(
            UNIFIED_SCENE_STEP,
        )
        log.log_info(f"step end unified_scene status=ok output={unified_scene_path}")
        log.log_info("step start unified_scene_gen")
        run_unified_scene_gen(
            request.output_root,
            unified_scene_result_path=unified_scene_path,
            llm_cfg=llm_cfg,
            gravity_settle_mode=request.gravity_settle_mode,
        )
        log.log_info("step end unified_scene_gen status=ok")

        log.log_info("step start gym_export")
        gym_config_path = export_gym_config(request.output_root)
        log.log_info(f"step end gym_export status=ok output={gym_config_path}")
        if request.prompt:
            log.log_info("step start scene_edit")
            run_scene_edit(
                SceneEditRequest(
                    output_root=request.output_root,
                    prompt=request.prompt,
                    gravity_settle_mode=request.gravity_settle_mode,
                ),
                llm_cfg=llm_cfg,
            )
            scene_edit_path = paths.step_result(SCENE_EDIT_STEP)
            log.log_info(
                f"step end scene_edit status=pending_implementation output={scene_edit_path}"
            )

    log.log_info(f"run end output_root={request.output_root}")

    return Prompt2SceneRunResult(
        output_root=request.output_root,
        manifest_path=manifest_path,
        scene_intake_path=scene_intake_path,
        image_segments_path=image_segments_path,
        image_spatial_relations_path=image_spatial_relations_path,
        unified_scene_path=unified_scene_path,
        gym_config_path=gym_config_path,
        scene_edit_path=scene_edit_path,
    )
