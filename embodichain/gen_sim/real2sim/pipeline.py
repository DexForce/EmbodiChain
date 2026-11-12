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
"""Assemble reusable Real2Sim processing with native Scene/Task Engine calls.

Source-specific loading and native settling execution are caller-owned.
Existing core APIs are invoked unchanged; rendering remains a separate consumer.
"""

from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager
import numpy as np
from embodichain.data_pipeline.real2sim.reconstruction.inputs import (
    ReconstructionInput,
    RegisteredScene,
    RefinedTrajectory,
)
from embodichain.gen_sim.services import GenSimServices
from embodichain.gen_sim.services.contracts import ChatService
from .image_edit import ImageEditor
from ._artifacts import write_json, digest

__all__ = ["Real2SimResult", "refine_recording", "run_real2sim"]


@dataclass(frozen=True)
class Real2SimResult:
    """Independent scene, nominal trajectory and optional source graph artifacts."""

    output: Path
    scene: RegisteredScene
    trajectory: RefinedTrajectory
    graph_binding: dict | None


def refine_recording(
    inputs: ReconstructionInput,
    scene: RegisteredScene,
    output: Path,
    *,
    approach_lead_frames: int = 120,
    grounding: tuple[dict, ...] = (),
) -> RefinedTrajectory:
    """Refine the demonstration and persist independent nominal outputs.

    Args:
        inputs: Original recording and calibrated robot model.
        scene: Generated geometry after initial physical settling.
        output: New directory for refined arrays and diagnostics.
        approach_lead_frames: Extra source samples for introducing grasp residuals.
        grounding: Validated source-video associations for detected gripper events.

    Returns:
        Refined joint reference without modifying the recorded demonstration.
    """
    from embodichain.data_pipeline.real2sim.reconstruction.refinement import (
        refine_trajectory,
    )

    output.mkdir(parents=True, exist_ok=False)
    result = refine_trajectory(
        inputs, scene, approach_lead_frames=approach_lead_frames, grounding=grounding
    )
    np.savez_compressed(
        output / "trajectory_refined.npz",
        qpos=result.qpos,
        timestamps=result.timestamps,
        joint_names=inputs.demonstration.joint_names,
        **{name + "_tcp": value for name, value in result.tcp_poses.items()},
    )
    write_json(output / "refinement.json", result.report)
    write_json(
        output / "provenance.json",
        dict(
            source_recording_sha256=inputs.demonstration.recording_sha256,
            robot_geometry_sha256=inputs.demonstration.robot_geometry_sha256,
            refined_sha256=digest(output / "trajectory_refined.npz"),
            origin="adapted_nominal_reference",
            source_arrays_overwritten=False,
            object_tracks_are_simulator_inputs=False,
        ),
    )
    return result


def run_real2sim(
    inputs: ReconstructionInput,
    output: Path,
    *,
    services: GenSimServices | None = None,
    editor: ImageEditor | None = None,
    chat: ChatService | None = None,
    prepared_image: Path | None = None,
    registered_scene: RegisteredScene | None = None,
    native_run: Path | None = None,
    reference_frames: tuple[int, ...] = (),
    annotate: bool = True,
    scene_lock: Callable[[], ContextManager] | None = None,
    settling_runner: Callable[[Path], Path] | None = None,
) -> Real2SimResult:
    """Process one demonstration or explicitly reuse generated scene artifacts.

    Args:
        inputs: Unified source observation, robot and calibrated video inputs.
        output: Fresh caller-owned output directory.
        services: Existing GenSim service injection for generation and graph work.
        editor: Image-edit service for robot/background/lighting cleanup.
        chat: Required source-event grounding and annotation model using the existing chat contract.
        prepared_image: Explicit reconstruction-only derivative to reuse.
        registered_scene: Explicit previously registered geometry to reuse.
        native_run: Existing full Scene Engine export to register instead of generate.
        reference_frames: Task-relevant same-view video reference frame IDs.
        annotate: Produce source annotation and graph; independent of simulation.
        scene_lock: Caller-owned resource lock around native scene generation.
        settling_runner: Run the original initial ten-second physics window for
            the output directory and return its rollout directory. Required so
            fresh scenes cannot silently bypass settling before grasp fitting.

    Returns:
        Registered scene, separate nominal reference, and optional annotated graph.
        Completion is not physical task success or an executable graph binding.
    """
    if annotate and (services is None or chat is None):
        raise ValueError("Annotation requires injected chat and GenSim services")
    if prepared_image is None and (services is None or editor is None):
        raise ValueError("Image preparation requires segmentation and editing services")
    if registered_scene is None and native_run is None and services is None:
        raise ValueError("Scene reconstruction requires GenSim services")
    if settling_runner is None:
        raise ValueError(
            "Supply a native settling_runner before trajectory refinement; "
            "use refine_recording directly for an already prepared scene"
        )
    if chat is None:
        raise ValueError("Source-event grounding requires injected chat")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "inputs.json",
        dict(
            source_id=inputs.demonstration.source_id,
            unit_id=inputs.demonstration.unit_id,
            recording_sha256=inputs.demonstration.recording_sha256,
            instruction=inputs.instruction,
            primary_camera=inputs.primary_camera,
            source_videos={k: str(v.video) for k, v in inputs.cameras.items()},
        ),
    )
    status = {
        "status": "running",
        "current_stage": "scene_input",
        "completed_stages": [],
        "physical_success_verified": False,
    }

    def completed(stage: str) -> None:
        status["completed_stages"].append(stage)
        write_json(output / "status.json", status)

    def starting(stage: str) -> None:
        status["current_stage"] = stage
        write_json(output / "status.json", status)

    try:
        from .scene import prepare_scene_input, reconstruct_scene, register_scene_export

        starting("scene_input")
        if prepared_image is None:
            camera = inputs.cameras[inputs.primary_camera]
            if reference_frames and camera.sample_frames[0] != 0:
                raise ValueError(
                    "Observed recovery currently anchors decoded frame zero"
                )
            prepared_image = prepare_scene_input(
                camera.first_image,
                output / "scene_input",
                services,
                editor,
                video=camera.video,
                reference_frames=reference_frames,
            )
        write_json(
            output / "scene_input.json",
            dict(
                path=str(prepared_image.resolve()),
                sha256=digest(prepared_image),
                reconstruction_only=True,
            ),
        )
        completed("scene_input")
        if registered_scene is None:
            if native_run is None:
                starting("scene_generation")
                with scene_lock() if scene_lock is not None else nullcontext():
                    native_run = reconstruct_scene(
                        prepared_image,
                        output / "scene_engine",
                        services,
                    )
            starting("registration")
            registered_scene = register_scene_export(inputs, native_run)
        if native_run is not None:
            write_json(
                output / "scene_export_source.json",
                {"path": str((native_run / "scene_export").resolve())},
            )
        write_json(output / "scene.json", registered_scene.description)
        np.savez_compressed(output / "scene_vertices.npz", **registered_scene.vertices)
        completed("registered_scene")
        from ._settling import settle_registered_scene

        starting("settling")
        registered_scene = settle_registered_scene(
            registered_scene, settling_runner(output), output
        )
        completed("scene_settled")
        from .annotation import ground_gripper_events

        starting("source_grounding")
        grounding = ground_gripper_events(
            inputs, registered_scene, output / "grounding", chat
        )
        completed("source_grounding")
        starting("trajectory_refinement")
        refined = refine_recording(
            inputs,
            registered_scene,
            output / "trajectory",
            grounding=tuple(grounding["events"]),
        )
        completed("refinement")
        binding = None
        if annotate:
            from .annotation import annotate_trajectory, generate_annotated_graph

            starting("annotation")
            annotation = annotate_trajectory(
                inputs, registered_scene, refined, output / "annotation", chat
            )
            completed("annotation")
            starting("semantic_graph")
            binding = generate_annotated_graph(
                prepared_image,
                annotation,
                output / "graph",
                services,
                chat,
                task_id=inputs.demonstration.unit_id,
            )
            binding["trajectory_sha256"] = digest(
                output / "trajectory/trajectory_refined.npz"
            )
            write_json(output / "graph/trajectory_binding.json", binding)
            completed("graph")
        status["status"] = (
            "completed" if annotate else "geometry_and_trajectory_completed"
        )
        status["current_stage"] = "completed"
        return Real2SimResult(output, registered_scene, refined, binding)
    except Exception as error:
        status.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        write_json(output / "status.json", status)
