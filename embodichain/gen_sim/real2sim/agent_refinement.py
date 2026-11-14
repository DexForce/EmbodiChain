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
"""Compose semantic review, bounded numerical correction and fixed-scene replay."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import numpy as np

from embodichain.data_pipeline.real2sim.adaptation.inputs import (
    ReconstructionInput,
    RegisteredScene,
)
from embodichain.data_pipeline.real2sim.adaptation.trajectory_corrections import (
    CORRECTION_LIMITS,
    apply_trajectory_corrections,
)
from .annotation import bound_trajectory_phases, validate_program_annotation
from ._artifacts import digest, write_json

__all__ = ["run_agent_refinement"]


def _apply_revision(
    current: np.ndarray,
    inputs: ReconstructionInput,
    scene: RegisteredScene,
    annotation: dict,
    review: dict,
    iteration: Path,
) -> np.ndarray:
    model = replace(
        inputs.robot,
        base_pose_world=np.asarray(
            scene.description.get("robot_base_pose_world", inputs.robot.base_pose_world)
        ),
    )
    times = inputs.demonstration.timestamps - inputs.demonstration.timestamps[0]
    if inputs.demonstration.time_unit == "ms":
        times = times / 1000
    candidate, patch = apply_trajectory_corrections(
        model,
        current,
        inputs.demonstration.qpos,
        times,
        bound_trajectory_phases(annotation),
        review["corrections"],
    )
    write_json(iteration / "revision.json", patch)
    return candidate


def run_agent_refinement(
    inputs: ReconstructionInput,
    scene: RegisteredScene,
    output: Path,
    *,
    agent: Callable[[str, dict, Path], dict],
    replay: Callable[[np.ndarray, Path], Path],
    initial_qpos: np.ndarray | None = None,
    max_revisions: int = 2,
    context: dict | None = None,
) -> dict:
    """Let an agent diagnose execution and request bounded numerical residuals.

    Args:
        inputs: Immutable real recording and original robot model.
        scene: Fixed registered scene; no reconstruction or placement runs here.
        output: Caller-owned directory for the annotation and replay iterations.
        agent: Semantic inference adapter called with annotate/review stage,
            evidence context and a fresh output directory. Review returns success,
            failure_phase_ids, explanation, evidence and structured corrections.
            It cannot submit code or joint trajectories; the numerical module
            generates commands from the permitted Cartesian/gripper parameters.
        replay: Host-owned native execution of commands in the same fixed scene.
            Returns a directory containing video and physical observations.
        initial_qpos: Explicit existing reference, or the original joints by default.
        max_revisions: Correction budget after the initial replay. The default
            allows at most three executions. Success ends the run immediately;
            budget exhaustion permits diagnosis only, not another correction.
            A rejected numerical edit may be resubmitted once against the same
            replay evidence, without consuming another physical execution.
        context: Additional backend-owned paths such as Python and robot metadata.

    Returns:
        Actual replay paths and the agent's task review. Numerical proximity is
        never used to declare completion. A successful review ends the run;
        bounded edit authority does not establish physical correctness.
    """
    if type(max_revisions) is not int or max_revisions < 0:
        raise ValueError("max_revisions must be a nonnegative integer")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "annotation").exists() or (output / "iterations").exists():
        raise FileExistsError("Agent refinement requires a fresh run directory")
    original = inputs.demonstration.qpos
    current = np.array(original if initial_qpos is None else initial_qpos, copy=True)
    if current.shape != original.shape or not np.isfinite(current).all():
        raise ValueError("Initial commands must be finite and match the recording")
    scene_path = output / "scene.json"
    vertices_path = output / "scene_vertices.npz"
    source = output / "source_trajectory.npz"
    source_arrays = dict(
        qpos=original,
        timestamps=inputs.demonstration.timestamps,
        joint_names=inputs.demonstration.joint_names,
        **{
            name + "_frames": camera.sample_frames
            for name, camera in inputs.cameras.items()
        },
    )
    write_json(scene_path, scene.description)
    np.savez_compressed(vertices_path, **scene.vertices)
    np.savez_compressed(source, **source_arrays)
    evidence = dict(context or {})
    evidence.update(
        instruction=inputs.instruction,
        source_trajectory=str(source),
        source_videos={
            name: dict(
                video=str(camera.video.resolve()), frame_indices_key=name + "_frames"
            )
            for name, camera in inputs.cameras.items()
        },
        scene=str(scene_path),
        scene_vertices=str(vertices_path),
        sample_count=len(original),
        joint_names=list(inputs.demonstration.joint_names),
        time_unit=inputs.demonstration.time_unit,
        primary_camera=inputs.primary_camera,
        correction_limits=CORRECTION_LIMITS,
    )
    fixed = {
        scene_path,
        vertices_path,
        source,
        inputs.robot.urdf.resolve(),
        *(Path(obj["path"]).resolve() for obj in scene.description["objects"]),
        *(Path(p).resolve() for p in evidence.get("fixed_files", [])),
    }
    baseline = {str(path): digest(path) for path in sorted(fixed)}
    write_json(output / "fixed_inputs.json", baseline)

    def assert_fixed() -> None:
        for path, expected in baseline.items():
            if not Path(path).is_file() or digest(Path(path)) != expected:
                raise ValueError("Fixed input changed during agent refinement: " + path)

    assert_fixed()
    status = dict(
        status="annotating",
        success=False,
        history=[],
        revisions=0,
        max_revisions=max_revisions,
    )
    annotation_path = output / "annotation/annotation.json"
    try:
        write_json(output / "status.json", status)
        annotation_dir = output / "annotation"
        annotation_dir.mkdir()
        annotation = agent("annotate", evidence, annotation_dir / "agent")
        assert_fixed()
        annotation = validate_program_annotation(
            annotation,
            len(original),
            {obj["id"] for obj in scene.description["objects"]},
            set(inputs.robot.side_names),
        )
        write_json(annotation_path, annotation)
        write_json(annotation_dir / "task_program.json", annotation["task_program"])
        write_json(
            annotation_dir / "trajectory_bindings.json",
            annotation["trajectory_bindings"],
        )
        write_json(
            annotation_dir / "scope.json",
            dict(
                execution_ready=False,
                scope="Observed Task Program semantics and source-time bindings; runtime integration and affordance grounding are not provided by annotation.",
            ),
        )
        evidence["annotation"] = str(annotation_path)
        phases = bound_trajectory_phases(annotation)
        ids = {phase["id"] for phase in phases}

        def validate_review(review: dict) -> dict:
            assert_fixed()
            if set(review) - {
                "success",
                "failure_phase_ids",
                "explanation",
                "evidence",
                "corrections",
            }:
                raise ValueError(
                    "Review accepts only diagnosis and structured corrections"
                )
            if not isinstance(review.get("corrections", []), list):
                raise ValueError("Review corrections must be a list")
            if type(review.get("success")) is not bool:
                raise ValueError("Agent review requires a boolean success assessment")
            if not set(review.get("failure_phase_ids", [])) <= ids:
                raise ValueError("Agent review references unknown failed phases")
            review["failure_segment_ids"] = list(
                dict.fromkeys(
                    phase["segment"]
                    for phase in phases
                    if phase["id"] in review.get("failure_phase_ids", [])
                )
            )
            return review

        while True:
            execution = len(status["history"])
            iteration = output / "iterations" / f"{execution:03d}"
            iteration.mkdir(parents=True)
            trajectory = iteration / "trajectory.npz"
            np.savez_compressed(
                trajectory,
                qpos=current,
                timestamps=inputs.demonstration.timestamps,
                joint_names=inputs.demonstration.joint_names,
            )
            status.update(status="replaying", iteration=execution)
            write_json(output / "status.json", status)
            print(f"Agent replay {execution}: {iteration}", flush=True)
            assert_fixed()
            rollout = replay(current.copy(), iteration).resolve()
            assert_fixed()
            evidence.update(
                current_trajectory=str(trajectory),
                rollout=str(rollout),
                iteration=execution,
                revision=status["revisions"],
                revisions_remaining=max_revisions - status["revisions"],
                history=status["history"],
            )
            write_json(iteration / "evidence.json", evidence)
            status["status"] = "reviewing"
            write_json(output / "status.json", status)
            review = validate_review(agent("review", evidence, iteration / "agent"))
            write_json(iteration / "review.json", review)
            status["history"].append(
                dict(
                    iteration=execution,
                    revision=status["revisions"],
                    rollout=str(rollout),
                    trajectory=str(trajectory),
                    review=str(iteration / "review.json"),
                    success=review["success"],
                    explanation=review.get("explanation", ""),
                )
            )
            status.update(
                latest_rollout=str(rollout), latest_trajectory=str(trajectory)
            )
            if review["success"]:
                status.update(
                    status="completed",
                    success=True,
                    assessment="agent review of native execution; repeatability not assessed",
                )
                break
            if status["revisions"] == max_revisions:
                status["status"] = "revision_budget_exhausted"
                break
            if not review.get("corrections"):
                status["status"] = "unresolved"
                break
            candidate, correction_dir = None, iteration
            for attempt in range(2):
                try:
                    candidate = _apply_revision(
                        current, inputs, scene, annotation, review, iteration
                    )
                    break
                except ValueError as error:
                    rejection = dict(
                        error=str(error), corrections=review["corrections"]
                    )
                    write_json(correction_dir / "rejection.json", rejection)
                    if attempt == 1:
                        status.update(status="correction_rejected", error=str(error))
                        break
                    correction_dir = iteration / "correction_retry"
                    correction_dir.mkdir()
                    retry_evidence = {**evidence, "correction_failure": rejection}
                    write_json(correction_dir / "evidence.json", retry_evidence)
                    status["status"] = "revising_parameters"
                    write_json(output / "status.json", status)
                    review = validate_review(
                        agent("review", retry_evidence, correction_dir / "agent")
                    )
                    if review["success"]:
                        raise ValueError(
                            "A rejected correction has no new replay establishing success"
                        )
                    write_json(correction_dir / "review.json", review)
                    status["history"][-1]["correction_review"] = str(
                        correction_dir / "review.json"
                    )
                    if not review.get("corrections"):
                        status.update(status="unresolved", error=str(error))
                        break
            if candidate is None:
                break
            current = candidate
            status["revisions"] += 1
        write_json(output / "result.json", status)
        return status
    except BaseException as error:
        status.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
            success=False,
        )
        raise
    finally:
        write_json(output / "status.json", status)
