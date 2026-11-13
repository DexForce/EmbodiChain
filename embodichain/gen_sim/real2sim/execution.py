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
"""Assemble prepared Real2Sim recordings, semantic review and native replay."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

from embodichain.data_pipeline.real2sim.adaptation.prepared import (
    load_prepared_source,
    load_registered_scene,
)
from .trajectory_agent import CodexTrajectoryAgent
from .replay import export_native_inputs
from ._artifacts import digest, write_json
from .pipeline import refine_recording

__all__ = ["run_prepared_real2sim", "refine_prepared_recording"]


def run_prepared_real2sim(
    source: Path,
    registered: Path,
    output: Path,
    *,
    instruction: str,
    model: str = "gpt-6-astra",
    reasoning_effort: str = "medium",
    max_revisions: int = 2,
    visual_urdf: Path | None = None,
    place_base: bool = False,
    continue_from: Path | None = None,
    initial_trajectory: Path | None = None,
) -> dict:
    """Compose the existing bounded agent loop for a fixed registered scene.

    Args:
        source: Prepared original recording directory.
        registered: Registered Scene Engine export directory.
        output: Fresh run directory.
        instruction: Observed task instruction.
        model: Inference model name.
        reasoning_effort: Caller-selected inference effort.
        max_revisions: Correction budget after the baseline replay.
        visual_urdf: Optional robot visuals for replay.
        place_base: Explicitly recompute mounting before starting the loop.
        continue_from: Completed run whose replay receipt verifies the unchanged
            executed trajectory and rollout before reusing its evidence.
        initial_trajectory: Optional existing joint reference with source timing.

    Returns:
        Bounded loop status and paths to its actual replay evidence.
    """
    if max_revisions < 0:
        raise ValueError("--max-revisions must be nonnegative")
    source, registered, output = (p.resolve() for p in (source, registered, output))
    previous = None
    if continue_from:
        if place_base:
            raise ValueError(
                "Changed mounting requires a new replay, not continuation evidence"
            )
        if initial_trajectory:
            raise ValueError(
                "--continue-from already selects the prior run's latest trajectory"
            )
        previous = continue_from.resolve()
        prior_annotation = json.loads(
            (previous / "annotation/annotation.json").read_text()
        )
        if "task_program" not in prior_annotation:
            raise ValueError(
                "Legacy phase-only annotations require a fresh run with Task Program annotation"
            )
        prior_context = json.loads((previous / "context.json").read_text())
        if (
            prior_context["prepared_source"],
            prior_context["registered_source"],
            prior_context["instruction"],
        ) != (str(source), str(registered), instruction):
            raise ValueError(
                "Continuation requires the same recording, scene and instruction"
            )
        for path, expected in json.loads(
            (previous / "fixed_inputs.json").read_text()
        ).items():
            if digest(Path(path)) != expected:
                raise ValueError("Prior run input changed: " + path)
        prior_result = json.loads((previous / "result.json").read_text())
        initial_trajectory = Path(prior_result["latest_trajectory"])
        prior_rollout = Path(prior_result["latest_rollout"])
        if not (prior_rollout / "diagnostics/physics_rollout.npz").is_file():
            raise ValueError("Continuation requires a completed native replay")
        prior_receipt = initial_trajectory.parent / "replay_receipt.json"
        if not prior_receipt.is_file():
            raise ValueError(
                "Continuation requires a replay receipt; run a fresh replay"
            )
        executed = json.loads(prior_receipt.read_text())
        if digest(initial_trajectory) != executed.get("trajectory_sha256"):
            raise ValueError(
                "Previously executed trajectory changed; run a fresh replay"
            )
        prior_physics = prior_rollout / "diagnostics/physics_rollout.npz"
        if digest(prior_physics) != executed.get("rollout_sha256"):
            raise ValueError("Previously completed rollout changed; run a fresh replay")
        if "visual_urdf" in prior_context:
            prior_visual = (
                Path(prior_context["visual_urdf"])
                if prior_context["visual_urdf"]
                else None
            )
        else:
            prior_command = json.loads(
                (initial_trajectory.parent / "replay_command.json").read_text()
            )
            prior_visual = (
                Path(prior_command[prior_command.index("--visual-urdf") + 1])
                if "--visual-urdf" in prior_command
                else None
            )
        if visual_urdf and visual_urdf.resolve() != prior_visual:
            raise ValueError("Continuation requires the same visual URDF")
        visual_urdf = prior_visual
    output.mkdir(parents=True, exist_ok=False)
    os.setpriority(os.PRIO_PROCESS, 0, max(15, os.getpriority(os.PRIO_PROCESS, 0)))
    cpus = set(sorted(os.sched_getaffinity(0))[:2])
    for thread in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(thread.name), cpus)
        except ProcessLookupError:
            pass
    inputs = load_prepared_source(source, instruction)
    initial_qpos = None
    if initial_trajectory:
        initial_trajectory = initial_trajectory.resolve()
        with np.load(initial_trajectory, allow_pickle=False) as initial:
            initial_qpos = np.asarray(initial["qpos"]).copy()
            if (
                initial_qpos.shape != inputs.demonstration.qpos.shape
                or not np.isfinite(initial_qpos).all()
            ):
                raise ValueError(
                    "Initial trajectory qpos must be finite and match the recording shape"
                )
            for key, expected in (
                ("timestamps", inputs.demonstration.timestamps),
                ("joint_names", inputs.demonstration.joint_names),
            ):
                if key in initial and not np.array_equal(initial[key], expected):
                    raise ValueError(
                        f"Initial trajectory {key} must match the recording"
                    )
    # Reused observations belong to the prior run's exact mounting and geometry.
    scene = load_registered_scene(previous if previous else registered)
    if place_base:
        from embodichain.data_pipeline.real2sim.adaptation.placement import (
            place_robot_base,
        )

        scene = place_robot_base(inputs, scene, replace_existing=True)
    receipt = registered / "scene_export_source.json"
    native = (
        Path(json.loads(receipt.read_text())["path"])
        if receipt.is_file()
        else registered / "native_inputs/native_scene_export"
    )
    if not native.is_absolute():
        native = registered / native
    if not (native / "scene_config.json").is_file():
        raise ValueError("Registered scene requires a native scene export receipt")
    robot_kwargs = asdict(inputs.robot)
    robot_kwargs["base_pose_world"] = scene.description.get(
        "robot_base_pose_world", np.eye(4)
    )
    write_json(output / "scene.json", scene.description)
    write_json(output / "robot_model.json", robot_kwargs)
    context = dict(
        native_scene_export=str(native.resolve()),
        fixed_files=[
            str((native / "scene_config.json").resolve()),
            str(registered / "scene.json"),
            str(source / "robot.json"),
            str(source / "trajectory_original.npz"),
            str(output / "robot_model.json"),
            *([str(visual_urdf.resolve())] if visual_urdf else []),
            *([str(initial_trajectory)] if initial_trajectory else []),
        ],
        initial_trajectory=(str(initial_trajectory) if initial_trajectory else None),
        prepared_source=str(source),
        registered_source=str(registered),
        instruction=instruction,
        scene_path=str(output / "scene.json"),
        robot_model_path=str(output / "robot_model.json"),
        visual_urdf=str(visual_urdf.resolve()) if visual_urdf else None,
        previous_run_reviews=(
            [
                *prior_context.get("previous_run_reviews", []),
                *(entry["review"] for entry in prior_result["history"]),
            ]
            if previous
            else []
        ),
    )
    if previous:
        context["continued_from"] = str(previous)
        context["fixed_files"].extend(
            [
                str(previous / "annotation/annotation.json"),
                str(prior_receipt),
                str(prior_physics),
                *context["previous_run_reviews"],
            ]
        )
    write_json(output / "context.json", context)

    def replay(qpos: np.ndarray, iteration_dir: Path) -> Path:
        iteration_dir = iteration_dir.resolve()
        iteration_dir.mkdir(parents=True, exist_ok=True)
        qpos = np.asarray(qpos)
        if qpos.shape != inputs.demonstration.qpos.shape or not np.isfinite(qpos).all():
            raise ValueError(
                "Replay requires finite commands with unchanged sample/joint shape"
            )
        trajectory_path = iteration_dir / "trajectory.npz"
        submitted_sha256 = digest(trajectory_path)

        def record_execution(render: Path) -> None:
            if digest(trajectory_path) != submitted_sha256:
                raise ValueError("Submitted trajectory changed during replay")
            write_json(
                iteration_dir / "replay_receipt.json",
                {
                    "trajectory_sha256": submitted_sha256,
                    "rollout_sha256": digest(
                        render / "diagnostics/physics_rollout.npz"
                    ),
                },
            )

        if previous and iteration_dir.name == "000":
            if not np.array_equal(qpos, initial_qpos):
                raise ValueError(
                    "Prior observations require exactly the executed commands"
                )
            render = iteration_dir / "native_render"
            render.symlink_to(prior_rollout, target_is_directory=True)
            record_execution(render)
            write_json(
                iteration_dir / "replay_reuse.json",
                {
                    "run": str(previous),
                    "rollout": str(prior_rollout),
                    "trajectory_sha256": digest(initial_trajectory),
                    "note": "Prior execution evidence, not a new replay under current code.",
                },
            )
            (output / "videos").symlink_to(
                (render / "videos").relative_to(output), target_is_directory=True
            )
            return render
        write_json(iteration_dir / "scene.json", scene.description)
        write_json(
            iteration_dir / "scene_export_source.json", {"path": str(native.resolve())}
        )
        trajectory = iteration_dir / "trajectory"
        trajectory.mkdir(exist_ok=True)
        np.savez_compressed(
            trajectory / "trajectory_refined.npz",
            qpos=qpos,
            timestamps=inputs.demonstration.timestamps,
            joint_names=inputs.demonstration.joint_names,
        )
        package = export_native_inputs(source, iteration_dir)
        render = iteration_dir / "native_render"
        command = [
            sys.executable,
            "-m",
            "embodichain.gen_sim.real2sim.replay",
            str(package),
            str(render),
        ]
        if visual_urdf:
            command += ["--visual-urdf", str(visual_urdf.resolve())]
        write_json(iteration_dir / "replay_command.json", command)
        with (iteration_dir / "replay.log").open("w") as log:
            subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[3],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        record_execution(render)
        # A single viewing entry points to the latest completed replay. Earlier
        # iterations retain their own videos, runtime assets and observations.
        latest = output / "videos"
        if latest.is_symlink():
            latest.unlink()
        latest.symlink_to(
            (render / "videos").relative_to(output), target_is_directory=True
        )
        return render

    from embodichain.gen_sim.real2sim.agent_refinement import run_agent_refinement

    backend = CodexTrajectoryAgent(model, reasoning_effort)

    def infer(stage: str, evidence: dict, directory: Path) -> dict:
        if previous and stage == "annotate":
            directory.mkdir(parents=True, exist_ok=True)
            write_json(
                directory / "reused_from.json",
                {"annotation": str(previous / "annotation/annotation.json")},
            )
            return json.loads((previous / "annotation/annotation.json").read_text())
        return backend(stage, evidence, directory)

    result = run_agent_refinement(
        inputs,
        scene,
        output,
        agent=infer,
        replay=replay,
        initial_qpos=initial_qpos,
        max_revisions=max_revisions,
        context=context,
    )
    write_json(output / "result.json", result)
    return result


def refine_prepared_recording(
    source: Path,
    registered: Path,
    output: Path,
    *,
    instruction: str,
    clearance: float = 0.004,
    grasp_seeds: Path | None = None,
    grasp_observations: Path | None = None,
) -> Path:
    """Run the earlier explicit contact-fit method, without the agent loop.

    Args:
        source: Prepared original recording directory.
        registered: Registered scene with existing source-event grounding.
        output: Fresh nominal-refinement directory.
        instruction: Observed task instruction.
        clearance: Existing local contact-fit clearance preference in metres.
        grasp_seeds: Optional JSON file of explicitly supplied grasp seeds.
        grasp_observations: Optional JSON file of observed object deltas.

    Returns:
        Prepared native replay package; this method does not execute it.
    """
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("--clearance must be finite and nonnegative")
    source, registered, output = (p.resolve() for p in (source, registered, output))
    output.mkdir(parents=True, exist_ok=False)
    if grasp_seeds:
        shutil.copy2(grasp_seeds, output / "grasp_seeds.json")
    if grasp_observations:
        shutil.copy2(grasp_observations, output / "grasp_observations.json")
    os.setpriority(os.PRIO_PROCESS, 0, max(15, os.getpriority(os.PRIO_PROCESS, 0)))
    cpus = set(sorted(os.sched_getaffinity(0))[:2])
    for thread in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(thread.name), cpus)
        except ProcessLookupError:
            pass
    status = dict(
        status="running",
        source=str(source),
        registered=str(registered),
        physical_success_verified=False,
    )
    try:
        inputs = load_prepared_source(source, instruction)
        scene = load_registered_scene(registered)
        from embodichain.data_pipeline.real2sim.adaptation.placement import (
            place_robot_base,
        )

        scene = place_robot_base(inputs, scene, clearance_m=clearance)
        write_json(output / "scene.json", scene.description)
        np.savez_compressed(output / "scene_vertices.npz", **scene.vertices)
        receipt = registered / "scene_export_source.json"
        native = (
            Path(json.loads(receipt.read_text())["path"])
            if receipt.is_file()
            else registered / "native_inputs/native_scene_export"
        )
        if not native.is_absolute():
            native = registered / native
        if not (native / "scene_config.json").is_file():
            raise ValueError("Registered scene requires a native scene export receipt")
        write_json(
            output / "scene_export_source.json", dict(path=str(native.resolve()))
        )
        grounding_path = registered / "grounding/grounding.json"
        grounding = json.loads(grounding_path.read_text())
        (output / "grounding").mkdir()
        shutil.copy2(grounding_path, output / "grounding/grounding.json")
        refined = refine_recording(
            inputs,
            scene,
            output / "trajectory",
            approach_clearance_m=clearance,
            grounding=tuple(grounding["events"]),
            grasp_seeds={
                (item["side"], item["grasp"]): tuple(
                    np.asarray(q, dtype=float) for q in item["qpos_candidates"]
                )
                for item in (json.loads(grasp_seeds.read_text()) if grasp_seeds else [])
            },
            grasp_object_deltas={
                (item["side"], item["grasp"]): {
                    uid: np.asarray(delta, dtype=float)
                    for uid, delta in item["object_deltas"].items()
                }
                for item in (
                    json.loads(grasp_observations.read_text())
                    if grasp_observations
                    else []
                )
            },
        )
        if len(refined.qpos) != len(inputs.demonstration.qpos):
            raise ValueError(
                "Refinement changed sample count; explicit source frame remapping is required"
            )
        package = export_native_inputs(source, output)
        status.update(
            status="refined", replay_inputs=str(package), samples=len(refined.qpos)
        )
        return package
    except Exception as error:
        diagnostics = getattr(error, "diagnostics", None)
        if diagnostics is not None:
            write_json(output / "failure_diagnostics.json", diagnostics)
        status.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "status.json", status)
