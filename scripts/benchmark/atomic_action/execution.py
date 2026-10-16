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

"""Direct planning, expansion, physical execution and scoring for Atomic skills.

Run: embodichain benchmark atomic-action --skill all --smoke
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback
from typing import Any

import psutil
import torch

from embodichain.lab.scripts._atomic_action.defaults import _scene_args
from embodichain.lab.scripts._atomic_action.expansion import _augment
from embodichain.lab.scripts._atomic_action.support import (
    create_affordance_sampling_context,
    run_tutorial,
)
from embodichain.lab.sim.atomic_actions._expansion_phases import FREE_PHASES
from embodichain.utils import set_seed

from .evaluation import evaluate_skill
from .scenes import build_scene

__all__ = ["run_batch"]


def _synchronize() -> None:
    """Finish queued planner work before reading a wall-clock boundary."""
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _memory(*, reset_peak: bool = False) -> dict[str, float]:
    """Sample process RSS and PyTorch GPU allocation, excluding native VRAM."""
    gpu = peak = 0.0
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        gpu = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "cpu": psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        "gpu": gpu,
        "peak": peak,
    }


def _run(
    args: argparse.Namespace, options: dict[str, Any], directory: Path
) -> dict[str, Any]:
    """Execute one independent physical batch; caller owns native cleanup."""
    scene = build_scene(args)
    sim, robot, engine = scene.sim, scene.robot, scene.engine
    context = engine.initial_context(
        control_dt=sim.sim_config.physics_dt,
        affordance_sampling=create_affordance_sampling_context(args),
        **({"scene": scene.initial_scene} if scene.initial_scene is not None else {}),
    )
    before = _memory(reset_peak=True)
    started = time.perf_counter()
    compiled = engine.compile(scene.invocations, context)
    _synchronize()
    planning_ms = (time.perf_counter() - started) * 1000
    augment_start = time.perf_counter()
    trajectory, records = _augment(compiled, robot, args)
    _synchronize()
    augmentation_ms = (time.perf_counter() - augment_start) * 1000
    metadata = dict(scene.metadata)
    skill = metadata["skill"]
    phases = []
    for index, plan in enumerate(compiled.action_plans):
        for segment in plan.segments:
            absolute = compiled.segment(index, segment.name)
            target_qpos = (
                trajectory.positions[:, absolute.stop - 1].detach().cpu().clone()
            )
            if skill == "move_joint":
                target_qpos[:, metadata["arm_joint_ids"]] = torch.tensor(
                    metadata["joint_targets"][index]
                )
            phases.append(
                {
                    "skill": plan.skill_id,
                    "name": segment.name,
                    "start": absolute.start,
                    "stop": absolute.stop,
                    "target_qpos": target_qpos.tolist(),
                }
            )
    if skill == "place":
        updates = compiled.action_plans[0].expected_effects.held_object_updates
        held = next((value for value in updates.values() if value is not None), None)
        release = torch.tensor(metadata["release_pose"], device=robot.device)
        target = release.expand(args.num_envs, -1, -1).clone()
        if held is not None:
            target = target @ torch.linalg.inv(held.object_to_eef)
        metadata["object_target_pose"] = (
            target[[record["reference_row"] for record in records]]
            .detach()
            .cpu()
            .tolist()
        )
    motion_parts = ("left_arm", "right_arm") if skill == "handover" else ("arm",)
    motion_ids = {
        joint for part in motion_parts for joint in robot.get_joint_ids(name=part)
    }
    locked_ids = [
        joint for joint in range(trajectory.robot_dof) if joint not in motion_ids
    ]
    for row, record in enumerate(records):
        reference = compiled.trajectory.positions[record["reference_row"]]
        delta = (trajectory.positions[row] - reference).abs()
        record["max_joint_delta_rad"] = float(delta.max())
        record["effective_spatial_expansion"] = (
            record["operator"] != "none" and record["max_joint_delta_rad"] > 0.0
        )
        protected = float(delta[:, locked_ids].max()) if locked_ids else 0.0
        for phase in phases:
            begin, end = phase["start"], phase["stop"]
            protected = max(
                protected, float(delta[begin].max()), float(delta[end - 1].max())
            )
            if phase["name"] not in FREE_PHASES[phase["skill"]]:
                protected = max(protected, float(delta[begin:end].max()))
        record["protected_max_joint_delta_rad"] = protected
        if protected != 0.0:
            raise RuntimeError(
                "Expansion changed a protected phase, waypoint or gripper command"
            )

    execution_start = time.perf_counter()
    observations = [scene.observe()]
    positions = trajectory.positions
    for index in range(trajectory.waypoint_count):
        # These are drive targets. Actual qpos is read separately for scoring.
        robot.set_qpos(positions[:, index], target=True)
        sim.update(step=options["steps_per_waypoint"])
        observations.append(scene.observe())
    # Two independent settling observations distinguish stable placement from
    # merely passing through the target during release.
    for _ in range(2):
        robot.set_qpos(positions[:, -1], target=True)
        sim.update(step=options["settle_steps"])
        observations.append(scene.observe())
    execution_ms = (time.perf_counter() - execution_start) * 1000
    after = _memory()
    trace = {
        key: torch.stack([item[key] for item in observations])
        for key in observations[0]
    }
    planned = compiled.plan_success.detach().cpu().tolist()
    rows = evaluate_skill(skill, trace, phases, metadata, planned)
    for env_index, (row, record) in enumerate(zip(rows, records, strict=True)):
        row["expansion"] = record
        row["frame_count"] = trajectory.waypoint_count
        row["metrics"]["max_tracking_error_rad"] = float(
            (
                trace["qpos"][1 : trajectory.waypoint_count + 1, env_index][
                    :, sorted(motion_ids)
                ]
                - positions[env_index, :, sorted(motion_ids)].detach().cpu()
            )
            .abs()
            .max()
        )
    if options["save_trajectories"]:
        torch.save(
            {
                "reference_positions": compiled.trajectory.positions.detach().cpu(),
                "positions": positions.detach().cpu(),
                "dt": trajectory.dt.detach().cpu(),
                "trace": trace,
                "phases": phases,
                "metadata": metadata,
                "rows": rows,
            },
            directory / f"batch_{args._batch_id:04d}.pt",
        )
    return {
        "seed": args._config["seed"],
        "recipe_indices": list(args._recipe_indices),
        "cost_time_ms": planning_ms + augmentation_ms + execution_ms,
        "planning_time_ms": planning_ms,
        "augmentation_time_ms": augmentation_ms,
        "execution_time_ms": execution_ms,
        "cpu_delta_mb": after["cpu"] - before["cpu"],
        "gpu_delta_mb": after["gpu"] - before["gpu"],
        "peak_gpu_mb": after["peak"],
        "physics_dt": sim.sim_config.physics_dt,
        "steps_per_waypoint": options["steps_per_waypoint"],
        "scene": metadata,
        "motion_strategies": [
            invocation.motion_policy.strategy for invocation in scene.invocations
        ],
        "rows": rows,
        "plan_diagnostics": [
            {"skill": plan.skill_id, "messages": list(plan.diagnostics.messages)}
            for plan in compiled.action_plans
        ],
    }


def run_batch(
    config: dict[str, Any],
    indices: tuple[int, ...],
    options: dict[str, Any],
    directory: Path,
    batch_id: int,
) -> dict[str, Any]:
    """Build, measure and release one fresh scene without a Gym/Task Program host.

    Args:
        config: Independent sample configuration, including seed and selectors.
        indices: Logical recipe indices assigned to this physical batch.
        options: Benchmark replay, settling and artifact settings.
        directory: Case artifact directory.
        batch_id: Unique batch identity within the case.

    Returns:
        Measured per-row outcomes, expansion proof and time/memory statistics.
    """
    args = _scene_args(config, indices, batch_id)
    args.num_envs = len(indices)
    result = None
    error = None

    def run() -> None:
        nonlocal result, error
        try:
            set_seed(config["seed"])
            result = _run(args, options, directory)
        except Exception:
            # Drop the traceback before destroying borrowed native wrappers.
            error = traceback.format_exc()

    run_tutorial(run)
    if error is not None:
        raise RuntimeError(error)
    if result is None:
        raise RuntimeError("Atomic benchmark produced no measured result")
    return result
