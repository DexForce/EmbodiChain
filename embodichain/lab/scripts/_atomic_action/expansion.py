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

"""Atomic plan adaptation, fixed-clock expansion and replay provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions import CompiledTrajectory, TimedTrajectory
from embodichain.lab.sim.motion.expansion import (
    SceneCase,
    TrajectoryAugmentationCfg,
    TrajectoryPhase,
    TrajectoryTemplate,
    expand_trajectory_variants,
)
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

from embodichain.lab.sim.atomic_actions._expansion_phases import (
    FREE_PHASES,
    _canonical_samples,
    _free_joint_indices,
)

OPERATORS = ("joint_residual", "via_points")


def _row_sample(value: Any, row: int, batch_size: int) -> Any:
    """Copy one reference row's affordance provenance without duplicating batches."""
    if isinstance(value, dict):
        return {key: _row_sample(item, row, batch_size) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[row]
    return value


def _template(
    compiled: CompiledTrajectory,
    row: int,
    joint_names: tuple[str, ...],
    joint_ids: tuple[int, ...],
    skill: str,
    branch: int,
) -> tuple[TrajectoryTemplate, torch.Tensor]:
    positions, dt, boundaries, restore = _canonical_samples(
        compiled.trajectory.positions[row],
        compiled.trajectory.dt[row],
    )
    phases = []
    for action_index, plan in enumerate(compiled.action_plans):
        editable = FREE_PHASES.get(plan.skill_id)
        if editable is None:
            raise ValueError(f"No expansion permissions for skill {plan.skill_id!r}")
        for segment in plan.segments:
            absolute = compiled.segment(action_index, segment.name)
            start, stop = int(boundaries[absolute.start]), int(
                boundaries[absolute.stop]
            )
            if stop <= start:
                raise ValueError(f"Timing normalization erased phase {segment.name!r}")
            free = segment.name in editable
            phases.append(
                TrajectoryPhase(
                    f"{action_index}_{segment.name}",
                    start,
                    stop,
                    "free" if free else "contact",
                    OPERATORS if free else (),
                )
            )
    template = TrajectoryTemplate(
        source_id=f"atomic_action:{skill}",
        source_revision="example_scene_v1",
        template_id=f"{skill}:grasp:{branch}",
        joint_names=joint_names,
        positions=positions,
        dt=dt,
        phases=tuple(phases),
        allowed_operators=OPERATORS,
        controlled_joint_indices=(
            _free_joint_indices(positions, tuple(phases), joint_ids)
            if skill == "handover"
            else joint_ids
        ),
    )
    return template, restore


def _augment(
    compiled: CompiledTrajectory, robot: Robot, args: argparse.Namespace
) -> tuple[TimedTrajectory, list[dict[str, Any]]]:
    trajectory = compiled.trajectory
    cfg = args._config
    settings = cfg["expansion"]["trajectory"]
    spatial = settings["spatial"]
    policy = TrajectoryAugmentationCfg.from_mapping(
        {
            "seed": cfg["seed"],
            "factors": {
                "spatial": {
                    "enabled": spatial["enabled"],
                    "method": spatial["operators"],
                    "joint_offset_scale": spatial["joint_offset_scale"],
                    "via_count": spatial["via_count"],
                },
                "ik": {"enabled": False},
                "timing": {"enabled": False},
            },
            "coverage": {"joint_dedup_normalized_tol": spatial["dedup_tolerance"]},
        }
    )
    arm_parts = ("left_arm", "right_arm") if cfg["skill"] == "handover" else ("arm",)
    joint_ids = tuple(
        dict.fromkeys(
            joint_id
            for part in arm_parts
            for joint_id in robot.get_joint_ids(name=part)
        )
    )
    rows = []
    records = []
    reference_rows: dict[int, int] = {}
    row_local_reference = any(
        plan.skill_id == "slide" for plan in compiled.action_plans
    )
    for row, (branch, variant_index) in enumerate(args._recipes):
        failed = not bool(compiled.plan_success[row])
        # Slide plans against each row's measured robot and handle state,
        # including after pull replay. Its branch identity remains shared in
        # affordance sampling, while geometry, timing and provenance stay local.
        # Other sample scenes retain one reference within each grasp branch.
        reference_row = (
            row
            if row_local_reference or failed
            else reference_rows.setdefault(branch, row)
        )
        record = {
            "row": row,
            "reference_row": reference_row,
            "recipe_index": args._recipe_indices[row],
            "affordance_branch": branch,
            "trajectory_variant": variant_index,
            "operator": "none",
            "rejected": {},
        }
        # Failed plans retain their own source hold and cannot seed a branch.
        if failed or variant_index == 0:
            rows.append(trajectory.positions[reference_row].clone())
        else:
            template, restore = _template(
                compiled,
                reference_row,
                tuple(robot.joint_names),
                joint_ids,
                cfg["skill"],
                branch,
            )
            spacing = template.dt[1:]
            control_dt = float(spacing[0])
            if not torch.allclose(
                spacing, torch.full_like(spacing, control_dt), atol=1e-6, rtol=1e-6
            ):
                raise ValueError(
                    "Spatial expansion requires a fixed-clock Atomic trajectory"
                )
            result = expand_trajectory_variants(
                template,
                cfg=policy,
                case=SceneCase(
                    f"atomic_action:{cfg['skill']}",
                    f"grasp:{branch}",
                    cfg["skill"],
                    cfg["skill"],
                    cfg["robot"],
                ),
                count=settings["variants_per_family"],
                joint_limits=robot.get_qpos_limits()[row].detach().cpu(),
                control_dt=control_dt,
                max_attempts=settings["max_proposals"],
            )
            if len(result.variants) <= variant_index:
                raise RuntimeError(
                    f"Recipe {args._recipe_indices[row]} needs variant {variant_index}, but only {len(result.variants)} survived: {result.rejected}"
                )
            candidate = result.candidates.positions[variant_index]
            if int(result.candidates.valid_length[variant_index]) != len(template.dt):
                raise RuntimeError(
                    "Spatial expansion changed the command-clock horizon"
                )
            restored = candidate[restore].to(
                device=trajectory.positions.device,
                dtype=trajectory.positions.dtype,
            )
            protected_ids = [
                index
                for index in range(trajectory.robot_dof)
                if index not in template.controlled_joint_indices
            ]
            restored[:, protected_ids] = trajectory.positions[
                reference_row, :, protected_ids
            ]
            # Canonicalization may merge near-identical zero-time samples.
            # Restore immutable phases from their actual source indices so
            # neither that merge nor resampling can alter a required target.
            for action_index, plan in enumerate(compiled.action_plans):
                for segment in plan.segments:
                    absolute = compiled.segment(action_index, segment.name)
                    for index in (absolute.start, absolute.stop - 1):
                        restored[index] = trajectory.positions[reference_row, index]
                    if segment.name not in FREE_PHASES[plan.skill_id]:
                        restored[absolute.start : absolute.stop] = trajectory.positions[
                            reference_row, absolute.start : absolute.stop
                        ]
            rows.append(restored)
            record.update(
                operator=result.variants[variant_index].spatial_operator,
                rejected=dict(result.rejected),
                attempted=result.attempted,
            )
        record["affordance_samples"] = [
            _row_sample(
                plan.diagnostics.metadata.get("affordance_sample", {}),
                reference_row,
                trajectory.batch_size,
            )
            for plan in compiled.action_plans
        ]
        records.append(record)
        logger.log_info(
            f"Atomic recipe {record['recipe_index']}: grasp branch={branch}, trajectory variant={variant_index}, operator={record['operator']}"
        )
    return (
        TimedTrajectory(
            positions=torch.stack(rows),
            velocities=None,
            accelerations=None,
            dt=trajectory.dt[[record["reference_row"] for record in records]].clone(),
            env_ids=trajectory.env_ids,
        ),
        records,
    )


def _save_replay(
    compiled: CompiledTrajectory,
    trajectory: TimedTrajectory,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    video_prefix: str,
) -> Path:
    cfg = args._config
    directory = Path(cfg["output"]["directory"]) / f"batch_{args._batch_id:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{args._replay_index:02d}_{video_prefix}"
    path = directory / f"{stem}.pt"
    if cfg["output"]["save_trajectories"]:
        torch.save(
            {
                "positions": trajectory.positions.detach().cpu(),
                "reference_positions": compiled.trajectory.positions[
                    [record["reference_row"] for record in records]
                ]
                .detach()
                .cpu(),
                "dt": trajectory.dt.detach().cpu(),
                "env_ids": trajectory.env_ids.detach().cpu(),
                "recipes": records,
            },
            path,
        )
    summary = {
        "segment": video_prefix,
        "batch_id": args._batch_id,
        "status": "planned",
        "measured_task_success": None,
        "trajectory_path": str(path) if cfg["output"]["save_trajectories"] else None,
        "rows": records,
    }
    (directory / f"{stem}.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    args._records.append(summary)
    args._replay_index += 1
    if cfg["output"]["save_video"]:
        args.auto_play_video_path = str(directory / f"{stem}.mp4")
    return directory / f"{stem}.json"


__all__: list[str] = []
