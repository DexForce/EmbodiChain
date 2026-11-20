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

"""Task-derived ordered reaching without hand actions or object dynamics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from embodichain.compute.kinematics import pose_error

from ..artifacts import _read_case_manifest
from ..config import (
    SuiteCfg,
    TrackCfg,
    stable_hash,
    _validate_affordance_reaching_track,
)
from ..embodiments import resolve_embodiment
from ..models import BenchmarkCase
from ..registry import register_scenario_provider
from ._atomic_reference import _qualified_reference_ik
from .base import ScenarioProvider

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["AffordanceReachingScenario"]

_STAGES = {
    "pick_up": ("pre_grasp", "grasp", "lift"),
    "move_held_object": ("pre_grasp", "grasp", "lift", "transport"),
    "place": (
        "pre_grasp",
        "grasp",
        "lift",
        "place_approach",
        "release_pose",
        "retract",
    ),
    "pour": ("pre_grasp", "grasp", "lift", "tilt", "return"),
    "axis_align": ("pre_grasp", "grasp", "lift", "align"),
}
_CONTROLS = frozenset({"move_end_effector", "move_joints"})


def _task_targets(case: BenchmarkCase) -> tuple[torch.Tensor, tuple[str, ...], int]:
    """Restore the frozen pickup prefix omitted by transport/place manifests."""
    targets = case.target_waypoints.clone()
    if case.skill_id in _CONTROLS:
        return targets, tuple(f"goal_{i + 1}" for i in range(case.num_waypoints)), 0
    if case.skill_id not in _STAGES:
        raise ValueError(f"Unsupported affordance source skill: {case.skill_id!r}.")
    prefix_count = 0
    if case.skill_id in {"move_held_object", "place"}:
        parameters = case.case_parameters
        grasp = torch.as_tensor(
            parameters["grasp_pose"], device=targets.device, dtype=targets.dtype
        )
        direction = torch.as_tensor(
            parameters["approach_direction"], device=targets.device, dtype=targets.dtype
        )
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if not torch.isfinite(norm).all() or bool((norm <= 0).any()):
            raise ValueError("Frozen approach_direction must be finite and nonzero.")
        pre_grasp, lift = grasp.clone(), grasp.clone()
        pre_grasp[:, :3, 3] -= (
            direction / norm * float(parameters["pre_grasp_distance_m"])
        )
        lift[:, 2, 3] += float(parameters["lift_height_m"])
        targets = torch.cat(
            (torch.stack((pre_grasp, grasp, lift), dim=1), targets), dim=1
        )
        prefix_count = 3
    stages = _STAGES[case.skill_id]
    if (
        targets.shape != (case.batch_size, len(stages), 4, 4)
        or not torch.isfinite(targets).all()
    ):
        raise ValueError(
            "Frozen affordance target count or values violate the declared sequence."
        )
    return targets, stages, prefix_count


def _project_case(
    source: BenchmarkCase,
    suite: SuiteCfg,
    track: TrackCfg,
    robot: Robot,
    control_part: str,
) -> BenchmarkCase:
    """Freeze one common request before any candidate planner is constructed."""
    targets, stages, prefix_count = _task_targets(source)
    references = source.reference_qpos.clone()
    if prefix_count:
        previous = source.start_qpos.clone()
        prefix = []
        for target in targets[:, :prefix_count].unbind(dim=1):
            valid, previous = _qualified_reference_ik(
                robot,
                target,
                previous,
                control_part=control_part,
                protocol=suite.protocol,
            )
            if not bool(valid.all()):
                raise ValueError(
                    f"Cannot certify frozen pickup prefix for {source.case_id}; inputs were not replaced."
                )
            prefix.append(previous.clone())
        references = torch.cat((torch.stack(prefix, dim=1), references), dim=1)

    # Witnesses qualify inputs only. Cartesian adapters do not receive them as goals.
    actual = robot.compute_batch_fk(qpos=references, name=control_part, to_matrix=True)
    dp, dr = pose_error(actual, targets)
    lower, upper = robot.get_qpos_limits(name=control_part).to(references).unbind(-1)
    violation = torch.maximum(
        lower[:, None] - references, references - upper[:, None]
    ).clamp_min(0)
    if not (
        torch.isfinite(references).all()
        and (dp.norm(dim=-1) <= suite.protocol.position_threshold_m).all()
        and (dr.norm(dim=-1) <= suite.protocol.rotation_threshold_rad).all()
        and (violation <= suite.protocol.joint_limit_tolerance_rad).all()
    ):
        raise ValueError(
            f"Frozen reference fails independent qualification: {source.case_id}."
        )

    parameters = deepcopy(source.case_parameters)
    parameters.update(
        measurement_purpose="affordance_reaching",
        affordance_protocol="task_space_affordance_reaching_v1",
        affordance_stage_names=list(stages),
        affordance_group="control" if source.skill_id in _CONTROLS else "task",
        source_case_id=source.case_id,
        source_cases_sha256=track.config["source_cases_sha256"],
        physical_execution="not_evaluated",
    )
    if source.skill_id == "axis_align":
        parameters["alignment_constraint"] = "frozen_full_pose_realization"
    return replace(
        source,
        suite_version=suite.suite_version,
        track=track.id,
        track_group=track.group_id or track.id,
        case_id=f"affordance_reaching:{source.case_id}",
        primary_success="geometry_valid",
        num_waypoints=targets.shape[1],
        target_waypoints=targets,
        reference_qpos=references,
        start_qpos=source.start_qpos.clone(),
        full_start_qpos=(
            None if source.full_start_qpos is None else source.full_start_qpos.clone()
        ),
        case_parameters=parameters,
        required_capabilities=frozenset(),
    )


class AffordanceReachingScenario(ScenarioProvider):
    """Project hash-bound Atomic cases into independently scored joint paths.

    Inputs retain their task frames, selected grasps, starts and strict tolerances.
    No object is spawned, hand command compiled or physical trajectory replayed.
    AxisAlign remains its frozen full-pose realization, not an axis-only request.
    """

    required_capabilities = frozenset({"batched", "empty_world"})

    def batch_sizes(self, suite: SuiteCfg, track: TrackCfg) -> list[int]:
        """Return the source batch size declared by the validated track."""
        _validate_affordance_reaching_track(track)
        return [track.config["batch_size"]]

    def generate_cases(
        self,
        suite: SuiteCfg,
        track: TrackCfg,
        robot: Robot,
        control_part: str,
        batch_size: int,
    ) -> list[BenchmarkCase]:
        """Load immutable task geometry and qualify missing prefix witnesses.

        Args:
            suite: Independently named reaching suite.
            track: Hash-bound source track configuration.
            robot: Same kinematic embodiment as the frozen source.
            control_part: Controlled arm group.
            batch_size: Source environment count per case.

        Returns:
            Common ordered requests; failed qualification raises without resampling.
        """
        _validate_affordance_reaching_track(track)
        path = Path(track.config["source_manifest"])
        payload = json.loads(path.read_text())
        if (
            payload.get("cases_sha256") != track.config["source_cases_sha256"]
            or stable_hash(payload.get("cases")) != track.config["source_cases_sha256"]
        ):
            raise ValueError("Affordance source manifest hash mismatch.")
        sources = _read_case_manifest(
            path,
            robot.device,
            expected_embodiment=resolve_embodiment(
                suite.embodiment, suite.robot, base_dir=Path.cwd()
            ),
        )
        selected = [
            case for case in sources if case.track == track.config["source_track"]
        ]
        if not selected or any(case.batch_size != batch_size for case in selected):
            raise ValueError(
                "Affordance source track is empty or its batch size differs."
            )
        return [
            _project_case(case, suite, track, robot, control_part) for case in selected
        ]

    def trial_metadata(self, case: BenchmarkCase) -> dict[str, object]:
        """Identify geometry-only evidence independently from physical Atomic."""
        return {
            **super().trial_metadata(case),
            "evaluation_scope": "task_space_affordance_reaching_v1",
            "primary_success": "geometry_valid",
            "physical_execution": "not_evaluated",
            "source_case_id": case.case_parameters["source_case_id"],
            "affordance_stage_names": case.case_parameters["affordance_stage_names"],
        }


register_scenario_provider("affordance_reaching", AffordanceReachingScenario)
