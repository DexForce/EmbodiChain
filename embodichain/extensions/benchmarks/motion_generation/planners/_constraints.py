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

"""Planner inputs use only the payload selected by each semantic modality."""

from __future__ import annotations

import torch

from embodichain.lab.sim.motion.planners import PlanState

from ..models import BenchmarkCase

__all__: list[str] = []


def _case_modalities(case: BenchmarkCase) -> list[str]:
    validity = case.case_parameters.get(
        "motion_validity", "ordered_cartesian_waypoints"
    )
    if validity == "ordered_joint_waypoints":
        return ["joint"] * case.num_waypoints
    if validity == "ordered_cartesian_waypoints":
        return ["pose"] * case.num_waypoints
    if validity != "ordered_constraint_waypoints":
        raise ValueError(f"Unsupported motion_validity mode {validity!r}.")
    modalities = list(case.case_parameters.get("constraint_modalities", ()))
    if len(modalities) != case.num_waypoints or any(
        value not in {"joint", "position", "orientation", "pose"}
        for value in modalities
    ):
        raise ValueError("Each ordered constraint requires one valid modality.")
    return modalities


def _case_targets(case: BenchmarkCase) -> list[PlanState]:
    targets = []
    for index, modality in enumerate(_case_modalities(case)):
        if modality == "joint":
            targets.append(PlanState.from_qpos(case.reference_qpos[:, index].clone()))
            continue
        # Unused fields can contain arbitrary witnesses or even nonfinite values.
        # They must not influence either optimization or target-frame conversion.
        pose = torch.eye(4, device=case.start_qpos.device, dtype=case.start_qpos.dtype)
        pose = pose.expand(case.batch_size, 4, 4).clone()
        if modality in {"position", "pose"}:
            pose[:, :3, 3] = case.target_waypoints[:, index, :3, 3]
        if modality in {"orientation", "pose"}:
            pose[:, :3, :3] = case.target_waypoints[:, index, :3, :3]
        targets.append(PlanState.from_xpos(pose))
    return targets
