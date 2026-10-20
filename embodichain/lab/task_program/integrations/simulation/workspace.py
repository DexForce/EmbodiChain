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

"""Compatibility adapter for Task Program workspace validation results."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.sim.scene.expansion import ScenePoseChange
from embodichain.lab.sim.scene.workspace import (
    RobotSceneWorkspace as _RobotSceneWorkspace,
    SceneWorkspaceCandidate,
)

__all__ = ["RobotSceneWorkspace", "SceneWorkspaceCandidate"]


class RobotSceneWorkspace(_RobotSceneWorkspace):
    """Preserve the Task Program import and its ValidationResult contract.

    New scene consumers should import the reusable service from
    :mod:`embodichain.lab.sim.scene.workspace`.
    """

    def check_object_pose(
        self,
        pose_change: ScenePoseChange,
        object_grasp_pose: torch.Tensor | Sequence[Sequence[float]],
        *,
        joint_seed: Sequence[float] | torch.Tensor | None = None,
    ) -> ValidationResult:
        """Convert scene checks to the existing execution validation contract.

        Args:
            pose_change: Object pose in the local arena frame.
            object_grasp_pose: TCP-to-object transform.
            joint_seed: Optional selected-control-part joint seed.

        Returns:
            Required Task Program validation evidence.
        """
        report = super().check_object_pose(
            pose_change, object_grasp_pose, joint_seed=joint_seed
        )
        return ValidationResult(
            tuple(
                ValidationCheck(
                    check.name,
                    "unavailable" if check.status == "unverified" else check.status,
                    check.message,
                    check.metrics,
                )
                for check in report.checks
            )
        )
