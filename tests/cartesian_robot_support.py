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

import torch


class _Solver:
    """Identity kinematics used to validate path derivative propagation."""

    def get_jacobian(self, qpos: torch.Tensor, jac_type: str = "full") -> torch.Tensor:
        assert jac_type == "full"
        dof = qpos.shape[-1]
        return torch.eye(6, dof, dtype=qpos.dtype).expand(qpos.shape[0], -1, -1)


class CartesianRobot:
    """Minimal robot interface used by the tutorial waypoint builders."""

    def __init__(
        self, batch_size: int = 2, dof: int = 6, *, dtype: torch.dtype = torch.float32
    ) -> None:
        self.qpos = torch.zeros(batch_size, dof, dtype=dtype)
        self.commands: list[torch.Tensor] = []
        self.solver = _Solver()
        self.fk_call_count = 0
        self.path_ik_call_count = 0

    def get_qpos(self, name: str) -> torch.Tensor:
        assert name == "left_arm"
        return self.qpos

    def get_joint_ids(self, name: str) -> list[int]:
        assert name == "left_arm"
        return list(range(self.qpos.shape[1]))

    def get_qpos_limits(self, joint_ids: list[int]) -> torch.Tensor:
        assert joint_ids == list(range(self.qpos.shape[1]))
        limits = torch.tensor((-1.0, 1.0)).repeat(self.qpos.shape[1], 1)
        return limits.unsqueeze(0).repeat(self.qpos.shape[0], 1, 1)

    def compute_fk(
        self,
        qpos: torch.Tensor,
        name: str,
        to_matrix: bool,
        env_ids: list[int] | None = None,
    ) -> torch.Tensor:
        assert name == "left_arm"
        self.fk_call_count += 1
        if env_ids is not None:
            assert len(env_ids) == qpos.shape[0]
        if to_matrix:
            pose = torch.eye(4, dtype=qpos.dtype).repeat(qpos.shape[0], 1, 1)
            pose[:, :3, 3] = qpos[:, :3]
            return pose
        quaternion = qpos.new_zeros((qpos.shape[0], 4))
        quaternion[:, 3] = 1.0
        return torch.cat((qpos[:, :3], quaternion), dim=-1)

    def compute_ik(
        self,
        pose: torch.Tensor,
        joint_seed: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert name == "left_arm"
        qpos = joint_seed.clone()
        qpos[:, :3] = pose[:, :3, 3]
        success = torch.ones(pose.shape[0], dtype=torch.bool)
        return success, qpos

    def compute_batch_ik(
        self,
        pose: torch.Tensor,
        joint_seed: torch.Tensor,
        name: str,
        *,
        continuous: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert name == "left_arm"
        assert continuous
        self.path_ik_call_count += 1
        qpos = joint_seed[:, None].expand(-1, pose.shape[1], -1).clone()
        qpos[:, :, :3] = pose[:, :, :3, 3]
        success = torch.ones(pose.shape[:2], dtype=torch.bool)
        return success, qpos

    def get_solver(self, name: str) -> _Solver:
        assert name == "left_arm"
        return self.solver

    def get_control_part_base_pose(self, name: str, to_matrix: bool) -> torch.Tensor:
        assert name == "left_arm"
        assert to_matrix
        return torch.eye(4, dtype=self.qpos.dtype).repeat(self.qpos.shape[0], 1, 1)

    def set_qpos(self, qpos: torch.Tensor, name: str) -> None:
        assert name == "left_arm"
        self.commands.append(qpos.clone())
