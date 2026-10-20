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

"""State restoration and environment isolation for robot randomizers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.gym.envs.managers import SceneEntityCfg
from embodichain.lab.gym.envs.managers.randomization.spatial import (
    randomize_rigid_object_pose,
    randomize_robot_eef_pose,
    randomize_robot_qpos,
)

pytestmark = pytest.mark.no_sim


class _Robot:
    def __init__(self) -> None:
        self.qpos = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        self.target = self.qpos.clone()
        self.velocity = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.dof = 2
        self.ik_success = True

    def get_qpos(self) -> torch.Tensor:
        return self.qpos.clone()

    def get_joint_ids(self, name: str | None = None) -> list[int]:
        return [0, 1]

    def set_qpos(
        self,
        qpos: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: list[int],
        target: bool = True,
    ) -> None:
        output = self.target if target else self.qpos
        output[env_ids[:, None], torch.tensor(joint_ids)] = qpos

    def clear_dynamics(self, env_ids: torch.Tensor) -> None:
        self.velocity[env_ids] = 0.0

    def compute_fk(
        self, *, name: str, qpos: torch.Tensor, env_ids: torch.Tensor, to_matrix: bool
    ) -> torch.Tensor:
        assert len(qpos) == len(env_ids)
        pose = torch.eye(4).repeat(len(env_ids), 1, 1)
        pose[:, :2, 3] = qpos
        return pose

    def compute_ik(
        self,
        *,
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(pose) == len(env_ids)
        return torch.full((len(env_ids),), self.ik_success), pose[:, :2, 3].clone()


def _environment() -> SimpleNamespace:
    robot = _Robot()
    return SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        robot=robot,
        sim=SimpleNamespace(
            get_robot=lambda _: robot,
            sync_render_state=Mock(),
            update=Mock(
                side_effect=AssertionError("Randomization must not step physics")
            ),
        ),
    )


@pytest.mark.parametrize("kind", ["joints", "eef"])
@pytest.mark.parametrize("selection", [None, [1], []])
def test_robot_randomization_preserves_unselected_state(
    kind: str, selection: list[int] | None
) -> None:
    env = _environment()
    rows = (
        torch.arange(2)
        if selection is None
        else torch.tensor(selection, dtype=torch.long)
    )
    env_ids = None if selection is None else rows
    before = env.robot.qpos.clone()
    velocity = env.robot.velocity.clone()
    entity = SceneEntityCfg(uid="robot", control_parts=["arm"])
    if kind == "joints":
        randomize_robot_qpos(
            env, env_ids, entity, qpos_range=([0.25, 0.25], [0.25, 0.25])
        )
    else:
        randomize_robot_eef_pose(
            env, env_ids, entity, position_range=([0.25, 0.25, 0.0], [0.25, 0.25, 0.0])
        )
    expected = before.clone()
    expected[rows] += 0.25
    velocity[rows] = 0.0
    torch.testing.assert_close(env.robot.qpos, expected)
    torch.testing.assert_close(env.robot.target, expected)
    torch.testing.assert_close(env.robot.velocity, velocity)
    assert env.sim.sync_render_state.call_count == int(len(rows) > 0)
    env.sim.update.assert_not_called()


def test_failed_ik_holds_selected_robot_position() -> None:
    env = _environment()
    env.robot.ik_success = False
    before = env.robot.qpos.clone()
    randomize_robot_eef_pose(
        env,
        torch.tensor([1]),
        SceneEntityCfg(uid="robot", control_parts=["arm"]),
        position_range=([0.25, 0.25, 0.0], [0.25, 0.25, 0.0]),
    )
    torch.testing.assert_close(env.robot.qpos, before)
    torch.testing.assert_close(env.robot.target, before)
    env.sim.update.assert_not_called()


def test_eef_randomization_requires_a_named_control_part_before_writes() -> None:
    env = _environment()
    before = env.robot.qpos.clone()
    with pytest.raises(ValueError, match="named control_parts"):
        randomize_robot_eef_pose(env, torch.tensor([0]), SceneEntityCfg(uid="robot"))
    torch.testing.assert_close(env.robot.qpos, before)
    env.sim.sync_render_state.assert_not_called()


def test_rigid_pose_randomization_preserves_unselected_velocity() -> None:
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[1, :3, 3] = torch.tensor([1.0, 1.0, 2.0])
    velocity = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def set_pose(value: torch.Tensor, env_ids: torch.Tensor) -> None:
        pose[env_ids] = value

    def clear(env_ids: torch.Tensor) -> None:
        velocity[env_ids] = 0.0

    obj = SimpleNamespace(
        cfg=SimpleNamespace(init_pos=[0.0, 0.0, 1.0], init_rot=[0.0, 0.0, 0.0]),
        set_local_pose=set_pose,
        clear_dynamics=clear,
    )
    env = SimpleNamespace(
        device=torch.device("cpu"),
        sim=SimpleNamespace(
            get_rigid_object_uid_list=lambda: ["cube"],
            get_rigid_object=lambda _: obj,
            update=Mock(
                side_effect=AssertionError("Pose randomization stepped physics")
            ),
        ),
    )
    randomize_rigid_object_pose(
        env,
        torch.tensor([0]),
        SceneEntityCfg(uid="cube"),
        position_range=([0.25, 0.0, 0.0], [0.25, 0.0, 0.0]),
    )
    torch.testing.assert_close(pose[0, :3, 3], torch.tensor([0.25, 0.0, 1.0]))
    torch.testing.assert_close(pose[1, :3, 3], torch.tensor([1.0, 1.0, 2.0]))
    torch.testing.assert_close(velocity[0], torch.zeros(3))
    torch.testing.assert_close(velocity[1], torch.tensor([4.0, 5.0, 6.0]))
    env.sim.update.assert_not_called()
