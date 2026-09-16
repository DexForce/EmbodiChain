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

"""Released-hand clearance binds poses without taking over execution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.release_clearance import (
    _ClearReleasedFactory,
    _ClearReleasedLowerer,
    _clearance_poses,
)
from embodichain.lab.sim.atomic_actions import (
    EndEffectorPoseGoal,
    MoveEndEffectorOptions,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall

__all__: list[str] = []


def test_clearance_raises_before_withdrawing_and_preserves_each_row() -> None:
    current = torch.eye(4).repeat(2, 1, 1)
    current[:, 2, 3] = torch.tensor([0.8, 1.4])
    root = torch.eye(4).repeat(2, 1, 1)
    root[:, 0, 3] = torch.tensor([1.0, -1.0])
    before = current.clone()
    poses = _clearance_poses(current, root, height=1.2, retreat=0.1)

    assert poses.shape == (2, 2, 4, 4)
    torch.testing.assert_close(
        poses[:, :, 2, 3], torch.tensor([[1.2, 1.2], [1.4, 1.4]])
    )
    torch.testing.assert_close(poses[:, 0, :2, 3], current[:, :2, 3])
    torch.testing.assert_close(poses[:, 1, 0, 3], torch.tensor([0.1, -0.1]))
    torch.testing.assert_close(
        poses[:, :, :3, :3], current[:, None, :3, :3].expand(-1, 2, -1, -1)
    )
    torch.testing.assert_close(current, before)


def _binding():
    endpoint = SimpleNamespace(
        task_state_key="left",
        runtime_target=SimpleNamespace(control_part="arm", joint_ids=(0, 1)),
    )
    return SimpleNamespace(
        binding=SimpleNamespace(
            resources={"primary": SimpleNamespace(endpoints={"motion": endpoint})}
        )
    )


def test_clearance_rejects_a_live_attachment_before_kinematics() -> None:
    robot = Mock()
    lowerer = _ClearReleasedLowerer(
        (("can", "safe", 1.2),), robot, retreat_distance=0.1
    )
    context = SimpleNamespace(
        task=SimpleNamespace(
            get_held_object=lambda key: SimpleNamespace(
                active_mask=torch.tensor([True])
            )
        )
    )
    call = RegisteredSemanticCall(
        call_id="gen_sim.clear_released",
        arguments={"object": "can", "target": "safe"},
    )
    with pytest.raises(ValueError, match="cannot move a held object"):
        lowerer.lower(
            call,
            context=context,
            bound=_binding(),
            option_template=MoveEndEffectorOptions(),
        )
    assert robot.mock_calls == []


def test_clearance_binds_one_shared_cartesian_goal_without_planning() -> None:
    current = torch.eye(4).unsqueeze(0)
    root = current.clone()
    root[:, 0, 3] = 1.0
    robot = Mock()
    robot.compute_fk.return_value = current
    robot.get_link_pose.return_value = root
    robot.cfg.solver_cfg = {"arm": SimpleNamespace(root_link_name="arm_root")}
    lowerer = _ClearReleasedLowerer(
        (("can", "safe", 1.2),), robot, retreat_distance=0.1
    )
    context = SimpleNamespace(
        task=SimpleNamespace(get_held_object=lambda key: None),
        robot=SimpleNamespace(qpos=torch.zeros(1, 2)),
        env_ids=torch.tensor([0]),
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id="gen_sim.clear_released",
            arguments={"object": "can", "target": "safe"},
        ),
        context=context,
        bound=_binding(),
        option_template=MoveEndEffectorOptions(),
    )
    assert type(result.goal) is EndEffectorPoseGoal
    assert result.goal.xpos.shape == (1, 2, 4, 4)
    assert result.registered_effect is None
    robot.compute_ik.assert_not_called()
    robot.plan.assert_not_called()
    robot.step.assert_not_called()


def test_clearance_factory_requires_the_same_robot() -> None:
    factory = _ClearReleasedFactory((("can", "safe", 1.2),))
    with pytest.raises(ValueError, match="factory's robot"):
        factory.create(
            simulation=None,
            robot=object(),
            scene_registry=Mock(),
            engine=SimpleNamespace(robot=object()),
        )
