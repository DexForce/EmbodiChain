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

from dataclasses import replace

import pytest
import torch

from embodichain.compute.kinematics import DifferentiableIKCfg
from embodichain.lab.sim.motion.expansion import (
    TaskSpaceDeformationConstraint,
    TrajectoryPhase,
    TrajectoryTemplate,
    deform_task_space_trajectory,
    project_task_space_trajectory,
    task_space_from_joint_template,
)

_SAMPLE_COUNT = 7
_CONTROL_DT = 0.1


def _prismatic_kinematics(qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map three prismatic joints directly onto Cartesian translation."""
    poses = torch.eye(4, dtype=qpos.dtype, device=qpos.device).repeat(
        qpos.shape[0], 1, 1
    )
    poses[:, :3, 3] = qpos
    jacobians = torch.zeros(qpos.shape[0], 6, 3, dtype=qpos.dtype, device=qpos.device)
    jacobians[:, :3] = torch.eye(3, dtype=qpos.dtype, device=qpos.device)
    return poses, jacobians


def _joint_template() -> TrajectoryTemplate:
    positions = torch.zeros(_SAMPLE_COUNT, 3, dtype=torch.float64)
    positions[:, 0] = torch.linspace(0, 0.6, _SAMPLE_COUNT, dtype=torch.float64)
    dt = torch.full((_SAMPLE_COUNT,), _CONTROL_DT, dtype=torch.float64)
    dt[0] = 0
    return TrajectoryTemplate(
        source_id="demo",
        source_revision="one",
        template_id="reach",
        joint_names=("x", "y", "z"),
        positions=positions,
        dt=dt,
        phases=(
            TrajectoryPhase("move", 0, _SAMPLE_COUNT, "free", ("joint_residual",)),
        ),
        allowed_operators=("joint_residual",),
        controlled_joint_indices=(0, 1, 2),
        validator_id="path_and_task",
    )


def _translation_constraint() -> TaskSpaceDeformationConstraint:
    return TaskSpaceDeformationConstraint(
        "move",
        translation_axes=(False, True, False),
        rotation_axes=(False, False, False),
        max_translation_m=0.08,
    )


def test_task_space_deformation_preserves_endpoints_and_locked_axes() -> None:
    source = _joint_template()
    task_space = task_space_from_joint_template(
        source,
        frame_id="tool",
        kinematics=_prismatic_kinematics,
        editable_phase_ids=("move",),
    )

    result = deform_task_space_trajectory(
        task_space,
        constraints={"move": _translation_constraint()},
        generator=torch.Generator().manual_seed(12),
    )

    torch.testing.assert_close(result.poses[[0, -1]], task_space.poses[[0, -1]])
    torch.testing.assert_close(result.poses[:, 0, 3], task_space.poses[:, 0, 3])
    torch.testing.assert_close(result.poses[:, 2, 3], task_space.poses[:, 2, 3])
    torch.testing.assert_close(result.poses[:, :3, :3], task_space.poses[:, :3, :3])
    assert not torch.equal(result.poses[1:-1, 1, 3], task_space.poses[1:-1, 1, 3])
    assert (
        float((result.poses[:, 1, 3] - task_space.poses[:, 1, 3]).abs().max()) <= 0.08
    )
    assert task_space.allowed_operators == (
        "joint_residual",
        "task_space_residual",
    )


def test_task_space_deformation_leaves_unselected_phases_unchanged() -> None:
    source = replace(
        _joint_template(),
        phases=(
            TrajectoryPhase("move", 0, 4, "free", ("joint_residual",)),
            TrajectoryPhase("settle", 4, _SAMPLE_COUNT, "hold"),
        ),
    )
    task_space = task_space_from_joint_template(
        source,
        frame_id="tool",
        kinematics=_prismatic_kinematics,
        editable_phase_ids=("move",),
    )

    result = deform_task_space_trajectory(
        task_space,
        constraints={"move": _translation_constraint()},
        generator=torch.Generator().manual_seed(5),
    )

    torch.testing.assert_close(result.poses[4:], task_space.poses[4:])


def test_task_space_deformation_is_reproducible_without_global_rng() -> None:
    task_space = task_space_from_joint_template(
        _joint_template(),
        frame_id="tool",
        kinematics=_prismatic_kinematics,
        editable_phase_ids=("move",),
    )
    before = torch.random.get_rng_state().clone()

    first = deform_task_space_trajectory(
        task_space,
        constraints={"move": _translation_constraint()},
        generator=torch.Generator().manual_seed(4),
    )
    second = deform_task_space_trajectory(
        task_space,
        constraints={"move": _translation_constraint()},
        generator=torch.Generator().manual_seed(4),
    )

    torch.testing.assert_close(first.poses, second.poses)
    assert torch.equal(before, torch.random.get_rng_state())


def test_batched_trajectory_ik_projects_only_editable_interior_samples() -> None:
    source = _joint_template()
    task_space = task_space_from_joint_template(
        source,
        frame_id="tool",
        kinematics=_prismatic_kinematics,
        editable_phase_ids=("move",),
    )
    target = deform_task_space_trajectory(
        task_space,
        constraints={"move": _translation_constraint()},
        generator=torch.Generator().manual_seed(7),
    )

    projection = project_task_space_trajectory(
        target,
        source,
        kinematics=_prismatic_kinematics,
        joint_limits=torch.tensor(
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64
        ),
        ik_cfg=DifferentiableIKCfg(
            max_iterations=8,
            damping=1e-4,
            step_size=1.0,
            pos_eps=1e-6,
            rot_eps=1e-6,
        ),
    )

    assert projection.accepted
    torch.testing.assert_close(
        projection.template.positions[[0, -1]], source.positions[[0, -1]]
    )
    achieved, _ = _prismatic_kinematics(projection.template.positions)
    torch.testing.assert_close(achieved, target.poses, atol=1e-6, rtol=0)


def test_trajectory_ik_reports_unreachable_rotation_without_moving_endpoints() -> None:
    source = _joint_template()
    task_space = task_space_from_joint_template(
        source,
        frame_id="tool",
        kinematics=_prismatic_kinematics,
        editable_phase_ids=("move",),
    )
    rotation_only = TaskSpaceDeformationConstraint(
        "move",
        translation_axes=(False, False, False),
        rotation_axes=(False, False, True),
        max_rotation_rad=0.2,
    )
    target = deform_task_space_trajectory(
        task_space,
        constraints={"move": rotation_only},
        generator=torch.Generator().manual_seed(9),
    )

    projection = project_task_space_trajectory(
        target,
        source,
        kinematics=_prismatic_kinematics,
        joint_limits=torch.tensor(
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64
        ),
    )

    assert not projection.accepted
    assert not bool(projection.success[1:-1].all())
    torch.testing.assert_close(
        projection.template.positions[[0, -1]], source.positions[[0, -1]]
    )


def test_task_space_contract_rejects_non_rigid_poses() -> None:
    source = _joint_template()
    task_space = task_space_from_joint_template(
        source,
        frame_id="tool",
        kinematics=_prismatic_kinematics,
    )
    invalid = task_space.poses.clone()
    invalid[2, 0, 0] = 2

    with pytest.raises(ValueError, match="SE\\(3\\)"):
        replace(task_space, poses=invalid)
