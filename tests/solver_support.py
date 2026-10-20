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

"""Small shared W1 scene and numerical sampling fixtures for solver tests."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot


def create_w1_solver_scene(solver_type: str) -> tuple[SimulationManager, Robot]:
    """Create the CPU W1 scene; the caller owns destruction and cleanup."""
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.data import get_data_path

    # Both consumers use the same CPU-backed W1 scene.
    config = SimulationManagerCfg(headless=True, device="cpu")
    sim = SimulationManager(config)

    # Load robot URDF file
    urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")
    assert os.path.isfile(urdf)

    cfg_dict = {
        "fpath": urdf,
        "control_parts": {
            "left_arm": [f"LEFT_J{i+1}" for i in range(7)],
            "right_arm": [f"RIGHT_J{i+1}" for i in range(7)],
        },
        "solver_cfg": {
            "left_arm": {
                "class_type": solver_type,
                "end_link_name": "left_ee",
                "root_link_name": "left_arm_base",
            },
            "right_arm": {
                "class_type": solver_type,
                "end_link_name": "right_ee",
                "root_link_name": "right_arm_base",
            },
        },
    }

    robot = sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))
    sim.prepare()
    return sim, robot


def grid_sample_qpos_from_limits(
    qpos_limits: torch.Tensor,
    steps_per_joint: int = 4,
    device: str | torch.device | None = None,
    max_samples: int = 4096,
    *,
    safe_margin: float,
) -> torch.Tensor:
    """Generate grid samples for qpos from qpos_limits.

    Args:
        qpos_limits: tensor of shape (1, n, 2) or (n, 2) where each row is [low, high].
        steps_per_joint: Number of values per joint.
        device: torch device to place the samples on.
        max_samples: cap the number of returned samples (take first N if grid is larger).
        safe_margin: Explicit solver-specific distance from limits, in radians.

    Returns:
        Tensor of shape (N, n) where N <= max_samples.
    """
    if device is None:
        device = qpos_limits.device

    limits = qpos_limits.squeeze(0) if qpos_limits.dim() == 3 else qpos_limits
    lows = limits[:, 0].to(device) + safe_margin * 1.01
    highs = limits[:, 1].to(device) - safe_margin * 1.01

    # create per-joint linspaces
    grids = [
        torch.linspace(l.item(), h.item(), steps_per_joint, device=device)
        for l, h in zip(lows, highs)
    ]

    # meshgrid and stack
    mesh = torch.meshgrid(*grids, indexing="ij")
    stacked = torch.stack([m.reshape(-1) for m in mesh], dim=1)

    if stacked.shape[0] > max_samples:
        return stacked[:max_samples]
    return stacked


def assert_pose_ik_roundtrip(
    robot: Robot, arm_name: str, *, safe_margin: float
) -> None:
    """Check both pose encodings and an unreachable target on one solver."""
    # Test inverse kinematics (IK) with a 1x4x4 homogeneous matrix pose and a joint_seed

    qpos_limit = robot.get_qpos_limits(name=arm_name)
    # generate a small grid of qpos samples from the joint limits (low/high)
    sample_qpos = grid_sample_qpos_from_limits(
        safe_margin=safe_margin,
        qpos_limits=qpos_limit,
        steps_per_joint=8,
        device=robot.device,
        max_samples=65536,
    )
    sample_qpos = sample_qpos[None, :, :]

    fk_xpos = robot.compute_batch_fk(qpos=sample_qpos, name=arm_name, to_matrix=True)
    fk_xpos_xyzquat = robot.compute_batch_fk(
        qpos=sample_qpos, name=arm_name, to_matrix=False
    )

    matrix_success, ik_qpos = robot.compute_batch_ik(
        pose=fk_xpos, joint_seed=sample_qpos, name=arm_name
    )

    xyzquat_success, ik_qpos_xyzquat = robot.compute_batch_ik(
        pose=fk_xpos_xyzquat, joint_seed=sample_qpos, name=arm_name
    )

    assert torch.equal(matrix_success, xyzquat_success)
    assert matrix_success.all()
    matrix_ik_xpos = robot.compute_batch_fk(qpos=ik_qpos, name=arm_name, to_matrix=True)
    xyzquat_ik_xpos = robot.compute_batch_fk(
        qpos=ik_qpos_xyzquat, name=arm_name, to_matrix=True
    )
    assert torch.allclose(
        fk_xpos, matrix_ik_xpos, atol=5e-3, rtol=5e-3
    ), f"Matrix-pose IK does not reconstruct FK for {arm_name}"
    assert torch.allclose(
        fk_xpos, xyzquat_ik_xpos, atol=5e-3, rtol=5e-3
    ), f"XYZ-quaternion IK does not reconstruct FK for {arm_name}"
    # test for failed xpos
    invalid_pose = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 10.0],
                [0.0, 0.0, 1.0, 10.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
        device=robot.device,
    )
    res, ik_qpos = robot.compute_ik(
        pose=invalid_pose, joint_seed=ik_qpos[:, 0, :], name=arm_name
    )
    dof = ik_qpos.shape[-1]
    assert res[0] == False
    assert ik_qpos.shape == (1, dof)
