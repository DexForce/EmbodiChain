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

"""GenSim-only prefiltering uses the unchanged shared PickUp implementation."""

from __future__ import annotations

import math
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.actions import GenSimPickUp
from embodichain.gen_sim.task_engine._task_program.motion import CheckedMotionGenerator
from embodichain.lab.sim.atomic_actions import (
    JointPositionTarget,
    PickUp,
    PickUpOptions,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.motion.solvers.qpos_seed_sampler import QposSeedSampler
from embodichain.lab.sim.objects.robot import Robot
from embodichain.utils.math import axis_angle_to_rotation_matrix

ARM_DOF = 6


def _bind(action_type, robot):
    action = action_type()
    action._planning_services = SimpleNamespace(robot=robot, device=torch.device("cpu"))
    return action


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("exemption", [None, "upright", "no_angle"])
@pytest.mark.parametrize("all_rejected", [False, True])
def test_prefilter_preserves_candidate_indices_rolls_and_environment_rows(
    num_envs, exemption, all_rejected
):
    robot = Mock()
    robot.compute_fk.side_effect = lambda **kw: torch.eye(4).repeat(num_envs, 1, 1)
    calibration = torch.eye(4)
    calibration[:3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([[0.0, math.pi / 2, 0.0]])
    )[0]
    poses = torch.eye(4).repeat(num_envs, 3, 1, 1)
    poses[..., 0, 3] = torch.arange(num_envs * 3).reshape(num_envs, 3)
    poses[:, 1, :3, :3] = calibration[:3, :3]
    if num_envs == 2:
        poses[0, 2, :3, :3] = calibration[:3, :3]
        poses[1, 0, :3, :3] = calibration[:3, :3]
    if all_rejected:
        poses[..., :3, :3] = calibration[:3, :3]
    options = PickUpOptions(
        grasp_frame_to_eef=calibration,
        approach_alignment_max_angle=None if exemption == "no_angle" else 0.1,
        rotate_upright=0.5 if exemption == "upright" else None,
        downstream_object_target_poses=(torch.eye(4),),
    )
    args = (
        poses,
        torch.zeros(num_envs, ARM_DOF),
        torch.eye(4).repeat(num_envs, 1, 1),
        JointPositionTarget("arm", tuple(range(ARM_DOF))),
        options,
        torch.tensor([1.0, 0.0, 0.0]),
    )
    results = []
    counts = []
    rng = []

    def solve(*, pose, joint_seed, name):
        assert pose.shape[0] == num_envs
        counts.append(pose.shape[1])
        return torch.ones(pose.shape[:2], dtype=torch.int32), joint_seed + torch.rand(1)

    robot.compute_batch_ik.side_effect = solve
    with torch.random.fork_rng(devices=[]):
        for action_type in (PickUp, GenSimPickUp, PickUp):
            counts.clear()
            torch.manual_seed(7)
            results.append(
                _bind(action_type, robot)._select_feasible_grasp_variants(*args)
            )
            rng.append(torch.get_rng_state())
            expected_count = 6
            if action_type is GenSimPickUp and exemption is None:
                expected_count = 0 if all_rejected else 4
            assert counts == ([expected_count] * 4 if expected_count else [])
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(rng[0], rng[2])
    if not all_rejected or exemption is not None:
        assert torch.equal(rng[0], rng[1])
    if exemption is None:
        expected = (
            [[False, False, False]] * num_envs
            if all_rejected
            else (
                [[True, False, True]]
                if num_envs == 1
                else [[True, False, False], [False, False, True]]
            )
        )
        assert results[1][1].tolist() == expected


def test_prefilter_keeps_real_robot_root_mapping_and_seed_stream():
    num_envs = 2
    robot = Mock()
    robot.device = torch.device("cpu")
    robot._all_indices = [0, 1]
    roots = torch.eye(4).repeat(num_envs, 1, 1)
    roots[:, 0, 3] = torch.tensor([2.0, 5.0])

    def root_pose(*, env_ids, **kwargs):
        assert env_ids == [0, 1]
        return roots[env_ids]

    robot.get_link_pose.side_effect = root_pose
    robot.compute_fk.side_effect = lambda **kw: roots.clone()
    sampler = QposSeedSampler(15, ARM_DOF, robot.device)
    targets = []

    def solve(*, target_xpos, qpos_seed, **kwargs):
        targets.append(target_xpos.clone())
        seeds = sampler.sample(
            qpos_seed, -torch.ones(ARM_DOF), torch.ones(ARM_DOF), len(qpos_seed)
        ).reshape(-1, 15, ARM_DOF)
        return torch.ones(len(qpos_seed), dtype=torch.bool), qpos_seed + seeds[:, 1]

    robot._solvers = {
        "arm": SimpleNamespace(dof=ARM_DOF, root_link_name="root", get_ik=solve)
    }
    robot.compute_batch_ik = MethodType(Robot.compute_batch_ik, robot)
    poses = roots[:, None].repeat(1, 3, 1, 1)
    poses[:, 1, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    poses[..., 0, 3] += torch.tensor([0.1, 0.2, 0.3])
    args = (
        poses,
        torch.zeros(num_envs, ARM_DOF),
        roots,
        JointPositionTarget("arm", tuple(range(ARM_DOF))),
        PickUpOptions(approach_alignment_max_angle=0.1),
        torch.tensor([0.0, 0.0, 1.0]),
    )
    results, streams = [], []
    with torch.random.fork_rng(devices=[]):
        for action_type in (PickUp, GenSimPickUp):
            torch.manual_seed(7)
            results.append(
                _bind(action_type, robot)._select_feasible_grasp_variants(*args)
            )
            streams.append(torch.get_rng_state())
    for before, after in zip(results[0], results[1]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert torch.equal(streams[0], streams[1])
    for before, after in zip(targets[:3], targets[3:]):
        expected = before.reshape(num_envs, 3, 2, 4, 4)[:, [0, 2]].reshape(-1, 4, 4)
        torch.testing.assert_close(after, expected, rtol=0, atol=0)
    assert (
        GenSimPickUp._compute_batch_candidate_ik is PickUp._compute_batch_candidate_ik
    )


def test_gensim_uses_unchanged_full_waypoint_loop():
    assert (
        CheckedMotionGenerator._solve_cartesian_waypoints
        is MotionGenerator._solve_cartesian_waypoints
    )
    for generator_type in (MotionGenerator, CheckedMotionGenerator):
        generator = object.__new__(generator_type)
        generator.device = torch.device("cpu")
        calls = []

        def solve(*, joint_seed, **kwargs):
            calls.append(1)
            return torch.tensor([len(calls) != 1]), joint_seed.clone()

        generator.robot = SimpleNamespace(compute_ik=solve)
        success, path = generator._solve_cartesian_waypoints(
            torch.eye(4).repeat(1, 4, 1, 1), torch.zeros(1, 1), "arm"
        )
        assert not success.any()
        assert torch.equal(path, torch.zeros(1, 4, 1))
        assert len(calls) == 4
