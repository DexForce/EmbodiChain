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

from types import SimpleNamespace as NS
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import motion


def _generator(monkeypatch, sampled=True):
    generator = object.__new__(motion.CheckedMotionGenerator)
    generator.device, generator.planner = torch.device("cpu"), NS()
    generator._motion_contract = (
        motion.SAMPLED_MOTION_CONTRACT if sampled else motion.SHARED_MOTION_CONTRACT
    )
    seeds = []

    def solve(*, pose, joint_seed, name):
        seeds.append(joint_seed.clone())
        torch.rand(1)
        return torch.ones(pose.shape[0], dtype=torch.bool), pose[:, :1, 3].clone()

    generator.robot = NS(
        compute_ik=solve,
        get_solver=lambda name: None,
        compute_batch_ik=lambda **kw: (
            torch.ones(kw["pose"].shape[:2], dtype=torch.bool),
            kw["pose"][:, :, :1, 3].clone(),
        ),
        get_qvel_limits=lambda **kw: torch.tensor([[2.0]]),
    )
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.tensor([[2.0]])
    )
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[:, 0, 3] = torch.tensor([0.01, 0.02])
    targets = [motion.PlanState.from_xpos(pose[None]) for pose in poses]
    options = motion.MotionGenOptions(
        strategy="ik_interp",
        control_part="arm",
        start_qpos=torch.zeros(1, 1),
        sample_count=3,
        interpolation_dt=0.04,
        preserve_cartesian_samples=True,
        acceleration_limit=0.5,
    )
    return generator, targets, options, seeds


def test_sampled_path_uses_shared_generate_and_preserves_grid_seed_and_rng(monkeypatch):
    generator, targets, options, seeds = _generator(monkeypatch)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        torch.rand(2)
        expected = torch.get_rng_state().clone()
        torch.manual_seed(17)
        result = generator.generate(targets, options)
        assert torch.equal(torch.get_rng_state(), expected)
    torch.testing.assert_close(
        result.positions, torch.tensor([[[0.0], [0.01], [0.02]]])
    )
    torch.testing.assert_close(result.dt, torch.tensor([[0.0, 0.04, 0.04]]))
    assert [s.item() for s in seeds] == pytest.approx([0, 0.01])
    assert options.acceleration_limit == 0.5
    assert result.success.all()


def test_sampled_over_velocity_is_rejected_not_retimed(monkeypatch):
    generator, targets, options, _ = _generator(monkeypatch)
    targets[-1].xpos[:, 0, 3] = 0.5
    result = generator.generate(targets, options)
    assert result.success.tolist() == [False]
    assert result.positions.shape[1] == 3


def test_unselected_generator_keeps_main_acceleration_planning(monkeypatch):
    generator, targets, options, _ = _generator(monkeypatch, False)
    result_positions = torch.zeros(1, 5, 1)
    limited = Mock(return_value=(torch.tensor([True]), result_positions))
    generator._interpolate_cartesian_with_limits = limited
    result = generator.generate(targets, options)
    limited.assert_called_once()
    assert result.positions.shape[1] == 5


@pytest.mark.parametrize(
    "field,value", [("velocity_limit", 0.2), ("acceleration_limit", 0.1)]
)
def test_sampled_never_discards_custom_new_limits(monkeypatch, field, value):
    generator, targets, options, _ = _generator(monkeypatch)
    setattr(options, field, value)
    with pytest.raises(ValueError, match="explicit motion limits"):
        generator.generate(targets, options)


@pytest.mark.parametrize(
    "bad", [None, False, 7, "legacy", " gen_sim.sampled_velocity/v1"]
)
def test_unknown_contract_is_rejected(bad):
    with pytest.raises(ValueError, match="motion_contract"):
        motion.validate_motion_contract(bad)


@pytest.mark.parametrize("joint", [False, True])
def test_ordinary_fixed_grid_uses_existing_interpolator(monkeypatch, joint):
    generator, targets, options, _ = _generator(monkeypatch)
    if joint:
        targets = [
            motion.PlanState.from_qpos(torch.tensor([[0.01]])),
            motion.PlanState.from_qpos(torch.tensor([[0.02]])),
        ]
    result = generator.generate(
        targets, replace(options, preserve_cartesian_samples=False)
    )
    assert result.positions.shape == (1, 3, 1)
    torch.testing.assert_close(result.positions[:, -1], torch.tensor([[0.02]]))
    torch.testing.assert_close(result.dt, torch.tensor([[0.0, 0.04, 0.04]]))


def test_direct_base_point_query_uses_same_contract_hooks(monkeypatch):
    generator, targets, options, seeds = _generator(monkeypatch)
    result = motion.MotionGenerator.generate(
        generator, [targets[0]], options=replace(options, sample_count=2)
    )
    torch.testing.assert_close(result.positions, torch.tensor([[[0.0], [0.01]]]))
    assert len(seeds) == 1


def test_known_start_and_place_branch_keep_their_authority(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import place_branch

    generator, targets, options, _ = _generator(monkeypatch, False)
    poses = torch.stack([s.xpos for s in targets], dim=1)
    known = torch.tensor([True])
    backend = Mock(return_value=(known, torch.zeros(1, 2, 1)))
    monkeypatch.setattr(motion.MotionGenerator, "_solve_cartesian_waypoints", backend)
    generator._solve_cartesian_waypoints(
        poses, options.start_qpos, "arm", known_start=known
    )
    assert backend.call_args.kwargs["known_start"] is known
    authority = (torch.tensor([False]), torch.ones(1, 2, 1))
    monkeypatch.setattr(place_branch, "solve_waypoints", lambda *args: authority)
    assert (
        generator._solve_cartesian_waypoints(poses, options.start_qpos, "arm")
        is authority
    )


def test_nonfinite_ik_row_holds_without_poisoning_successful_peer(monkeypatch):
    generator, targets, options, _ = _generator(monkeypatch)
    generator.robot.get_qvel_limits = lambda **kw: torch.tensor([[2.0], [2.0]])
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.tensor([[2.0], [2.0]])
    )
    generator.robot.compute_ik = lambda **kw: (
        torch.tensor([True, True]),
        torch.tensor([[0.01], [float("nan")]]),
    )
    targets = [motion.PlanState.from_xpos(t.xpos.expand(2, -1, -1)) for t in targets]
    result = generator.generate(targets, replace(options, start_qpos=torch.zeros(2, 1)))
    assert result.success.tolist() == [True, False]
    assert torch.isfinite(result.positions).all()
    assert torch.equal(result.positions[1], torch.zeros(3, 1))
