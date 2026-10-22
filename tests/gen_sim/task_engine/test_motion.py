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

from types import SimpleNamespace
import pytest
import torch
from embodichain.gen_sim.task_engine._task_program import motion, place_branch


@pytest.mark.parametrize("failure", [None, "invalid", "nonfinite"])
def test_sampled_ik_preserves_seed_order_failure_rows_and_random_stream(failure):
    generator = object.__new__(motion.CheckedMotionGenerator)
    generator.device = torch.device("cpu")
    seeds = []

    def solve(*, pose, joint_seed, name):
        assert name == "arm"
        index = len(seeds)
        seeds.append(joint_seed.clone())
        torch.rand(1)
        qpos = joint_seed + pose[:, :1, 3]
        valid = torch.ones(2, dtype=torch.bool)
        if index == 1 and failure == "invalid":
            valid[1] = False
        if index == 1 and failure == "nonfinite":
            qpos[1] = float("nan")
        return valid, qpos

    generator.robot = SimpleNamespace(compute_ik=solve)
    poses = torch.eye(4).repeat(2, 3, 1, 1)
    poses[:, :, 0, 3] = torch.tensor([0.1, 0.2, 0.3])
    start = torch.tensor([[0.0], [1.0]])
    with torch.random.fork_rng():
        torch.manual_seed(17)
        torch.rand(3)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(17)
        valid, positions = generator._solve_seeded_cartesian_waypoints(
            poses, start, "arm"
        )
        assert torch.equal(torch.get_rng_state(), expected_rng)
    assert valid.tolist() == [True, failure is None]
    torch.testing.assert_close(positions[0, :, 0], torch.tensor([0.1, 0.3, 0.6]))
    expected = [1.1, 1.3, 1.6] if failure is None else [1.0, 1.0, 1.0]
    torch.testing.assert_close(positions[1, :, 0], torch.tensor(expected))
    torch.testing.assert_close(seeds[0], start)
    torch.testing.assert_close(seeds[1], torch.tensor([[0.1], [1.1]]))
    torch.testing.assert_close(
        seeds[2], torch.tensor([[0.3], [1.3 if failure is None else 1.1]])
    )


def test_gen_sim_waypoint_dispatch_preserves_place_and_known_start(monkeypatch):
    generator = object.__new__(motion.CheckedMotionGenerator)
    generator._motion_contract = motion.SAMPLED_MOTION_CONTRACT
    poses, start, known = (
        torch.eye(4)[None, None],
        torch.zeros(1, 1),
        torch.tensor([True]),
    )
    authority = (known, start[:, None])
    monkeypatch.setattr(place_branch, "solve_waypoints", lambda *args: authority)
    assert generator._solve_cartesian_waypoints(poses, start, "arm") is authority
    monkeypatch.setattr(place_branch, "solve_waypoints", lambda *args: None)

    def shared(self, *args, known_start=None):
        assert known_start is known
        return authority

    monkeypatch.setattr(motion.MotionGenerator, "_solve_cartesian_waypoints", shared)
    assert (
        generator._solve_cartesian_waypoints(poses, start, "arm", known_start=known)
        is authority
    )
