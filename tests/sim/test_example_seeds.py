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

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.no_sim


def test_curobo_obstacle_seed_is_local_and_reproducible() -> None:
    from examples.sim.motion.planners.curobo_planner import _perturb_obstacles

    class Obstacle:
        uid = "block"

        def get_local_pose(self, *, to_matrix):
            return torch.eye(4).repeat(4, 1, 1)

        def set_local_pose(self, pose):
            self.pose = pose

    def sample(seed):
        return _perturb_obstacles(
            [Obstacle()],
            num_envs=4,
            xy_perturbation=0.02,
            yaw_perturbation_deg=5.0,
            seed=seed,
        )["block"]

    state = torch.get_rng_state().clone()
    first = sample(17)
    torch.testing.assert_close(first, sample(17))
    assert not torch.equal(first, sample(18))
    assert torch.equal(state, torch.get_rng_state())


def test_grasp_scene_seed_does_not_change_numpy_global_stream() -> None:
    from examples.sim.demo.grasp_cup_to_caffe import apply_random_xy_perturbation

    class Object:
        def get_local_pose(self, *, to_matrix):
            return torch.eye(4).unsqueeze(0)

        def set_local_pose(self, pose):
            self.pose = pose

    state = np.random.get_state()
    first, second, third = Object(), Object(), Object()
    apply_random_xy_perturbation(first, rng=np.random.default_rng(17))
    apply_random_xy_perturbation(second, rng=np.random.default_rng(17))
    apply_random_xy_perturbation(third, rng=np.random.default_rng(18))
    torch.testing.assert_close(first.pose, second.pose)
    assert not torch.equal(first.pose, third.pose)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
