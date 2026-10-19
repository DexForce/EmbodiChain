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

import numpy as np
import pytest
import torch

from embodichain.lab.sim.utility import atom_action_utils
from embodichain.lab.sim.motion.planners.toppra_planner import retime_joint_paths
from embodichain.lab.sim.motion.planners.utils import TrajectorySampleMethod


@pytest.mark.parametrize("sample_count", [20, 37])
def test_legacy_plan_trajectory_keeps_requested_sample_count(monkeypatch, sample_count):
    """Count-based expert actions must not interpret their count as seconds."""

    class RetimeMotionGenerator:
        def __init__(self, cfg):
            self.cfg = cfg

        def generate(self, *, target_states, options):
            plan = options.plan_opts
            assert plan.sample_method is TrajectorySampleMethod.QUANTITY
            # This standalone fake has no robot from which to resolve caps.
            limits = {
                name: (
                    fallback
                    if plan.constraints[name] is None
                    else plan.constraints[name]
                )
                for name, fallback in (("velocity", 0.2), ("acceleration", 0.5))
            }
            solved = retime_joint_paths(
                torch.stack([state.qpos.reshape(-1) for state in target_states])[None],
                constraints=limits,
                sample_method=plan.sample_method,
                sample_interval=plan.sample_interval,
            )
            assert solved.is_all_success()
            return solved

    monkeypatch.setattr(atom_action_utils, "MotionGenerator", RetimeMotionGenerator)
    env = SimpleNamespace(
        robot=SimpleNamespace(uid="robot"), sim=SimpleNamespace(instance_id="unused")
    )
    poses, hands = [], []
    atom_action_utils.plan_trajectory(
        env, "right_arm", [[0.0] * 7, [0.2] * 7], sample_count, [0.0, 0.0], poses, hands
    )
    assert np.asarray(poses).shape == (sample_count, 7)
    np.testing.assert_allclose(poses[0], 0.0)
    np.testing.assert_allclose(poses[-1], 0.2)
    assert hands == [[0.0, 0.0]] * sample_count
