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

"""Real-renderer check of LeWM goal-image state restoration."""

from __future__ import annotations

import os

import numpy as np
import pytest

from embodichain.lab.sim.scene import SceneView, capture_state
from examples.learning.lewm._sim import close_env, frame, goal_image, make_env, reset


@pytest.mark.gpu
@pytest.mark.requires_sim
def test_goal_image_restores_scene_without_advancing_time() -> None:
    env, _ = make_env(os.environ.get("LEWM_TEST_DEVICE", "cuda:0"))
    try:
        obs, _ = reset(env, 200_000)
        raw = env.unwrapped
        before_image = frame(obs)
        before = capture_state(SceneView(raw.sim))
        before_step = raw.elapsed_steps.clone()
        goal = goal_image(env)
        after = capture_state(SceneView(raw.sim))
        assert goal.shape == before_image.shape
        assert np.mean(np.abs(goal.astype(np.float32) - before_image)) > 0.1
        assert raw.elapsed_steps.equal(before_step)
        for left, right in zip(before.batches[0], after.batches[0]):
            assert left.uid == right.uid
            for name in (
                "pose",
                "velocity",
                "qpos",
                "qvel",
                "target_qpos",
                "target_qvel",
                "qf",
            ):
                np.testing.assert_allclose(
                    getattr(left, name), getattr(right, name), rtol=0, atol=1e-4
                )
    finally:
        close_env(env)
