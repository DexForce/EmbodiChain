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

"""Exercise the packaged Tool Hang scene, including a physical support contact."""

from __future__ import annotations

from pathlib import Path
import math
import subprocess
import sys

import pytest
import torch
import yaml


def _exercise_hanging_scene() -> None:
    import gymnasium as gym
    import embodichain.extensions.tasks
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.sim import SimulationManager

    path = (
        Path(__file__).resolve().parents[4]
        / "embodichain/extensions/tasks/configs/tasks/manipulation/assembly/tool_hang/task.panda.yaml"
    )
    declaration = yaml.safe_load(path.read_text())
    cfg = config_to_cfg(declaration, source_path=path)
    env = gym.make(declaration["id"], cfg=cfg).unwrapped
    try:
        env.reset(seed=7)
        frame = env.sim.get_articulation("frame")
        initial = frame.get_local_pose(to_matrix=True)[0]
        # The frame lies flat. Reset yaw varies its heading between 140 and
        # 160 degrees without tipping the grip off the work surface.
        assert initial[2, 1].item() == pytest.approx(-1, abs=1e-5)
        heading = math.degrees(math.atan2(initial[1, 0].item(), initial[0, 0].item()))
        assert 140 <= heading <= 160
        zero = torch.zeros((1, 7), device=env.device)
        obs, reward, terminated, truncated, info = env.step(zero)
        assert reward.item() == 0 and not terminated.item() and not truncated.item()
        assert not info["success"].item()
        for name in ("stand", "frame", "tool"):
            assert obs[name].shape == (1, 7)

        # Work in arena coordinates. Move the hand out of the fixture, set an
        # upright rod on the socket axis and rotate the ring axis along the bar.
        joints = env.robot.get_qpos().clone()
        joints[:, 0] = 1.2
        env.robot.set_qpos(joints, target=False)
        env.robot.set_qpos(joints)
        stand = env.sim.get_articulation("stand")
        tool = env.sim.get_articulation("tool")
        stand_xyz = stand.get_local_pose()[0, :3]
        root = stand_xyz + stand_xyz.new_tensor([-0.04375, 0.045, 0.06545])
        frame_pose = torch.cat((root, root.new_tensor([0.0, 0.0, 0.0, 1.0])))[None]
        half = math.sqrt(0.5)
        tool_pose = torch.cat(
            (
                root + root.new_tensor([-0.001875, 0, 0.08625 - 0.0935328]),
                root.new_tensor([0, half, 0, half]),
            )
        )[None]

        def measure(steps: int) -> tuple[bool, dict]:
            env.sim.update(step=steps)
            env.termination_manager.compute()
            accepted, _, metrics = env.compute_task_state()
            return accepted.item(), {
                key.removeprefix("success.hanging."): value.item()
                for key, value in metrics.items()
            }

        frame.set_local_pose(frame_pose)
        frame.set_root_velocity(torch.zeros((1, 6), device=env.device))
        tool.set_local_pose(tool_pose)
        tool.set_root_velocity(torch.zeros((1, 6), device=env.device))
        accepted, metrics = measure(1)
        assert metrics["frame_assembled"] and metrics["hole_threaded"], metrics
        assert not metrics["ring_bar_contact"] and not accepted, metrics

        # Gravity brings the lower ring edge onto the horizontal bar.
        for settling_steps in (80, 20, 20):
            accepted, metrics = measure(settling_steps)
            assert (
                metrics["ring_bar_contact"] and not metrics["hand_tool_contact"]
            ), metrics
            assert accepted, metrics

        # Save the supported pose before the normal Gym transition auto-resets.
        frame_pose = frame.get_local_pose().clone()
        tool_pose = tool.get_local_pose().clone()
        _, reward, terminated, truncated, info = env.step(zero)
        assert info["success"].item() and reward.item() == 1
        assert terminated.item() and not truncated.item()

        # Both objects keep their relative pose, but the frame misses the socket.
        frame_pose[:, 0] += 0.03
        tool_pose[:, 0] += 0.03
        frame.set_local_pose(frame_pose)
        tool.set_local_pose(tool_pose)
        frame.set_root_velocity(torch.zeros((1, 6), device=env.device))
        tool.set_root_velocity(torch.zeros((1, 6), device=env.device))
        accepted, metrics = measure(1)
        assert not metrics["frame_assembled"] and not accepted, metrics

        # Touch the wrench handle with just one finger. A two-pad grasp detector
        # would miss this contact, which independently rules out a released tool.
        finger = env.robot.get_link_pose("fr3_leftfinger", to_matrix=True)
        contact = finger @ finger.new_tensor([0, 0.015, 0.022, 1])
        tool_pose[:, :3] = contact[:, :3]
        tool.set_local_pose(tool_pose)
        tool.set_root_velocity(torch.zeros((1, 6), device=env.device))
        accepted, metrics = measure(1)
        assert metrics["hand_tool_contact"] and not accepted, metrics
    finally:
        env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.gpu
@pytest.mark.subprocess_sim
@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Newton MuJoCo-Warp needs CUDA"
)
def test_native_hanging_requires_both_stages_and_a_released_supported_ring() -> None:
    """Use a fresh process so the deployment owns its NoRender world."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _exercise_hanging_scene()
