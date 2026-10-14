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

"""Fluid replay launcher renderer and final recording commit contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys
import pytest
import torch

from embodichain.lab.scripts import run_fluid_task_program as runner

pytestmark = pytest.mark.no_sim
TASK = (
    Path(__file__).resolve().parents[3]
    / "embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.fluid.yaml"
)


@pytest.mark.parametrize("renderer", [None, "hybrid", "fast-rt"])
def test_launcher_selects_supported_renderers(renderer, monkeypatch, tmp_path):
    args = [
        "fluid",
        "--trajectory",
        str(tmp_path / "source.pt"),
        "--output",
        str(tmp_path),
    ]
    if renderer is not None:
        args += ["--renderer", renderer]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner, "discover_task_packages", lambda: None)
    replay = Mock()
    monkeypatch.setattr(runner, "_replay", replay)
    runner.main()
    assert replay.call_args.args[0].renderer == (renderer or "hybrid")


def test_final_episode_is_committed_through_record_camera(monkeypatch, tmp_path):
    from embodichain.lab.gym.envs import fluid_replay
    from embodichain.lab.gym.envs.managers import observations

    recorder = SimpleNamespace(save_and_clear=Mock())
    created = []

    class Environment:
        def __init__(self, cfg, **kwargs):
            created.append(cfg)
            self.num_envs = cfg.num_envs
            self.device = torch.device("cpu")
            self.step_dt = 1 / 24
            self.control_frequency = 24
            self.replay_duration = self.step_dt
            self.active_joint_ids = [0]
            self.event_manager = SimpleNamespace(get_functor=lambda name: recorder)
            self.sim = SimpleNamespace(
                get_fluid_object=lambda uid: SimpleNamespace(num_particles=8)
            )
            self.source_qualification = {"accepted": True}
            self.close = Mock()

        def reset(self, **kwargs):
            return {}, {}

        def step(self, action):
            info = {
                "fluid_replay_time_s": torch.full((self.num_envs,), self.step_dt),
                "fluid_replay_finished": torch.ones(self.num_envs, dtype=torch.bool),
            }
            return {}, None, None, None, info

    monkeypatch.setattr(fluid_replay, "FluidReplayEnv", Environment)
    monkeypatch.setattr(
        observations,
        "get_fluid_summary",
        lambda env, *args: torch.ones(env.num_envs, 5),
    )
    monkeypatch.setattr(
        observations,
        "get_fluid_volume_in_region",
        lambda env, *args, **kwargs: torch.zeros(env.num_envs, 1),
    )
    args = argparse.Namespace(
        replay_task=str(TASK),
        num_envs=3,
        renderer="fast-rt",
        particle_spacing=None,
        speed=2,
        seconds=None,
        partial_reset_step=None,
    )
    runner._replay(args, tmp_path / "source.pt", tmp_path)
    recorder.save_and_clear.assert_called_once_with()
    cfg = created[0]
    assert cfg.events.record_camera.params["max_env_num"] == 3
    assert cfg.events.record_camera.params["save_path"] == str(tmp_path / "videos")
    result = json.loads((tmp_path / "fluid_gym_result.json").read_text())
    assert result["renderer"] == "fast-rt"
    assert result["video_fps"] == 24
    assert result["videos"] == [
        f"videos/episode_0_fluid_gym_env{i}.mp4" for i in range(3)
    ]
