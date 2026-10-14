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


@pytest.fixture(autouse=True)
def task_mesh_cache(tmp_path, monkeypatch):
    """Resolve task meshes locally so configuration tests never download them."""
    from types import SimpleNamespace
    from embodichain.data import dataset

    directory = tmp_path / "PourCupAssets"
    directory.mkdir()
    for name in (
        "cup.glb",
        "mug_wall.obj",
        "receiver_wall.obj",
        "fluid_table.obj",
        "fluid_floor.obj",
    ):
        (directory / name).touch()
    original = dataset.get_data_class

    def resolve(name, extra_modules=None):
        if name == "PourCupAssets":
            return lambda: SimpleNamespace(extract_dir=str(directory))
        return original(name, extra_modules)

    monkeypatch.setattr(dataset, "get_data_class", resolve)
    return directory


@pytest.mark.parametrize("renderer", [None, "fast-rt"])
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
    assert replay.call_args.args[0].renderer == "fast-rt"


@pytest.mark.parametrize("spacing", [None, 0.003, 0.006])
def test_final_episode_is_committed_through_record_camera(
    spacing, monkeypatch, tmp_path
):
    from embodichain.lab.gym.envs import fluid_replay
    from embodichain.lab.gym.envs.managers import observations

    recorder = SimpleNamespace(save_and_clear=Mock())
    created = []
    decode_cfg = runner.config_to_cfg

    def configured_render_settings(*args, **kwargs):
        cfg = decode_cfg(*args, **kwargs)
        cfg.sim_cfg.render_cfg.spp = 8
        return cfg

    monkeypatch.setattr(runner, "config_to_cfg", configured_render_settings)

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
        particle_spacing=spacing,
        speed=2,
        seconds=None,
        partial_reset_step=None,
    )
    runner._replay(args, tmp_path / "source.pt", tmp_path)
    recorder.save_and_clear.assert_called_once_with()
    cfg = created[0]
    assert cfg.sim_cfg.render_cfg.spp == 8
    assert cfg.events.record_camera.params["max_env_num"] == 3
    assert cfg.events.record_camera.params["save_path"] == str(tmp_path / "videos")
    result = json.loads((tmp_path / "fluid_gym_result.json").read_text())
    assert result["renderer"] == "fast-rt"
    assert result["fluid_source"]["kind"] == "revolved_volume"
    assert result["fluid_source"]["file_path"] is None
    expected_spacing = spacing or 0.0015
    assert result["particle_spacing_m"] == expected_spacing
    fluid = cfg.fluid_object[0]
    scale = expected_spacing / 0.0015
    assert fluid.render.render_radius == pytest.approx(0.00045 * scale)
    assert fluid.render.mesh_surface["cell_size"] == pytest.approx(0.00081 * scale)
    assert fluid.render.mesh_surface["kernel_radius"] == pytest.approx(0.00225 * scale)
    assert cfg.sim_cfg.physics_cfg.solver_cfg[
        "particle_support_radius"
    ] == pytest.approx(2.25 * expected_spacing)
    assert result["video_fps"] == 24
    assert result["videos"] == [
        f"videos/episode_0_fluid_gym_env{i}.mp4" for i in range(3)
    ]


def test_spacing_override_rejects_fixed_points_before_scene_creation(
    monkeypatch, tmp_path
):
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.sim.cfg import FluidSpawnCfg
    from embodichain.utils.utility import load_config

    cfg = config_to_cfg(load_config(TASK), source_path=TASK)
    cfg.fluid_object[0].spawn = FluidSpawnCfg(positions=[[0, 0, 0]])
    monkeypatch.setattr(runner, "config_to_cfg", lambda *args, **kwargs: cfg)
    args = argparse.Namespace(
        replay_task=str(TASK),
        num_envs=2,
        renderer="fast-rt",
        particle_spacing=0.006,
        speed=2,
    )
    with pytest.raises(ValueError, match="fixed sampling"):
        runner._replay(args, tmp_path / "unused.pt", tmp_path)


def test_fluid_launcher_rejects_invisible_hybrid_backend(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fluid", "--renderer", "hybrid"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
