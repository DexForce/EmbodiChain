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

"""Coverage for train-rl EnvProfiler CLI flags."""

from __future__ import annotations

import random
import json
from unittest.mock import Mock
import importlib

import numpy as np
import pytest
import torch

from embodichain.learning.rl.train import (
    _event_params,
    _resolve_profile_output,
    _seed_training_rng,
    parse_args,
    train_from_config,
)


def test_parse_args_profile_flags():
    args = parse_args(
        [
            "--config",
            "dummy.yaml",
            "--profile",
            "--profile_output",
            "outputs/rl_profile.json",
        ]
    )
    assert args.profile is True
    assert args.profile_output == "outputs/rl_profile.json"

    defaults = parse_args(["--config", "dummy.yaml"])
    assert defaults.profile is False
    assert defaults.profile_output is None


def test_resolve_profile_output_disambiguates_ranks():
    assert _resolve_profile_output("out/prof.json", rank=0, world_size=1) == (
        "out/prof.json"
    )
    assert _resolve_profile_output("out/prof.json", rank=1, world_size=2) == (
        "out/prof_rank1.json"
    )


def test_seed_training_rng_replays_all_policy_rngs() -> None:
    _seed_training_rng(73, torch.device("cpu"))
    first = (random.random(), float(np.random.random()), torch.rand(3))

    _seed_training_rng(73, torch.device("cpu"))
    second = (random.random(), float(np.random.random()), torch.rand(3))

    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])


def test_learning_env_rejects_profile(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "trainer:\n  learning_env:\n    name: PointMassRL\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="learning_env"):
        train_from_config(str(config_path), profile=True)

    with pytest.raises(ValueError, match="--profile_output requires --profile"):
        train_from_config(str(config_path), profile_output="prof.json")


@pytest.mark.parametrize("renderer", [None, "no-render", "hybrid"])
def test_training_passes_only_explicit_renderer_override(
    tmp_path, monkeypatch, renderer
):
    module = importlib.import_module("embodichain.learning.rl.train")
    trainer = {"gym_config": "task.yaml", "device": "cpu"}
    if renderer is not None:
        trainer["renderer"] = renderer
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps({"trainer": trainer, "policy": {}, "algorithm": {}})
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "SummaryWriter", Mock())
    build_runtime = Mock(side_effect=RuntimeError("stop at environment construction"))
    monkeypatch.setattr(module, "build_gym_policy_runtime", build_runtime)

    with pytest.raises(RuntimeError, match="stop at environment construction"):
        train_from_config(str(config_path))

    assert build_runtime.call_args.kwargs["renderer"] == renderer


@pytest.mark.requires_tasks
def test_training_resolves_packaged_gym_components_from_outside_checkout(
    tmp_path, monkeypatch
) -> None:
    from embodichain.data import constants
    from embodichain.lab.sim.cfg.physics import NewtonPhysicsCfg
    from embodichain.utils.utility import load_config

    module = importlib.import_module("embodichain.learning.rl.train")
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    embodiment = load_config(
        "embodichain/extensions/tasks/configs/components/embodiments/"
        "flexiv_rizon4s_robotiq2f85.nominal.yaml"
    )
    urdf = data_root / embodiment["simulation"]["fpath"]
    urdf.parent.mkdir(parents=True)
    urdf.write_text('<robot name="test"><link name="base"/></robot>', encoding="utf-8")
    monkeypatch.setattr(constants, "EMBODICHAIN_DEFAULT_DATA_ROOT", str(data_root))

    recipe = load_config(
        "embodichain/extensions/tasks/configs/tasks/manipulation/assembly/"
        "gear_meshing/agents/ppo.study.nominal.dexuni.yaml"
    )
    recipe["trainer"].update(device="cpu", num_envs=1)
    config_path = tmp_path / "agent.json"
    config_path.write_text(json.dumps(recipe), encoding="utf-8")
    monkeypatch.setattr(module, "SummaryWriter", Mock())

    def stop_at_simulator(env_id, *, base_env_cfg):
        assert env_id == "BlindGearMeshingRL-StudyNominal-DexUni-v1"
        physics = base_env_cfg.sim_cfg.physics_cfg
        assert isinstance(physics, NewtonPhysicsCfg)
        assert physics.solver_cfg["solver_type"] == "dexuni"
        raise RuntimeError("resolved deployment reached the simulator boundary")

    monkeypatch.setattr("embodichain.learning.rl.runtime.build_env", stop_at_simulator)
    with pytest.raises(
        RuntimeError, match="resolved deployment reached the simulator boundary"
    ):
        train_from_config(str(config_path))


@pytest.mark.parametrize(
    "absolute_reference", [False, True], ids=["relative", "absolute"]
)
def test_training_preserves_config_symlink_suffixes_and_components(
    tmp_path, monkeypatch, absolute_reference: bool
) -> None:
    from embodichain.learning.rl.train import _freeze_gym_config

    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    gym_config = {
        "id": "SymlinkTask-v0",
        "environment": {"component": "env.yaml"},
        "robot": {"uid": "robot"},
    }
    gym_target = shared / "gym-data"
    gym_target.write_text(json.dumps(gym_config), encoding="utf-8")
    gym_link = local / "gym.yaml"
    gym_link.symlink_to(gym_target)
    for directory, steps in ((local, 23), (shared, 127)):
        (directory / "env.yaml").write_text(
            json.dumps(
                {
                    "environment_id": "symlink_env",
                    "physics": "default",
                    "max_episode_steps": steps,
                    "simulation": {},
                    "env": {},
                }
            ),
            encoding="utf-8",
        )

    agent_target = tmp_path / "agent-data"
    gym_ref = gym_link if absolute_reference else gym_link.relative_to(tmp_path)
    agent_target.write_text(
        json.dumps(
            {
                "trainer": {"gym_config": str(gym_ref), "device": "cpu"},
                "policy": {},
                "algorithm": {},
            }
        ),
        encoding="utf-8",
    )
    agent_link = tmp_path / "agent.yaml"
    agent_link.symlink_to(agent_target)
    monkeypatch.chdir(local)
    monkeypatch.setattr("embodichain.learning.rl.train.SummaryWriter", Mock())

    def stop_at_simulator(env_id, *, base_env_cfg):
        assert env_id == "SymlinkTask-v0"
        assert base_env_cfg.max_episode_steps == 23
        raise RuntimeError("symlink deployment reached the simulator boundary")

    monkeypatch.setattr("embodichain.learning.rl.runtime.build_env", stop_at_simulator)
    with pytest.raises(
        RuntimeError, match="symlink deployment reached the simulator boundary"
    ):
        train_from_config(str(agent_link))

    assert _freeze_gym_config(gym_config, gym_link)["max_episode_steps"] == 23


def test_camera_recording_defaults_to_the_run_directory(tmp_path):
    params = _event_params(
        {"func": "record_camera_data_async", "params": {"name": "main"}},
        run_base=tmp_path / "run",
        phase="eval",
    )

    assert params["save_path"] == str(tmp_path / "run" / "videos" / "eval")


@pytest.mark.parametrize(
    "phase,enabled,expected",
    [("train", False, 1), ("eval", True, 1), ("eval", False, 0)],
)
def test_active_recording_events_are_resolved_before_first_environment(
    tmp_path, monkeypatch, phase, enabled, expected
) -> None:
    from embodichain.lab.gym.envs.managers.record import record_camera_data_async

    module = importlib.import_module("embodichain.learning.rl.train")
    trainer = {
        "gym_config": "task.yaml",
        "device": "cpu",
        "enable_eval": enabled,
        "events": {phase: {"record": {"func": "record_camera_data_async"}}},
    }
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps({"trainer": trainer, "policy": {}, "algorithm": {}})
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "SummaryWriter", Mock())
    build = Mock(side_effect=RuntimeError("environment boundary"))
    monkeypatch.setattr(module, "build_gym_policy_runtime", build)
    with pytest.raises(RuntimeError, match="environment boundary"):
        train_from_config(str(config_path))
    events = build.call_args.kwargs["trainer_events"]
    assert len(events) == expected
    if expected:
        assert events[0].func is record_camera_data_async
