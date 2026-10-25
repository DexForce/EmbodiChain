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
from types import SimpleNamespace
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
    with pytest.raises(ValueError, match="on_trainer_ready requires"):
        train_from_config(str(config_path), on_trainer_ready=lambda trainer: None)


@pytest.mark.parametrize(
    "observer_present,observer_fails", [(True, False), (True, True), (False, False)]
)
def test_native_instrumentation_runs_after_resume_with_normal_cleanup(
    tmp_path, monkeypatch, observer_present: bool, observer_fails: bool
) -> None:
    from embodichain.learning.rl.algo import RolloutKind

    module = importlib.import_module("embodichain.learning.rl.train")
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "trainer": {
                    "gym_config": "task.yaml",
                    "device": "cpu",
                    "iterations": 1,
                    "buffer_size": 2,
                    "use_wandb": False,
                },
                "policy": {"name": "proprioceptive_actor_critic"},
                "algorithm": {"name": "ppo", "cfg": {}},
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    writer = Mock()
    monkeypatch.setattr(module, "SummaryWriter", Mock(return_value=writer))
    env = SimpleNamespace(num_envs=2)
    runtime = SimpleNamespace(
        env=env,
        policy=Mock(),
        gym_config_path="task.yaml",
        gym_config={"id": "TestRL"},
        env_cfg=SimpleNamespace(
            num_envs=2,
            sim_cfg=SimpleNamespace(
                headless=True,
                device="cpu",
                render_cfg=SimpleNamespace(renderer="no-render"),
            ),
        ),
        close=Mock(),
    )
    monkeypatch.setattr(module, "build_gym_policy_runtime", Mock(return_value=runtime))
    monkeypatch.setattr(
        module,
        "build_algo",
        Mock(
            return_value=SimpleNamespace(
                rollout_kind=RolloutKind.STANDARD, cfg=SimpleNamespace(batch_size=4)
            )
        ),
    )
    order = []
    trainer = Mock(env=env)
    trainer.train.side_effect = lambda total: order.append("collect")
    monkeypatch.setattr(module, "Trainer", Mock(return_value=trainer))
    monkeypatch.setattr(module, "_configure_checkpoints", Mock())
    monkeypatch.setattr(
        module, "_resume_training", lambda *args: order.append("restore")
    )
    monkeypatch.setattr(module, "_save_final_checkpoint", Mock())
    synchronize_errors = Mock(wraps=module._sync_errors)
    monkeypatch.setattr(module, "_sync_errors", synchronize_errors)

    def observe(actual) -> None:
        assert actual is trainer
        order.append("observe")
        if observer_fails:
            raise RuntimeError("diagnostic capacity unavailable")

    if observer_fails:
        with pytest.raises(RuntimeError, match="diagnostic capacity unavailable"):
            train_from_config(str(config), on_trainer_ready=observe)
        assert order == ["restore", "observe"]
        trainer.train.assert_not_called()
    else:
        train_from_config(
            str(config), on_trainer_ready=observe if observer_present else None
        )
        assert order == (
            ["restore", "observe", "collect"]
            if observer_present
            else ["restore", "collect"]
        )
        trainer.train.assert_called_once_with(4)
    # Ranks without a callback must still join the same error collective.
    synchronize_errors.assert_called_once()
    runtime.close.assert_called_once()
    writer.close.assert_called_once()


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
