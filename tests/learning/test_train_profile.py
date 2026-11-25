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
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

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


def test_learning_entry_imports_do_not_load_simulator() -> None:
    script = """
import importlib.abc
import sys

class RejectSimulation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"embodichain.lab.gym", "embodichain.lab.sim"}:
            raise AssertionError(f"Unexpected simulator import: {fullname}")

sys.meta_path.insert(0, RejectSimulation())
from embodichain.learning.rl.runtime import build_learning_policy_runtime
from embodichain.learning.rl.train import parse_args
assert parse_args(["--config", "learning.yaml"]).config == "learning.yaml"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
    module.SummaryWriter.return_value.close.assert_called_once_with()


def test_camera_recording_defaults_to_the_run_directory(tmp_path):
    params = _event_params(
        {"func": "record_camera_data_async", "params": {"name": "main"}},
        run_base=tmp_path / "run",
        phase="eval",
    )

    assert params["save_path"] == str(tmp_path / "run" / "videos" / "eval")


@pytest.mark.parametrize(
    "phases,enabled,active_phases",
    [
        (("train",), False, ("train",)),
        (("eval",), True, ("eval",)),
        (("eval",), False, ()),
        (("train", "eval"), True, ("train", "eval")),
    ],
)
def test_active_recording_events_are_resolved_before_first_environment(
    tmp_path, monkeypatch, phases, enabled, active_phases
) -> None:
    from embodichain.lab.gym.envs.managers.record import record_camera_data_async

    module = importlib.import_module("embodichain.learning.rl.train")
    trainer = {
        "gym_config": "task.yaml",
        "device": "cpu",
        "enable_eval": enabled,
        "events": {
            phase: {"record": {"func": "record_camera_data_async"}} for phase in phases
        },
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
    assert len(events) == len(active_phases)
    for event, phase in zip(events, active_phases):
        assert event.func is record_camera_data_async
        assert Path(event.params["save_path"]).parts[-2:] == ("videos", phase)


@pytest.mark.parametrize("failure_stage", ["evaluation", "algorithm", "trainer"])
def test_learning_setup_failure_releases_acquired_resources(
    tmp_path, monkeypatch, failure_stage
):
    module = importlib.import_module("embodichain.learning.rl.train")
    config = {
        "trainer": {"learning_env": "example", "enable_eval": True, "use_wandb": True},
        "policy": {},
        "algorithm": {"name": "ppo"},
    }
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)
    runtime = SimpleNamespace(
        env=object(), policy=object(), env_id="example", close=Mock()
    )
    evaluation = SimpleNamespace(close=Mock())
    build_eval = Mock(return_value=("example", evaluation))
    build_algorithm = Mock(
        return_value=SimpleNamespace(cfg=SimpleNamespace(batch_size=4))
    )
    build_trainer = Mock(side_effect=ValueError("setup failed"))
    monkeypatch.setattr(
        module, "build_learning_policy_runtime", Mock(return_value=runtime)
    )
    monkeypatch.setattr(module, "_build_learning_environment", build_eval)
    monkeypatch.setattr(module, "build_algo", build_algorithm)
    monkeypatch.setattr(module, "get_trainer_class", lambda _: build_trainer)
    monkeypatch.setattr(module, "Trainer", build_trainer)
    writer = Mock()
    # A failing teardown must neither hide the setup error nor skip the environments.
    writer.close.side_effect = RuntimeError("writer cleanup failed")
    monkeypatch.setattr(module, "SummaryWriter", Mock(return_value=writer))
    tracking = SimpleNamespace(run=None, init=Mock(return_value=Mock()))
    monkeypatch.setattr(module, "wandb", tracking)
    warn = Mock()
    monkeypatch.setattr(module.logger, "log_warning", warn)
    if failure_stage == "evaluation":
        build_eval.side_effect = ValueError("setup failed")
    elif failure_stage == "algorithm":
        build_algorithm.side_effect = ValueError("setup failed")

    with pytest.raises(ValueError, match="setup failed"):
        train_from_config(str(config_path))

    runtime.close.assert_called_once_with()
    assert evaluation.close.call_count == (failure_stage != "evaluation")
    assert writer.close.call_count == (failure_stage == "trainer")
    assert tracking.init.return_value.finish.call_count == (failure_stage == "trainer")
    if failure_stage == "trainer":
        assert "writer cleanup failed" in warn.call_args.args[0]


@pytest.mark.parametrize("borrowed", [False, True])
def test_training_only_finishes_its_own_tracking_run(tmp_path, monkeypatch, borrowed):
    module = importlib.import_module("embodichain.learning.rl.train")
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "trainer": {"gym_config": "task.yaml", "use_wandb": True},
                "policy": {},
                "algorithm": {},
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    tracking_run = Mock()
    tracking = SimpleNamespace(
        run=tracking_run if borrowed else None, init=Mock(return_value=tracking_run)
    )
    monkeypatch.setattr(module, "wandb", tracking)
    monkeypatch.setattr(module, "SummaryWriter", Mock())
    monkeypatch.setattr(
        module, "build_gym_policy_runtime", Mock(side_effect=ValueError("setup failed"))
    )

    with pytest.raises(ValueError, match="setup failed"):
        train_from_config(str(config_path))

    assert tracking_run.finish.call_count == (not borrowed)
    assert tracking.init.call_count == (not borrowed)
    module.SummaryWriter.return_value.close.assert_called_once_with()


@pytest.mark.parametrize("borrowed", [False, True])
def test_training_only_destroys_its_own_process_group(tmp_path, monkeypatch, borrowed):
    module = importlib.import_module("embodichain.learning.rl.train")
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "trainer": {"gym_config": "task.yaml", "distributed": True},
                "policy": {},
                "algorithm": {},
            }
        )
    )
    monkeypatch.setenv("LOCAL_RANK", "0")
    for name, value in (
        ("is_available", True),
        ("device_count", 1),
        ("current_device", 0),
    ):
        monkeypatch.setattr(torch.cuda, name, Mock(return_value=value))
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    for name, value in (
        ("is_available", True),
        ("is_initialized", borrowed),
        ("get_rank", 0),
        ("get_world_size", 1),
    ):
        monkeypatch.setattr(torch.distributed, name, Mock(return_value=value))
    initialize, destroy = Mock(), Mock()
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)
    monkeypatch.setattr(
        module, "_seed_training_rng", Mock(side_effect=ValueError("setup failed"))
    )

    with pytest.raises(ValueError, match="setup failed"):
        train_from_config(str(config_path))

    assert initialize.call_count == (not borrowed)
    assert destroy.call_count == (not borrowed)
