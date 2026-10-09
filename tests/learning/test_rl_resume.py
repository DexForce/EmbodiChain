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

import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from embodichain.learning.rl.algo import build_algo
from embodichain.learning.rl.checkpoint import _atomic_write
from embodichain.learning.rl.models import ActorCritic
from embodichain.learning.rl.policy_evaluation.manifest import RunManifest
from embodichain.learning.rl.train import parse_args, train_from_config
from embodichain.learning.rl.utils.trainer import Trainer
from embodichain_tasks.classic_control.point_mass import PointMassEnv


def _trainer(
    directory: Path, algorithm: str = "ppo", scheduler: dict | None = None, **cfg
) -> Trainer:
    env = PointMassEnv(num_envs=4, max_episode_steps=4)
    policy = ActorCritic(
        14,
        2,
        torch.device("cpu"),
        torch.nn.Linear(14, 2),
        torch.nn.Linear(14, 1),
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algo_cfg = dict(
        n_epochs=1, batch_size=8, optimizer=dict(name="adam", learning_rate=1e-3), **cfg
    )
    if scheduler is not None:
        algo_cfg["lr_scheduler"] = scheduler
    algo = build_algo(algorithm, algo_cfg, policy, torch.device("cpu"))
    return Trainer(
        policy,
        env,
        algo,
        buffer_size=4,
        batch_size=8,
        writer=None,
        eval_freq=32,
        save_freq=32,
        checkpoint_dir=str(directory),
        exp_name="resume",
        use_wandb=False,
    )


def _assert_state_equal(first, second) -> None:
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_state_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _assert_state_equal(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("algorithm", ["ppo", "grpo"])
@pytest.mark.parametrize(
    "scheduler",
    [
        None,
        dict(name="linear", kwargs=dict(total_iters=4)),
        dict(name="cosine", kwargs=dict(T_max=4)),
        dict(name="step", kwargs=dict(step_size=2, gamma=0.5)),
        dict(name="exponential", kwargs=dict(gamma=0.9)),
    ],
)
def test_restored_next_update_matches_fixed_rollout(tmp_path, algorithm, scheduler):
    torch.manual_seed(7)
    trainer = _trainer(tmp_path / "first", algorithm, scheduler)
    trainer.train(16)
    rollout = trainer.collector.collect(4, trainer.buffer.start_rollout()).clone()
    path = trainer.save_checkpoint()
    rng_expected = (random.random(), np.random.rand(), torch.rand(3))
    trainer.load_checkpoint(path)
    expected = trainer.algorithm.update(rollout)
    policy_expected = deepcopy(trainer.policy.state_dict())
    optimizer_expected = deepcopy(trainer.algorithm.state_dict())

    restored = _trainer(tmp_path / "second", algorithm, scheduler)
    restored.load_checkpoint(path)
    rng_actual = (random.random(), np.random.rand(), torch.rand(3))
    _assert_state_equal(rng_expected, rng_actual)
    # The baseline update starts at the checkpoint RNG, before the draws above.
    restored.load_checkpoint(path)
    actual = restored.algorithm.update(rollout)
    _assert_state_equal(expected, actual)
    _assert_state_equal(policy_expected, restored.policy.state_dict())
    _assert_state_equal(optimizer_expected, restored.algorithm.state_dict())
    assert restored.global_step == 16
    assert restored.num_updates == 1
    assert restored._next_eval_step == 32
    assert restored._next_save_step == 32


def test_resume_throughput_counts_only_new_steps(tmp_path, monkeypatch):
    import embodichain.learning.rl.utils.trainer as training

    trainer = _trainer(tmp_path / "first")
    trainer.train(16)
    path = trainer.save_checkpoint()
    restored = _trainer(tmp_path / "restored")
    clock = [200.0]
    monkeypatch.setattr(training.time, "time", lambda: clock[0])
    restored.load_checkpoint(path)
    assert restored.get_summary()["training_fps"] == 0
    update = restored.algorithm.update

    def timed_update(rollout):
        result = update(rollout)
        clock[0] += 2.0
        return result

    monkeypatch.setattr(restored.algorithm, "update", timed_update)
    summary = restored.train(48)
    assert summary["global_step"] == 48
    assert summary["num_updates"] == 3
    assert summary["training_fps"] == 8
    assert summary["last_train_metrics"]["charts/SPS"] == 8


def test_resume_trains_only_remaining_budget_and_restarts_schedule(tmp_path):
    trainer = _trainer(tmp_path / "first", scheduler=dict(name="linear"))
    trainer.train(16)
    path = trainer.save_checkpoint()
    original = deepcopy(trainer.algorithm.optimizer.state_dict()["state"])
    restored = _trainer(tmp_path / "second", scheduler=dict(name="linear"))
    restored.load_checkpoint(path)
    with pytest.raises(ValueError, match="resume-scheduler restart"):
        restored.train(48)
    restored.load_checkpoint(path, resume_scheduler="restart")
    assert restored.algorithm.current_learning_rate() == 1e-3
    _assert_state_equal(original, restored.algorithm.optimizer.state_dict()["state"])
    restored.train(48)
    assert restored.num_updates == 3
    assert restored.global_step == 48
    assert restored.algorithm._lr_scheduler_cfg.kwargs["total_iters"] == 2
    restarted = restored.save_checkpoint()
    again = _trainer(tmp_path / "third", scheduler=dict(name="linear"))
    again.load_checkpoint(restarted)
    _assert_state_equal(restored.algorithm.state_dict(), again.algorithm.state_dict())
    again.train(48)
    assert again.num_updates == 3


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c: c.update(schema_version=999), "schema version"),
        (lambda c: c.pop("rng_by_rank"), "required fields"),
        (lambda c: c["metadata"].update(environment="different"), "environment"),
        (lambda c: c.update(world_size=2), "world_size"),
        (
            lambda c: c["metadata"]["policy_attributes"].update(
                actor_obs_groups=["different"]
            ),
            "actor_obs_groups",
        ),
    ],
)
def test_invalid_checkpoint_fails_before_policy_changes(tmp_path, mutation, match):
    trainer = _trainer(tmp_path)
    trainer.train(16)
    checkpoint = torch.load(trainer.save_checkpoint(), weights_only=True)
    mutation(checkpoint)
    path = tmp_path / "invalid.pt"
    torch.save(checkpoint, path)
    restored = _trainer(tmp_path / "restored")
    initial = deepcopy(restored.policy.state_dict())
    with pytest.raises(ValueError, match=match):
        restored.load_checkpoint(path)
    _assert_state_equal(initial, restored.policy.state_dict())


def test_legacy_ppo_loads_available_state_and_grpo_requires_reference(tmp_path):
    trainer = _trainer(tmp_path / "ppo")
    trainer.train(16)
    checkpoint = torch.load(trainer.save_checkpoint(), weights_only=True)
    for field in (
        "schema_version",
        "metadata",
        "rng_by_rank",
        "normalizer_by_rank",
        "world_size",
        "best_checkpoint",
    ):
        checkpoint.pop(field)
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)
    restored = _trainer(tmp_path / "restored")
    with pytest.warns(UserWarning, match="Legacy checkpoint"):
        restored.load_checkpoint(path)
    restored.train(32)
    assert restored.num_updates == 2
    grpo = _trainer(tmp_path / "grpo", "grpo")
    with pytest.raises(ValueError, match="ref_policy"):
        grpo.load_checkpoint(path)


def _config(name: str) -> dict:
    actor = dict(type="mlp", network_cfg=dict(hidden_sizes=[8], activation="tanh"))
    cfg = dict(
        trainer=dict(
            exp_name=f"resume_{name}",
            learning_env=dict(name="PointMassRL", cfg=dict(max_episode_steps=4)),
            device="cpu",
            num_envs=4,
            iterations=1,
            buffer_size=4,
            segment_length=2,
            update_horizon=4,
            save_frequency_updates=1,
            enable_eval=False,
            use_wandb=False,
        ),
        policy=dict(
            name="actor_only" if name == "apg" else "actor_critic", actor=actor
        ),
        algorithm=dict(
            name=name,
            cfg=dict(optimizer=dict(name="adam", learning_rate=1e-3), batch_size=8),
        ),
    )
    if name != "apg":
        cfg["policy"].update(critic=actor, actor_obs_normalization=True)
        cfg["algorithm"]["cfg"].update(n_epochs=1)
    return cfg


@pytest.mark.parametrize("algorithm", ["ppo", "grpo", "apg"])
@pytest.mark.parametrize("source", ["file", "latest", "best"])
def test_cli_contract_resumes_all_algorithms(tmp_path, monkeypatch, algorithm, source):
    monkeypatch.chdir(tmp_path)
    config = _config(algorithm)
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    first = train_from_config(str(config_path))
    path = Path(first["latest_checkpoint_path"]).resolve()
    run = path.parents[1]
    manifest = RunManifest.load(run)
    assert manifest.select_checkpoint("latest")[1] == path
    config["trainer"]["iterations"] = 3
    config_path.write_text(json.dumps(config))
    if source == "file":
        second = train_from_config(str(config_path), resume=path)
    else:
        second = train_from_config(str(config_path), resume=source, run=run)
    assert second["global_step"] == 48
    assert second["num_updates"] == 3
    assert Path(second["latest_checkpoint_path"]).resolve().parents[1] != run


def test_periodic_manifest_survives_interrupted_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _config("ppo")
    config["trainer"]["iterations"] = 3
    path = tmp_path / "train.json"
    path.write_text(json.dumps(config))
    original = Trainer._collect_rollout

    def interrupt_second(self):
        if self.num_updates == 1:
            raise KeyboardInterrupt
        return original(self)

    monkeypatch.setattr(Trainer, "_collect_rollout", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        train_from_config(str(path))
    manifests = list((tmp_path / "outputs").glob("*/run-manifest.json"))
    assert len(manifests) == 1
    checkpoint = RunManifest.load(manifests[0].parent).select_checkpoint()[1]
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["global_step"] == 16
    assert payload["num_updates"] == 1


def test_atomic_write_preserves_previous_contents_on_error(tmp_path):
    path = tmp_path / "checkpoint"
    path.write_bytes(b"complete")

    def broken(stream):
        stream.write(b"incomplete")
        raise OSError("injected failure")

    with pytest.raises(OSError, match="injected"):
        _atomic_write(path, broken)
    assert path.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "args",
    [
        ["--resume", "latest"],
        ["--run", "run"],
        ["--resume", "file.pt", "--run", "run"],
        ["--resume-scheduler", "restart"],
    ],
)
def test_cli_rejects_invalid_resume_arguments(args):
    with pytest.raises(SystemExit):
        parse_args(["--config", "train.yaml", *args])


def test_legacy_manifest_rejects_changed_algorithm_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _config("ppo")
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    first = train_from_config(str(config_path))
    path = Path(first["latest_checkpoint_path"])
    payload = torch.load(path, weights_only=True)
    for field in (
        "schema_version",
        "metadata",
        "rng_by_rank",
        "normalizer_by_rank",
        "world_size",
        "best_checkpoint",
    ):
        payload.pop(field)
    torch.save(payload, path)
    config["algorithm"]["cfg"]["gamma"] = 0.5
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="legacy checkpoint configuration"):
        train_from_config(str(config_path), resume=path)


def test_scheduler_validation_does_not_mutate_policy(tmp_path):
    trainer = _trainer(
        tmp_path, scheduler=dict(name="linear", kwargs=dict(total_iters=4))
    )
    trainer.train(16)
    payload = torch.load(trainer.save_checkpoint(), weights_only=True)
    payload["lr_scheduler_cfg"]["name"] = "unknown"
    path = tmp_path / "bad-scheduler.pt"
    torch.save(payload, path)
    restored = _trainer(
        tmp_path / "new", scheduler=dict(name="linear", kwargs=dict(total_iters=4))
    )
    original = deepcopy(restored.policy.state_dict())
    with pytest.raises(ValueError, match="Unsupported lr_scheduler"):
        restored.load_checkpoint(path)
    _assert_state_equal(original, restored.policy.state_dict())


def test_cli_rejects_changed_environment_and_policy_semantics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _config("ppo")
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    first = train_from_config(str(config_path))
    for changed, match in [
        (
            lambda cfg: cfg["trainer"]["learning_env"]["cfg"].update(
                max_episode_steps=8
            ),
            "learning_env",
        ),
        (
            lambda cfg: cfg["policy"]["actor"]["network_cfg"].update(activation="relu"),
            "modules|context.policy",
        ),
    ]:
        candidate = deepcopy(config)
        changed(candidate)
        config_path.write_text(json.dumps(candidate))
        with pytest.raises(ValueError, match=match):
            train_from_config(str(config_path), resume=first["latest_checkpoint_path"])


def test_gym_resume_error_closes_runtime_without_process_exit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import embodichain.learning.rl.train as training

    config = _config("ppo")
    config["trainer"].pop("learning_env")
    config["trainer"]["gym_config"] = "gym.json"
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    source = tmp_path / "source.pt"
    source.touch()
    components = _trainer(tmp_path / "components")
    sim_cfg = SimpleNamespace(
        headless=True,
        render_cfg=SimpleNamespace(renderer="hybrid"),
        device="cpu",
    )
    runtime = SimpleNamespace(
        env=components.env,
        policy=components.policy,
        gym_config_path=tmp_path / "gym.json",
        gym_config={"id": "PointMassRL"},
        env_cfg=SimpleNamespace(num_envs=4, sim_cfg=sim_cfg),
        close=Mock(),
    )
    monkeypatch.setattr(
        training, "build_gym_policy_runtime", lambda *args, **kwargs: runtime
    )

    def incompatible(self, path, **kwargs):
        raise ValueError("incompatible resume configuration")

    monkeypatch.setattr(Trainer, "load_checkpoint", incompatible)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="incompatible resume"):
        train_from_config(str(config_path), resume=source)
    runtime.close.assert_called_once_with()


def test_cli_flushes_cleanup_and_propagates_resume_error(monkeypatch):
    import embodichain.learning.rl.train as training
    from embodichain.lab.sim.sim_manager import SimulationManager

    calls = []

    def incompatible(*args, **kwargs):
        raise ValueError("incompatible resume configuration")

    monkeypatch.setattr(training, "discover_task_packages", lambda: None)
    monkeypatch.setattr(training, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(training, "train_from_config", incompatible)
    monkeypatch.setattr(
        SimulationManager, "flush_cleanup_queue", lambda: calls.append("cleanup")
    )
    with pytest.raises(ValueError, match="incompatible resume"):
        training.cli(["--config", "train.json", "--resume", "source.pt"])
    assert calls == ["cleanup"]
