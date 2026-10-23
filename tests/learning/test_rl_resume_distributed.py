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

import datetime
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from embodichain.learning.rl.algo import build_algo
from embodichain.learning.rl.policies import ActorCritic
from embodichain.learning.rl.policy_evaluation.manifest import (
    RunManifest,
    write_run_manifest,
)
from embodichain.learning.rl.utils.trainer import Trainer
from embodichain.learning.rl.train import _save_final_checkpoint
from embodichain_tasks.classic_control.point_mass import PointMassEnv


def _components(rank: int, directory: Path, mode: str) -> Trainer:
    env = PointMassEnv(num_envs=4, max_episode_steps=4)
    env.level = rank + 2
    env.get_training_state = lambda: {"rank": rank, "level": env.level}

    def validate(state):
        if state["rank"] != rank:
            raise ValueError("Environment state belongs to another rank")

    def load(state):
        validate(state)
        env.level = state["level"]

    env.validate_training_state = validate
    env.load_training_state = load
    policy = ActorCritic(
        14,
        2,
        torch.device("cpu"),
        torch.nn.Linear(14, 2),
        torch.nn.Linear(14, 1),
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    kwargs = dict(n_epochs=2, batch_size=8)
    if mode == "ppo_adaptive":
        kwargs.update(schedule="adaptive", desired_kl=0.01)
    else:
        kwargs["lr_scheduler"] = dict(name="linear", kwargs=dict(total_iters=4))
    algo = build_algo(
        "grpo" if mode == "grpo" else "ppo",
        kwargs,
        policy,
        torch.device("cpu"),
        distributed=True,
    )
    return Trainer(
        policy,
        env,
        algo,
        buffer_size=4,
        batch_size=8,
        writer=None,
        eval_freq=32,
        save_freq=32,
        save_frequency_updates=1,
        checkpoint_dir=str(directory / "checkpoints"),
        exp_name="distributed",
        use_wandb=False,
        distributed=True,
        rank=rank,
        world_size=2,
        eval_env=PointMassEnv(num_envs=2, max_episode_steps=4) if rank == 0 else None,
        num_eval_episodes=2,
        eval_seed=17,
    )


def _assert_shared(trainer: Trainer) -> None:
    parameters = torch.cat([p.detach().flatten() for p in trainer.policy.parameters()])
    copies = [torch.empty_like(parameters) for _ in range(2)]
    torch.distributed.all_gather(copies, parameters)
    torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
    momentum = torch.cat(
        [
            state["exp_avg"].flatten()
            for state in trainer.algorithm.optimizer.state.values()
        ]
    )
    copies = [torch.empty_like(momentum) for _ in range(2)]
    torch.distributed.all_gather(copies, momentum)
    torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
    lr = torch.tensor(trainer.algorithm.current_learning_rate())
    rates = [torch.empty_like(lr) for _ in range(2)]
    torch.distributed.all_gather(rates, lr)
    assert rates[0] == rates[1]


def _worker(rank: int, group_file: str, directory: str) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{group_file}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    directory = Path(directory)
    try:
        for mode in ("ppo_fixed", "ppo_adaptive", "grpo"):
            random.seed(3 + rank)
            np.random.seed(3 + rank)
            torch.manual_seed(3 + rank)
            run = directory / mode
            trainer = _components(rank, run, mode)
            train_config = directory / "train.json"
            trainer._checkpoint_callback = lambda summary: write_run_manifest(
                run,
                train_config=train_config,
                latest_checkpoint=summary["latest_checkpoint_path"],
                best_checkpoint=summary["best_checkpoint_path"],
            )
            trainer.train(64)
            _assert_shared(trainer)
            path = trainer.save_checkpoint()
            expected_rng = random.random(), np.random.rand(), torch.rand(3)
            expected_normalizers = {
                k: v.clone()
                for k, v in trainer.policy.state_dict().items()
                if "normalizer" in k
            }
            expected_reference = (
                deepcopy(trainer.algorithm.ref_policy.state_dict())
                if mode == "grpo"
                else None
            )
            restored = _components(rank, directory / f"{mode}_resumed", mode)
            restored.env.level = -1
            restored.load_checkpoint(path)
            assert restored.env.level == rank + 2
            assert restored.best_eval_value == trainer.best_eval_value
            assert restored.best_checkpoint_path is None
            actual_rng = random.random(), np.random.rand(), torch.rand(3)
            assert actual_rng[:2] == expected_rng[:2]
            torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0, atol=0)
            for key, value in expected_normalizers.items():
                torch.testing.assert_close(
                    restored.policy.state_dict()[key], value, rtol=0, atol=0
                )
            if expected_reference is not None:
                for key, value in expected_reference.items():
                    torch.testing.assert_close(
                        restored.algorithm.ref_policy.state_dict()[key],
                        value,
                        rtol=0,
                        atol=0,
                    )
            assert restored.num_updates == 2
            assert restored.global_step == 64
            restored.train(96)
            assert restored.num_updates == 3
            _assert_shared(restored)
            if rank == 0:
                manifest = RunManifest.load(run)
                assert manifest.checkpoints["best"] is not None
                if restored.best_checkpoint_path is not None:
                    assert Path(restored.best_checkpoint_path).is_file()
                payload = torch.load(path, weights_only=True)
                assert len(payload["rng_by_rank"]) == 2
                assert payload["environment_state_by_rank"] == [
                    {"rank": 0, "level": 2},
                    {"rank": 1, "level": 3},
                ]
                if mode != "grpo":
                    assert not torch.equal(
                        payload["normalizer_by_rank"][0]["actor_obs_normalizer.mean"],
                        payload["normalizer_by_rank"][1]["actor_obs_normalizer.mean"],
                    )
            torch.distributed.barrier()
            # Rank zero owns output files; final-save validation must fail on all ranks.
            if rank == 0:
                Path(restored.latest_checkpoint_path).unlink()
            torch.distributed.barrier()
            with pytest.raises(RuntimeError, match="rank 0.*latest checkpoint"):
                _save_final_checkpoint(restored)
            torch.distributed.barrier()
            # Only rank 1 loses its input; both ranks must report the failure.
            try:
                restored.load_checkpoint(
                    path if rank == 0 else directory / "missing.pt"
                )
            except RuntimeError as exc:
                assert "rank 1" in str(exc)
            else:
                raise AssertionError("All ranks must fail on a rank-local read error")
            torch.distributed.barrier()
            if rank == 0:

                def fail_index(summary):
                    raise OSError("injected manifest failure")

                trainer._checkpoint_callback = fail_index
            previous = trainer.latest_checkpoint_path
            try:
                trainer.save_checkpoint(run / "checkpoints" / "failed-index.pt")
            except RuntimeError as exc:
                assert "injected manifest failure" in str(exc)
            else:
                raise AssertionError("All ranks must fail on rank-zero write errors")
            if rank == 0:
                assert trainer.latest_checkpoint_path == previous
                assert (
                    RunManifest.load(run).select_checkpoint()[1]
                    == Path(previous).resolve()
                )
            torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="Requires torch.distributed"
)
def test_two_rank_training_and_resume(tmp_path: Path) -> None:
    (tmp_path / "train.json").write_text(json.dumps({"test": "distributed resume"}))
    context = mp.spawn(
        _worker,
        args=(str(tmp_path / "group"), str(tmp_path)),
        nprocs=2,
        join=False,
    )
    # A bounded join detects collective regressions without leaving child jobs.
    deadline = datetime.datetime.now() + datetime.timedelta(seconds=100)
    try:
        while not context.join(timeout=1):
            if datetime.datetime.now() > deadline:
                pytest.fail("Distributed checkpoint test exceeded 100 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
