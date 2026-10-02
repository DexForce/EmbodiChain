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

"""G1/Go2 CUDA inference, partial reset, and PPO checkpoint integration tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

import embodichain
from embodichain.learning.rl.evaluation import (
    convert_policy_action_for_env,
    infer_policy_action,
)
from embodichain.learning.rl.runtime import build_gym_policy_runtime
from scripts.benchmark.rl.locomotion_ppo import (
    _load_config,
    _build_trainer,
    _check_backend,
)

ROOT = Path(embodichain.__file__).resolve().parent.parent
TASK_DIR = ROOT / "embodichain_tasks/configs/tasks/locomotion/velocity"


def _finite(value: object) -> bool:
    """Check tensor and array leaves of a nested observation."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return bool(np.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(child) for child in value)
    return True


@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.requires_sim
@pytest.mark.requires_tasks
@pytest.mark.parametrize("robot", ["g1", "go2"])
@pytest.mark.parametrize("backend", ["default", "newton"])
def test_locomotion_cuda_reset_step_and_partial_reset(robot: str, backend: str) -> None:
    """Exercise each robot/backend combination in its own process."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), robot, backend],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_locomotion_smoke(robot: str, backend: str) -> None:
    """Check inference and selected-row reset with the official task config."""
    import embodichain_tasks  # noqa: F401 — register task manager functors

    config_dir = TASK_DIR / f"{robot}_flat"
    config = _load_config(config_dir, backend, 8, 24)
    runtime = build_gym_policy_runtime(
        config,
        device=torch.device("cuda:0"),
        num_envs=8,
        headless=True,
        renderer=config["trainer"]["renderer"],
        gpu_id=0,
        seed=42,
        config_dir=config_dir,
    )
    try:
        env = runtime.env
        raw = env.unwrapped
        newton = _check_backend(env, backend)
        observation, _ = env.reset(seed=42)
        assert _finite(observation)
        runtime.policy.eval()
        with torch.inference_mode():
            for _ in range(32):
                action = infer_policy_action(
                    runtime.policy,
                    observation,
                    device=torch.device("cuda:0"),
                    num_envs=8,
                )
                assert action.shape[0] == 8
                assert _finite(action)
                observation, reward, terminated, truncated, _ = env.step(
                    convert_policy_action_for_env(env, action)
                )
                assert _finite((observation, reward))
                assert reward.shape[0] == 8
                assert terminated.shape[0] == truncated.shape[0] == 8
        before = raw._elapsed_steps.clone()
        if newton is not None:
            assert newton.cuda_graph_status == "captured"
        observation, _ = env.reset(
            options={"reset_ids": torch.tensor([0, 3], device=raw.device)}
        )
        assert _finite(observation)
        assert bool((raw._elapsed_steps[[0, 3]] == 0).all())
        remaining = [1, 2, 4, 5, 6, 7]
        assert torch.equal(raw._elapsed_steps[remaining], before[remaining])
    finally:
        runtime.close()


@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.requires_sim
@pytest.mark.requires_tasks
@pytest.mark.parametrize("backend", ["default", "newton"])
def test_g1_ppo_checkpoint_resume(tmp_path: Path, backend: str) -> None:
    """Train through the benchmark CLI, then resume in a separate process."""
    output = tmp_path / backend
    commands = [
        [
            sys.executable,
            "-m",
            "scripts.benchmark.rl.locomotion_ppo",
            "--backend",
            backend,
            "--output",
            str(output),
            "--envs",
            "32",
            "--steps",
            "24",
            "--warmup",
            "1",
            "--updates",
            "3",
        ],
        [sys.executable, str(Path(__file__).resolve()), "resume", backend, str(output)],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "passed"
    assert result["summary"]["ppo_sps"] > 0
    assert result["parameter_max_abs_delta"] > 0


def _resume_checkpoint(backend: str, output: Path) -> None:
    import embodichain_tasks  # noqa: F401

    device = torch.device("cuda:0")
    config_dir = TASK_DIR / "g1_flat"
    config = _load_config(config_dir, backend, 8, 24)
    config["trainer"]["gym_config"] = str((output / "env.yaml").resolve())
    runtime = build_gym_policy_runtime(
        config,
        device=device,
        num_envs=8,
        headless=True,
        renderer=config["trainer"]["renderer"],
        gpu_id=0,
        seed=42,
        config_dir=config_dir,
    )
    try:
        _check_backend(runtime.env, backend)
        saved = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
        assert saved["num_updates"] == 4
        assert saved["global_step"] == 32 * 24 * 4
        trainer = _build_trainer(runtime, config, output)
        runtime.policy.load_state_dict(saved["policy"])
        trainer.algorithm.optimizer.load_state_dict(saved["optimizer"])
        for actual, expected in (
            (runtime.policy.state_dict(), saved["policy"]),
            (trainer.algorithm.optimizer.state_dict(), saved["optimizer"]),
        ):
            torch.testing.assert_close(
                actual, expected, rtol=0, atol=0, check_device=False
            )
        optimizer_steps = [
            float(state["step"]) for state in saved["optimizer"]["state"].values()
        ]
        assert optimizer_steps
        runtime.policy.eval()
        observation, _ = runtime.env.reset(seed=42)
        with torch.no_grad():
            for _ in range(32):
                action = infer_policy_action(
                    runtime.policy, observation, device=device, num_envs=8
                )
                assert _finite(action)
                observation, reward, *_ = runtime.env.step(
                    convert_policy_action_for_env(runtime.env, action)
                )
                assert _finite((observation, reward))
        runtime.policy.train()
        trainer.global_step = saved["global_step"]
        trainer.num_updates = saved["num_updates"]
        trainer.best_eval_value = saved["best_eval_value"]
        summary = trainer.train(trainer.global_step + 8 * 24)
        losses = [
            value
            for key, value in summary["last_train_metrics"].items()
            if key.startswith("train/")
        ]
        assert losses and all(np.isfinite(value) for value in losses)
        trainer.save_checkpoint(str(output / "resumed.pt"))
        resumed = torch.load(
            output / "resumed.pt", map_location="cpu", weights_only=False
        )
        assert resumed["num_updates"] == 5
        assert resumed["global_step"] == saved["global_step"] + 8 * 24
        assert any(
            not torch.equal(saved["policy"][key], value)
            for key, value in resumed["policy"].items()
        )
        assert all(_finite(value) for value in resumed["policy"].values())
        after_steps = [
            float(state["step"]) for state in resumed["optimizer"]["state"].values()
        ]
        assert all(
            after > before
            for before, after in zip(optimizer_steps, after_steps, strict=True)
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    if sys.argv[1] == "resume":
        _resume_checkpoint(sys.argv[2], Path(sys.argv[3]))
    else:
        _run_locomotion_smoke(sys.argv[1], sys.argv[2])
