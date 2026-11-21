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

"""Fair reset pairing, physical-domain admission and complete-episode reporting."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.benchmark.rl.nerd_policy import (
    BenchmarkCfg,
    InitialStates,
    _evaluate,
    _validate_physics,
)
from scripts.benchmark.rl._nerd_policy_report import _load_complete, _mean_std

pytestmark = pytest.mark.no_sim


def test_reset_pairing_is_independent_of_completion_order_and_policy_rng() -> None:
    rng = torch.get_rng_state().clone()
    first = InitialStates("ant", 2, 4, 123, torch.device("cpu"))
    second = InitialStates("ant", 2, 4, 123, torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), rng)
    assert first.sha256 == second.sha256
    both = torch.tensor([0, 1])
    for a, b in zip(first(both), second(both)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Different episode completion order must not change either world's next start.
    a0 = first(torch.tensor([0]))
    a1 = first(torch.tensor([1]))
    b1 = second(torch.tensor([1]))
    b0 = second(torch.tensor([0]))
    for a, b in zip((*a0, *a1), (*b0, *b1)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("missing", [False, True])
def test_physical_identity_rejects_codec_drift_or_missing_metadata(
    missing: bool,
) -> None:
    expected = {
        "physics_signature": "robot-and-contact-digest",
        "physics_signature_schema": "v2",
        "dt": 1 / 60,
        "gravity": [0, 0, -9.81],
        "joint_names": ["slider", "pole"],
        "contact_schema": "plane-v1",
        "codec": {"robot": "cartpole", "base_com": [0, 0, 0]},
    }
    actual = copy.deepcopy(expected)
    _validate_physics(expected, actual)
    if missing:
        actual.pop("codec")
    else:
        actual["codec"]["base_com"] = [0, 1, 0]
    with pytest.raises(ValueError, match="codec"):
        _validate_physics(expected, actual)


class _Actor(torch.nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.full((observation.shape[0], 1), 0.5)


class _Teacher:
    """Two independent cartpole episodes, one failing on the first step."""

    robot = "cartpole"
    device = torch.device("cpu")
    num_envs = 2
    profile = {"qd_dim": 2}

    def __init__(self, nonfinite: bool) -> None:
        self.nonfinite = nonfinite
        self.actions: list[torch.Tensor] = []

    def reset(self, q: torch.Tensor, qd: torch.Tensor) -> None:
        self.steps = 0

    def step(self, effort: torch.Tensor) -> None:
        self.actions.append(effort.clone())
        self.steps += 1

    def snapshot(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        position = (
            float("nan") if self.nonfinite and self.steps >= 2 else float(self.steps)
        )
        q = torch.tensor([[5.0 if self.steps else 0.0, 0.0], [position, 0.0]])
        return {"q": q, "qd": torch.zeros(2, 2)}, {}


@pytest.mark.parametrize("nonfinite", [False, True])
def test_evaluation_keeps_each_fixed_episode_and_counts_nonfinite(
    nonfinite: bool,
) -> None:
    teacher = _Teacher(nonfinite)
    policy = torch.nn.Module()
    policy.actor = _Actor()
    result = _evaluate(teacher, policy, torch.zeros(2, 2), torch.zeros(2, 2), 3)
    # -0.05*x^2: world 0 ends at x=5; world 1 visits x=1,2,3.
    assert result["episodes"][0]["return"] == pytest.approx(-1.25)
    assert result["episodes"][0]["length"] == 1
    assert len(result["episodes"]) == 2
    assert torch.equal(teacher.actions[1][0], torch.zeros(2))
    if nonfinite:
        assert result["nonfinite_episodes"] == 1
        assert result["return_mean"] is None
        assert result["episodes"][1]["return"] is None
    else:
        assert result["return_mean"] == pytest.approx((-1.25 - 0.7) / 2)
        assert result["length_mean"] == 2
        assert result["failure_rate"] == 0.5


def test_report_rejects_unpaired_runs_and_uses_seed_dispersion(tmp_path: Path) -> None:
    cfg = BenchmarkCfg(
        robots=("cartpole",), seeds=(0,), updates=2, checkpoints=1, eval_episodes=1
    )
    for algorithm in cfg.algorithms:
        identity = {
            "robot": "cartpole",
            "algorithm": algorithm,
            "seed": 0,
            "recipe": cfg.to_dict(),
            **{
                key: "paired"
                for key in (
                    "model_weights_sha256",
                    "model_manifest_sha256",
                    "physical_signature",
                    "evaluation_initial_states_sha256",
                    "initial_actor_sha256",
                    "initial_state_bank_sha256",
                )
            },
        }
        path = tmp_path / "cartpole" / algorithm / "seed-0" / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "identity": identity,
                    "evaluations": [
                        {"global_step": step, "episodes": [{}]} for step in (0, 2048)
                    ],
                }
            )
        )
    assert len(_load_complete(tmp_path, cfg)) == 2
    result = json.loads(path.read_text())
    result["identity"]["initial_actor_sha256"] = "different-actor"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="unpaired"):
        _load_complete(tmp_path, cfg)
    assert _mean_std([1.0, 3.0, 5.0]) == (3.0, 2.0)
    assert not torch.isfinite(torch.tensor(_mean_std([1.0, None])[0]))
