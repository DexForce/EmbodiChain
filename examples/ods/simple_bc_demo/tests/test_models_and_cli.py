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

import argparse
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from examples.ods.simple_bc_demo.models import GeometryRecurrentBC, MLPBC
from examples.ods.simple_bc_demo.checkpoint import load_policy, save_checkpoint
from examples.ods.simple_bc_demo.observations import (
    _geometry_statistics,
    policy_observations,
)


def test_direct_entry_resolves_its_worktree_without_pythonpath(tmp_path) -> None:
    """The documented file entry reaches its own runner before any simulation."""
    (tmp_path / "existing.txt").write_text("Do not overwrite")
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "examples/ods/simple_bc_demo/train.py"),
            "--smoke",
            "--output",
            str(tmp_path),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": "", "EMBODICHAIN_SIM_EXIT_PROCESS": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Use a fresh output directory" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def _mlp(mode: str = "pose") -> MLPBC:
    width = 7 if mode == "pose" else 8
    return MLPBC(
        torch.zeros(51),
        torch.ones(51),
        -torch.ones(width),
        torch.ones(width),
        arm_command_mode=mode,
    )


def test_geometry_features_are_measured_and_quaternion_sign_invariant() -> None:
    obs = {
        "robot": {"qpos": torch.zeros(2, 9), "qvel": torch.zeros(2, 9)},
        "tcp_pose": torch.eye(4).repeat(2, 1, 1),
        "cube_pose": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        "goal_position": torch.tensor([[4.0, 6.0, 8.0]]).repeat(2, 1),
        "cube_velocity": torch.zeros(2, 6),
    }
    values = policy_observations(obs, "geometry")
    assert values.shape == (2, 51)
    torch.testing.assert_close(values[:, 39:42], obs["cube_pose"][:, :3])
    torch.testing.assert_close(
        values[:, 42:45], torch.tensor([[3.0, 4.0, 5.0]]).repeat(2, 1)
    )
    obs["cube_pose"][:, 3:] *= -1
    torch.testing.assert_close(values, policy_observations(obs, "geometry"))
    mean, std = _geometry_statistics(values)
    assert torch.isfinite(std).all() and (std > 0).all()
    torch.testing.assert_close(std[21:27], torch.ones(6))
    obs["cube_pose"].zero_()
    assert torch.isfinite(policy_observations(obs, "geometry")).all()
    obs["cube_pose"][:, 3] = float("nan")
    assert not torch.isfinite(policy_observations(obs, "geometry")).all()


def test_mlp_pose_loss_is_in_unclamped_controller_space() -> None:
    model = _mlp()
    for value in model.parameters():
        torch.nn.init.zeros_(value)
    model.net[-1].bias.data[:] = 2
    obs = torch.zeros(2, 51)
    arm, hand = model.loss(obs, torch.zeros(2, 7))
    torch.testing.assert_close(arm, torch.tensor(4.0))
    (arm + hand).backward()
    assert model.net[-1].bias.grad.abs().min() > 0
    actions, hidden = model.action(obs)
    assert hidden is None
    torch.testing.assert_close(actions, torch.ones(2, 7))


def test_mlp_joint_residual_loss_uses_residual_scale_not_joint_span() -> None:
    model = _mlp("residual")
    for value in model.parameters():
        torch.nn.init.zeros_(value)
    obs = torch.zeros(2, 51)
    targets = torch.zeros(2, 8)
    targets[:, :7] = 0.125
    arm, hand = model.loss(obs, targets)
    torch.testing.assert_close(arm, torch.tensor(0.25))
    torch.testing.assert_close(hand, torch.tensor(0.0))


@pytest.mark.parametrize("mode", ["pose", "residual"])
def test_mlp_checkpoint_reload_preserves_actions_and_normalization(
    tmp_path, mode: str
) -> None:
    model = _mlp(mode)
    path = tmp_path / "mlp.pt"
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(path, model, optimizer, 10, argparse.Namespace(policy="mlp"), 123)
    restored, state = load_policy(str(path), "cpu")
    values = torch.randn(2, 51)
    torch.testing.assert_close(model.action(values)[0], restored.action(values)[0])
    assert isinstance(restored, MLPBC) and state["policy_config"]["type"] == "mlp"


def test_budget_deadline_and_baseline_checkpoint_selection() -> None:
    from examples.ods.simple_bc_demo.trainer import (
        _BudgetExpired,
        _selection,
        _timeout,
    )
    from types import SimpleNamespace

    initial = {"success_rate": 0.9, "lift_rate": 1.0, "goal_error": 0.01}
    worse = {"success_rate": 0.1, "lift_rate": 0.2, "goal_error": 0.3}
    assert _selection(initial) > _selection(worse)
    with pytest.raises(_BudgetExpired):
        _timeout(
            SimpleNamespace(wall_time_budget=5.0),
            SimpleNamespace(elapsed_seconds=lambda: 6.0),
        )
    assert (
        _timeout(
            SimpleNamespace(wall_time_budget=5.0),
            SimpleNamespace(elapsed_seconds=lambda: 3.0),
        )
        == 2.0
    )


def test_qualification_reuse_checks_physics_sources_and_independent_scenes(
    tmp_path,
) -> None:
    import json
    from examples.ods.simple_bc_demo.trainer import _reuse_qualification

    path = tmp_path / "expert_qualification.json"
    records = [{"seed": i, "success": True} for i in range(40)]
    path.write_text(json.dumps({"records": records, "success_rate": 1.0}))
    config = {
        "device": "cuda:1",
        "gpu_id": 1,
        "seed": 42,
        "physics": "default",
        "env": {"actions": "dls"},
    }
    (tmp_path / "task_config.json").write_text(json.dumps(config))
    hashes = {
        "embodichain/lab/gym/envs/managers/actions.py": "a",
        "examples/ods/simple_bc_demo/environment.py": "b",
    }
    (tmp_path / "environment.json").write_text(json.dumps({"source_sha256": hashes}))
    assert (
        _reuse_qualification(path, {**config, "device": "cuda:3", "gpu_id": 3}, hashes)[
            "success_rate"
        ]
        == 1.0
    )
    with pytest.raises(ValueError, match="configuration differs"):
        _reuse_qualification(path, {**config, "physics": "newton"}, hashes)
    with pytest.raises(ValueError, match="source differs"):
        _reuse_qualification(
            path,
            config,
            {**hashes, "embodichain/lab/gym/envs/managers/actions.py": "changed"},
        )
    records[1]["seed"] = records[0]["seed"]
    path.write_text(json.dumps({"records": records, "success_rate": 1.0}))
    with pytest.raises(ValueError, match="independent scenes"):
        _reuse_qualification(path, config, hashes)


def test_geometry_recurrent_prefix_matches_stepwise_and_ignores_padding() -> None:
    """Full causal training preserves online hidden state and gradients through history."""
    torch.manual_seed(17)
    model = GeometryRecurrentBC(
        torch.zeros(51), torch.ones(51), -torch.ones(7), torch.ones(7)
    )
    obs = torch.randn(2, 12, 51, requires_grad=True)
    predictions, full_hidden = model(obs)
    hidden = None
    for index in range(obs.shape[1]):
        action, hidden = model.action(obs[:, index], hidden)
        torch.testing.assert_close(action, predictions[:, index].clamp(-1, 1))
    torch.testing.assert_close(hidden, full_hidden)

    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, 4:8], mask[1, 8:] = True, True
    targets = torch.zeros(2, 12, 7)
    arm, hand = model.loss(obs, targets, loss_mask=mask)
    (arm + hand).backward()
    assert obs.grad[:, :4].abs().sum() > 0
    assert obs.grad[0, 8:].count_nonzero() == 0
    padded_obs, padded_targets = obs.detach().clone(), targets.clone()
    padded_obs[0, 8:] = 1000
    padded_targets[~mask] = -1000
    torch.testing.assert_close(
        model.loss(padded_obs, padded_targets, loss_mask=mask), (arm, hand)
    )


def test_geometry_recurrent_checkpoint_reload_preserves_history_and_contract(
    tmp_path,
) -> None:
    """The geometry GRU has an explicit checkpoint identity and identical resumed actions."""
    torch.manual_seed(19)
    model = GeometryRecurrentBC(
        torch.linspace(-0.5, 0.5, 51),
        torch.linspace(0.5, 1.5, 51),
        -torch.ones(7),
        torch.ones(7),
    ).eval()
    path = tmp_path / "geometry_gru.pt"
    save_checkpoint(
        path,
        model,
        torch.optim.AdamW(model.parameters()),
        12,
        argparse.Namespace(policy="gru_pose"),
        456,
    )
    restored, state = load_policy(str(path), "cpu")
    assert isinstance(restored, GeometryRecurrentBC)
    assert state["policy_config"] == {
        "type": "gru_pose",
        "observation_mode": "geometry",
        "arm_command_mode": "pose",
        "schema": None,
    }
    history, current = torch.randn(2, 4, 51), torch.randn(2, 51)
    _, hidden = model(history)
    _, restored_hidden = restored(history)
    torch.testing.assert_close(
        model.action(current, hidden), restored.action(current, restored_hidden)
    )
    torch.testing.assert_close(model.mean, restored.mean)
    torch.testing.assert_close(model.std, restored.std)
    assert not restored.training


def test_geometry_recurrent_loss_uses_unclamped_normalized_requests() -> None:
    """MSE supervises linear requests even when execution clamps them at the bounds."""
    model = GeometryRecurrentBC(
        torch.zeros(51), torch.ones(51), -torch.ones(7), torch.ones(7)
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        model.head[-1].bias.fill_(2)
    obs = torch.zeros(2, 3, 51)
    arm, hand = model.loss(obs, torch.zeros(2, 3, 7))
    torch.testing.assert_close((arm, hand), (torch.tensor(4.0), torch.tensor(4.0)))
    (arm + hand).backward()
    assert model.head[-1].bias.grad.abs().min() > 0
    action, _ = model.action(obs[:, 0])
    torch.testing.assert_close(action, torch.ones(2, 7))


def test_offline_reset_diagnostic_uses_original_causal_frame_zero() -> None:
    from examples.ods.simple_bc_demo.trainer import _offline_reset_positions

    observations = torch.randn(4, 3, 51)
    names = [f"feature_{i}" for i in range(51)]
    names[27:30] = ["cube_0", "cube_1", "cube_2"]
    corpus = SimpleNamespace(
        observations=observations,
        valid=torch.ones(4, 3, dtype=torch.bool),
        frame_ids=torch.zeros(4, 3, 3, dtype=torch.int64),
        manifest={
            "schema": {
                "observation": {"version": "pick_place.geometry@2", "names": names}
            }
        },
    )
    assert _offline_reset_positions(corpus, 4) == observations[:, 0, 27:30].tolist()
    with pytest.raises(ValueError, match="exactly the requested episodes"):
        _offline_reset_positions(corpus, 1)
    corpus.valid[2, 0] = False
    with pytest.raises(ValueError, match="frame zero"):
        _offline_reset_positions(corpus, 4)
    corpus.valid[2, 0] = True
    corpus.frame_ids[1, 0, 2] = 1
    with pytest.raises(ValueError, match="frame zero"):
        _offline_reset_positions(corpus, 4)
