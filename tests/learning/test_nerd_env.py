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

"""FK-only NeRD factory, graph-safe episodes and production trainer routing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from embodichain.learning.rl.env import DifferentiableEnvProtocol, build_learning_env
from examples.learning.nerd._support import _ant_env, _cartpole_env
from nerd_support import small_nerd_export


@pytest.fixture
def export(tmp_path, request):
    pytest.importorskip("newton")
    pytest.importorskip("dexsim")
    from examples.learning.nerd._resources import prepare_reference_assets

    try:
        prepare_reference_assets(local_files_only=True)
    except (OSError, ValueError):
        pytest.skip("Pinned NeRD reference assets have not been prepared")
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    robot = getattr(request, "param", parameters.get("robot", "ant"))
    return robot, small_nerd_export(tmp_path / robot, robot)


def _env(export, **kwargs):
    robot, checkpoint = export
    return build_learning_env(
        f"{robot.title()}Dynamics-v1",
        num_envs=2,
        device="cpu",
        checkpoint=str(checkpoint),
        random_reset=False,
        **kwargs,
    )


@pytest.mark.parametrize("export", ["cartpole", "ant"], indirect=True)
def test_factory_has_no_world_scene_or_solver_execution(export, monkeypatch):
    import dexsim
    import newton

    def forbidden(*args, **kwargs):
        raise AssertionError("Learning-env attempted native lifecycle allocation")

    monkeypatch.setattr(dexsim, "World", forbidden)
    monkeypatch.setattr(newton.solvers.SolverBase, "__init__", forbidden)
    env = _env(export, max_episode_steps=4)
    try:
        assert isinstance(env, DifferentiableEnvProtocol)
        assert not hasattr(env, "sim")
        action = torch.full(
            (2, *env.single_action_space.shape), 0.02, requires_grad=True
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        assert obs.shape == (2, *env.single_observation_space.shape)
        assert reward.shape == terminated.shape == truncated.shape == (2,)
        (reward.sum() + 0.001 * obs.square().sum()).backward()
        assert action.grad is not None and torch.isfinite(action.grad).all()
        assert action.grad.abs().sum() > 0
        assert all(p.grad is None for p in env.transition.network.parameters())
    finally:
        env.close()


@pytest.mark.parametrize("export", ["cartpole", "ant"], indirect=True)
def test_partial_reset_keeps_other_row_graph_and_previous_reward(export):
    env = _env(export, max_episode_steps=10)
    try:
        action = torch.full(
            (2, *env.single_action_space.shape), 0.02, requires_grad=True
        )
        _, first_reward, *_ = env.step(action)
        old = env.dynamics_state
        old_q, old_reward = old.q.detach().clone(), first_reward.detach().clone()
        env.reset(options={"reset_ids": [0]})
        assert env.dynamics_state.valid_lengths.tolist() == [0, 1]
        torch.testing.assert_close(env.dynamics_state.q[1], old_q[1])
        _, next_reward, *_ = env.step(torch.zeros_like(action))
        gradient = torch.autograd.grad(next_reward.sum(), action, retain_graph=True)[0]
        torch.testing.assert_close(gradient[0], torch.zeros_like(gradient[0]))
        assert gradient[1].abs().sum() > 0
        first_reward.sum().backward()
        torch.testing.assert_close(first_reward, old_reward)
        torch.testing.assert_close(old.q, old_q)
    finally:
        env.close()


@pytest.mark.parametrize("export", ["cartpole", "ant"], indirect=True)
def test_truncation_keeps_full_terminal_graph_and_detach_preserves_values(export):
    env = _env(export, max_episode_steps=1)
    try:
        for _ in range(2):
            action = torch.full(
                (2, *env.single_action_space.shape), 0.02, requires_grad=True
            )
            _, reward, _, truncated, info = env.step(action)
            assert truncated.tolist() == [True, True]
            terminal = info["terminal_observation"]
            saved = terminal.detach().clone()
            assert terminal.requires_grad
            assert env.dynamics_state.valid_lengths.tolist() == [0, 0]
            assert not env.dynamics_state.q.requires_grad
            assert all(
                not value.requires_grad for value in env.dynamics_state.history.values()
            )
            before = env.get_obs().detach().clone()
            (reward.sum() + 0.001 * terminal.square().sum()).backward()
            assert action.grad is not None and action.grad.abs().sum() > 0
            torch.testing.assert_close(terminal, saved)
            torch.testing.assert_close(env.detach_state(), before, rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.parametrize("export", ["cartpole", "ant"], indirect=True)
def test_manual_full_reset_starts_independent_graphs_after_backward(export):
    env = _env(export, max_episode_steps=10)
    try:
        for _ in range(2):
            observation, _ = env.reset()
            assert not observation.requires_grad
            action = torch.full(
                (2, *env.single_action_space.shape), 0.02, requires_grad=True
            )
            obs, reward, *_ = env.step(action)
            (reward.sum() + 0.001 * obs.square().sum()).backward()
            assert action.grad is not None and action.grad.abs().sum() > 0
    finally:
        env.close()


def test_ant_fk_contact_directional_gradient_uses_current_initial_state(export):
    env = _env(export, max_episode_steps=5)
    try:
        q = env.dynamics_state.q.detach().clone().requires_grad_()
        qd = env.dynamics_state.qd.detach().clone()
        direction = torch.zeros_like(q)
        direction[:, 2], direction[:, 7] = 0.4, 0.3

        def value(position):
            contacts = env.transition.geometry.contacts(position, qd)
            return (
                contacts["contact_depths"].sum()
                + 0.1 * contacts["contact_points_1"].sum()
            )

        derivative = (torch.autograd.grad(value(q), q)[0] * direction).sum()
        epsilon = 1e-3
        expected = (
            value(q.detach() + epsilon * direction)
            - value(q.detach() - epsilon * direction)
        ) / (2 * epsilon)
        torch.testing.assert_close(derivative, expected, atol=2e-3, rtol=2e-3)
    finally:
        env.close()


@pytest.mark.parametrize(
    "field,bad", [("physics_signature", "wrong"), ("asset_sha256", "wrong")]
)
def test_factory_rejects_domain_mismatch_before_weight_allocation(
    export, monkeypatch, field, bad
):
    from examples.learning.nerd._support import _model_factory

    _, checkpoint = export
    path = checkpoint / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = bad
    path.write_text(json.dumps(manifest))

    def forbidden(*args, **kwargs):
        raise AssertionError("Loaded weights before validating physical identity")

    monkeypatch.setattr(_model_factory, "load_model", forbidden)
    with pytest.raises(ValueError, match="identity|physics_signature"):
        _env(export)


@pytest.mark.parametrize(
    "robot,algorithm",
    [("cartpole", "apg"), ("ant", "apg"), ("ant", "apg_actor_critic")],
)
def test_existing_training_entry_performs_consecutive_updates(
    export, tmp_path, monkeypatch, robot, algorithm
):
    from embodichain.learning.rl.train import train_from_config
    from examples.learning.nerd._resources import prepare_reference_assets

    prepare_reference_assets(local_files_only=True)
    _, checkpoint = export
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(
        (
            root / f"examples/learning/nerd/configs/{robot}/agents/apg.nerd.yaml"
        ).read_text()
    )
    cfg["trainer"].update(
        device="cpu",
        num_envs=2,
        iterations=2,
        segment_length=2,
        update_horizon=2,
        enable_eval=False,
        use_wandb=False,
        save_frequency_updates=0,
    )
    cfg["trainer"]["learning_env"]["cfg"].update(
        checkpoint=str(checkpoint), max_episode_steps=1, random_reset=False
    )
    cfg["policy"]["actor"]["network_cfg"] = {
        "hidden_sizes": [8],
        "activation": "tanh",
        "last_activation": "tanh",
    }
    if algorithm == "apg_actor_critic":
        cfg["policy"]["name"] = "actor_critic"
        cfg["policy"]["critic"] = {
            "type": "mlp",
            "network_cfg": {"hidden_sizes": [8], "activation": "tanh"},
        }
        cfg["algorithm"]["name"] = algorithm
        cfg["algorithm"]["cfg"].update(critic_epochs=1, critic_minibatches=1)
    path = tmp_path / "train.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path)
    result = train_from_config(path)
    assert result["global_step"] == 8
    assert result["last_train_metrics"]["train/skipped_update"] == 0
    assert list((tmp_path / "outputs").rglob("*.pt"))
