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

"""Sharpa learned-environment state, reward, reset and gradient contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from embodichain.learning.rl.env import build_learning_env
from embodichain_models.dynamics.rwm.robot_state import (
    integrate_robot_state,
    rotation_exp,
)
from embodichain.learning.sl import SupervisedBatch
from embodichain_tasks.manipulation.dexterous.sharpa_inhand import (
    SharpaInHandCfg,
    SharpaInHandModelEnv,
    SharpaInHandWorldModel,
    _REFERENCE_HAND_POSE,
    _REFERENCE_OBJECT_POSITION,
    _reward,
    _WorldObjective,
    _joint_limit_aware_action,
    _cylinder_axis_observation,
    _stable_rotation_mask,
)

pytestmark = pytest.mark.no_sim


class ControlledModel(SharpaInHandWorldModel):
    """Known differentiable plant for checking the task/learner boundary."""

    def __init__(self) -> None:
        super().__init__((8,))

    def forward(self, state, action, context=None):
        change = action - state[:, 18:40]
        angular_step = torch.cat(
            (
                torch.zeros(len(state), 2),
                (state[:, 17:18] + change[:, :1] * 0.5) * 0.05,
            ),
            -1,
        )
        increment = torch.cat(
            (
                torch.zeros(len(state), 3),
                angular_step,
                torch.zeros(len(state), 5),
                change[:, :1] * 0.5,
                change,
                change * 20 - state[:, 40:62],
            ),
            -1,
        )
        return integrate_robot_state(state, increment)


def _bank(count=2):
    states = torch.zeros(count, 62)
    states[:, :3] = torch.tensor(_REFERENCE_OBJECT_POSITION)
    states[:, 3:12] = torch.eye(3).flatten()
    states[:, 18:40] = torch.tensor(_REFERENCE_HAND_POSE)
    contexts = torch.zeros(count, 47)
    contexts[:, :22] = states[:, 18:40]
    contexts[:, 22:25] = states[:, :3]
    contexts[:, 25:47] = states[:, 18:40]
    limits = torch.tensor([[-2.0, 2.0]]).expand(22, 2).clone()
    return states, contexts, limits


def _env(steps=8, num_envs=2, model=None, **task_options):
    states, contexts, limits = _bank()
    return build_learning_env(
        "SharpaInHandModel",
        num_envs=num_envs,
        device="cpu",
        max_episode_steps=steps,
        model=model or ControlledModel(),
        initial_states=states,
        initial_contexts=contexts,
        joint_limits=limits,
        **task_options,
    )


def test_recurrent_action_gradient_matches_finite_difference():
    env = _env()
    controls = torch.zeros(2, 2, 22, requires_grad=True)

    def objective(actions):
        env.reset(seed=11)
        total = 0
        for action in actions:
            _, reward, _, _, _ = env.step(action)
            total = total + reward.sum()
        return total

    value = objective(controls)
    (gradient,) = torch.autograd.grad(value, controls)
    assert gradient[0, 0, 0] > 0
    assert torch.isfinite(gradient).all()
    epsilon = 1e-3
    plus, minus = controls.detach().clone(), controls.detach().clone()
    plus[0, 0, 0] += epsilon
    minus[0, 0, 0] -= epsilon
    numerical = (objective(plus) - objective(minus)) / (2 * epsilon)
    torch.testing.assert_close(gradient[0, 0, 0], numerical, atol=2e-5, rtol=0.02)
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in env.model.parameters()
    )
    env.close()
    env.close()  # Closing twice must retain restored caller ownership.
    assert all(parameter.requires_grad for parameter in env.model.parameters())


def test_reference_reward_and_terminal_autoreset_are_separate():
    env = _env(steps=1)
    env.state[:, 17] = 0.2
    terminal_state = env.state
    observations, reward, terminated, truncated, info = env.step(torch.zeros(2, 22))
    # rotation=2.5*.2; position=0.003/0.001; all penalties zero; dt=.05
    torch.testing.assert_close(reward, torch.full((2,), 0.175))
    assert not terminated.any() and truncated.all()
    assert observations.shape == (2, 109)
    assert info["_episode"].all() and (info["episode"]["l"] == 1).all()
    assert (env.episode_step == 0).all()
    assert (env.state[:, 17] == 0).all()
    assert (terminal_state[:, 17] == 0.2).all()


def test_actor_observes_absolute_pose_and_separate_reset_control_references():
    env = _env()
    env.state[:] = env.state[0].clone()
    env.context[:] = env.context[0].clone()
    before = env.detach_state().clone()
    # Reset references remain observable for height termination and control,
    # while the reference reward uses the fixed source object and hand poses.
    env.context[1, 22] += 0.01
    env.context[1, 25] += 0.2
    after = env.detach_state().clone()
    torch.testing.assert_close(after[:, :84], before[:, :84])
    assert not torch.equal(after[0, 84:], after[1, 84:])
    reward = _reward(env.state, env.context, env.cfg, previous_state=env.state)
    torch.testing.assert_close(reward[0], reward[1])
    # Moving the actual cylinder changes the physical-pose channels regardless
    # of whether its task reset anchor moves along with it.
    env.state[1, 0] += 0.01
    moved = env.detach_state()
    torch.testing.assert_close(moved[1, 0] - after[1, 0], torch.tensor(0.1))


def test_source_fixed_reward_pose_and_control_period_rates():
    # Independent source constants: reference default hand deviation for
    # q=(1,.1,0,...,0) is5.167486224524282 rad² at pinned upstream9816885.
    state = torch.zeros(1, 62)
    state[:, :3] = torch.tensor((-0.09559, -0.00517, 0.61906))
    state[:, 3:12] = torch.eye(3).flatten()
    state[:, 18:20] = torch.tensor((1.0, 0.1))
    previous = state.clone()
    # Instantaneous velocities and reset anchors must not change this
    # transition's finite-difference reward or its fixed pose references.
    state[:, 12:18] = 50.0
    state[:, 40:62] = 100.0
    context = torch.zeros(1, 47)
    context[:, :22] = state[:, 18:40]
    context[:, 22:25] = torch.tensor((0.8, 0.8, 0.7))
    context[:, 25:47] = state[:, 18:40]
    reward = _reward(state, context, SharpaInHandCfg(), previous_state=previous)
    torch.testing.assert_close(reward, torch.tensor([0.04665027550951435]))


def test_stable_rotation_uses_inclusive_positive_counts_and_all_requirements():
    result = _stable_rotation_mask(
        torch.tensor([True, True, True, True, False, True]),
        torch.tensor([7.0, 7.0, 7.0, 7.0, 7.0, 6.0]),
        torch.tensor([320, 319, 320, 320, 400, 400]),
        torch.tensor([80, 100, 79, 81, 100, 100]),
        steps=400,
        final_window_steps=100,
    )
    assert result.tolist() == [True, False, False, True, False, False]


def test_axis_policy_features_keep_tilt_and_remove_cylinder_phase():
    observation = torch.arange(109, dtype=torch.float32).repeat(3, 1)
    observation[:, 3:12] = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, -1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, -1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    original = observation.clone()
    features = _cylinder_axis_observation(observation)
    torch.testing.assert_close(features[0], features[1])
    torch.testing.assert_close(
        features[2, 3:12], torch.tensor([0, 0, 0, 0, 0, -1, 0, 0, 0]).float()
    )
    torch.testing.assert_close(features[:, :3], observation[:, :3])
    torch.testing.assert_close(features[:, 12:], observation[:, 12:])
    torch.testing.assert_close(observation, original)


def test_joint_limit_aware_policy_is_feasible_and_differentiable_at_bounds():
    lower, upper = torch.full((22,), -1.0), torch.full((22,), 1.0)
    lower, upper = lower.double(), upper.double()
    observation = torch.zeros(2, 109, dtype=torch.float64)
    observation[0, 62:84] = -1
    observation[1, 62:84] = 1
    proposal = torch.full((2, 22), 0.9, dtype=torch.float64)
    proposal[0] = -0.9
    proposal.requires_grad_()

    def projected_target(value):
        action = _joint_limit_aware_action(value, observation, lower, upper)
        current = observation[:, 62:84]
        return (current + action / 24).clamp(lower, upper)

    target = projected_target(proposal)
    assert (target > lower).all() and (target < upper).all()
    assert (target - observation[:, 62:84]).abs().max() < 1 / 24
    (gradient,) = torch.autograd.grad(target.sum(), proposal)
    assert (gradient > 0).all() and torch.isfinite(gradient).all()
    plus, minus = proposal.detach().clone(), proposal.detach().clone()
    plus[0, 0] += 1e-5
    minus[0, 0] -= 1e-5
    numerical = (projected_target(plus)[0, 0] - projected_target(minus)[0, 0]) / 2e-5
    torch.testing.assert_close(gradient[0, 0], numerical, rtol=1e-5, atol=1e-9)


def test_optional_height_objective_has_a_restoring_gradient_before_drop():
    env = _env(height_penalty=1.0, rotation_weight=0.0)
    state = env.state.detach().clone()
    state[:, 2] += 0.01
    env.state = state.requires_grad_()
    _, reward, terminated, _, _ = env.step(torch.zeros(2, 22))
    # At 10 mm excursion: position=.003/(.01+.001), barrier=(.01/.02)^2.
    expected = 0.05 * (0.003 / 0.011 - 0.25)
    torch.testing.assert_close(reward, torch.full((2,), expected), rtol=1e-4, atol=1e-6)
    (gradient,) = torch.autograd.grad(reward.sum(), state)
    restoring = -0.05 * (0.003 / 0.011**2 + 2 * 0.01 / 0.02**2)
    torch.testing.assert_close(
        gradient[:, 2], torch.full((2,), restoring), rtol=1e-4, atol=1e-5
    )
    assert not terminated.any()


def test_drop_and_selective_reset_do_not_change_other_rows_or_old_graphs():
    env = _env()
    actions = torch.zeros(2, 22, requires_grad=True)
    env.state[0, 2] += 0.025  # Reference total band is 40 mm: 20 mm each side.
    _, reward, terminated, truncated, info = env.step(actions)
    assert terminated.tolist() == [True, False] and not truncated.any()
    assert info["episode"]["dropped"].tolist() == [1.0, 0.0]
    assert env.episode_step.tolist() == [0, 1]
    observation_before = env.detach_state().clone()
    previous_state = env.state
    env.reset(options={"env_ids": torch.tensor([0])})
    torch.testing.assert_close(previous_state[1], env.state[1])
    torch.testing.assert_close(observation_before[1], env.detach_state()[1])
    reward.sum().backward()
    assert actions.grad is not None and torch.isfinite(actions.grad).all()


def test_detach_preserves_state_targets_and_sampling_rng():
    env = _env()
    env.step(torch.full((2, 22), 0.2, requires_grad=True))
    state, context = env.state.detach().clone(), env.context.detach().clone()
    rng = env._generator.get_state().clone()
    detached = env.detach_state()
    assert not detached.requires_grad and not env.state.requires_grad
    torch.testing.assert_close(env.state, state)
    torch.testing.assert_close(env.context, context)
    assert torch.equal(env._generator.get_state(), rng)


@pytest.mark.parametrize("servo_context", [False, True])
def test_fixed_palm_model_derives_context_from_its_predicted_pose(servo_context):
    model = SharpaInHandWorldModel((8,), servo_context=servo_context)
    states, _, _ = _bank()
    states[:, 18:40] = 0.1
    torch.nn.init.normal_(model.net[-1].weight, std=0.05)
    action = torch.full((2, 22), 0.25)
    from embodichain_models.dynamics.rwm import RobotWorldModel

    context = (
        torch.cat((states[:, :12], torch.full((2, 22), 0.15)), -1)
        if servo_context
        else states[:, :12]
    )
    expected = RobotWorldModel.forward(model, states, action, context)
    torch.testing.assert_close(model(states, action), expected)
    with pytest.raises(ValueError, match="current prediction"):
        model(states, action, torch.zeros(2, 3))
    # The canonical RWM drops global yaw. A fixed palm requires that yaw back:
    # identical hand joints at different cylinder yaw must be distinguishable.
    rotated = states.clone()
    rotated[:, 3:12] = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    rotated_context = context.clone()
    rotated_context[:, 3:12] = rotated[:, 3:12]
    original_features = model._features(states, action, context)
    rotated_features = model._features(rotated, action, rotated_context)
    base_width = model.input_dim - model.cfg.context_dim
    torch.testing.assert_close(
        original_features[:, :base_width], rotated_features[:, :base_width]
    )
    assert not torch.equal(
        original_features[:, base_width:], rotated_features[:, base_width:]
    )
    if servo_context:
        controls = action.clone().requires_grad_()
        (gradient,) = torch.autograd.grad(model(states, controls)[:, 2].sum(), controls)
        plus, minus = action.clone(), action.clone()
        plus[0, 0] += 1e-3
        minus[0, 0] -= 1e-3
        numerical = (
            model(states, plus)[:, 2].sum() - model(states, minus)[:, 2].sum()
        ) / 0.002
        torch.testing.assert_close(gradient[0, 0], numerical, rtol=0.03, atol=2e-4)


@pytest.mark.parametrize("servo_context", [False, True])
@pytest.mark.parametrize(
    "world_frame,axial_symmetry", [(False, False), (True, False), (True, True)]
)
def test_model_export_loads_measured_starts_and_rejects_wrong_layout(
    tmp_path: Path, servo_context, world_frame, axial_symmetry
):
    model = SharpaInHandWorldModel(
        (8,),
        servo_context=servo_context,
        world_frame=world_frame,
        axial_symmetry=axial_symmetry,
    )
    states, contexts, limits = _bank()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format": "sharpa-inhand-world-model-v1",
            "hidden_dims": [8],
            "servo_context": servo_context,
            "world_frame": world_frame,
            "axial_symmetry": axial_symmetry,
            "model": model.state_dict(),
            "initial_states": states,
            "initial_contexts": contexts,
            "joint_limits": limits,
        },
        checkpoint,
    )
    env = SharpaInHandModelEnv(2, model_checkpoint=str(checkpoint))
    torch.testing.assert_close(env.state, states)
    old = env.state
    env.replace_start_bank(states + torch.tensor([0.01] + [0.0] * 61), contexts)
    torch.testing.assert_close(env.state, old)
    env.reset()
    torch.testing.assert_close(env.state[:, 0], states[:, 0] + 0.01)
    with pytest.raises(ValueError, match="measured start"):
        env.replace_start_bank(states[:, :-1], contexts)


@pytest.mark.parametrize("servo_context", [False, True])
def test_native_cylinder_phase_symmetry_and_action_gradients(servo_context):
    model = SharpaInHandWorldModel(
        (8,), world_frame=True, axial_symmetry=True, servo_context=servo_context
    )
    torch.nn.init.normal_(model.net[-1].weight, std=0.05)
    state, _, _ = _bank()
    state[:, 3:12] = rotation_exp(torch.tensor([[0.2, -0.1, 0.3]])).flatten()
    twist = rotation_exp(torch.tensor([[0.0, 0.0, 0.7]]))
    changed = state.clone()
    changed[:, 3:12] = (state[:, 3:12].reshape(-1, 3, 3) @ twist).flatten(1)
    action = torch.full((2, 22), 0.25, requires_grad=True)
    first, second = model(state, action), model(changed, action)
    torch.testing.assert_close(first[:, :3], second[:, :3], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(first[:, 12:], second[:, 12:], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        second[:, 3:12],
        (first[:, 3:12].reshape(-1, 3, 3) @ twist).flatten(1),
        atol=2e-6,
        rtol=2e-5,
    )
    (gradient,) = torch.autograd.grad(first[:, 2].sum(), action)
    plus, minus = action.detach().clone(), action.detach().clone()
    plus[0, 0] += 1e-3
    minus[0, 0] -= 1e-3
    numerical = (model(state, plus)[0, 2] - model(state, minus)[0, 2]) / 0.002
    torch.testing.assert_close(gradient[0, 0], numerical, rtol=0.03, atol=2e-4)


def test_world_frame_statistics_and_fitting_use_matching_error_axes():
    model = SharpaInHandWorldModel((8,), world_frame=True, axial_symmetry=True)
    state, _, _ = _bank()
    state[:, 3:12] = rotation_exp(torch.tensor([[0.0, torch.pi / 2, 0.0]])).flatten()
    action = state[:, 18:40].clone()
    target = state.clone()
    target[:, 0] += 0.001
    batch = SupervisedBatch(
        {"initial_state": state, "actions": action[:, None]},
        target[:, None],
        {"valid": torch.ones(2, 1, dtype=torch.bool)},
    )
    # One1 mm WORLD-X error, scaled by2 mm, root weight8 / total weight140.
    # Body-frame normalization would wrongly use the1 mm WORLD-Z scale.
    result = _WorldObjective().compute_loss(model, batch)
    torch.testing.assert_close(result.loss, torch.tensor(1 / 70), rtol=1e-4, atol=1e-7)
    increment = state.new_tensor(
        [0.001, 0.002, -0.003, 0.01, 0.02, -0.03, 0.1, 0.2, -0.3, 0.2, 0.1, 0.05]
    )
    target[:, :3] = state[:, :3] + increment[:3]
    target[:, 3:12] = (
        rotation_exp(increment[None, 3:6]) @ state[:, 3:12].reshape(-1, 3, 3)
    ).flatten(1)
    target[:, 12:18] = state[:, 12:18] + increment[6:12]
    model.update_normalization(state, action, target)
    torch.testing.assert_close(
        model.output_normalizer.mean[0, :12], increment, rtol=1e-4, atol=1e-7
    )


def test_packaged_apg_config_routes_and_saves_checkpoint(tmp_path: Path, monkeypatch):
    from embodichain.learning.rl.train import train_from_config

    model = SharpaInHandWorldModel((8,))
    states, contexts, limits = _bank()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format": "sharpa-inhand-world-model-v1",
            "hidden_dims": [8],
            "model": model.state_dict(),
            "initial_states": states,
            "initial_contexts": contexts,
            "joint_limits": limits,
        },
        checkpoint,
    )
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/sharpa_inhand/agents/apg.yaml"
    )
    config = yaml.safe_load(path.read_text())
    config["trainer"].update(
        device="cpu",
        num_envs=2,
        iterations=2,
        segment_length=2,
        update_horizon=2,
        save_frequency_updates=0,
    )
    config["trainer"]["learning_env"]["cfg"].update(
        model_checkpoint=str(checkpoint), max_episode_steps=2
    )
    config["policy"]["actor"]["network_cfg"]["hidden_sizes"] = [8]
    run_config = tmp_path / "apg.yaml"
    run_config.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    result = train_from_config(str(run_config))
    assert result["num_updates"] == 2
    assert Path(result["latest_checkpoint_path"]).is_file()
