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

"""Replay-seeded imagination, gradient boundaries and clip validity."""

from __future__ import annotations

import numpy as np
import torch
import pytest
from tensordict import TensorDict
from embodichain.compute.kinematics import TreeKinematics
from embodichain.data_pipeline.datasets.motion import MotionLibrary
from embodichain_models.dynamics.rwm import KinematicWorldModel, KinematicWorldModelCfg
from embodichain_models.dynamics.rwm.model import frozen_world_model
from embodichain_models.dynamics.rwm.robot_state import pack_robot_state
from embodichain_models.policies.motion_tracking import (
    MotionTrackingPolicy,
    MotionTrackingPolicyCfg,
)
from embodichain_tasks.locomotion.tracking._task import (
    MotionTrackingTask,
    MotionTrackingCfg,
)
from embodichain_tasks.locomotion.tracking.motion_model import (
    MotionModelEnv,
    MotionModelCfg,
    MotionModelStart,
)

URDF = """<robot name="one"><link name="base"/><link name="tip"/>
<joint name="joint" type="revolute"><parent link="base"/><child link="tip"/>
<origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint></robot>"""


def make_env(noise=False, compile_mode=None):
    qpos = np.zeros((8, 8), dtype=np.float32)
    qpos[:, 2] = 1
    qpos[:, 6] = 1
    motion = MotionLibrary(
        qpos, np.zeros((8, 7), dtype=np.float32), [8], ["stand"], ["joint"], 50
    )
    kin = TreeKinematics(URDF, ["joint"])
    task = MotionTrackingTask(
        kin,
        torch.zeros(1),
        torch.ones(1) * 0.5,
        torch.tensor([[-1.0, 1.0]]),
        torch.ones(1),
        ["base", "tip"],
        MotionTrackingCfg(history_steps=2, future_offsets=(1, 2)),
    )
    wm = KinematicWorldModel(
        URDF,
        KinematicWorldModelCfg(
            joint_names=("joint",),
            resting_position=(0.0,),
            action_dim=1,
            context_dim=5,
            hidden_dims=(16, 16),
        ),
    )
    env = MotionModelEnv(
        2,
        "cpu",
        model=wm,
        motion=motion,
        task=task,
        cfg=MotionModelCfg(
            horizon=3,
            observation_noise=noise,
            goal_position_noise=0.05 if noise else 0.0,
            action_noise=0.0,
            compile_mode=compile_mode,
        ),
    )
    frame = torch.tensor([0, 6])
    state = motion.frame(frame)
    prev = torch.zeros(2, 1)
    starts = MotionModelStart(
        pack_robot_state(state),
        torch.zeros(2, 5),
        frame,
        task.observation(state, prev, torch.zeros_like(prev))[:, None].expand(
            -1, 2, -1
        ),
        prev,
        torch.ones(2, dtype=torch.bool),
    )
    env.set_starts(starts)
    return env, wm


def test_model_rollout_preserves_early_action_gradient_and_clip_end_validity():
    torch.manual_seed(14)
    env, wm = make_env()
    env.reset()
    actions = torch.full((3, 2, 1), 0.02, requires_grad=True)
    with frozen_world_model(wm):
        rewards = []
        valid = []
        for action in actions:
            _, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            valid.append(info["next_valid"])
            assert not terminated.any() and not truncated.any()
        torch.testing.assert_close(
            torch.stack(valid),
            torch.tensor([[True, True], [True, False], [True, False]]),
        )
        torch.stack(rewards)[-1].sum().backward()
    assert torch.isfinite(actions.grad).all() and actions.grad[0].abs().sum() > 0
    assert all(p.grad is None for p in wm.parameters())
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(actions[0])


def test_detach_preserves_noisy_observation_and_reference_progress():
    env, wm = make_env(noise=True)
    first, _ = env.reset(seed=43)
    before, _, _, _, _ = env.step(torch.ones(2, 1, requires_grad=True) * 0.01)
    before = before.clone()
    state = env.state.detach().clone()
    after = env.detach_state()
    for key in ("policy", "critic"):
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
        assert not after[key].requires_grad
    torch.testing.assert_close(env.state, state, rtol=0, atol=0)
    _, _, _, _, info = env.step(torch.zeros(2, 1))
    assert info["next_valid"].tolist() == [True, False]


def test_policy_actor_and_critic_have_independent_inputs_and_statistics():
    cfg = MotionTrackingPolicyCfg(
        history_dim=18,
        goal_dim=34,
        critic_dim=57,
        action_dim=1,
        latent_dim=4,
        encoder_hidden=(8,),
        decoder_hidden=(8,),
        critic_hidden=(8,),
    )
    p = MotionTrackingPolicy(cfg)
    x = torch.randn(3, 52, requires_grad=True)
    p.update_actor_normalization(x.detach())
    count = p.critic_obs_normalizer.count.clone()
    p.get_differentiable_action(TensorDict({"obs": x}, [3]))["action"].sum().backward()
    assert x.grad[:, :18].abs().sum() > 0 and x.grad[:, 18:].abs().sum() > 0
    assert all(v.grad is None for v in p.critic.parameters())
    assert p.critic_obs_normalizer.count == count
    p.update_critic_normalization(torch.randn(3, 57))
    assert p.critic_obs_normalizer.count == 3
    clone = MotionTrackingPolicy(cfg)
    clone.load_state_dict(p.state_dict())
    torch.testing.assert_close(clone.actor(x), p.actor(x), rtol=0, atol=0)


def test_compiled_transition_matches_eager_values_noise_and_multistep_gradients(
    monkeypatch,
):
    compile_fn = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: compile_fn(fn, backend="aot_eager", fullgraph=True),
    )
    results = []
    for mode in (None, "default"):
        torch.manual_seed(14)
        env, wm = make_env(noise=True, compile_mode=mode)
        env.cfg.action_noise = 0.1
        env.reset(seed=42)
        actions = torch.full((3, 2, 1), 0.02, requires_grad=True)
        with frozen_world_model(wm):
            rewards = []
            for action in actions:
                observation, reward, _, _, info = env.step(action)
                rewards.append(reward)
            objective = torch.stack(rewards).sum() + 0.01 * observation["policy"].sum()
            gradient = torch.autograd.grad(objective, actions)[0]
        results.append(
            (
                env.state.detach(),
                env.history.detach(),
                observation["policy"].detach(),
                info["continuation"].detach(),
                objective.detach(),
                gradient,
            )
        )
    for eager, compiled in zip(*results):
        torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("compiled", [False, True])
def test_task_analytics_preserves_values_and_state_gradients(compiled):
    """The physical task fusion retains the independently callable objectives."""
    env, _ = make_env()
    task = env.task
    frame = torch.tensor([1, 3])
    reference = env.motion.frame(frame)
    future = env.motion.future(frame, task.future_offsets)
    state = {k: v.clone() for k, v in reference.items()}
    state["q"] = (state["q"] + 0.17).requires_grad_()
    state["v"] = (state["v"] + 0.12).requires_grad_()
    action = torch.full((2, 1), 0.08, requires_grad=True)
    previous = torch.full_like(action, -0.03)
    bias = torch.full_like(action, 0.01)
    expected_reward, expected_terms = task.reward(state, reference, action, previous)
    expected_failed, expected_continuation = task.termination(state, reference)
    expected = {
        "reward": expected_reward,
        "reward_terms": expected_terms,
        "failed": expected_failed,
        "continuation": expected_continuation,
        "observation": task.observation(state, action, bias),
        "goal": task.goal(state, future),
        "features": task.features(state),
    }
    evaluate = task.analytics
    if compiled:
        evaluate = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    actual = evaluate(state, reference, future, action, previous, bias)
    torch.testing.assert_close(actual, expected)
    inputs = (state["q"], state["v"], action)
    expected_gradient = torch.autograd.grad(
        expected_reward.sum(), inputs, retain_graph=True
    )
    actual_gradient = torch.autograd.grad(actual["reward"].sum(), inputs)
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("compiled", [False, True])
def test_physical_transition_excludes_nonfinite_rows(compiled):
    """Divergent state rows terminate with finite features and zero return."""
    from types import SimpleNamespace
    from embodichain_tasks.locomotion.tracking.g1_motion import UnitreeG1MotionEnv

    env, _ = make_env()
    frame = torch.tensor([1, 3])
    reference = env.motion.frame(frame)
    state = {key: value.clone() for key, value in reference.items()}
    state["q"][0] += 0.17
    state["q"][1] = float("nan")
    state["q"].requires_grad_()
    future = env.motion.future(frame, env.task.future_offsets)
    action = torch.full((2, 1), 0.08)
    zero = torch.zeros_like(action)
    owner = SimpleNamespace(task=env.task)
    evaluate = lambda *args: UnitreeG1MotionEnv._compute_transition(owner, *args)
    if compiled:
        evaluate = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    result = evaluate(state, reference, future, action, zero, zero)
    assert result["valid"].tolist() == [True, False]
    assert result["failed"][1]
    assert result["reward"][1] == 0
    assert result["continuation"][1] == 0
    for key in ("observation", "goal", "features"):
        assert torch.isfinite(result[key]).all()
    for key, value in reference.items():
        torch.testing.assert_close(result["state"][key][1], value[1])
    expected, _ = env.task.reward(result["state"], reference, action, zero)
    torch.testing.assert_close(result["reward"][0], expected[0])
    gradient = torch.autograd.grad(result["reward"].sum(), state["q"])[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]))


def test_physical_episode_metrics_own_storage_across_task_invocations(monkeypatch):
    """Reset metrics remain writable when a compiled producer reuses its output."""
    from types import SimpleNamespace
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain_tasks.locomotion.tracking.g1_motion import UnitreeG1MotionEnv

    model_env, _ = make_env()
    env = object.__new__(UnitreeG1MotionEnv)
    env.motion, env.task = model_env.motion, model_env.task
    env.motion_cfg = SimpleNamespace(gamma=0.99)
    env.motion_frame = torch.tensor([1, 3])
    state = env.motion.frame(env.motion_frame)
    env.read_state = lambda: state
    zero = torch.zeros(2, 1)
    env.position_action = SimpleNamespace(
        raw_actions=zero, previous_raw_actions=zero, position_bias=zero
    )
    env._episode_steps = torch.zeros(2, dtype=torch.long)
    env._episode_discounted_return = torch.zeros(2)
    env._reward = torch.zeros(2)
    env._failed = torch.zeros(2, dtype=torch.bool)
    env._continuation = torch.ones(2)
    env._refresh = lambda *args, **kwargs: None
    outputs = {}

    def evaluate(*args):
        outputs.update(env._compute_transition(*args))
        return outputs

    env._task_analytics = evaluate
    monkeypatch.setattr(EmbodiedEnv, "_update_sim_state", lambda *args, **kw: None)
    env._update_sim_state()
    expected_reward = env._reward.clone()
    outputs["reward"].zero_()
    outputs["failed"].fill_(True)
    outputs["continuation"].zero_()
    torch.testing.assert_close(env._reward, expected_reward)
    assert not env._failed.any()
    assert (env._continuation > 0).all()
    env._reward.zero_()
    env._continuation.fill_(1)
    assert not outputs["continuation"].any()


def test_configured_model_policy_update_shares_workflow_resources():
    from embodichain.learning.rl.policies import build_policy
    from embodichain.learning.rl.algo import build_algo
    from embodichain.learning.rl.differentiable_trainer import (
        DifferentiableTrainer,
        DifferentiableTrainerCfg,
    )
    from embodichain.learning.rl.runtime import build_learning_policy_runtime

    source, model = make_env()
    with pytest.raises(ValueError, match="dimensions differ"):
        build_policy(
            {"name": "motion_tracking", "cfg": {"action_dim": 29}},
            source.single_observation_space,
            source.single_action_space,
            torch.device("cpu"),
        )
    task = source.task
    config = {
        "trainer": {
            "learning_env": {
                "name": "MotionTrackingModelRL",
                "cfg": {"horizon": 3, "observation_noise": False},
            },
        },
        "policy": {
            "name": "motion_tracking",
            "cfg": {
                "history_dim": task.observation_dim * task.cfg.history_steps,
                "goal_dim": task.goal_dim,
                "critic_dim": source.single_observation_space["critic"].shape[0],
                "action_dim": 1,
                "latent_dim": 4,
                "encoder_hidden": [8],
                "decoder_hidden": [8],
                "critic_hidden": [8],
            },
        },
        "algorithm": {
            "name": "apg_actor_critic",
            "cfg": {
                "optimizer": {"learning_rate": 0.002},
                "critic_optimizer": {"learning_rate": 0.003},
                "critic_epochs": 1,
                "critic_minibatches": 1,
            },
        },
    }
    starts = source._start
    provider = lambda: starts
    resources = {
        "model": model,
        "motion": source.motion,
        "task": task,
        "start_provider": provider,
    }

    runtime = build_learning_policy_runtime(
        config, device=torch.device("cpu"), num_envs=2, resources=resources
    )
    algorithm = build_algo(
        config["algorithm"]["name"],
        config["algorithm"]["cfg"],
        runtime.policy,
        runtime.device,
    )
    trainer = DifferentiableTrainer(
        DifferentiableTrainerCfg(rollout_mode="complete", deterministic_actions=True),
        runtime.env,
        runtime.policy,
        algorithm,
    )
    assert runtime.env is not source
    assert runtime.env.model is model
    assert runtime.env.motion is source.motion
    assert runtime.env.task is task
    assert runtime.env.start_provider is provider
    assert runtime.env.cfg.horizon == 3
    assert trainer.algorithm.critic_optimizer.param_groups[0]["lr"] == 0.003
    original_actor = [
        parameter.detach().clone() for parameter in runtime.policy.actor.parameters()
    ]
    original_model = [parameter.detach().clone() for parameter in model.parameters()]
    with frozen_world_model(model):
        trainer.update_once()
    assert trainer.num_updates == 1
    assert trainer.global_step == 6
    assert any(
        not torch.equal(old, new)
        for old, new in zip(original_actor, runtime.policy.actor.parameters())
    )
    for old, new in zip(original_model, model.parameters()):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
        assert new.grad is None
    runtime.close()
