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

"""Shared skill recipes and RewardManager lifecycle regression cases."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch

from embodichain.lab.sim.atomic_skills.reward_profiles import (
    default_skill_reward_profile,
)
from embodichain.lab.gym.envs.managers.reward_composition import compose_reward_terms
from embodichain.lab.gym.envs.managers.reward_manager import RewardManager
from embodichain.lab.gym.envs.managers.reward_context import RewardTransitionContext


def _frame(values, revision=0):
    return SimpleNamespace(
        revision=revision,
        episode_ids=(0, 0),
        fields={
            "score": SimpleNamespace(
                values=torch.tensor(values), valid=torch.ones(2, dtype=torch.bool)
            )
        },
    )


def _manager(program=False):
    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), sim=SimpleNamespace())
    skill = {"skill_id": "move_joints", "metrics": {"joints": {"field": "score"}}}
    if program:
        skill["stage"] = "move"
        env.task_program_progress = SimpleNamespace(
            ordered_progress=lambda: (torch.zeros(2, dtype=torch.long), 1),
            indices=lambda stage: (0,),
        )
    env.reward_manager = RewardManager(
        compose_reward_terms({"discount": 0.9, "skills": [skill]}, program=program), env
    )
    return env.reward_manager


def test_direct_and_program_use_identical_recipe_and_repeat_reads_are_pure():
    rewards = []
    for program in (False, True):
        manager = _manager(program)
        manager.initialize_observation(_frame([0.2, 0.4]), [0, 1])
        manager.transition_context = RewardTransitionContext(
            _frame([0.8, 0.6], 1), torch.tensor([False, True])
        )
        value, _ = manager.compute({}, torch.zeros(2, 1), {})
        torch.testing.assert_close(value, torch.tensor([0.52, -0.4]))
        value.zero_()
        repeat, _ = manager.compute({}, torch.zeros(2, 1), {})
        torch.testing.assert_close(repeat, torch.tensor([0.52, -0.4]))
        rewards.append(repeat)
    torch.testing.assert_close(*rewards)


def test_partial_reset_changes_only_selected_reward_baselines():
    manager = _manager()
    manager.initialize_observation(_frame([0.2, 0.4]), [0, 1])
    manager.reset([0])
    manager.initialize_observation(_frame([0.5, 0.9]), [0])
    manager.transition_context = RewardTransitionContext(
        _frame([0.8, 0.6], 2), torch.zeros(2, dtype=torch.bool)
    )
    value, _ = manager.compute({}, torch.zeros(2, 1), {})
    torch.testing.assert_close(value, torch.tensor([0.22, 0.14]))


@pytest.mark.parametrize(
    "skill,roles,expected",
    [
        ("pick_up", {"reach": 0.6, "held": 1.0, "lifted": 0.5}, 0.7),
        ("place", {"pose": 0.6, "released": 1.0, "stable": 0.5}, 0.5),
        ("move_held_object", {"pose": 0.8, "held": 0.0}, 0.0),
        ("move_end_effector", {"position": 0.8, "orientation": 0.2}, 0.5),
        ("move_joints", {"joints": 0.7}, 0.7),
    ],
)
def test_recipe_values_and_invalid_measurements(skill, roles, expected):
    profile = default_skill_reward_profile(skill)
    scores = {
        name: torch.tensor([value, float("nan")]) for name, value in roles.items()
    }
    value, valid = profile.potential(scores)
    assert value.tolist() == pytest.approx([expected, 0.0])
    assert valid.tolist() == [True, False]


def test_program_switch_preserves_global_potential_boundary():
    manager = _manager(True)
    manager.initialize_observation(_frame([0.8, 0.6]), [0, 1])
    manager._env.task_program_progress.ordered_progress = lambda: (
        torch.ones(2, dtype=torch.long),
        1,
    )
    manager.transition_context = RewardTransitionContext(
        _frame([0.0, 0.0], 1), torch.tensor([False, True])
    )
    value, _ = manager.compute({}, torch.zeros(2, 1), {})
    torch.testing.assert_close(value, torch.tensor([0.1, -0.6]))


@pytest.mark.parametrize("discount", [None, True, 1.1])
def test_composition_rejects_implicit_or_invalid_discount(discount):
    with pytest.raises(ValueError, match="discount"):
        compose_reward_terms({"discount": discount}, program=True)


def test_unavailable_reward_evidence_terminates_without_erasing_potential():
    from embodichain.lab.gym.envs.managers.termination_manager import TerminationManager
    from embodichain.lab.gym.envs.managers.cfg import RLTaskCfg

    manager = _manager()
    owner = SimpleNamespace(episode_ids=(0, 0))
    manager._env.episode_context = owner
    manager.initialize_observation(_frame([0.2, 0.4]), [0, 1])
    termination = TerminationManager(RLTaskCfg(), manager._env)
    termination.compute()
    frame = _frame([float("nan"), 0.6], 1)
    frame.episode_owner = owner
    valid = manager.prepare_observation(frame)
    termination.invalidate_reward_evidence(valid)
    assert termination.terminated.tolist() == [True, False]
    manager.transition_context = RewardTransitionContext(frame, termination.terminated)
    value, _ = manager.compute({}, torch.zeros(2, 1), {})
    torch.testing.assert_close(value, torch.tensor([-0.2, 0.14]))
    assert termination.snapshot()["invalid_measurement"].tolist() == [True, False]


def test_foreign_and_repeated_acquisitions_are_rejected_before_rewards():
    manager = _manager()
    owner = SimpleNamespace(episode_ids=(0, 0))
    manager._env.episode_context = owner
    manager.initialize_observation(_frame([0.2, 0.4]), [0, 1])
    frame = _frame([0.5, 0.6], 1)
    frame.episode_owner = owner
    assert manager.prepare_observation(frame).tolist() == [True, True]
    assert manager.prepare_observation(frame).tolist() == [False, False]
    frame.revision = 2
    frame.episode_owner = SimpleNamespace(episode_ids=(0, 0))
    assert manager.prepare_observation(frame).tolist() == [False, False]


def test_invalid_initial_measured_potential_still_yields_finite_terminal_reward():
    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), sim=SimpleNamespace())
    cfg = compose_reward_terms(
        {
            "discount": 0.9,
            "skills": [
                {
                    "skill_id": "move_joints",
                    "profile": "measured_potential",
                    "metrics": {"potential": {"field": "score", "index": 0}},
                }
            ],
        },
        program=False,
    )
    manager = env.reward_manager = RewardManager(cfg, env)
    frame = _frame([[float("nan"), 0.0], [0.5, 0.0]])
    manager.initialize_observation(frame, [0, 1])
    manager.transition_context = RewardTransitionContext(
        _frame([[0.2, 0.0], [0.6, 0.0]], 1), torch.tensor([True, False])
    )
    value, _ = manager.compute({}, torch.zeros(2, 1), {})
    torch.testing.assert_close(value, torch.tensor([0.0, 0.04]))


def test_named_templates_are_shared_and_saved_revisions_are_enforced(monkeypatch):
    from embodichain.lab.sim.atomic_skills.reward_profiles import (
        SkillRewardProfile,
        register_skill_reward_profile,
        _PROFILES,
    )

    monkeypatch.setitem(_PROFILES, "test_skill", {})
    register_skill_reward_profile(
        "product", SkillRewardProfile("test_skill", 1, ("a", "b"), "product")
    )
    declaration = {
        "discount": 0.9,
        "skills": [
            {
                "skill_id": "test_skill",
                "profile": "product",
                "metrics": {"a": {"field": "score"}, "b": {"field": "score"}},
            }
        ],
    }
    terms = compose_reward_terms(declaration, program=False)
    assert terms["composed_potential"].extra["profile_revisions"] == [
        ("test_skill", "product", 1)
    ]
    declaration["profiles"] = [["test_skill", "product", 1]]
    compose_reward_terms(declaration, program=False)
    monkeypatch.setitem(
        _PROFILES["test_skill"],
        "product",
        SkillRewardProfile("test_skill", 2, ("a", "b"), "product"),
    )
    with pytest.raises(ValueError, match="saved reward contract"):
        compose_reward_terms(declaration, program=False)


@pytest.mark.parametrize("target", [[float("inf")], [float("nan")]])
def test_nonfinite_metric_targets_reject_before_reward_binding(target):
    with pytest.raises(ValueError, match="target.*finite"):
        compose_reward_terms(
            {
                "discount": 0.99,
                "skills": [
                    {
                        "skill_id": "move_joints",
                        "metrics": {
                            "joints": {
                                "field": "joint_position",
                                "target": target,
                                "scale": 0.1,
                            }
                        },
                    }
                ],
            },
            program=False,
        )
