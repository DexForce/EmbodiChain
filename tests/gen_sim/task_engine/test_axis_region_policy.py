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

"""Axis-region reinterpretation is explicit and local to derived GenSim ends."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import actions
from embodichain.gen_sim.task_engine._task_program.adaptive_grasp import (
    ADAPTIVE_GRASP,
    grasp_strategies,
    select_grasp,
)
from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_actions.affordance_sampling import AffordanceSample
from embodichain.lab.sim.atomic_actions.primitives.hand_over import HandOverOptions
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp, PickUpOptions


def test_default_affordance_retains_legacy_generator_signature() -> None:
    calls = []

    def generate(
        *,
        mesh_vertices,
        mesh_triangles,
        obj_poses,
        approach_direction,
        obj_longest_axis,
        is_positive_part,
    ):
        calls.append((obj_longest_axis, is_positive_part))
        return [(obj_poses.clone(), torch.zeros(1))]

    affordance = AntipodalAffordance()
    generator = SimpleNamespace(get_valid_grasp_poses=generate)
    candidates = affordance.get_grasp_candidates(
        generator, torch.eye(4)[None], torch.tensor([0.0, 0.0, -1.0])
    )
    assert candidates.valid.tolist() == [[True]]
    assert calls == [(None, True)]


def test_affordance_policy_is_call_local_and_never_retries() -> None:
    generator = SimpleNamespace(
        get_valid_grasp_poses=Mock(return_value=[(torch.eye(4)[None], torch.zeros(1))])
    )
    affordance = AntipodalAffordance()
    args = (generator, torch.eye(4)[None], torch.tensor([0.0, 0.0, -1.0]))
    affordance.get_grasp_candidates(*args, axis_region_policy="supported_end")
    affordance.get_grasp_candidates(*args)
    first, second = generator.get_valid_grasp_poses.call_args_list
    assert first.kwargs["axis_region_policy"] == "supported_end"
    assert "axis_region_policy" not in second.kwargs
    generator.get_valid_grasp_poses.side_effect = TypeError("unsupported policy")
    with pytest.raises(TypeError, match="unsupported policy"):
        affordance.get_grasp_candidates(*args, axis_region_policy="supported_end")
    assert generator.get_valid_grasp_poses.call_count == 3


@pytest.mark.parametrize("policy", ["mesh_extent", "supported_end"])
def test_select_grasp_forwards_only_explicit_policy_without_new_sampling(
    policy,
) -> None:
    affordance = AntipodalAffordance()
    pose = torch.eye(4)[None]
    generator = SimpleNamespace(
        get_valid_grasp_poses=Mock(return_value=[(pose, torch.tensor([0.25]))])
    )
    affordance.sample_candidates = Mock(wraps=affordance.sample_candidates)
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    choices = grasp_strategies(axis, pose, pose, ends=True)
    context = SimpleNamespace(affordance_sampling=None, env_ids=torch.tensor([0]))
    state = torch.random.get_rng_state().clone()
    sample, direction = select_grasp(
        affordance, generator, pose, choices, context, "same", axis_region_policy=policy
    )
    generator.get_valid_grasp_poses.assert_called_once()
    kwargs = generator.get_valid_grasp_poses.call_args.kwargs
    assert kwargs.get("axis_region_policy", "mesh_extent") == policy
    assert ("axis_region_policy" in kwargs) == (policy == "supported_end")
    assert kwargs["obj_longest_axis"] is choices[0].axis
    assert kwargs["is_positive_part"] is choices[0].positive
    assert sample.metadata["candidate_ids"] == [0]
    assert (
        sample.metadata["adaptive_grasp"].get("axis_region_policy", "mesh_extent")
        == policy
    )
    assert torch.equal(direction, choices[0].direction)
    assert torch.equal(state, torch.random.get_rng_state())
    affordance.sample_candidates.assert_called_once()


@pytest.mark.parametrize("policy", [None, "whole_mesh", True])
def test_unknown_policy_fails_before_generator(policy) -> None:
    affordance = AntipodalAffordance()
    generator = SimpleNamespace(get_valid_grasp_poses=Mock())
    with pytest.raises(ValueError, match="axis_region_policy"):
        affordance.get_grasp_candidates(
            generator,
            torch.eye(4)[None],
            torch.tensor([0.0, 0.0, -1.0]),
            axis_region_policy=policy,
        )
    generator.get_valid_grasp_poses.assert_not_called()


@pytest.mark.parametrize("mode", ["ends", "ordinary", "pour"])
@pytest.mark.parametrize("part", ["center", "top", "bottom"])
@pytest.mark.parametrize("hard_rule", [False, True])
def test_source_policy_only_marks_derived_center_ends(
    monkeypatch, mode, part, hard_rule
) -> None:
    affordance = AntipodalAffordance(custom_config={ADAPTIVE_GRASP: mode})
    if hard_rule:
        affordance.set_custom_config(actions._PICK_GRASP_RULE, ("object", "grasp"))
    action = actions.GenSimPickUp()
    action._planning_services = SimpleNamespace(
        robot=SimpleNamespace(compute_fk=Mock(return_value=torch.eye(4)[None])),
        grasp_pose_generator=Mock(return_value=object()),
    )
    select = Mock(return_value=object())
    monkeypatch.setattr(action, "_select_adaptive_pick", select)
    parent = Mock(return_value=object())
    monkeypatch.setattr(PickUp, "_resolve_grasp_pose", parent)
    monkeypatch.setattr(
        actions, "observed_axis", lambda *_: torch.tensor([[0.0, 0.0, 1.0]])
    )
    options = PickUpOptions(pick_object_part=part)
    token = actions._PICK_SELECTION.set(actions._PickSelection(request=object()))
    try:
        if hard_rule:
            with pytest.raises(ValueError, match="call-local grasp filter"):
                action._resolve_grasp_pose(
                    SimpleNamespace(affordance=affordance, entity_id="object"),
                    torch.eye(4)[None],
                    torch.zeros(1, 1),
                    SimpleNamespace(control_part="arm"),
                    "grasp",
                    options,
                    torch.tensor([0.0, 0.0, -1.0]),
                    object(),
                    sample_key="source",
                )
        else:
            action._resolve_grasp_pose(
                SimpleNamespace(affordance=affordance, entity_id="object"),
                torch.eye(4)[None],
                torch.zeros(1, 1),
                SimpleNamespace(control_part="arm"),
                "grasp",
                options,
                torch.tensor([0.0, 0.0, -1.0]),
                object(),
                sample_key="source",
            )
    finally:
        actions._PICK_SELECTION.reset(token)
    if part == "center" and not hard_rule:
        select.assert_called_once()
        assert select.call_args.kwargs.get("axis_region_policy", "mesh_extent") == (
            "supported_end" if mode == "ends" else "mesh_extent"
        )
    else:
        select.assert_not_called()


@pytest.mark.parametrize("part", ["center", "top", "bottom"])
@pytest.mark.parametrize("hard_rule", [False, True])
def test_receiver_policy_only_marks_derived_center_ends(
    monkeypatch, part, hard_rule
) -> None:
    affordance = AntipodalAffordance()
    if hard_rule:
        affordance.set_custom_config(actions._PICK_GRASP_RULE, ("object", "grasp"))
    action = actions.GenSimHandOver()
    action._planning_services = SimpleNamespace(
        robot=SimpleNamespace(compute_fk=Mock(return_value=torch.eye(4)[None])),
        grasp_pose_generator=Mock(return_value=object()),
    )
    monkeypatch.setattr(
        actions, "observed_axis", lambda *_: torch.tensor([[0.0, 0.0, 1.0]])
    )
    sample = AffordanceSample(
        torch.tensor([True]),
        torch.eye(4)[None],
        {"candidate_ids": [0], "adaptive_grasp": {}},
    )
    select = Mock(return_value=(sample, torch.tensor([[0.0, 0.0, -1.0]])))
    monkeypatch.setattr(actions, "select_grasp", select)
    resources = SimpleNamespace(
        second=SimpleNamespace(arm=SimpleNamespace(joint_ids=(0,), control_part="arm"))
    )
    selection = actions._ReceiveSelection(
        resources,
        SimpleNamespace(object_to_eef=torch.eye(4)[None]),
        torch.tensor([False]),
        SimpleNamespace(skill_options=HandOverOptions(receive_pick_object_part=part)),
        None,
    )
    token = actions._RECEIVE_SELECTION.set(selection)
    try:
        action._resolve_grasp(
            affordance,
            torch.eye(4)[None],
            torch.tensor([[0.0, 0.0, -1.0]]),
            "grasp",
            obj_longest_axis=None,
            is_positive_part=True,
            context=SimpleNamespace(
                robot=SimpleNamespace(qpos=torch.zeros(1, 1)), affordance_sampling=None
            ),
            sample_key="receive",
        )
    finally:
        actions._RECEIVE_SELECTION.reset(token)
    select.assert_called_once()
    assert select.call_args.kwargs.get("axis_region_policy", "mesh_extent") == (
        "supported_end" if part == "center" and not hard_rule else "mesh_extent"
    )
