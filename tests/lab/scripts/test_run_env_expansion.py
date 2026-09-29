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

"""Tests for task-bound Expansion Policy resolution in ``run-task``."""

from __future__ import annotations

import pytest

from embodichain.lab.scripts.run_env import (
    CollectionPlan,
    CollectionSelection,
    _create_parser,
    _resolve_expansion_request,
    resolve_collection_plan,
)
from embodichain.lab.gym.utils.gym_utils import _apply_expansion_runtime_overlay
from embodichain.lab.sim.motion.expansion import CombinedExpansionProfile
from embodichain.utils.config_paths import resolve_config_path
from embodichain.utils.utility import load_config


def _resolve(path: str):
    args = _create_parser().parse_args(
        ["--gym-config", path, "--headless", "--device", "cpu"]
    )
    config = load_config(resolve_config_path(path))
    return _resolve_expansion_request(args, config)


def test_repeated_pick_place_policy_component_merges_task_binding() -> None:
    request = _resolve(
        "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/"
        "task.ur5.yaml"
    )
    assert request is not None
    profile, candidate_indices, _ = request
    assert isinstance(profile, CombinedExpansionProfile)
    assert profile.source.source_id == "repeated_cube_pick_place"
    assert profile.source.call_count == 6
    assert profile.scene_randomization.reference_family_count == 4
    assert profile.trajectory.spatial.joint_offset_scale == 0.005
    assert candidate_indices == ()
    plan = resolve_collection_plan(
        args=_create_parser().parse_args(
            ["--gym-config", "task.yaml", "--headless", "--device", "cpu"]
        ),
        gym_config={},
        expansion_collection={"target_episodes": 64},
    )
    assert plan.target_episodes == 64
    assert plan.selection.take(0, 4) == (0, 1, 2, 3)


def test_open_drawer_policy_component_merges_phase_binding() -> None:
    request = _resolve(
        "embodichain_tasks/configs/tasks/manipulation/open_drawer/" "task.ur5.yaml"
    )
    assert request is not None
    profile, candidate_indices, _ = request
    assert isinstance(profile, CombinedExpansionProfile)
    assert profile.source.source_id == "slide_open_drawer"
    assert profile.source.call_count == 1
    assert profile.source.phase_permissions["pull"] == ("joint_residual",)
    assert profile.visual.enabled is False
    assert candidate_indices == ()


def test_task_expansion_config_accepts_local_candidate_override() -> None:
    task_path = resolve_config_path(
        "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/"
        "task.ur5.yaml"
    )
    config = load_config(task_path)
    config["expansion"]["collection"] = {
        "target_episodes": 1,
        "selection": {"mode": "explicit", "recipe_indices": [32]},
    }
    args = _create_parser().parse_args(
        ["--gym-config", str(task_path), "--headless", "--device", "cpu"]
    )

    request = _resolve_expansion_request(args, config)

    assert request is not None
    assert request[1] == ()
    assert request[2].name == "repeated_pick_place.yaml"


def test_collection_plan_separates_target_from_parallel_capacity() -> None:
    plan = CollectionPlan(64, 3, CollectionSelection("sequential"))
    assert plan.batch_count(16) == 4
    assert plan.selection.take(16, 4) == (16, 17, 18, 19)


def test_collection_plan_rejects_explicit_target_mismatch() -> None:
    with pytest.raises(ValueError, match="equal target_episodes"):
        CollectionPlan(
            3,
            selection=CollectionSelection("explicit", recipe_indices=(0, 1)),
        )


def test_collection_plan_prefers_task_target_over_legacy_environment_default() -> None:
    args = _create_parser().parse_args(
        ["--gym-config", "task.yaml", "--headless", "--device", "cpu"]
    )
    plan = resolve_collection_plan(
        args,
        {"max_episodes": 1, "collection": {"target_episodes": 4}},
    )
    assert plan.target_episodes == 4


def test_collection_plan_rejects_conflicting_cli_target() -> None:
    args = _create_parser().parse_args(
        [
            "--gym-config",
            "task.yaml",
            "--headless",
            "--device",
            "cpu",
            "--max_episodes",
            "5",
        ]
    )
    with pytest.raises(ValueError, match="CLI --max-episodes"):
        resolve_collection_plan(
            args,
            {"collection": {"target_episodes": 4}},
        )


def test_collection_plan_rejects_conflicting_task_and_expansion_targets() -> None:
    args = _create_parser().parse_args(
        ["--gym-config", "task.yaml", "--headless", "--device", "cpu"]
    )
    with pytest.raises(ValueError, match="declared more than once"):
        resolve_collection_plan(
            args,
            {"collection": {"target_episodes": 4}},
            expansion_collection={"target_episodes": 8},
        )


def test_collection_plan_maps_legacy_max_episodes_as_fallback() -> None:
    args = _create_parser().parse_args(
        ["--gym-config", "task.yaml", "--headless", "--device", "cpu"]
    )
    plan = resolve_collection_plan(args, {"max_episodes": 5})

    assert plan.target_episodes == 5
    assert plan.selection.mode == "sequential"


def test_task_expansion_config_applies_runtime_overlay() -> None:
    task_path = resolve_config_path(
        "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/"
        "task.ur5.yaml"
    )
    with pytest.raises(ValueError, match="unsupported fields.*simulation"):
        _apply_expansion_runtime_overlay(
            {
                "num_envs": 1,
                "arena_space": 1.0,
                "simulation": {"light": {"direct": []}},
                "env": {"events": {}},
                "expansion": {"runtime": {"simulation": {"light": {}}}},
            },
            base_dir=task_path.parent,
        )

    resolved = _apply_expansion_runtime_overlay(
        {
            "num_envs": 1,
            "arena_space": 1.0,
            "env": {"events": {}},
            "expansion": {
                "runtime": {
                    "num_envs": 16,
                    "arena_space": 2.5,
                    "env": {"events": {"expansion_profile_reset": {"mode": "reset"}}},
                }
            },
        },
        base_dir=task_path.parent,
    )

    assert resolved["num_envs"] == 16
    assert resolved["arena_space"] == 2.5
    assert resolved["env"]["events"]["expansion_profile_reset"]["mode"] == "reset"


def test_cli_expansion_profile_overrides_task_expansion_policy() -> None:
    task_path = resolve_config_path(
        "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/"
        "task.ur5.yaml"
    )
    policy_path = resolve_config_path(
        "embodichain_tasks/configs/components/expansion_policies/"
        "task_program_episode.yaml"
    )
    args = _create_parser().parse_args(
        [
            "--gym-config",
            str(task_path),
            "--expansion-profile",
            str(policy_path),
            "--headless",
            "--device",
            "cpu",
        ]
    )

    request = _resolve_expansion_request(args, load_config(task_path))

    assert request is not None
    assert request[0].source.source_id == "task_program_source"
