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

from embodichain.lab.scripts.run_task import (
    CollectionPlan,
    CollectionSelection,
    _create_parser,
    _pad_expansion_recipe_indices,
    _resolve_expansion_request,
    _run_expansion,
    _validate_expansion_collection_plan,
    resolve_collection_plan,
)
from embodichain.lab.gym.utils.gym_utils import (
    _apply_expansion_runtime_overlay,
    build_env_cfg_from_args,
)
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
        "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/"
        "task.ur5.yaml"
    )
    assert request is not None
    profile, candidate_indices, _, collection = request
    assert isinstance(profile, CombinedExpansionProfile)
    assert profile.source.source_id == "repeated_cube_pick_place"
    assert profile.source.call_count == 6
    assert profile.scene_randomization.reference_family_count == 1
    assert not profile.scene_randomization.enabled
    assert not profile.visual.enabled
    assert profile.trajectory.spatial.joint_offset_scale == 0.005
    # Without a smaller tolerance the declared amplitude would measure as a
    # duplicate of the reference and the variants would collapse into nominal.
    assert profile.trajectory.spatial.dedup_tolerance == 0.002
    assert (
        profile.trajectory.spatial.dedup_tolerance
        < profile.trajectory.spatial.joint_offset_scale
    )
    assert candidate_indices == ()
    assert collection["target_episodes"] == 16
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
        "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_slide/"
        "task.ur5.yaml"
    )
    assert request is not None
    profile, candidate_indices, _, collection = request
    assert isinstance(profile, CombinedExpansionProfile)
    assert profile.source.source_id == "slide_open_drawer"
    assert profile.source.call_count == 2
    assert not profile.source.phase_permissions
    assert profile.source.affordance_call_indices == (0, 1)
    assert profile.source.trajectory_call_indices == (0, 1)
    assert profile.visual.enabled is False
    assert (
        profile.trajectory.spatial.dedup_tolerance
        < profile.trajectory.spatial.joint_offset_scale
    )
    assert candidate_indices == ()
    assert collection["target_episodes"] == 48


def test_task_expansion_config_accepts_local_candidate_override() -> None:
    task_path = resolve_config_path(
        "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/"
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
    assert request[3]["target_episodes"] == 1
    assert request[3]["selection"]["recipe_indices"] == [32]


def test_collection_plan_cli_recipe_override_uses_explicit_mode() -> None:
    args = _create_parser().parse_args(
        [
            "--gym-config",
            "task.yaml",
            "--max_episodes",
            "4",
            "--expansion-recipe-indices",
            "0",
            "16",
            "32",
            "48",
            "--headless",
            "--device",
            "cpu",
        ]
    )
    plan = resolve_collection_plan(
        args,
        {},
        expansion_collection={
            "target_episodes": 64,
            "selection": {"mode": "sequential", "start_recipe_index": 0},
        },
        legacy_recipe_indices=(0, 16, 32, 48),
    )

    assert plan.target_episodes == 4
    assert plan.selection.mode == "explicit"
    assert plan.selection.recipe_indices == (0, 16, 32, 48)


def test_collection_plan_rejects_non_integer_numeric_values() -> None:
    args = _create_parser().parse_args(
        ["--gym-config", "task.yaml", "--headless", "--device", "cpu"]
    )
    with pytest.raises(ValueError, match="target_episodes"):
        resolve_collection_plan(
            args,
            {"collection": {"target_episodes": 4.5}},
        )


def test_expansion_recipe_padding_preserves_selected_rows() -> None:
    assert _pad_expansion_recipe_indices((4, 16), num_envs=4) == (4, 16, 16, 16)


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


def test_collection_plan_cli_target_overrides_task_target() -> None:
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
    plan = resolve_collection_plan(args, {"collection": {"target_episodes": 4}})

    assert plan.target_episodes == 5


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


def test_cli_collection_target_overrides_task_target() -> None:
    args = _create_parser().parse_args(
        [
            "--gym-config",
            "task.yaml",
            "--max_episodes",
            "5",
            "--headless",
            "--device",
            "cpu",
        ]
    )

    plan = resolve_collection_plan(
        args,
        {"collection": {"target_episodes": 64}},
    )

    assert plan.target_episodes == 5


def test_collection_plan_maps_legacy_max_episodes_as_fallback() -> None:
    args = _create_parser().parse_args(
        ["--gym-config", "task.yaml", "--headless", "--device", "cpu"]
    )
    plan = resolve_collection_plan(args, {"max_episodes": 5})

    assert plan.target_episodes == 5
    assert plan.selection.mode == "sequential"


def test_task_expansion_config_applies_runtime_overlay() -> None:
    task_path = resolve_config_path(
        "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/"
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
        "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/"
        "task.ur5.yaml"
    )
    policy_path = resolve_config_path(
        "extensions/embodichain_tasks/configs/components/expansion_policies/"
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


def test_explicit_num_envs_updates_expansion_capacity():
    path = "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    args = _create_parser().parse_args(["--gym-config", path, "--num-envs", "4"])
    profile, _, _, _ = _resolve_expansion_request(
        args, load_config(resolve_config_path(path))
    )
    profile.validate_for_num_envs(4)
    assert profile.execution.max_inflight == 4


def test_cyclic_selection_wraps_without_changing_recipe_ids() -> None:
    selection = CollectionSelection("cyclic", start_recipe_index=2)
    assert selection.take(0, 3, recipe_count=4) == (2, 3, 0)
    assert selection.take(3, 5, recipe_count=4) == (1, 2, 3, 0, 1)
    with pytest.raises(ValueError, match="positive recipe_count"):
        selection.take(0, 1)


def test_cyclic_collection_requires_an_expansion_profile() -> None:
    args = _create_parser().parse_args(["--gym-config", "task.yaml"])
    with pytest.raises(ValueError, match="requires an expansion profile"):
        resolve_collection_plan(
            args,
            {"collection": {"selection": {"mode": "cyclic"}}},
        )


def test_cyclic_collection_rejects_a_start_outside_recipe_budget() -> None:
    profile = _resolve(
        "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )[0]
    with pytest.raises(ValueError, match="starts outside"):
        _validate_expansion_collection_plan(
            profile,
            CollectionPlan(48, 3, CollectionSelection("cyclic", start_recipe_index=16)),
            num_envs=2,
        )


def test_cyclic_expansion_resets_before_each_repeated_batch(monkeypatch, tmp_path):
    """A wrapped recipe gets a fresh host reset and its own collected episode."""
    import json
    from types import SimpleNamespace

    path = "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_move_joint/task.ur5.yaml"
    args = _create_parser().parse_args(
        ["--gym-config", path, "--max_episodes", "10", "--output-dir", str(tmp_path)]
    )
    resets = []
    batches = []
    env = SimpleNamespace(num_envs=2, reset=lambda **kwargs: resets.append(kwargs))

    def generate(env, **kwargs):
        assert len(resets) == len(batches) + 1
        assert resets[-1] == {"seed": None, "options": {"save_data": False}}
        assert kwargs["reset_before"] is False
        batches.append(kwargs["expansion_recipe_indices"])
        return True

    monkeypatch.setattr("embodichain.lab.scripts.run_task.generate_function", generate)
    _run_expansion(env, args, load_config(resolve_config_path(path)))
    assert batches == [(0, 1), (2, 3), (0, 1), (2, 3), (0, 1)]
    manifest = json.loads((tmp_path / "expansion_manifest.json").read_text())
    assert manifest["committed_episodes"] == 10
    assert manifest["prepare_reset_count"] == 5
    assert manifest["selection"]["mode"] == "cyclic"


def test_reused_cube_scene_keeps_move_held_object_recording_identity() -> None:
    path = "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_move_held_object/task.ur5.yaml"
    args = _create_parser().parse_args(
        ["--gym-config", path, "--headless", "--disable-sensor", "--num-envs", "1"]
    )
    _, config, _ = build_env_cfg_from_args(args)
    params = config["env"]["dataset"]["lerobot"]["params"]
    assert (
        params["save_path"]
        == "/tmp/embodichain-manipulation/repeated_move_held_object/datasets"
    )
    assert params["extra"]["task_description"] == "repeated_move_held_object"
