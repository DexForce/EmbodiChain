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

"""Tests for config-created Expert Program environment runtimes."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from gymnasium.envs.registration import registry as gym_registry
import pytest

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.expert_program.configured_runtime import (
    _decode_configured_expert_program_runtime,
    _decode_grasp_generator,
    _register_configured_expert_program_runtime,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS

_REPOSITORY_ROOT = Path(__file__).parents[4]
_CONFIG_DIRECTORY = _REPOSITORY_ROOT / "embodichain_tasks/configs/tasks/manipulation"
_TASKS = {
    "repeated_pick_place": (
        "expert_program_repeated_pick_place",
        "expert_program_ur5_pick_place",
        frozenset({"pick", "place", "hand_over"}),
    ),
    "open_drawer": (
        "expert_program_open_drawer",
        "expert_program_ur5_slide",
        frozenset(
            {
                "pick",
                "place",
                "hand_over",
                "simulation.articulation_link_slide",
            }
        ),
    ),
    "hand_over": (
        "dual_ur5_handover_v1",
        "dual_ur5_handover_v1",
        frozenset({"pick", "place", "hand_over"}),
    ),
}
_TEST_ENV_ID = "ConfiguredExpertProgramRuntimeTest-v1"


def _config_path(task_name: str) -> Path:
    """Return one official task's Gym config path."""
    return _CONFIG_DIRECTORY / task_name / "env.json"


def _gym_config(task_name: str) -> dict[str, object]:
    """Return an independently owned production Gym configuration."""
    payload = json.loads(_config_path(task_name).read_text(encoding="utf-8"))
    assert type(payload) is dict
    return payload


def _runtime_payload(task_name: str) -> dict[str, object]:
    """Return an independently owned production runtime declaration."""
    payload = _gym_config(task_name)["expert_program_runtime"]
    assert type(payload) is dict
    return payload


@pytest.fixture
def registered_test_ids() -> Iterator[list[str]]:
    """Track and remove exact test-owned runtime registrations."""
    identifiers: list[str] = []
    yield identifiers
    for env_id in identifiers:
        REGISTERED_ENVS.pop(env_id, None)
        gym_registry.pop(env_id, None)


@pytest.mark.parametrize(
    ("task_name", "expected_scene", "expected_profile", "expected_calls"),
    tuple(
        (task_name, scene_id, profile_id, calls)
        for task_name, (scene_id, profile_id, calls) in _TASKS.items()
    ),
)
def test_all_examples_decode_through_one_composable_runtime_schema(
    task_name: str,
    expected_scene: str,
    expected_profile: str,
    expected_calls: frozenset[str],
) -> None:
    """Each official example uses the same scene/profile/services decoder."""
    runtime = _decode_configured_expert_program_runtime(_runtime_payload(task_name))
    registration = runtime.registration

    assert registration.scene_binding.registry_id == expected_scene
    assert registration.robot_profile_binding.profile_id == expected_profile
    assert frozenset(registration.call_catalog.descriptors) == expected_calls
    assert runtime.adapter_factory.registration is registration


@pytest.mark.parametrize("task_name", tuple(_TASKS))
def test_all_examples_register_plain_embodied_env_under_config_selected_ids(
    task_name: str,
    registered_test_ids: list[str],
) -> None:
    """Config loading creates a runnable ID without a task environment class."""
    env_id = f"Configured-{task_name.replace('_', '-')}-v1"
    registered_test_ids.append(env_id)
    config = _gym_config(task_name)
    config["id"] = env_id

    cfg = config_to_cfg(config, source_path=_config_path(task_name))
    spec = REGISTERED_ENVS[env_id]

    assert spec.cls is EmbodiedEnv
    assert spec.max_episode_steps == config["max_episode_steps"]
    assert spec.expert_program_adapter_factory is not None
    assert spec.expert_program_registration is not None
    assert spec.expert_program_adapter_factory.registration is (
        spec.expert_program_registration
    )
    assert cfg.expert_program is not None
    assert cfg.expert_program.integration.scene_registry == (
        spec.expert_program_registration.scene_binding.registry_id
    )
    assert cfg.expert_program.integration.robot_profile == (
        spec.expert_program_registration.robot_profile_binding.profile_id
    )
    assert env_id in gym_registry


def test_trajectory_examples_disable_validation_and_recovery_layers() -> None:
    """The two showcase profiles contain only open-loop trajectory execution."""
    for task_name in ("repeated_pick_place", "open_drawer"):
        runtime = _decode_configured_expert_program_runtime(_runtime_payload(task_name))
        preset = runtime.registration.robot_profile_binding.presets[0]

        assert preset.motion_policy.sample_count == 40
        assert dict(preset.effect_monitors) == {}
        assert preset.recovery_policy.max_replans == 0
        assert preset.recovery_policy.max_action_retries == 0
        assert preset.workflow_recovery_policy.max_recovery_attempts == 0
        assert preset.runner_cfg.minimum_cycle_time == 0.0


def test_grasp_generator_resolves_named_model_and_library_defaults() -> None:
    """Serialized services name reusable geometry and omit policy defaults."""
    factory = _decode_grasp_generator(
        {
            "kind": "antipodal_parallel_jaw",
            "model": "dh_pgi_140_80",
        },
        path="generator",
    )

    assert factory.model_id == "dh_pgi_140_80"
    assert factory.min_opening_width == pytest.approx(0.005)
    assert factory.palm_depth == pytest.approx(0.096)
    assert factory.sample_count is None
    assert factory.approach_direction_samples is None
    assert factory.opening_margin is None
    assert factory.point_sample_density is None
    assert factory.filter_ground_collision is None
    assert factory.force_refresh is None
    assert not hasattr(factory, "viser_port")


def test_grasp_generator_factory_defers_to_toolkit_policy_defaults() -> None:
    """Omitted runtime policy fields remain owned by the grasp toolkit."""
    generator = _decode_grasp_generator(
        {
            "kind": "antipodal_parallel_jaw",
            "model": "dh_pgi_140_80",
        },
        path="generator",
    )()

    assert generator.algorithm_cfg.sample_count == 20_000
    assert generator.algorithm_cfg.approach_direction_samples == 4
    assert generator.collision_cfg.opening_margin == pytest.approx(0.01)
    assert generator.collision_cfg.point_sample_density == pytest.approx(0.01)
    assert generator.collision_cfg.filter_ground_collision is True
    assert generator.annotation_cfg.selection_mode == "whole_mesh"
    assert generator.annotation_cfg.force_refresh is False


def test_grasp_generator_inline_model_uses_geometry_defaults() -> None:
    """Custom inline models only override geometry that differs from defaults."""
    factory = _decode_grasp_generator(
        {
            "kind": "antipodal_parallel_jaw",
            "model": {
                "model_id": "custom_parallel_jaw",
                "palm_depth": 0.09,
            },
        },
        path="generator",
    )

    assert factory.model_id == "custom_parallel_jaw"
    assert factory.min_opening_width == pytest.approx(0.001)
    assert factory.max_opening_width == pytest.approx(0.1)
    assert factory.finger_length == pytest.approx(0.08)
    assert factory.finger_width == pytest.approx(0.03)
    assert factory.finger_thickness == pytest.approx(0.01)
    assert factory.palm_depth == pytest.approx(0.09)


def test_grasp_generator_rejects_unknown_models_and_removed_viser_port() -> None:
    """The runtime keeps a closed model catalog and no interactive port."""
    with pytest.raises(ValueError, match="available.*dh_pgi_140_80"):
        _decode_grasp_generator(
            {
                "kind": "antipodal_parallel_jaw",
                "model": "missing_gripper",
            },
            path="generator",
        )

    with pytest.raises(ValueError, match="unsupported fields.*viser_port"):
        _decode_grasp_generator(
            {
                "kind": "antipodal_parallel_jaw",
                "model": "dh_pgi_140_80",
                "viser_port": 11_801,
            },
            path="generator",
        )


def test_scene_shorthand_derives_native_ids_and_affordance_metadata() -> None:
    """Common scene identities need not be repeated in serialized bindings."""
    payload = deepcopy(_runtime_payload("repeated_pick_place"))
    scene = payload["scene"]
    assert type(scene) is dict
    rigid_object = scene["rigid_objects"][0]
    grasp = scene["antipodal_grasps"][0]
    for field in ("simulation_uid", "default_grasp_affordance"):
        rigid_object.pop(field, None)
    for field in ("native_name", "revision"):
        grasp.pop(field, None)

    binding = _decode_configured_expert_program_runtime(
        payload
    ).registration.scene_binding

    assert binding.rigid_objects[0].simulation_uid == "cube"
    assert binding.rigid_objects[0].default_grasp_affordance is None
    assert binding.antipodal_grasps[0].native_name == "cube_grasp"
    assert binding.antipodal_grasps[0].revision == "1"


def test_official_generators_use_named_models_without_default_fields() -> None:
    """Reference configs expose only task-specific grasp-generator tuning."""
    forbidden = {"force_refresh", "viser_port"}
    for task_name in _TASKS:
        services = _runtime_payload(task_name)["runtime_services"]
        generators = services["grasp_pose_generators"]
        for generator in generators.values():
            assert generator["model"] == "dh_pgi_140_80"
            assert forbidden.isdisjoint(generator)
            if task_name == "hand_over":
                assert generator["approach_direction_samples"] == 1
            else:
                assert "approach_direction_samples" not in generator


def test_open_drawer_trajectory_runtime_selects_a_snapshot_target() -> None:
    """The showcase does not install dynamic-target recovery monitoring."""
    runtime = _decode_configured_expert_program_runtime(_runtime_payload("open_drawer"))
    factories = runtime.registration.registered_semantic_lowerer_factories

    assert len(factories) == 1
    assert factories[0].target_pose_mode == "snapshot"


def test_runtime_config_rejects_unknown_slide_target_pose_mode() -> None:
    """Configured lowerers accept only explicit live or snapshot targets."""
    payload = deepcopy(_runtime_payload("open_drawer"))
    services = payload["runtime_services"]
    assert type(services) is dict
    lowerers = services["registered_semantic_lowerers"]
    assert type(lowerers) is list
    lowerer = lowerers[0]
    assert type(lowerer) is dict
    lowerer["target_pose_mode"] = "moving"

    with pytest.raises(ValueError, match="target_pose_mode"):
        _decode_configured_expert_program_runtime(payload)


def test_runtime_config_rejects_unknown_fields_before_live_construction() -> None:
    """Misspelled or unsupported runtime fields fail closed."""
    payload = _runtime_payload("repeated_pick_place")
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="unsupported fields.*unexpected"):
        _decode_configured_expert_program_runtime(payload)


def test_runtime_config_rejects_removed_schema_version() -> None:
    """The runtime declaration has no development-history version field."""
    payload = _runtime_payload("repeated_pick_place")
    payload["schema_version"] = 1

    with pytest.raises(ValueError, match="unsupported fields.*schema_version"):
        _decode_configured_expert_program_runtime(payload)


def test_runtime_registration_is_idempotent_for_the_same_config(
    registered_test_ids: list[str],
) -> None:
    """Loading the same config twice preserves its exact runtime registration."""
    registered_test_ids.append(_TEST_ENV_ID)
    config = _gym_config("repeated_pick_place")
    config["id"] = _TEST_ENV_ID

    first_cfg = config_to_cfg(config, source_path=_config_path("repeated_pick_place"))
    first = REGISTERED_ENVS[_TEST_ENV_ID]
    second_cfg = config_to_cfg(
        deepcopy(config),
        source_path=_config_path("repeated_pick_place"),
    )

    assert REGISTERED_ENVS[_TEST_ENV_ID] is first
    assert first_cfg.expert_program is not None
    assert second_cfg.expert_program is not None


def test_invalid_config_does_not_leave_a_runtime_registration(
    registered_test_ids: list[str],
) -> None:
    """Registration occurs only after program and environment parsing succeeds."""
    registered_test_ids.append(_TEST_ENV_ID)
    config = _gym_config("repeated_pick_place")
    config["id"] = _TEST_ENV_ID
    config["expert_program_path"] = "missing-program.yaml"

    with pytest.raises(FileNotFoundError):
        config_to_cfg(config, source_path=_config_path("repeated_pick_place"))

    assert _TEST_ENV_ID not in REGISTERED_ENVS
    assert _TEST_ENV_ID not in gym_registry


def test_runtime_registration_rejects_reusing_an_id_for_changed_config(
    registered_test_ids: list[str],
) -> None:
    """Changing runtime data requires a distinct ID instead of silent override."""
    registered_test_ids.append(_TEST_ENV_ID)
    runtime = _decode_configured_expert_program_runtime(
        _runtime_payload("repeated_pick_place")
    )
    _register_configured_expert_program_runtime(
        _TEST_ENV_ID,
        runtime,
        max_episode_steps=321,
    )
    changed_payload = deepcopy(_runtime_payload("repeated_pick_place"))
    profile = changed_payload["robot_profile"]
    assert type(profile) is dict
    presets = profile["presets"]
    assert type(presets) is list
    preset = presets[0]
    assert type(preset) is dict
    motion = preset["motion"]
    assert type(motion) is dict
    motion["sample_count"] = 41
    changed_runtime = _decode_configured_expert_program_runtime(changed_payload)

    with pytest.raises(ValueError, match="different environment or runtime"):
        _register_configured_expert_program_runtime(
            _TEST_ENV_ID,
            changed_runtime,
            max_episode_steps=321,
        )


def test_examples_have_no_importable_task_environment_modules() -> None:
    """All three environment implementations are now serialized configuration."""
    for task_name in _TASKS:
        module_name = f"embodichain_tasks.manipulation.{task_name}"
        assert importlib.util.find_spec(module_name) is None


__all__: list[str] = []
