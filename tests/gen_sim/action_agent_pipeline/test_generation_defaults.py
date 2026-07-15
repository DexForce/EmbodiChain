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

from pathlib import Path
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.action_agent_pipeline.defaults import (
    ACTION_AGENT_CONFIG_DEFAULTS,
    CONVEX_HULL_DEFAULTS,
    DEFAULT_MAX_EPISODES,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
    DEFAULT_TARGET_BODY_SCALE,
    DEFAULT_TASK_NAME,
    generation_defaults_section,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_blocks import (
    _container_rigid_object_max_convex_hull_num,
    _make_background_config,
    _make_extra_rigid_object_config,
    _make_target_object_config,
    _moved_rigid_object_max_convex_hull_num,
    _relative_rigid_object_max_convex_hull_num,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.glb_geometry_baking import (
    GlbGeometryNormalizer,
)

_EXPECTED_RIGID_OBJECT_PHYSICS = {
    "mass": 0.2,
    "static_friction": 0.95,
    "dynamic_friction": 0.9,
    "linear_damping": 0.9,
    "angular_damping": 0.9,
    "contact_offset": 0.002,
    "rest_offset": 0.001,
    "restitution": 0.05,
    "max_depenetration_velocity": 0.8,
    "max_linear_velocity": 0.5,
    "max_angular_velocity": 0.5,
}


def test_public_generation_defaults_are_loaded_from_yaml() -> None:
    task = ACTION_AGENT_CONFIG_DEFAULTS["task"]
    geometry = ACTION_AGENT_CONFIG_DEFAULTS["geometry"]

    assert DEFAULT_TASK_NAME == task["default_name"]
    assert DEFAULT_MAX_EPISODES == task["max_episodes"]
    assert DEFAULT_MAX_EPISODE_STEPS == task["max_episode_steps"]
    assert DEFAULT_TARGET_BODY_SCALE == geometry["target_body_scale"]
    assert DEFAULT_SURFACE_RELEASE_CLEARANCE == geometry["surface_release_clearance"]


def test_generation_defaults_expose_required_hyperparameter_sections() -> None:
    expected_sections = {
        "action",
        "arrangement",
        "geometry",
        "grasp",
        "physics",
        "relative_placement",
        "stacking",
        "task",
    }

    assert expected_sections <= ACTION_AGENT_CONFIG_DEFAULTS.keys()
    convex_hulls = ACTION_AGENT_CONFIG_DEFAULTS["physics"]["convex_hulls"]
    assert {
        "target",
        "container",
        "moved",
        "extra_rigid",
        "table",
    } == set(convex_hulls)


def test_affordance_stabilization_steps_is_a_non_negative_integer() -> None:
    stabilization_steps = ACTION_AGENT_CONFIG_DEFAULTS["grasp"][
        "affordance_stabilization_steps"
    ]

    assert isinstance(stabilization_steps, int)
    assert stabilization_steps >= 0


def test_generated_rigid_object_uses_complete_physics_defaults(
    tmp_path: Path,
) -> None:
    obj = _SceneObject(
        "cube",
        "rigid_object",
        {"shape": {"shape_type": "Box"}},
    )

    config = _make_target_object_config(
        tmp_path,
        obj,
        "cube",
        [1.0, 1.0, 1.0],
        GlbGeometryNormalizer(output_dir=tmp_path / "normalized"),
    )

    assert {
        key: config["attrs"][key] for key in _EXPECTED_RIGID_OBJECT_PHYSICS
    } == _EXPECTED_RIGID_OBJECT_PHYSICS
    assert config["max_convex_hull_num"] == CONVEX_HULL_DEFAULTS["target"]


def test_generation_defaults_section_rejects_unknown_section() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        generation_defaults_section("missing")


def test_convex_hull_limits_are_loaded_from_yaml() -> None:
    convex_hulls = ACTION_AGENT_CONFIG_DEFAULTS["physics"]["convex_hulls"]

    assert CONVEX_HULL_DEFAULTS["target"] == convex_hulls["target"]
    assert CONVEX_HULL_DEFAULTS["container"] == convex_hulls["container"]
    assert CONVEX_HULL_DEFAULTS["moved"] == convex_hulls["moved"]
    assert CONVEX_HULL_DEFAULTS["extra_rigid"] == convex_hulls["extra_rigid"]
    assert CONVEX_HULL_DEFAULTS["table"] == convex_hulls["table"]


def test_scene_role_limits_override_source_convex_hull_limit(tmp_path: Path) -> None:
    moved_object = _SceneObject("cube", "rigid_object", {})
    source_limited_object = _SceneObject(
        "cup",
        "rigid_object",
        {"max_convex_hull_num": 4},
    )
    table = _SceneObject(
        "table",
        "background",
        {"shape": {"shape_type": "Box"}, "max_convex_hull_num": 4},
    )
    extra = _SceneObject(
        "spoon",
        "rigid_object",
        {"shape": {"shape_type": "Box"}, "max_convex_hull_num": 4},
    )
    normalizer = GlbGeometryNormalizer(output_dir=tmp_path / "normalized")
    table_config = _make_background_config(tmp_path, table, normalizer)
    extra_config = _make_extra_rigid_object_config(
        tmp_path,
        extra,
        [1.0, 1.0, 1.0],
        normalizer,
    )

    assert _moved_rigid_object_max_convex_hull_num(moved_object) == (
        CONVEX_HULL_DEFAULTS["moved"]
    )
    assert _moved_rigid_object_max_convex_hull_num(source_limited_object) == (
        CONVEX_HULL_DEFAULTS["moved"]
    )
    assert _container_rigid_object_max_convex_hull_num(source_limited_object) == (
        CONVEX_HULL_DEFAULTS["container"]
    )
    assert table_config["max_convex_hull_num"] == CONVEX_HULL_DEFAULTS["table"]
    assert extra_config["max_convex_hull_num"] == CONVEX_HULL_DEFAULTS["extra_rigid"]


def test_relative_object_roles_use_distinct_yaml_limits() -> None:
    spec = SimpleNamespace(
        placements=(
            SimpleNamespace(
                relation="inside",
                moved_runtime_uid="apple",
                reference_runtime_uid="basket",
            ),
            SimpleNamespace(
                relation="on",
                moved_runtime_uid="cup",
                reference_runtime_uid="pad",
            ),
        )
    )

    assert _relative_rigid_object_max_convex_hull_num("basket", spec) == (
        CONVEX_HULL_DEFAULTS["container"]
    )
    assert _relative_rigid_object_max_convex_hull_num("apple", spec) == (
        CONVEX_HULL_DEFAULTS["moved"]
    )
    assert _relative_rigid_object_max_convex_hull_num("pad", spec) == (
        CONVEX_HULL_DEFAULTS["target"]
    )
    assert _relative_rigid_object_max_convex_hull_num("spoon", spec) == (
        CONVEX_HULL_DEFAULTS["extra_rigid"]
    )
