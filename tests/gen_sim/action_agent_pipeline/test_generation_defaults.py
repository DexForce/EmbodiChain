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

import pytest

from embodichain.gen_sim.action_agent_pipeline.defaults import (
    ACTION_AGENT_CONFIG_DEFAULTS,
    DEFAULT_MAX_EPISODES,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
    DEFAULT_TARGET_BODY_SCALE,
    DEFAULT_TASK_NAME,
    generation_defaults_section,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_blocks import (
    _make_target_object_config,
    _moved_rigid_object_max_convex_hull_num,
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
    "linear_damping": 0.7,
    "angular_damping": 0.7,
    "contact_offset": 0.002,
    "rest_offset": 0.001,
    "restitution": 0.05,
    "max_depenetration_velocity": 5.0,
    "max_linear_velocity": 1.0,
    "max_angular_velocity": 1.0,
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
    assert "moved" in ACTION_AGENT_CONFIG_DEFAULTS["physics"]["convex_hulls"]


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


def test_generation_defaults_section_rejects_unknown_section() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        generation_defaults_section("missing")


def test_moved_convex_hull_limit_is_loaded_from_defaults() -> None:
    moved_default = ACTION_AGENT_CONFIG_DEFAULTS["physics"]["convex_hulls"]["moved"]
    moved_object = _SceneObject("cube", "rigid_object", {})
    source_limited_object = _SceneObject(
        "cup",
        "rigid_object",
        {"max_convex_hull_num": 4},
    )

    assert _moved_rigid_object_max_convex_hull_num(moved_object) == moved_default
    assert _moved_rigid_object_max_convex_hull_num(source_limited_object) == 4
