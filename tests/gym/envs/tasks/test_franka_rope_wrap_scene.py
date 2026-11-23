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
"""Check that the packaged scene and the custom-scene deployment share one geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from dexsim.kit.usd import parse_usd

from embodichain.lab.task_program.integrations._configured_composition import (
    _resolve_task_program_components,
)
from embodichain.extensions.tasks.manipulation.cable.franka_rope_wrap import (
    cable_points,
    load_config,
    materialize_deployment,
    task_directory,
)


def test_default_deployment_matches_materialized_scene(tmp_path: Path) -> None:
    root = task_directory()
    defaults = load_config()
    urdf = tmp_path / "derived.urdf"
    alternate = materialize_deployment(defaults, urdf, tmp_path)
    _, default_integration, default_policy = _resolve_task_program_components(
        yaml.safe_load((root / "task.franka.yaml").read_text())["task_program"],
        base_dir=root,
    )
    _, integration, policy = _resolve_task_program_components(
        yaml.safe_load(alternate.read_text())["task_program"],
        base_dir=alternate.parent,
    )
    assert integration == default_integration
    assert policy == default_policy
    stored = yaml.safe_load((root / "env.yaml").read_text())
    compiled = yaml.safe_load((alternate.parent / "env.yaml").read_text())
    event = compiled["env"]["events"]["settle_rope"]
    assert load_config(Path(event["params"].pop("scenario_path"))) == defaults
    assert compiled == stored
    assert compiled["enable_ground_plane"] is True
    assert compiled["env"]["sim_steps_per_control"] == 32
    assert compiled["physics_config"]["num_substeps"] == 1
    np.testing.assert_allclose(compiled["physics_config"]["physics_dt"] * 32, 1 / 60)
    scene = compiled["simulation"]["usd_scene"]
    cable = scene["asset_cfgs"]["/World/cable"]
    asset = alternate.parent / scene["source"]["file_path"]
    assert asset.read_bytes() == (root / "cable.usda").read_bytes()
    (desc,) = parse_usd(str(asset)).cables
    assert desc.usd.prim_path == "/World/cable"
    assert desc.enable_collision
    material = desc.render_material
    assert material.color_space == "linear"
    original_srgb = np.array([0.93, 0.94, 0.92])
    np.testing.assert_allclose(
        material.base_color[:3], ((original_srgb + 0.055) / 1.055) ** 2.4
    )
    np.testing.assert_allclose(material.roughness, 0.6)
    np.testing.assert_allclose(desc.radius, defaults["cable_radius_m"])
    points = np.asarray(desc.points)
    np.testing.assert_array_equal(points, cable_points(defaults))
    assert points.shape == (97, 3)
    np.testing.assert_allclose(
        points[[0, -1]], np.asarray(defaults["cable_waypoints_m"])[[0, -1]], atol=1e-7
    )
    delta = np.r_[defaults["post_center_xy_m"], points[0, 2]] - points[0]
    np.testing.assert_allclose(
        (points[1] - points[0]) / np.linalg.norm(points[1] - points[0]),
        delta / np.linalg.norm(delta),
        atol=1e-6,
    )
    assert cable["fixed_segments"] == [0]
    assert cable["collision_filter_span"] == 2


def test_custom_scene_rescales_windows_and_uses_requested_assets(
    tmp_path: Path,
) -> None:
    scenario = load_config()
    scenario.update(
        duration_s=72.0, floor_height_m=0.013, cable_segments=64, control_hz=30
    )
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps(scenario))
    effective = load_config(custom)
    assert (effective["grasp_complete_s"], effective["lift_complete_s"]) == (10.0, 14.0)
    urdf = tmp_path / "padded_franka.urdf"
    deployment = materialize_deployment(effective, urdf, tmp_path)
    _, integration, policy = _resolve_task_program_components(
        yaml.safe_load(deployment.read_text())["task_program"],
        base_dir=deployment.parent,
    )
    pickup = integration["profile"]["skill_options"]["simulation.cable_pickup"]
    assert pickup["hand_interp_steps"] == 30
    assert pickup["grasp_settle_steps"] == 8
    settle = integration["runtime_services"]["cable_material"]["targets"]["settle_lift"]
    assert settle["samples"] == 4
    assert policy["motion"]["sample_count"] == 20
    # 20–56 s is a single continuous orbit, plus two control ticks to acknowledge it.
    np.testing.assert_allclose(policy["recovery"]["skill_timeout"], 36 + 2 / 30)
    env = yaml.safe_load(deployment.with_name("env.yaml").read_text())
    dt = env["physics_config"]["physics_dt"] * env["env"]["sim_steps_per_control"]
    np.testing.assert_allclose(dt, 1 / 30)
    cylinder = next(
        body for body in env["simulation"]["background"] if body["uid"] == "post"
    )
    assert cylinder["init_pos"] == [0.48, 0.0, 0.073]
    scene = env["simulation"]["usd_scene"]
    (desc,) = parse_usd(str(deployment.parent / scene["source"]["file_path"])).cables
    np.testing.assert_array_equal(desc.points, cable_points(effective))
    assert len(desc.points) == 65
    assert np.min(desc.points[:, 2]) >= 0.017 - 1e-8
    embodiment = yaml.safe_load(deployment.with_name("embodiment.yaml").read_text())
    assert embodiment["simulation"]["fpath"] == str(urdf)
    assert embodiment["simulation"]["init_pos"] == [0, 0, 0.013]


def test_custom_deployment_rejects_unrepresentable_pickup_hold(tmp_path: Path) -> None:
    scenario = load_config()
    scenario["control_hz"] = 1
    with pytest.raises(ValueError, match="control_hz.*pickup settle hold"):
        materialize_deployment(scenario, tmp_path / "derived.urdf", tmp_path)
    assert not (tmp_path / "deployment").exists()
