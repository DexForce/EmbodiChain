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
"""Verify the USD geometry of the declarative rope-wrap deployment."""

from __future__ import annotations

import json

import numpy as np
import yaml
from dexsim.kit.usd import parse_usd

from embodichain.extensions.tasks.configs import get_config_path


def test_default_scene_preserves_the_authored_cable() -> None:
    root = get_config_path("tasks/manipulation/cable/franka_rope_wrap")
    defaults = json.loads((root / "config.json").read_text())
    env = yaml.safe_load((root / "env.yaml").read_text())
    assert env["enable_ground_plane"] is True
    assert env["env"]["sim_steps_per_control"] == 32
    np.testing.assert_allclose(env["physics_config"]["physics_dt"] * 32, 1 / 60)
    scene = env["simulation"]["usd_scene"]
    cable = scene["asset_cfgs"]["/World/cable"]
    asset = root / scene["source"]["file_path"]
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
