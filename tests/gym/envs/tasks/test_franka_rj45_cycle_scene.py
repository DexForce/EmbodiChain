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
"""Verify the authored RJ45 scene against the original physics inputs."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import yaml

from embodichain.lab.sim.cfg import ArticulationCfg, CableObjectCfg

from embodichain.extensions.tasks.configs import get_config_path


def test_rj45_deployment_preserves_the_original_cable_scene() -> None:
    path = get_config_path("tasks/manipulation/cable/franka_rj45_cycle/env.yaml")
    environment = yaml.safe_load(path.read_text())
    assert environment["physics"] == "newton"
    assert environment["env"]["sim_steps_per_control"] == 32
    dt = environment["physics_config"]["physics_dt"]
    np.testing.assert_allclose(dt * 32, 1 / 60)
    assert environment["max_episode_steps"] == 2401

    scene = environment["simulation"]
    (raw_cable,) = scene["cable_object"]
    cable = CableObjectCfg.from_dict(raw_cable)
    plug = ArticulationCfg.from_dict(
        next(item for item in scene["articulation"] if item["uid"] == "plug")
    )
    points = np.asarray(cable.points, dtype=np.float32)
    # SHA-256 of the little-endian float32 array from the approved scene_inputs.npz.
    assert sha256(points.astype("<f4").tobytes()).hexdigest() == (
        "fb460bef62ce64eaf7a9918ed981dfaa385f35e80c6b4af23d614bc6fe713b27"
    )
    assert points.shape == (97, 3)
    assert cable.fixed_segments == [0]
    assert cable.collision_filter_span == 2
    assert (cable.end_attachment.target, cable.end_attachment.link_name) == (
        "plug",
        "plug",
    )
    assert plug.root_props.fixed_base is False
    np.testing.assert_allclose(
        plug.init_local_pose[:3, :3] @ cable.end_attachment.position
        + plug.init_local_pose[:3, 3],
        points[-1],
        atol=1e-8,
    )
    assert plug.joint_drive_props.limit_stiffness == 3.0
    assert plug.joint_drive_props.limit_damping == 0.0002
