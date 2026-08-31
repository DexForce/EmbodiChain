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

"""Regression coverage for robots configured by simulation tutorials."""

from __future__ import annotations

import numpy as np
import pytest

import dexsim
from embodichain.lab.sim.spawn.descriptors import (
    articulation_desc_from_cfg,
    configure_articulation_desc,
)
from embodichain.lab.sim.spawn.scene import SpawnScene
from scripts.tutorials.sim.create_sensor import create_robot as create_sensor_robot
from scripts.tutorials.sim.create_robot import create_robot

pytestmark = pytest.mark.requires_sim

ARM_BASE_MASS = 3.167  # SR5 base_link inertial mass from the source URDF.
ARM_BASE_INERTIA = (5.677594, 30.912516, 31.167990)
ARM_STIFFNESS = 1.0e4
ARM_DAMPING = 1.5e3
ARM_MAX_EFFORT = 1.0e4


class _ConfigCapture:
    def add_robot(self, cfg):
        return cfg


def _resolve_tutorial_properties(world, cfg):
    scene = SpawnScene(world, num_envs=1)
    scene.builder.prepare_arenas()
    descriptor = articulation_desc_from_cfg(cfg, per_env=False)
    scene.declare(
        "articulation",
        "robot",
        descriptor,
        configure_source=lambda value: configure_articulation_desc(value, cfg),
    )
    result = scene.commit()
    descriptor = scene.handles("robot")[0].desc

    base = descriptor.get_link_desc("arm_base_link")
    joint = descriptor.get_joint_desc("joint1")
    properties = (
        base.rigid_body.mass,
        base.rigid_body.inertia.copy(),
        joint.dexsim.stiffness,
        joint.dexsim.damping,
        joint.dexsim.max_force,
        joint.newton.target_ke,
        joint.newton.target_kd,
        joint.effort_limit,
    )
    result.close()
    return properties


def test_create_robot_preserves_source_inertia_and_arm_drive() -> None:
    cfg = create_robot(_ConfigCapture())
    cfg.fpath = cfg.urdf_cfg.assemble_urdf()

    config = dexsim.WorldConfig()
    config.open_windows = False
    config.renderer = dexsim.types.Renderer.HYBRID
    config.backend = dexsim.types.Backend.VULKAN
    world = dexsim.World(config)

    (
        mass,
        inertia,
        stiffness,
        damping,
        max_effort,
        newton_ke,
        newton_kd,
        common_max_effort,
    ) = _resolve_tutorial_properties(world, cfg)

    assert mass == pytest.approx(ARM_BASE_MASS)
    np.testing.assert_allclose(
        inertia,
        ARM_BASE_INERTIA,
        rtol=1.0e-5,
    )
    assert stiffness == pytest.approx(ARM_STIFFNESS)
    assert damping == pytest.approx(ARM_DAMPING)
    assert max_effort == pytest.approx(ARM_MAX_EFFORT)
    assert newton_ke == pytest.approx(ARM_STIFFNESS)
    assert newton_kd == pytest.approx(ARM_DAMPING)
    assert common_max_effort == pytest.approx(ARM_MAX_EFFORT)


def test_create_sensor_uses_the_matched_arm_drive() -> None:
    """Keep the sensor tutorial's arm controller aligned across backends."""
    cfg = create_sensor_robot(_ConfigCapture())

    assert cfg.joint_drive_props is not None
    assert cfg.joint_drive_props.max_effort == {
        "joint[1-6]": ARM_MAX_EFFORT,
        "LEFT_.*": ARM_MAX_EFFORT,
    }
    cfg.fpath = cfg.urdf_cfg.assemble_urdf()

    config = dexsim.WorldConfig()
    config.open_windows = False
    config.renderer = dexsim.types.Renderer.HYBRID
    config.backend = dexsim.types.Backend.VULKAN
    world = dexsim.World(config)

    (
        _,
        _,
        stiffness,
        damping,
        max_effort,
        newton_ke,
        newton_kd,
        common_max_effort,
    ) = _resolve_tutorial_properties(world, cfg)

    assert (
        stiffness,
        damping,
        max_effort,
        newton_ke,
        newton_kd,
        common_max_effort,
    ) == pytest.approx(
        (
            ARM_STIFFNESS,
            ARM_DAMPING,
            ARM_MAX_EFFORT,
            ARM_STIFFNESS,
            ARM_DAMPING,
            ARM_MAX_EFFORT,
        )
    )
