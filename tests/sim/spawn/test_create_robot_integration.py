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

"""Regression coverage for the robot configured by create_robot.py."""

from __future__ import annotations

import numpy as np
import pytest

import dexsim
from embodichain.lab.sim.spawn.descriptors import (
    articulation_desc_from_cfg,
    configure_articulation_desc,
)
from scripts.tutorials.sim.create_robot import create_robot

pytestmark = pytest.mark.requires_sim


class _ConfigCapture:
    def add_robot(self, cfg):
        return cfg


def _resolve_tutorial_properties(world, cfg):
    builder = dexsim.spawn.SceneBuilder(world)
    descriptor = articulation_desc_from_cfg(cfg, per_env=False)
    builder.add_articulation(descriptor)
    builder.resolve_sources()
    configure_articulation_desc(descriptor, cfg)

    base = descriptor.get_link_desc("arm_base_link")
    joint = descriptor.get_joint_desc("joint1")
    return (
        base.rigid_body.mass,
        base.rigid_body.inertia.copy(),
        joint.dexsim.stiffness,
        joint.dexsim.damping,
        joint.newton.target_ke,
        joint.newton.target_kd,
    )


def test_create_robot_preserves_source_inertia_and_arm_drive() -> None:
    cfg = create_robot(_ConfigCapture())
    cfg.fpath = cfg.urdf_cfg.assemble_urdf()

    config = dexsim.WorldConfig()
    config.open_windows = False
    config.renderer = dexsim.types.Renderer.HYBRID
    config.backend = dexsim.types.Backend.VULKAN
    world = dexsim.World(config)

    mass, inertia, stiffness, damping, newton_ke, newton_kd = (
        _resolve_tutorial_properties(world, cfg)
    )

    assert mass == pytest.approx(1.0)
    np.testing.assert_allclose(
        inertia,
        [5.677594, 30.912516, 31.167990],
        rtol=1.0e-5,
    )
    assert stiffness == pytest.approx(1.0e4)
    assert damping == pytest.approx(1.0e3)
    assert newton_ke == pytest.approx(1.0e4)
    assert newton_kd == pytest.approx(1.0e3)
