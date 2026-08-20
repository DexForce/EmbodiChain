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

"""Tests for solver-aware Spawn descriptor translation."""

from __future__ import annotations

import pytest

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.spawn.descriptors import (
    articulation_desc_from_cfg,
    rigid_desc_from_cfg,
)

pytestmark = pytest.mark.no_sim

RESTITUTION = 0.25


@pytest.mark.parametrize(
    ("solver_type", "expected_restitution"),
    [
        ("mujoco_warp", None),
        ("semi_implicit", None),
        ("featherstone", None),
        ("xpbd", RESTITUTION),
        (None, RESTITUTION),
    ],
)
def test_rigid_descriptor_projects_restitution_only_to_supported_solvers(
    solver_type: str | None,
    expected_restitution: float | None,
) -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyAttributesCfg(restitution=RESTITUTION),
    )

    descriptor, _ = rigid_desc_from_cfg(
        cfg,
        newton_solver_type=solver_type,
    )

    assert descriptor.collisions[0].newton.restitution == expected_restitution


def test_rigid_descriptor_preserves_default_backend_restitution() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyAttributesCfg(restitution=RESTITUTION),
    )

    descriptor, _ = rigid_desc_from_cfg(
        cfg,
        newton_solver_type="mujoco_warp",
    )

    assert descriptor.collisions[0].dexsim.restitution == RESTITUTION


def test_articulation_descriptor_omits_restitution_for_mujoco_warp() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        attrs=RigidBodyAttributesCfg(restitution=RESTITUTION),
    )

    descriptor = articulation_desc_from_cfg(
        cfg,
        newton_solver_type="mujoco_warp",
    )

    assert descriptor.newton_collision.restitution is None
