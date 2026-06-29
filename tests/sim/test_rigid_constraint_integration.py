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

"""Real-sim integration smoke test for rigid constraints.

Mirrors the dexsim ``test_constraint.py`` contract at the EmbodiChain
:class:`SimulationManager` layer: two dynamic objects welded by a fixed
constraint keep their relative transform under physics, and detaching lets
them separate.

Skipped when the required asset is not present on disk (it is downloaded on
demand by ``embodichain.data.get_data_path``) or when CUDA is unavailable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    RigidObjectCfg,
    RigidConstraintCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.shapes import MeshCfg

DUCK_PATH = "ToyDuck/toy_duck.glb"


def _can_run_sim(device: str) -> bool:
    """True only if the device is usable and the test asset is present."""
    if device == "cuda" and not torch.cuda.is_available():
        return False
    try:
        return os.path.isfile(get_data_path(DUCK_PATH))
    except Exception:
        return False


class BaseRigidConstraintTest:
    """Shared setup for the CPU and CUDA integration tests."""

    def _delta_z(self) -> float:
        """Return duck_b.z - duck_a.z (env 0) from the bodies' current poses."""
        pose_a = self.duck_a.get_local_pose(to_matrix=True)
        pose_b = self.duck_b.get_local_pose(to_matrix=True)
        return float(pose_b[0, 2, 3] - pose_a[0, 2, 3])

    def setup_simulation(self, sim_device: str) -> None:
        if not _can_run_sim(sim_device):
            pytest.skip(
                f"Cannot run rigid-constraint integration test on {sim_device}."
            )
        config = SimulationManagerCfg(headless=True, sim_device=sim_device, num_envs=1)
        self.sim = SimulationManager(config)
        self.sim.enable_physics(False)

        duck_path = get_data_path(DUCK_PATH)
        # Two dynamic ducks at different heights, welded at identity frames.
        attrs_a = RigidBodyAttributesCfg()
        attrs_a.mass = 0.2
        attrs_b = RigidBodyAttributesCfg()
        attrs_b.mass = 0.1
        self.duck_a = self.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="duck_a",
                shape=MeshCfg(fpath=duck_path),
                body_type="dynamic",
                init_pos=[0.0, 0.0, 1.4],
                attrs=attrs_a,
            ),
        )
        self.duck_b = self.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="duck_b",
                shape=MeshCfg(fpath=duck_path),
                body_type="dynamic",
                init_pos=[0.0, 0.0, 1.2],
                attrs=attrs_b,
            ),
        )

        if sim_device == "cuda" and getattr(self.sim, "is_use_gpu_physics", False):
            self.sim.init_gpu_physics()
        self.sim.enable_physics(True)

    def test_fixed_constraint_holds_relative_pose(self):
        """Welded objects keep their relative transform; detaching lets them move."""
        constraint = self.sim.create_rigid_constraint(
            RigidConstraintCfg(
                name="weld",
                rigid_object_a_uid="duck_a",
                rigid_object_b_uid="duck_b",
            )
        )
        assert constraint.is_valid() == [True]

        pose_a = self.duck_a.get_local_pose(to_matrix=True)
        pose_b = self.duck_b.get_local_pose(to_matrix=True)
        initial_delta_z = float(pose_b[0, 2, 3] - pose_a[0, 2, 3])

        # Step physics; the constraint should hold the relative transform.
        for _ in range(120):
            self.sim.update(step=1)

        rel = constraint.get_relative_transform()[0]
        np.testing.assert_allclose(rel[:3, 3], np.zeros(3), atol=2e-2)
        pose_a2 = self.duck_a.get_local_pose(to_matrix=True)
        pose_b2 = self.duck_b.get_local_pose(to_matrix=True)
        delta_z = float(pose_b2[0, 2, 3] - pose_a2[0, 2, 3])
        assert abs(delta_z - initial_delta_z) < 2e-2

        # Detach and confirm the relative pose is no longer held: once free,
        # the lower duck (duck_b) lands first and the gap closes, so the
        # relative z drifts away from the value the constraint was holding.
        self.sim.remove_rigid_constraint("weld")
        assert self.sim.get_rigid_constraint("weld") is None
        held_delta_z = self._delta_z()
        for _ in range(120):
            self.sim.update(step=1)
        final_delta_z = self._delta_z()
        assert abs(final_delta_z - held_delta_z) > 2e-2


class TestRigidConstraintCPU(BaseRigidConstraintTest):
    def setup_method(self) -> None:
        self.setup_simulation("cpu")


class TestRigidConstraintCUDA(BaseRigidConstraintTest):
    def setup_method(self) -> None:
        self.setup_simulation("cuda")
