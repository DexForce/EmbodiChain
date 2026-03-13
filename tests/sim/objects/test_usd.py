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

import pytest
import torch

from embodichain.lab.sim import (
    SimulationManager,
    SimulationManagerCfg,
)
from embodichain.lab.sim.objects import Articulation, RigidObject
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    RigidObjectCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.data import get_data_path

NUM_ARENAS = 1


class BaseUsdTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, sim_device):
        config = SimulationManagerCfg(
            headless=True, sim_device=sim_device, num_envs=NUM_ARENAS, enable_rt=False
        )
        self.sim = SimulationManager(config)

        if sim_device == "cuda" and getattr(self.sim, "is_use_gpu_physics", False):
            self.sim.init_gpu_physics()

    def test_import_rigid(self):
        default_attr = RigidBodyAttributesCfg()
        sugar_box_path = get_data_path("SugarBox/sugar_box_usd/sugar_box.usda")
        sugar_box: RigidObject = self.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="sugar_box",
                shape=MeshCfg(fpath=sugar_box_path),
                body_type="dynamic",
                use_usd_properties=False,
                init_pos=[0.0, 1.0, 0.1],
                attrs=default_attr,
            )
        )
        body0 = sugar_box._entities[0].get_physical_body()
        print(sugar_box._entities[0].get_physical_attr())
        assert pytest.approx(body0.get_mass()) == default_attr.mass
        assert pytest.approx(body0.get_linear_damping()) == default_attr.linear_damping
        assert (
            pytest.approx(body0.get_angular_damping()) == default_attr.angular_damping
        )
        assert body0.get_solver_iteration_counts() == (
            default_attr.min_position_iters,
            default_attr.min_velocity_iters,
        )

    def test_import_articulation(self):
        default_drive = JointDrivePropertiesCfg()
        h1_path = get_data_path("UnitreeH1Usd/H1_usd/h1.usd")
        h1: Articulation = self.sim.add_articulation(
            cfg=ArticulationCfg(
                uid="h1",
                fpath=h1_path,
                build_pk_chain=False,
                use_usd_properties=False,
                init_pos=[0.0, 0.0, 1.2],
                drive_pros=default_drive,
            )
        )

        stiffness = h1.body_data.joint_stiffness
        damping = h1.body_data.joint_damping
        max_force = h1.body_data.qf_limits
        print(f"All joint stiffness: {stiffness}")
        print(f"All joint damping: {damping}")
        print(f"All joint max force: {max_force}")
        expected_stiffness = default_drive.stiffness
        assert torch.allclose(
            stiffness, torch.tensor(expected_stiffness, dtype=torch.float32)
        )
        expectied_damping = default_drive.damping
        assert torch.allclose(
            damping, torch.tensor(expectied_damping, dtype=torch.float32)
        )

    def test_usd_properties(self):
        """In this test, we set use_usd_properties=True to verify that the USD properties are correctly applied."""
        h1_path = get_data_path("UnitreeH1Usd/H1_usd/h1.usd")
        h1: Articulation = self.sim.add_articulation(
            cfg=ArticulationCfg(
                uid="h1_beta",
                fpath=h1_path,
                build_pk_chain=False,
                use_usd_properties=True,
                init_pos=[1.0, 0.0, 1.2],
            )
        )

        stiffness = h1.body_data.joint_stiffness
        damping = h1.body_data.joint_damping
        max_force = h1.body_data.qf_limits
        print(f"All joint stiffness: {stiffness}")
        print(f"All joint damping: {damping}")
        print(f"All joint max force: {max_force}")
        expected_stiffness = 10000000.0
        assert torch.allclose(
            stiffness, torch.tensor(expected_stiffness, dtype=torch.float32)
        )

        joint_names = h1.joint_names
        print(f"Joint names: {joint_names}")

        target_joint_name = "left_hip_yaw_joint"
        if target_joint_name in joint_names:
            joint_idx = joint_names.index(target_joint_name)
            # check for the first instance
            assert torch.isclose(
                stiffness[0, joint_idx], torch.tensor(10000000.0, dtype=torch.float32)
            )
            assert torch.isclose(
                damping[0, joint_idx], torch.tensor(0.0, dtype=torch.float32)
            )
            assert torch.isclose(
                max_force[0, joint_idx], torch.tensor(200.0, dtype=torch.float32)
            )

        sugar_box_path = get_data_path("SugarBox/sugar_box_usd/sugar_box.usda")
        sugar_box: RigidObject = self.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="sugar_box_beta",
                shape=MeshCfg(fpath=sugar_box_path),
                body_type="dynamic",
                use_usd_properties=True,
                init_pos=[1.0, 1.0, 0.1],
            )
        )
        body0 = sugar_box._entities[0].get_physical_body()
        print(sugar_box._entities[0].get_physical_attr())
        assert pytest.approx(body0.get_mass(), 0.001) == 0.514
        # TODO: nvidia physx attrs in usd currently are not fully suported
        # assert(body0.get_linear_damping()==0)
        # assert(body0.get_angular_damping()==0.05)
        # assert(body0.get_solver_iteration_counts()==(4, 1))
        # assert(body0.get_max_angular_velocity()==100)

    def export_usd(self):
        self.sim.export_usd("test_export.usda")

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


class TestUsdCPU(BaseUsdTest):
    def setup_method(self):
        self.setup_simulation("cpu")


@pytest.mark.skip(reason="Skipping CUDA tests temporarily")
class TestUsdCUDA(BaseUsdTest):
    def setup_method(self):
        self.setup_simulation("cuda")


if __name__ == "__main__":
    test = TestUsdCPU()
    test.setup_method()
    test.test_import_rigid()
    test.test_import_articulation()
    test.export_usd()
    test.test_usd_properties()

    # from IPython import embed
    # embed()
