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

import os
from unittest.mock import Mock

import torch
import pytest

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import RigidBodyGroupData, RigidObjectGroup
from embodichain.lab.sim.cfg import (
    RigidObjectGroupCfg,
    RigidObjectCfg,
    physics_cfg_for_backend,
)
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.data import get_data_path
from dexsim.types import ActorType

DUCK_PATH = "ToyDuck/toy_duck.glb"
TABLE_PATH = "ShopTableSimple/shop_table_simple.ply"
NUM_ARENAS = 4
Z_TRANSLATION = 2.0


def _teardown_newton_physics() -> None:
    from dexsim.engine.newton_physics import teardown_newton_physics

    teardown_newton_physics()


@pytest.mark.no_sim
def test_cpu_body_data_reads_angular_velocity_from_angular_api():
    """CPU rigid-object groups must not report linear velocity as angular."""
    expected = torch.tensor([[[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]]])
    body_view = Mock()
    body_view.fetch_angular_velocity.side_effect = lambda out: out.copy_(
        expected.reshape(-1, 3)
    )
    body_data = RigidBodyGroupData(
        body_view,
        num_instances=1,
        num_objects=2,
        device=torch.device("cpu"),
    )

    angular_velocity = body_data.ang_vel

    assert torch.equal(angular_velocity, expected)
    body_view.fetch_angular_velocity.assert_called_once()
    body_view.fetch_linear_velocity.assert_not_called()


class BaseRigidObjectGroupTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, device: str, physics: str = "default") -> None:
        config = SimulationManagerCfg(
            headless=True,
            device=device,
            num_envs=NUM_ARENAS,
            physics_cfg=physics_cfg_for_backend(physics),
        )
        self.sim = SimulationManager(config)
        self.physics = physics

        duck_path = get_data_path(DUCK_PATH)
        assert os.path.isfile(duck_path)
        table_path = get_data_path(TABLE_PATH)
        assert os.path.isfile(table_path)

        cfg_dict = {
            "uid": "group",
            "rigid_objects": {
                "duck1": {
                    "shape": {
                        "shape_type": "Mesh",
                        "fpath": duck_path,
                    },
                },
                "duck2": {
                    "shape": {
                        "shape_type": "Mesh",
                        "fpath": duck_path,
                    },
                },
            },
        }
        self.obj_group: RigidObjectGroup = self.sim.add_rigid_object_group(
            cfg=RigidObjectGroupCfg.from_dict(cfg_dict)
        )

        self.sim.prepare()

        self.sim.enable_physics(True)

    def test_local_pose_behavior(self):

        # Set initial poses
        pose_duck1 = torch.eye(4, device=self.sim.device)
        pose_duck1[2, 3] = Z_TRANSLATION
        pose_duck1 = pose_duck1.unsqueeze(0).repeat(NUM_ARENAS, 1, 1)

        pose_duck2 = torch.eye(4, device=self.sim.device)
        pose_duck2[2, 3] = Z_TRANSLATION
        pose_duck2 = pose_duck2.unsqueeze(0).repeat(NUM_ARENAS, 1, 1)

        combined_pose = torch.stack([pose_duck1, pose_duck2], dim=1)

        self.obj_group.set_local_pose(combined_pose)
        group_pos = self.obj_group.get_local_pose()[..., :3]
        assert torch.allclose(
            group_pos,
            combined_pose[..., :3, 3],
            atol=1e-5,
        ), "FAIL: Local poses do not match after setting."

    def test_body_data_exposes_mass_properties(self):
        """Current and initialization-time properties use [env, object] layout."""
        data = self.obj_group.body_data
        expected_prefix = (NUM_ARENAS, self.obj_group.num_objects)

        assert data.mass.shape == expected_prefix
        assert data.inertia.shape == (*expected_prefix, 3)
        assert data.com_pose.shape == (*expected_prefix, 7)
        assert data.default_mass.shape == data.mass.shape
        assert data.default_inertia.shape == data.inertia.shape
        assert data.default_com_pose.shape == data.com_pose.shape

    def test_reset_restores_default_mass_properties(self):
        """Partial reset restores Group mass properties only in selected envs."""
        data = self.obj_group.body_data
        env_ids = [0, 1]
        obj_ids = [0]
        default_mass = data.default_mass[env_ids, :1].clone()
        default_inertia = data.default_inertia[env_ids, :1].clone()
        default_com_pose = data.default_com_pose[env_ids, :1].clone()
        changed_mass = default_mass + 0.5
        changed_inertia = default_inertia * 1.25
        changed_com_pose = default_com_pose.clone()
        changed_com_pose[..., 0] += 0.02

        self.obj_group.set_mass(changed_mass, env_ids=env_ids, obj_ids=obj_ids)
        self.obj_group.set_inertia(
            changed_inertia,
            env_ids=env_ids,
            obj_ids=obj_ids,
        )
        self.obj_group.set_com_pose(
            changed_com_pose,
            env_ids=env_ids,
            obj_ids=obj_ids,
        )

        assert torch.allclose(data.default_mass[env_ids, :1], default_mass)
        assert torch.allclose(data.default_inertia[env_ids, :1], default_inertia)
        assert torch.allclose(data.default_com_pose[env_ids, :1], default_com_pose)

        self.obj_group.reset(env_ids=[env_ids[0]])
        mass_after_partial = self.obj_group.get_mass(env_ids=env_ids, obj_ids=obj_ids)
        inertia_after_partial = self.obj_group.get_inertia(
            env_ids=env_ids, obj_ids=obj_ids
        )
        com_after_partial = self.obj_group.get_com_pose(
            env_ids=env_ids, obj_ids=obj_ids
        )

        assert torch.allclose(mass_after_partial[0], default_mass[0], atol=1e-5)
        assert torch.allclose(mass_after_partial[1], changed_mass[1], atol=1e-5)
        assert torch.allclose(inertia_after_partial[0], default_inertia[0], atol=1e-5)
        assert torch.allclose(inertia_after_partial[1], changed_inertia[1], atol=1e-5)
        assert torch.allclose(com_after_partial[0], default_com_pose[0], atol=1e-5)
        assert torch.allclose(com_after_partial[1], changed_com_pose[1], atol=1e-5)

        self.obj_group.reset(env_ids=[env_ids[1]])
        assert torch.allclose(
            self.obj_group.get_mass(env_ids=env_ids, obj_ids=obj_ids),
            default_mass,
            atol=1e-5,
        )
        assert torch.allclose(
            self.obj_group.get_inertia(env_ids=env_ids, obj_ids=obj_ids),
            default_inertia,
            atol=1e-5,
        )
        assert torch.allclose(
            self.obj_group.get_com_pose(env_ids=env_ids, obj_ids=obj_ids),
            default_com_pose,
            atol=1e-5,
        )

    def test_get_user_ids(self):
        """Test get_user_ids method."""
        user_ids = self.obj_group.get_user_ids()

        assert user_ids.shape == (NUM_ARENAS, self.obj_group.num_objects), (
            f"Unexpected user_ids shape: {user_ids.shape}, "
            f"expected ({NUM_ARENAS}, {self.obj_group.num_objects})"
        )

    def test_get_object_mesh_geometry(self):
        """Test constituent render geometry used by external visualizers."""
        vertices = self.obj_group.get_object_vertices(
            0,
            env_ids=[0],
            scale=True,
        )
        triangles = self.obj_group.get_object_triangles(0, env_ids=[0])

        assert vertices.ndim == 3 and vertices.shape[0] == 1
        assert triangles.ndim == 3 and triangles.shape[0] == 1
        assert int(triangles.max()) < vertices.shape[1]

    def test_remove(self):
        self.sim.remove_asset(self.obj_group.uid)

        assert (
            self.obj_group.uid not in self.sim.asset_uids
        ), "Object group UID still present after removal"

    def test_set_physical_visible(self):
        self.obj_group.set_physical_visible(visible=True)
        self.obj_group.set_physical_visible(visible=False)

    def test_set_visible(self):
        self.obj_group.set_visible(visible=True)
        self.obj_group.set_visible(visible=False)

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        self.__dict__.clear()
        import gc

        gc.collect()


class TestRigidObjectGroupCPU(BaseRigidObjectGroupTest):
    def setup_method(self):
        self.setup_simulation("cpu")


class TestRigidObjectGroupCUDA(BaseRigidObjectGroupTest):
    def setup_method(self):
        self.setup_simulation("cuda")


class TestRigidObjectGroupNewton(BaseRigidObjectGroupTest):
    def setup_method(self):
        self.setup_simulation("cuda", physics="newton")

    def teardown_method(self):
        super().teardown_method()
        _teardown_newton_physics()


if __name__ == "__main__":
    # pytest.main(["-s", __file__])
    test = TestRigidObjectGroupCPU()
    test.setup_method()
    test.test_local_pose_behavior()
