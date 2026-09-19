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
import torch

from embodichain.lab.sim import (
    SimulationManager,
    SimulationManagerCfg,
)
from embodichain.lab.sim.objects import Articulation, RigidObject
from embodichain.lab.sim.cfg import (
    RenderCfg,
    ArticulationCfg,
    RigidObjectCfg,
    JointDrivePropertiesCfg,
    RigidBodyPhysicsCfg,
    MassPropertiesCfg,
    DefaultRigidBodyPropertiesCfg,
    physics_cfg_for_backend,
)
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.data import get_data_path

NUM_ARENAS = 2


@pytest.mark.parametrize("physics", ["default", "newton"])
def test_usdc_fk_matches_simulation_at_named_joint_states(
    tmp_path: Path, physics: str
) -> None:
    """Resolved USD FK agrees with physics and does not change joint state."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "kinematic_tree.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    assembly = UsdGeom.Xform.Define(stage, "/assembly")
    stage.SetDefaultPrim(assembly.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(assembly.GetPrim())
    for name in ("base", "drawer", "handle"):
        body = UsdGeom.Xform.Define(stage, f"/assembly/{name}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
        cube = UsdGeom.Cube.Define(stage, f"/assembly/{name}/mesh")
        cube.CreateSizeAttr(0.1)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    for name, schema, parent, child in (
        ("slide", UsdPhysics.PrismaticJoint, "base", "drawer"),
        ("hinge", UsdPhysics.RevoluteJoint, "drawer", "handle"),
    ):
        joint = schema.Define(stage, f"/assembly/{name}")
        joint.CreateBody0Rel().SetTargets([f"/assembly/{parent}"])
        joint.CreateBody1Rel().SetTargets([f"/assembly/{child}"])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(-1.0 if name == "slide" else -90.0)
        joint.CreateUpperLimitAttr(1.0 if name == "slide" else 90.0)
        # Nonzero child-side frames exercise the USD-specific FK conversion.
        for attr in (joint.CreateLocalPos0Attr(), joint.CreateLocalPos1Attr()):
            attr.Set(Gf.Vec3f(0.0, 0.0, 0.2))
        rotation = Gf.Quatf(0.70710678, Gf.Vec3f(0.0, 0.0, 0.70710678))
        joint.CreateLocalRot0Attr(rotation)
        joint.CreateLocalRot1Attr(rotation)
    stage.GetRootLayer().Save()

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=1,
            physics_cfg=physics_cfg_for_backend(physics),
        )
    )
    obj = None
    try:
        obj = sim.add_articulation(
            ArticulationCfg(
                uid="usd_fk",
                fpath=str(path),
                init_pos=(0.5, -0.3, 1.0),
                init_rot=(10.0, 20.0, 30.0),
            )
        )
        sim.prepare()
        assert obj.pk_chain is not None
        states = ((0.0, 0.0), (0.13, 0.4), (-0.07, -0.2))
        if physics == "newton":
            states = ((0.13, 0.4),)
        for slide, hinge in states:
            state = {"slide": slide, "hinge": hinge}
            qpos = torch.tensor([[state[name] for name in obj.joint_names]])
            obj.set_qpos(qpos, target=False)
            # Default publishes physical link poses on the next update. Compare
            # with the measured state after that update, not the command.
            sim.update()
            before = obj.get_qpos().clone()
            fk = obj.compute_fk(
                before, link_names=obj.link_names, qpos_joint_names=obj.joint_names
            )
            root = obj.get_link_pose("base", to_matrix=True)
            for index, name in enumerate(obj.link_names):
                observed = torch.linalg.inv(root) @ obj.get_link_pose(
                    name, to_matrix=True
                )
                torch.testing.assert_close(fk[:, index], observed, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(obj.get_qpos(), before)
    finally:
        sim.destroy(exit_process=False)
        del obj
        SimulationManager.flush_cleanup_queue()


class BaseUsdTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, device):
        config = SimulationManagerCfg(
            headless=True,
            device=device,
            num_envs=NUM_ARENAS,
        )
        self.sim = SimulationManager(config)

    def test_import_rigid(self):
        default_attr = RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(mass=1.25),
            rigid_props=DefaultRigidBodyPropertiesCfg(
                linear_damping=0.2,
                angular_damping=0.3,
                min_position_iters=8,
                min_velocity_iters=2,
            ),
        )
        sugar_box_path = get_data_path("SugarBox/sugar_box_usd/sugar_box.usda")
        sugar_box: RigidObject = self.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="sugar_box",
                shape=MeshCfg(fpath=sugar_box_path),
                body_type="dynamic",
                asset_physics_mode="overlay",
                init_pos=[0.0, 1.0, 0.1],
                attrs=default_attr,
            )
        )
        self.sim.prepare()
        torch.testing.assert_close(
            sugar_box.get_mass(),
            torch.full_like(sugar_box.get_mass(), default_attr.mass_props.mass),
        )
        expected_damping = torch.tensor(
            [
                default_attr.rigid_props.linear_damping,
                default_attr.rigid_props.angular_damping,
            ],
            device=self.sim.device,
        ).expand(NUM_ARENAS, -1)
        torch.testing.assert_close(sugar_box.get_damping(), expected_damping)
        for entity in sugar_box._entities:
            attr = entity.get_physical_attr()
            assert (
                attr.min_position_iters == default_attr.rigid_props.min_position_iters
            )
            assert (
                attr.min_velocity_iters == default_attr.rigid_props.min_velocity_iters
            )
        assert len(sugar_box._entities) == NUM_ARENAS
        handles = {entity.get_native_handle() for entity in sugar_box._entities}
        assert len(handles) == NUM_ARENAS

    def test_import_articulation(self):
        default_drive = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness=1e4,
            damping=1e3,
            max_effort=1e10,
            max_velocity=1e10,
            friction=0.0,
            armature=0.0,
        )
        h1_path = get_data_path("UnitreeH1Usd/H1_usd/h1.usd")
        h1: Articulation = self.sim.add_articulation(
            cfg=ArticulationCfg(
                uid="h1",
                fpath=h1_path,
                build_pk_chain=False,
                asset_physics_mode="overlay",
                init_pos=[0.0, 0.0, 1.2],
                joint_drive_props=default_drive,
            )
        )
        self.sim.prepare()

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
        """Verify that preserve mode keeps physics authored in USD assets."""
        h1_path = get_data_path("UnitreeH1Usd/H1_usd/h1.usd")
        h1: Articulation = self.sim.add_articulation(
            cfg=ArticulationCfg(
                uid="h1_beta",
                fpath=h1_path,
                build_pk_chain=False,
                asset_physics_mode="preserve",
                init_pos=[1.0, 0.0, 1.2],
            )
        )
        self.sim.prepare()

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
                asset_physics_mode="preserve",
                init_pos=[1.0, 1.0, 0.1],
            )
        )
        self.sim.prepare()
        torch.testing.assert_close(
            sugar_box.get_mass(),
            torch.full_like(sugar_box.get_mass(), 0.514),
            rtol=0.001,
            atol=0.0,
        )

    def export_usd(self):
        self.sim.export_usd("test_export.usda")

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        self.__dict__.clear()
        import gc

        gc.collect()


# @pytest.mark.skip(reason="Skipping CUDA tests temporarily")
class TestUsdCPU(BaseUsdTest):
    def setup_method(self):
        self.setup_simulation("cpu")


# @pytest.mark.skip(reason="Skipping CUDA tests temporarily")
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
