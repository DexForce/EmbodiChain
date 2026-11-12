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
"""Native Wuji physical collection; no upstream simulator is constructed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
    MassPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    RobotCfg,
    UsdSourceCfg,
)
from embodichain.lab.sim.physics._cylinder_variants import _shrink_cylinder_variants
from embodichain.lab.sim.sensors.contact._geometry_contact_sensor import (
    _GeometryNetForceSensor,
)
from embodichain.utils.math import quat_apply

__all__: list[str] = []

JOINTS = tuple(f"right_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5))
SCALES = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)


class _WujiPhysics:
    """Own one prepared native hand/cylinder per arena and measured state."""

    def __init__(
        self, assets: Path, output: Path, *, num_envs: int, seed: int, gain: float = 1.0
    ) -> None:
        if num_envs < 8 or num_envs % 8:
            raise ValueError("Use a positive multiple of eight arenas")
        from pxr import Usd, UsdGeom, UsdPhysics

        self.num_envs = num_envs
        self.device = torch.device("cuda:0")
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.width = num_envs // 8
        self.scale_ids = torch.arange(8, device=self.device).repeat_interleave(
            self.width
        )
        self.scales = torch.tensor(SCALES, dtype=torch.float32, device=self.device)[
            self.scale_ids
        ]
        self.profile = json.loads((assets / "compiled/source.json").read_text())
        root_pose = np.eye(4)
        quat = np.asarray(self.profile["root_quat_wxyz"])
        root_pose[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        root_pose[:3, 3] = self.profile["root_position"]
        if not np.allclose(root_pose[:3, 0], [0, 0, 1], atol=1e-6):
            raise ValueError(
                "Wuji v2 requires its local palmar +X normal to face world +Z"
            )
        self.tip_centers = torch.tensor(
            self.profile["tip_geometry_centers"],
            dtype=torch.float32,
            device=self.device,
        )
        self.root_rotation = torch.tensor(
            root_pose[:3, :3], dtype=torch.float32, device=self.device
        )
        authored = self.profile["joints"]
        self.kp = torch.tensor(
            [authored[n]["kp"] * gain for n in JOINTS], device=self.device
        )
        self.kd = torch.tensor(
            [authored[n]["kd"] * gain**0.5 for n in JOINTS], device=self.device
        )
        self.effort = {n: authored[n]["max_effort"] for n in JOINTS}
        self.lower = torch.tensor(
            [authored[n]["ctrl_range"][0] for n in JOINTS], device=self.device
        )
        self.upper = torch.tensor(
            [authored[n]["ctrl_range"][1] for n in JOINTS], device=self.device
        )
        self.cache = [
            torch.as_tensor(
                np.load(assets / f"candidates_{s:g}.npy"),
                dtype=torch.float32,
                device=self.device,
            )
            for s in SCALES
        ]
        output.mkdir(parents=True, exist_ok=True)
        cylinder = output / "cylinder.usda"
        stage = Usd.Stage.CreateNew(str(cylinder))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        body = UsdGeom.Xform.Define(stage, "/CylinderBody")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.05)
        geom = UsdGeom.Cylinder.Define(stage, "/CylinderBody/Geometry")
        geom.CreateRadiusAttr(0.03)
        geom.CreateHeightAttr(0.048)
        geom.CreateAxisAttr(UsdGeom.Tokens.z)
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        stage.SetDefaultPrim(body.GetPrim())
        stage.GetRootLayer().Save()
        self.sim: SimulationManager | None = None
        self.robot: Any = None
        self.object: Any = None
        self.contact: Any = None
        try:
            self.sim = SimulationManager(
                SimulationManagerCfg(
                    headless=True,
                    device=str(self.device),
                    num_envs=num_envs,
                    arena_space=1.0,
                    physics_dt=1 / 240,
                    render_cfg=RenderCfg(renderer="no-render"),
                    physics_cfg=NewtonPhysicsCfg(
                        device=str(self.device),
                        gravity=(0.0, 0.0, 0.0),
                        num_substeps=1,
                        use_cuda_graph=True,
                        solver_cfg={
                            "solver_type": "mujoco_warp",
                            "solver": "newton",
                            "integrator": "euler",
                            "iterations": 100,
                            "ls_iterations": 50,
                            "tolerance": 1e-8,
                            "ls_tolerance": 0.01,
                            "save_to_mjcf": str(output / "native_scene.xml"),
                        },
                    ),
                )
            )
            self.robot = self.sim.add_robot(
                RobotCfg(
                    uid="hand",
                    source=UsdSourceCfg(
                        file_path=self.profile["native_source"],
                        source_prim_path="/Hand",
                    ),
                    init_local_pose=root_pose,
                    root_props=ArticulationRootPropertiesCfg(
                        fixed_base=True, self_collision_enabled=True
                    ),
                    asset_physics_mode="overlay",
                    joint_drive_props=JointDrivePropertiesCfg(
                        drive_type="force",
                        stiffness={n: authored[n]["kp"] * gain for n in JOINTS},
                        damping={n: authored[n]["kd"] * gain**0.5 for n in JOINTS},
                        max_effort=self.effort,
                        max_velocity=100.0,
                        armature={n: authored[n]["armature"] for n in JOINTS},
                        friction={n: authored[n]["friction"] for n in JOINTS},
                    ),
                    attrs=RigidBodyPhysicsCfg(
                        collision_props=NewtonCollisionPropertiesCfg(
                            condim=6, margin=0.0, gap=0.0
                        ),
                        material_props=NewtonRigidBodyMaterialCfg(
                            dynamic_friction=2.0,
                            torsional_friction=0.005,
                            rolling_friction=0.0001,
                            ke=2500.0,
                            kd=100.0,
                        ),
                    ),
                )
            )
            self.object = self.sim.add_rigid_object(
                RigidObjectCfg(
                    uid="cylinder",
                    source=UsdSourceCfg(
                        file_path=str(cylinder), source_prim_path="/CylinderBody"
                    ),
                    init_pos=(-0.1, 0.0, 0.55),
                    asset_physics_mode="overlay",
                    attrs=RigidBodyPhysicsCfg(
                        mass_props=MassPropertiesCfg(mass=0.05),
                        collision_props=NewtonCollisionPropertiesCfg(
                            condim=6, margin=0.0, gap=0.0
                        ),
                        material_props=NewtonRigidBodyMaterialCfg(
                            dynamic_friction=2.0,
                            torsional_friction=0.005,
                            rolling_friction=0.0001,
                            ke=10000.0,
                            kd=200.0,
                        ),
                    ),
                )
            )
            self.sim.prepare()
            palm = self.robot.get_link_pose("right_palm_link")
            axis = torch.zeros(num_envs, 3, device=self.device)
            axis[:, 0] = 1
            normal = quat_apply(palm[:, 3:7], axis)
            if not bool((normal[:, 2] > 0.999).all()):
                raise RuntimeError("Native Wuji palm is not facing upward")
            self.arena_offsets = palm[:, :3] - torch.tensor(
                root_pose[:3, 3], dtype=torch.float32, device=self.device
            )
            self.joint_ids = [self.robot.joint_names.index(n) for n in JOINTS]
            _shrink_cylinder_variants(self.sim, "cylinder", self.scales / 1.5)
            self.contact = _GeometryNetForceSensor(
                self.sim,
                "hand",
                "cylinder",
                tuple(
                    (f"right_finger{i}_link4", f"right_finger{i}_link4_geom_3")
                    for i in range(1, 6)
                ),
            )
            self.force = torch.zeros(num_envs, 3, device=self.device)
            self.force[:, 2] = -0.05 * 9.81
            self.target = torch.zeros(num_envs, 20, device=self.device)
        except BaseException:
            self.close()
            raise
        self.identity = {
            "hand": "Wuji Hand V1 right",
            "physical_dof": 20,
            "task_version": "wuji-palm-up-v2",
            "palm_normal_world": root_pose[:3, 0].tolist(),
            "joint_names": JOINTS,
            "host": "EmbodiChain SimulationManager",
            "physics": "newton",
            "solver": "mujoco_warp/newton/euler",
            "upstream_environment_constructed": False,
            "scales": SCALES,
            "pd_gain_multiplier": gain,
            "torque_limit_nm": self.effort,
            "gravity": "compensated hand; object weight applied each substep",
            "object_mass_kg": 0.05,
            "physics_dt": 1 / 240,
            "control_dt": 0.05,
            "source_profile": self.profile,
            "tactile": "native fingertip-geometry/object net-force",
            "task_scope": "Native Wuji task, no claimed Sharpa physics equivalence",
        }
        (output / "native_identity.json").write_text(
            json.dumps(self.identity, indent=2) + "\n"
        )

    @torch.no_grad()
    def reset(
        self,
        ids: torch.Tensor | None = None,
        *,
        choices: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        if seed is not None:
            self.generator.manual_seed(seed)
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        poses = torch.empty(len(ids), self.cache[0].shape[1], device=self.device)
        for scale in range(8):
            local = (self.scale_ids[ids] == scale).nonzero().flatten()
            if not len(local):
                continue
            selected = (
                torch.randint(
                    len(self.cache[scale]),
                    (len(local),),
                    device=self.device,
                    generator=self.generator,
                )
                if choices is None
                else choices[local] % len(self.cache[scale])
            )
            poses[local] = self.cache[scale][selected]
        q = poses[:, :20]
        targets = poses[:, 27:47] if poses.shape[1] == 47 else q
        self.robot.set_state(
            qpos=q,
            target_qpos=targets,
            qvel=torch.zeros_like(q),
            joint_ids=self.joint_ids,
            env_ids=ids,
            clear_dynamics=True,
            clamp_qpos=False,
            clamp_target_qpos=False,
        )
        pose = torch.cat((poses[:, 20:23], poses[:, 24:27], poses[:, 23:24]), -1)
        self.object.clear_dynamics(env_ids=ids)
        self.object.set_local_pose(pose, env_ids=ids)
        self.target[ids] = targets
        self.contact.reset(ids)
        return poses

    def state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body = self.object.body_state.clone()
        joints = self.robot.body_data.fetch_state()["qpos"][:, self.joint_ids].clone()
        quat = torch.cat((body[:, 6:7], body[:, 3:6]), -1)
        return body[:, :3], quat, joints

    def tip_positions(self) -> torch.Tensor:
        """Return source fingertip-mesh centers in each arena's local world frame."""
        poses = torch.stack(
            [self.robot.get_link_pose(f"right_finger{i}_link4") for i in range(1, 6)], 1
        )
        return (
            poses[..., :3]
            + quat_apply(
                poses[..., 3:7], self.tip_centers.expand(self.num_envs, -1, -1)
            )
            - self.arena_offsets[:, None]
        )

    def grasp_contact(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Measure thumb plus two opposing non-thumb fingertip contacts."""
        force = self.contact.measure()
        self.last_tip_positions = self.tip_positions()
        delta = self.last_tip_positions - self.object.body_state[:, None, :3]
        radial = delta[..., :2]
        radial = torch.nn.functional.normalize(radial, dim=-1, eps=1e-6)
        side = delta[..., 2].abs() < (0.8 * 0.016 * self.scales[:, None])
        opposite = (radial[:, 1:] * radial[:, :1]).sum(-1) < -0.15
        touch = (force > 0.02) & side
        valid = touch[:, 0] & ((touch[:, 1:] & opposite).sum(-1) >= 2)
        return force, valid

    @torch.no_grad()
    def step(self, target: torch.Tensor) -> None:
        self.robot.set_state(
            target_qpos=target, joint_ids=self.joint_ids, clamp_target_qpos=False
        )
        for _ in range(12):
            self.object.add_force_torque(self.force)
            self.sim.update(step=1)

    def close(self) -> None:
        if self.contact is not None:
            self.contact.close()
            self.contact = None
        self.robot = self.object = None
        manager, self.sim = self.sim, None
        if manager is not None:
            manager.destroy(exit_process=False)
