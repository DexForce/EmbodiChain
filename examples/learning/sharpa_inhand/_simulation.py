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

"""Newton/SimulationManager acquisition for the Sharpa APG example."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch
from embodichain_tasks.manipulation.dexterous.sharpa_inhand import (
    _JOINT_NAMES,
    _REFERENCE_HAND_POSE,
    _REFERENCE_OBJECT_POSITION,
    SharpaInHandCfg,
    _observation,
    _reward,
    _target,
)

from embodichain.data import get_data_path
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
from embodichain.utils.math import matrix_from_quat, quat_mul

__all__ = ["SharpaSimulation"]


def _cylinder_source(output: Path, scale: float = 1.0) -> UsdSourceCfg:
    """Author the source task's analytic cylinder through the existing USD route."""
    from pxr import Usd, UsdGeom, UsdPhysics

    output.mkdir(parents=True, exist_ok=True)
    path = output / "cylinder.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Xform.Define(stage, "/CylinderBody")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.05)
    cylinder = UsdGeom.Cylinder.Define(stage, "/CylinderBody/Geometry")
    cylinder.CreateRadiusAttr(0.02 * scale)
    cylinder.CreateHeightAttr(0.032 * scale)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    stage.SetDefaultPrim(body.GetPrim())
    stage.GetRootLayer().Save()
    return UsdSourceCfg(file_path=str(path), source_prim_path="/CylinderBody")


class SharpaSimulation:
    """Acquire physical transitions before reset through project simulation APIs."""

    def __init__(
        self,
        hand_urdf: str,
        grasp_cache: Path,
        output: Path,
        *,
        num_envs: int,
        device: str,
        seed: int,
        renderer: str = "no-render",
        randomize_yaw: bool = False,
        reference_profile: dict[str, Any] | None = None,
    ) -> None:
        self.num_envs, self.device = num_envs, torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.cfg = SharpaInHandCfg()
        self.reference_profile = reference_profile
        self.randomize_yaw = randomize_yaw
        armature, friction = {}, {}
        for name in _JOINT_NAMES:
            if "thumb_CMC" in name:
                values = (0.0032, 0.132)
            elif "pinky_CMC" in name:
                values = (0.00012, 0.012)
            elif "MCP" in name:
                values = (0.00265, 0.07456)
            elif name.endswith("DIP"):
                values = (0.00042, 0.00378738)
            else:
                values = (0.0006, 0.01276)
            armature[name], friction[name] = values
        stiffness: float | dict[str, float] = self.cfg.stiffness
        damping: float | dict[str, float] = self.cfg.damping
        if reference_profile is not None:
            authored = reference_profile["joints"]
            stiffness = {n: authored[n]["kp"] for n in _JOINT_NAMES}
            damping = {
                n: authored[n]["kd"]
                + (
                    0.0
                    if reference_profile.get("source_task_profile", False)
                    else authored[n]["passive_damping"]
                )
                for n in _JOINT_NAMES
            }
            armature = {n: authored[n]["armature"] for n in _JOINT_NAMES}
            friction = {n: authored[n]["friction"] for n in _JOINT_NAMES}
        source = Path(get_data_path(hand_urdf)).resolve()
        raw = np.load(grasp_cache, allow_pickle=False)
        if (
            raw.ndim != 2
            or raw.shape[1] != 29
            or len(raw) < 1
            or not np.isfinite(raw).all()
        ):
            raise ValueError(
                "Grasp cache must contain finite (N,29) hand/object states"
            )
        if not np.allclose(np.linalg.norm(raw[:, 25:29], axis=-1), 1, atol=1e-4):
            raise ValueError("Grasp cache quaternions must be normalized WXYZ")
        self.grasps = torch.as_tensor(raw, device=self.device, dtype=torch.float32)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        urdf = ET.parse(source).getroot()
        dependencies = {}
        for element in urdf.iter("mesh"):
            name = element.attrib["filename"]
            path = (source.parent / name).resolve()
            dependencies[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        names = [
            j.attrib["name"]
            for j in urdf.findall("joint")
            if j.attrib["type"] != "fixed"
        ]
        if set(names) != set(_JOINT_NAMES):
            raise ValueError("The task requires the right Sharpa Wave 22-joint model")
        limit = {
            j.attrib["name"]: [
                float(j.find("limit").attrib[k]) for k in ("lower", "upper")
            ]
            for j in urdf.findall("joint")
            if j.attrib["type"] != "fixed"
        }
        self.joint_limits = torch.tensor(
            [limit[n] for n in _JOINT_NAMES], device=self.device
        )
        self.lower, self.upper = self.joint_limits[:, 0], self.joint_limits[:, 1]
        valid = (
            (self.grasps[:, :22] >= self.lower) & (self.grasps[:, :22] <= self.upper)
        ).all(-1)
        # The upstream reset writes every cached pose, including compliant
        # poses just beyond physical hinge limits. Preserve that distribution.
        if reference_profile is not None:
            valid.fill_(True)
        self.valid_grasp_rows = valid.nonzero().flatten()
        if len(self.valid_grasp_rows) == 0:
            raise ValueError("No grasp-cache rows satisfy this URDF's joint limits")
        self.grasps = self.grasps[self.valid_grasp_rows]
        cylinder_source = _cylinder_source(
            output,
            reference_profile.get("object_scale", 1.0) if reference_profile else 1.0,
        )
        self.sim: SimulationManager | None = None
        self.robot: Any = None
        self.object: Any = None
        try:
            self.sim = SimulationManager(
                SimulationManagerCfg(
                    headless=True,
                    device=str(self.device),
                    num_envs=num_envs,
                    arena_space=1.0,
                    physics_dt=1.0 / 240.0,
                    render_cfg=RenderCfg(renderer=renderer),
                    physics_cfg=NewtonPhysicsCfg(
                        device=str(self.device),
                        gravity=(
                            (0.0, 0.0, 0.0) if reference_profile else (0.0, 0.0, -9.81)
                        ),
                        num_substeps=1,
                        use_cuda_graph=True,
                        solver_cfg={
                            "solver_type": "mujoco_warp",
                            "solver": (
                                "newton"
                                if reference_profile
                                and reference_profile.get("source_task_profile", False)
                                else "cg"
                            ),
                            **(
                                {
                                    "integrator": "euler",
                                    "ls_iterations": 50,
                                    "tolerance": 1e-8,
                                    "ls_tolerance": 0.01,
                                }
                                if reference_profile
                                and reference_profile.get("source_task_profile", False)
                                else {}
                            ),
                            "iterations": 100,
                            **(
                                {"save_to_mjcf": str(output / "native_scene.xml")}
                                if reference_profile
                                else {}
                            ),
                        },
                    ),
                )
            )
            self.robot = self.sim.add_robot(
                RobotCfg(
                    uid="hand",
                    fpath=str(source) if reference_profile is None else None,
                    source=(
                        None
                        if reference_profile is None
                        else UsdSourceCfg(
                            file_path=reference_profile["native_source"],
                            source_prim_path="/Hand",
                        )
                    ),
                    init_pos=(0.0, 0.0, 0.5),
                    init_rot=(0.0, -70.0, 0.0),
                    root_props=ArticulationRootPropertiesCfg(
                        fixed_base=True,
                        self_collision_enabled=reference_profile is not None,
                    ),
                    asset_physics_mode="overlay",
                    joint_drive_props=JointDrivePropertiesCfg(
                        drive_type="force",
                        stiffness=stiffness,
                        damping=damping,
                        max_effort=1000.0 if reference_profile else 1.0,
                        max_velocity=1000.0 if reference_profile else 10.0,
                        armature=armature,
                        friction=friction,
                    ),
                    attrs=RigidBodyPhysicsCfg(
                        collision_props=(
                            NewtonCollisionPropertiesCfg(condim=6, margin=0.0, gap=0.0)
                            if reference_profile
                            else NewtonCollisionPropertiesCfg(condim=6)
                        ),
                        material_props=NewtonRigidBodyMaterialCfg(
                            dynamic_friction=None if reference_profile else 2.0,
                            torsional_friction=0.1,
                            rolling_friction=0.02,
                            ke=2500.0,
                            kd=100.0,
                        ),
                    ),
                )
            )
            self.object = self.sim.add_rigid_object(
                RigidObjectCfg(
                    uid="cylinder",
                    source=cylinder_source,
                    asset_physics_mode="overlay",
                    attrs=RigidBodyPhysicsCfg(
                        mass_props=MassPropertiesCfg(mass=0.05),
                        collision_props=(
                            NewtonCollisionPropertiesCfg(condim=6, margin=0.0, gap=0.0)
                            if reference_profile
                            else NewtonCollisionPropertiesCfg(condim=6)
                        ),
                        material_props=NewtonRigidBodyMaterialCfg(
                            dynamic_friction=2.0,
                            torsional_friction=0.1,
                            rolling_friction=0.02,
                            ke=10000.0,
                            kd=200.0,
                        ),
                    ),
                    init_pos=_REFERENCE_OBJECT_POSITION,
                )
            )
            self.sim.prepare()
            self.joint_ids = [
                self.robot.joint_names.index(name) for name in _JOINT_NAMES
            ]
            self.context = torch.empty(num_envs, 47, device=self.device)
            self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
            self.episode_ids = torch.zeros_like(self.steps)
            self.reset()
            self.gravity_force = torch.zeros(num_envs, 3, device=self.device)
            self.gravity_force[:, 2] = -0.05 * 9.81
        except Exception:
            self.robot = self.object = None
            if self.sim is not None:
                self.sim.destroy(exit_process=False)
                self.sim = None
            raise
        self.identity = {
            "profile": "sharpa-reference-native-cylinder-v3",
            "object_geometry": "native-cylinder",
            "hand_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "hand_mesh_sha256": dependencies,
            "grasp_cache_sha256": hashlib.sha256(grasp_cache.read_bytes()).hexdigest(),
            "grasp_rows_accepted": len(self.valid_grasp_rows),
            "grasp_rows_rejected_joint_limits": len(raw) - len(self.valid_grasp_rows),
            "joint_names": list(_JOINT_NAMES),
            "physics": "newton",
            "solver": "mujoco_warp",
            "solver_algorithm": "cg",
            "solver_iterations": 100,
            "physics_substeps": 1,
            "physics_dt": 1 / 240,
            "control_dt": 0.05,
            "object_radius_m": 0.02,
            "object_height_m": 0.032,
            "object_mass_kg": 0.05,
            "hand_stiffness": self.cfg.stiffness,
            "hand_damping": self.cfg.damping,
            "hand_effort_cap_nm": 1.0,
            "joint_armature": armature,
            "joint_passive_friction": friction,
            "contact_dimension": 6,
            "torsional_friction": 0.1,
            "rolling_friction": 0.02,
            "hand_contact_ke_kd": [2500.0, 100.0],
            "object_contact_ke_kd": [10000.0, 200.0],
            "hand_self_collision": False,
            "hand_gravity": True,
            "friction": 2.0,
            "action_scale_rad": self.cfg.action_scale,
            "drop_height_band_m": self.cfg.drop_height_band,
            "observation": "privileged-fixed-palm-and-reset-109-v3",
            "randomization": (
                "grasp-cache-and-cylinder-32fold-yaw"
                if randomize_yaw
                else "grasp-cache-only"
            ),
        }
        if reference_profile is not None:
            self.identity.update(
                profile="native-compiled-reference-v1",
                reference_source=reference_profile,
                hand_gravity="compensated; zero world gravity plus object weight each physics step",
                hand_effort_cap_nm=1000.0,
                hand_stiffness=stiffness,
                hand_damping=damping,
                hand_self_collision=True,
                observation="native-task-owns-HORA-frames",
                randomization="grasp-cache-only; fixed-scale-1 calibration",
            )
        self.reward_identity = {
            "version": "sharpa-fixed-references-control-period-v3",
            "object_position_reference": list(_REFERENCE_OBJECT_POSITION),
            "hand_pose_reference_rad": list(_REFERENCE_HAND_POSE),
            "velocities": "completed control-period finite differences",
            "drop_reference": "sampled reset height",
        }

    def read_state(self) -> torch.Tensor:
        """Copy canonical object-root and ordered hand state, in local arena axes."""
        body = self.object.body_state
        rotation = matrix_from_quat(body[:, 3:7]).flatten(1)
        hand = self.robot.body_data.fetch_state()
        return torch.cat(
            (
                body[:, :3],
                rotation,
                body[:, 7:13],
                hand["qpos"][:, self.joint_ids],
                hand["qvel"][:, self.joint_ids],
            ),
            -1,
        ).clone()

    def observation(self) -> torch.Tensor:
        """Return the exact same privileged actor inputs used in imagination."""
        return _observation(self.read_state(), self.context, self.lower, self.upper)

    @torch.no_grad()
    def reset(
        self, env_ids: torch.Tensor | None = None, *, seed: int | None = None
    ) -> torch.Tensor:
        """Reset selected rows from the cache; clear physics before writing state."""
        if seed is not None:
            self.generator.manual_seed(seed)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids
        )
        if len(ids) == 0:
            return self.read_state()
        self.sim.reset_objects_state(env_ids=ids)
        choices = torch.randint(
            len(self.grasps), (len(ids),), generator=self.generator, device=self.device
        )
        grasp = self.grasps[choices]
        q = (
            grasp[:, :22]
            if self.reference_profile
            else grasp[:, :22].clamp(self.lower, self.upper)
        )
        clamp = not (
            self.reference_profile
            and self.reference_profile.get("source_task_profile", False)
        )
        self.robot.set_state(
            qpos=q,
            clamp_qpos=clamp,
            clamp_target_qpos=clamp,
            target_qpos=q,
            qvel=torch.zeros_like(q),
            joint_ids=self.joint_ids,
            env_ids=ids,
        )
        pose = torch.cat((grasp[:, 22:25], grasp[:, 26:29], grasp[:, 25:26]), -1)
        if self.randomize_yaw:
            # The native cylinder is axially symmetric. Sample 32 body phases
            # while preserving its world axis and geometry; every resulting
            # transition is still measured in simulation.
            phase = torch.randint(
                32, (len(ids),), device=self.device, generator=self.generator
            ) * (2 * torch.pi / 32)
            twist = torch.zeros(len(ids), 4, device=self.device)
            twist[:, 2], twist[:, 3] = (phase / 2).sin(), (phase / 2).cos()
            pose[:, 3:7] = quat_mul(pose[:, 3:7], twist)
        self.object.set_local_pose(pose, env_ids=ids)
        zeros = torch.zeros(len(ids), 3, device=self.device)
        self.object.set_velocity(zeros, zeros, env_ids=ids)
        self.context[ids] = torch.cat((q, grasp[:, 22:25], q), -1)
        self.steps[ids] = 0
        self.episode_ids[ids] += 1
        return self.read_state()

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Capture current/next physical states before caller-owned reset."""
        if action.shape != (self.num_envs, 22) or not torch.isfinite(action).all():
            raise ValueError("Physical actions must have finite shape (num_envs,22)")
        state, before = self.read_state(), self.context.clone()
        target = _target(before, action, self.lower, self.upper, self.cfg)
        self.robot.set_qpos(target, joint_ids=self.joint_ids, target=True)
        self.sim.update(step=12)
        next_state = self.read_state()
        if not torch.isfinite(next_state).all():
            raise RuntimeError("Nonfinite physical transition")
        self.context = torch.cat((target, before[:, 22:]), -1)
        self.steps += 1
        return {
            "state": state,
            "action": target,
            "next_state": next_state,
            "context": before,
            "reward": _reward(next_state, self.context, self.cfg, previous_state=state),
            "terminated": (next_state[:, 2] - before[:, 24]).abs()
            > self.cfg.drop_height_band,
            "truncated": self.steps >= self.cfg.max_episode_steps,
        }

    def close(self) -> None:
        """Drop native children and queue cleanup; flush after caller frames exit."""
        self.robot = self.object = None
        manager, self.sim = self.sim, None
        if manager is not None:
            manager.destroy(exit_process=False)
            manager = None
