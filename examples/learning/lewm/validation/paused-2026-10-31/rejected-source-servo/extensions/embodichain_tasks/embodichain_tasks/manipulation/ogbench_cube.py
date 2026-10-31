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

"""Official OGBench single-Cube control and scene in the DexSim backend.

Compiled reference assets are produced by the LeWM ``official-reference`` tool.
This module does not import MuJoCo or the upstream environment at runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import gymnasium as gym
import numpy as np
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation
import torch
from tensordict import TensorDict

from embodichain.compute.dynamics._inertia import (
    _generalized_mass_matrix,
    _implicit_velocity_step,
)
from embodichain.lab.gym.envs import BaseEnv, EnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    DefaultRigidBodyPropertiesCfg,
    DefaultRigidBodyMaterialCfg,
    DefaultPhysicsCfg,
    JointDrivePropertiesCfg,
    LightCfg,
    LinkPhysicsOverrideCfg,
    MassPropertiesCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RigidConstraintCfg,
    RigidObjectCfg,
    RobotCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import configclass
from embodichain.utils.math import matrix_from_quat, xyz_quat_to_4x4_matrix

__all__ = ["OGBenchCubeCfg", "OGBenchCubeEnv"]


def _base_mimic_compliance(
    solref: Sequence[float], solimp: Sequence[float]
) -> tuple[float, float]:
    """Map the base MuJoCo equality force law to native mimic compliance.

    MuJoCo's regularizer is ``R = (1-d)/d * A0`` and its spring reference
    includes another impedance factor. A direct inverse-time-constant mapping
    omits both factors. PhysX uses angular frequency directly, without a
    ``2*pi`` conversion. The official driver pair has matching actual and
    approximate inverse inertia. This preserves its base force response;
    position-dependent ``solimp`` remains a separate alignment limitation.
    """
    timeconst, ratio = solref[:2]
    base_impedance, maximum_impedance = solimp[:2]
    if not (
        timeconst > 0 and ratio > 0 and 0 < base_impedance <= maximum_impedance < 1
    ):
        raise ValueError("official Cube requires positive, soft equality parameters")
    scale = np.sqrt(1 - base_impedance)
    frequency = base_impedance / (maximum_impedance * timeconst * ratio * scale)
    return float(frequency), float(ratio / scale)


def _base_point_compliance(
    solref: Sequence[float],
    solimp: Sequence[float],
    inverse_weight: float,
) -> tuple[float, float]:
    """Map source point regularization to spring/damping force coefficients.

    MuJoCo uses the compiled approximate inverse weight in ``R``. A native
    acceleration drive applies a different mass normalization, including a
    configuration-dependent response on the gripper. Use force units instead.
    This preserves the base KBIP/R force law; nonlinear impedance and coupled
    solver dynamics still require independent validation.
    """
    timeconst, ratio = solref[:2]
    d0, dmax = solimp[:2]
    if not (
        timeconst > 0
        and ratio > 0
        and 0 < d0 <= dmax < 1
        and np.isfinite(inverse_weight)
        and inverse_weight > 0
    ):
        raise ValueError("official Cube requires positive soft point inverse weight")
    regularizer = (1 - d0) / d0 * inverse_weight
    stiffness = d0 / (dmax * timeconst * ratio) ** 2 / regularizer
    damping = 2 / (dmax * timeconst) / regularizer
    return float(stiffness), float(damping)


def _source_body_external_wrench(
    pose: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    incoming: torch.Tensor,
    mass: torch.Tensor,
    com: torch.Tensor,
    inertia: torch.Tensor,
    joint_frames: torch.Tensor,
    body_index: int,
    children: Sequence[int],
    gravity: torch.Tensor,
) -> torch.Tensor:
    """Recover the source body's external torque/force about robot subtree COM.

    Inputs use the public link order. Incoming wrenches are force/torque in
    child-joint axes, motion is world-frame at COM, and poses are xyz+xyzw.
    RNE removes gravity/inertia and outgoing child reactions; an incoming
    joint wrench alone is not the source external/contact wrench.
    """
    rotations = matrix_from_quat(pose[..., 3:])
    centers = pose[..., :3] + (rotations @ com[None, ..., None]).squeeze(-1)
    subtree_com = (centers * mass[None, :, None]).sum(1) / mass.sum()

    def joint_wrench(index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame = joint_frames[:, index]
        rotation = rotations[:, index] @ frame[:, :3, :3]
        position = pose[:, index, :3] + (
            rotations[:, index] @ frame[:, :3, 3, None]
        ).squeeze(-1)
        force = (rotation @ incoming[:, index, :3, None]).squeeze(-1)
        torque = (rotation @ incoming[:, index, 3:, None]).squeeze(-1)
        torque = torque + torch.linalg.cross(
            position - centers[:, body_index], force, dim=-1
        )
        return force, torque

    force, torque = joint_wrench(body_index)
    external_force = (
        mass[body_index] * (acceleration[:, body_index, :3] - gravity) - force
    )
    world_inertia = (
        rotations[:, body_index] @ inertia @ rotations[:, body_index].transpose(-2, -1)
    )
    omega = velocity[:, body_index, 3:]
    external_torque = (world_inertia @ acceleration[:, body_index, 3:, None]).squeeze(
        -1
    )
    external_torque = (
        external_torque
        + torch.linalg.cross(
            omega, (world_inertia @ omega[..., None]).squeeze(-1), dim=-1
        )
        - torque
    )
    for child in children:
        child_force, child_torque = joint_wrench(child)
        external_force = external_force + child_force
        external_torque = external_torque + child_torque
    external_torque = external_torque + torch.linalg.cross(
        centers[:, body_index] - subtree_com, external_force, dim=-1
    )
    return torch.cat([external_torque, external_force], dim=-1)


def _source_state_observation(
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    pinch: torch.Tensor,
    cube: torch.Tensor,
    pad_wrench: torch.Tensor,
) -> torch.Tensor:
    """Assemble the original single-cube 28-D state (source quaternion order)."""
    center = qpos.new_tensor([0.425, 0, 0])
    yaw = torch.atan2(pinch[:, 1, 0], pinch[:, 0, 0])
    rotation = matrix_from_quat(cube[:, 3:])
    cube_yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return torch.cat(
        [
            qpos[:, :6],
            qvel[:, :6],
            (pinch[:, :3, 3] - center) * 10,
            yaw.cos()[:, None],
            yaw.sin()[:, None],
            (qpos[:, 6:7] / 0.8).clamp(0, 1) * 3,
            (torch.linalg.vector_norm(pad_wrench, dim=-1, keepdim=True) / 50).clamp(
                0, 1
            ),
            (cube[:, :3] - center) * 10,
            torch.roll(cube[:, 3:], 1, dims=-1),
            cube_yaw.cos()[:, None],
            cube_yaw.sin()[:, None],
        ],
        dim=-1,
    )


class _SourceDynamics:
    """Convert the compiled source actuator law and fixed-base inertia.

    Topology and inertial data are immutable for this task. Native link poses
    provide the current kinematics; no source simulator runs in the native task.
    The source implicitfast derivative uses mass-tree sparsity, including its
    omission of sibling tendon cross-terms.
    """

    def __init__(self, env: OGBenchCubeEnv) -> None:
        self.dtype = torch.float64
        device = env.device
        links, joints = env.robot.link_names, env._names
        bodies = {body["name"]: body for body in env.asset["bodies"]}

        def tensor(value: Any) -> torch.Tensor:
            return torch.as_tensor(value, dtype=self.dtype, device=device)

        self.mass = tensor([bodies[name]["mass"] for name in links])
        self.com = tensor([bodies[name]["ipos"] for name in links])
        self.inertia = tensor([bodies[name]["inertia_matrix"] for name in links])
        self.armature = tensor(env.contract["dof_armature"][:14])
        self.passive_damping = tensor(env.contract["dof_damping"][:14])
        self.stiffness = tensor(env.contract["jnt_stiffness"][:14])
        self.resting = tensor(env.contract["qpos_spring"][:14])
        self.ancestors = torch.zeros(
            (len(links), len(joints)), dtype=self.dtype, device=device
        )
        metadata = {}
        for link_index, name in enumerate(links):
            for joint in env.robot.get_parent_joint_chain(name):
                if joint.name in joints:
                    if joint.joint_type not in {"revolute", "continuous"}:
                        raise ValueError(
                            "official Cube dynamics requires scalar rotary joints"
                        )
                    self.ancestors[link_index, joints.index(joint.name)] = 1
                    metadata[joint.name] = joint
        if set(metadata) != set(joints):
            raise ValueError("official Cube dynamics has incomplete joint topology")
        self.parent_ids = torch.tensor(
            [links.index(metadata[name].parent_link_name) for name in joints],
            dtype=torch.int64,
            device=device,
        )
        origins = torch.stack([metadata[name].origin_pose for name in joints]).to(
            device=device, dtype=self.dtype
        )
        axes = torch.stack([metadata[name].axis for name in joints]).to(
            device=device, dtype=self.dtype
        )
        self.joint_offset = origins[:, :3, 3]
        self.axis_parent = (origins[:, :3, :3] @ axes[..., None]).squeeze(-1)
        self.mass_pattern = (self.ancestors.T @ self.ancestors) > 0
        actuators = env.contract["actuators"]
        self.gain = tensor([a["gainprm"][0] for a in actuators])
        self.bias = tensor([a["biasprm"][:3] for a in actuators])
        self.limits = tensor([a["forcerange"] for a in actuators])
        self.ctrl_limits = tensor([a["ctrlrange"] for a in actuators])
        self.tendon = torch.zeros(14, dtype=self.dtype, device=device)
        for side in ("right", "left"):
            self.tendon[joints.index(f"ur5e/robotiq/{side}_driver_joint")] = 0.5

    def mass_matrix(self, pose: torch.Tensor) -> torch.Tensor:
        """Assemble all 14 generalized coordinates, including armature."""
        pose = pose.to(dtype=self.dtype)
        rotations = matrix_from_quat(pose[..., 3:])
        centers = pose[..., :3] + (rotations @ self.com[None, ..., None]).squeeze(-1)
        parent_rotation = rotations[:, self.parent_ids]
        origins = pose[:, self.parent_ids, :3] + (
            parent_rotation @ self.joint_offset[None, ..., None]
        ).squeeze(-1)
        axes = (parent_rotation @ self.axis_parent[None, ..., None]).squeeze(-1)
        lever = centers[:, :, None] - origins[:, None]
        linear = torch.linalg.cross(axes[:, None].expand_as(lever), lever, dim=-1)
        linear = linear * self.ancestors[None, :, :, None]
        angular = axes[:, None] * self.ancestors[None, :, :, None]
        world_inertia = rotations @ self.inertia[None] @ rotations.transpose(-2, -1)
        return _generalized_mass_matrix(
            linear, angular, self.mass, world_inertia, self.armature
        )

    def forces_and_derivative(
        self, qpos: torch.Tensor, qvel: torch.Tensor, control: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate old-state force clipping and the source smooth derivative."""
        control = control.clamp(self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
        length = torch.cat((qpos[:, :6], (qpos @ self.tendon)[:, None]), dim=1)
        velocity = torch.cat((qvel[:, :6], (qvel @ self.tendon)[:, None]), dim=1)
        raw = (
            self.gain * control
            + self.bias[:, 0]
            + self.bias[:, 1] * length
            + self.bias[:, 2] * velocity
        )
        force = raw.clamp(self.limits[:, 0], self.limits[:, 1])
        active = (raw > self.limits[:, 0]) & (raw < self.limits[:, 1])
        generalized = (
            -self.stiffness * (qpos - self.resting) - self.passive_damping * qvel
        )
        generalized[:, :6] += force[:, :6]
        generalized += force[:, 6:7] * self.tendon
        derivative = torch.diag_embed(
            self.passive_damping.expand(len(qpos), -1)
        ).clone()
        derivative[:, :6, :6] += torch.diag_embed(-self.bias[:6, 2] * active[:, :6])
        derivative += (-self.bias[6, 2] * active[:, 6, None, None]) * torch.outer(
            self.tendon, self.tendon
        )
        return generalized, derivative * self.mass_pattern


class _CubeController:
    """Apply the reference IK iteration to the compiled arm kinematic chain."""

    def __init__(self, root: Path, robot: dict[str, Any]) -> None:
        bindings = {frame["name"]: frame for frame in robot["source_frames"]}
        attach = bindings["ur5e/attachment_site"]
        self.chain = pk.build_serial_chain_from_urdf(
            (root / "robot/robot.urdf").read_text(), attach["link"], "ur5e/base"
        ).to(dtype=torch.float64)
        self.root = np.array(robot["root_pose"])
        self.tool = pk.Transform3d(
            dtype=torch.float64,
            matrix=torch.tensor(attach["transform"], dtype=torch.float64)[None],
        )

    def solve(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        qpos = np.array(seed, dtype=np.float64, copy=True)
        for _ in range(20):
            jac, pose = self.chain.jacobian(
                torch.from_numpy(qpos[None]), locations=self.tool, ret_eef_pose=True
            )
            current = self.root @ pose[0].numpy()
            error = np.concatenate(
                (
                    target[:3, 3] - current[:3, 3],
                    Rotation.from_matrix(
                        target[:3, :3] @ current[:3, :3].T
                    ).as_rotvec(),
                )
            )
            if np.linalg.norm(error[:3]) <= 1e-4 and np.linalg.norm(error[3:]) <= 1e-4:
                break
            jacobian = jac[0].numpy().copy()
            jacobian[:3] = self.root[:3, :3] @ jacobian[:3]
            jacobian[3:] = self.root[:3, :3] @ jacobian[3:]
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + np.eye(6) * 1e-12, error
            )
            largest = np.max(np.abs(update))
            if largest > np.pi / 4:
                update *= (np.pi / 4) / largest
            qpos += update
        return qpos


@configclass
class OGBenchCubeCfg(EnvCfg):
    """Configure the compiled official single-Cube task."""

    reference_root: str = MISSING
    """Directory containing the audited physical contract and robot assets."""
    enable_images: bool = True
    """Acquire the calibrated 224-pixel reference camera."""
    enable_source_state: bool = True
    """Emit the original 28-D state using retained fixed-pad link measurements."""
    sim_steps_per_control: int = 25
    sim_cfg: SimulationManagerCfg = SimulationManagerCfg(
        physics_cfg=DefaultPhysicsCfg(
            physics_dt=0.002, device="cuda:0", cache_material=True, enable_tgs=False
        ),
        headless=True,
        enable_default_plane=False,
    )
    max_episode_steps: int = 100
    seed: int = 42


@register_env("OGBenchCube-v0", max_episode_steps=100)
class OGBenchCubeEnv(BaseEnv):
    """Single cube with reference XYZ/yaw/gripper control and point linkages."""

    def __init__(self, cfg: OGBenchCubeCfg, **kwargs: Any) -> None:
        cfg.validate()
        if cfg.enable_images and cfg.num_envs > 32:
            raise ValueError(
                "the Cube camera path supports at most 32 environments per World; "
                "use official-eval-dexsim's persistent workers for 50 queries"
            )
        if (
            not isinstance(cfg.sim_cfg.physics_cfg, DefaultPhysicsCfg)
            or not cfg.sim_cfg.physics_cfg.cache_material
        ):
            raise ValueError(
                "official Cube requires DefaultPhysicsCfg(cache_material=True) for GPU contact materials"
            )
        if cfg.sim_cfg.physics_cfg.enable_tgs:
            raise ValueError(
                "official Cube point compliance requires enable_tgs=False (PGS)"
            )
        self.root = Path(cfg.reference_root)
        self.contract = json.loads((self.root / "physical-contract.json").read_text())
        if not {"body_invweight0", "sleep_enabled"}.issubset(self.contract):
            raise ValueError(
                "point inverse weights and sleep intent are missing; re-export official-reference"
            )
        if self.contract["sleep_enabled"]:
            raise ValueError("official Cube requires source sleeping to be disabled")
        if not np.allclose(
            cfg.sim_cfg.physics_cfg.gravity,
            self.contract["gravity"],
            rtol=1e-7,
            atol=1e-9,
        ):
            raise ValueError("official Cube requires the compiled source gravity")
        self.asset = json.loads((self.root / "robot/robot.json").read_text())
        if cfg.enable_source_state:
            required = {"ur5e/robotiq/right_pad", "ur5e/robotiq/right_silicone_pad"}
            if not required.issubset(body["name"] for body in self.asset["bodies"]):
                raise ValueError(
                    "source state requires positive fixed-pad bodies; re-export official-reference"
                )
            if torch.device(cfg.sim_cfg.device).type != "cuda":
                raise ValueError(
                    "source state link-wrench measurements require Default Direct GPU"
                )
        if (
            hashlib.sha256((self.root / "robot/robot.urdf").read_bytes()).hexdigest()
            != self.asset["sha256"]
        ):
            raise ValueError("official Cube robot checksum mismatch")
        if (
            cfg.num_envs < 1
            or cfg.sim_steps_per_control != 25
            or abs(cfg.sim_cfg.physics_dt - 0.002) > 1e-12
        ):
            raise ValueError("official Cube requires 0.002 s physics and 25 substeps")
        if cfg.sim_cfg.enable_default_plane:
            raise ValueError(
                "official Cube owns its ground; set enable_default_plane=False"
            )
        self.bindings = {frame["name"]: frame for frame in self.asset["source_frames"]}
        self.controller = _CubeController(self.root, self.asset)
        self._replay_no_auto_reset = True
        self._names = self.asset["joint_names"]
        self._goal = np.tile([0.425, -0.1, 0.02], (cfg.num_envs, 1))
        self._control = np.zeros((cfg.num_envs, 7))
        self._control[:, :6] = self.contract["home_qpos"]
        super().__init__(cfg, **kwargs)

    def _requires_native_renderer(self) -> bool:
        return self.cfg.enable_images or super()._requires_native_renderer()

    def _declare_robot(self, **kwargs: Any):
        attrs = {}
        actuators = self.contract["actuators"]
        efforts = {
            name: (
                max(abs(x) for x in actuators[index]["forcerange"])
                if index < 6
                else 0.0
            )
            for index, name in enumerate(self._names)
        }
        for body in self.asset["bodies"]:
            attrs[body["name"]] = LinkPhysicsOverrideCfg(
                link_names_expr=[re.escape(body["name"])],
                attrs=RigidBodyPhysicsCfg(
                    mass_props=MassPropertiesCfg(
                        mass=body["mass"],
                        inertia=np.array(body["inertia_matrix"]),
                        com_position=body["ipos"],
                    )
                ),
            )
        for index, geom in enumerate(self.contract["geoms"]):
            if geom["name"].endswith(
                ("/right_pad1", "/right_pad2", "/left_pad1", "/left_pad2")
            ):
                link = self.bindings[geom["body"]]["link"]
                friction = geom["friction"][0]
                # The pads' higher source priority selects their friction in
                # cube/floor contacts; the native min rule expresses that pair.
                attrs[link].collision_materials[f"geom_{index}"] = (
                    DefaultRigidBodyMaterialCfg(
                        static_friction=friction,
                        dynamic_friction=friction,
                        friction_combine_mode="min",
                    )
                )
        return self.sim.add_robot(
            RobotCfg(
                uid="ogbench_robot",
                fpath=str((self.root / "robot/robot.urdf").resolve()),
                urdf_override_material=True,
                init_local_pose=np.array(self.asset["root_pose"]),
                control_parts={"arm": self._names[:6], "eef": self._names[6:]},
                init_qpos=self.contract["home_qpos"] + [0.0] * 8,
                solver_cfg=None,
                build_pk_chain=False,
                asset_physics_mode="overlay",
                attrs=RigidBodyPhysicsCfg(
                    rigid_props=DefaultRigidBodyPropertiesCfg(
                        linear_damping=0, angular_damping=0, has_gravity=True
                    ),
                    material_props=RigidBodyMaterialCfg(
                        static_friction=1, dynamic_friction=1, restitution=0
                    ),
                    collision_props=CollisionPropertiesCfg(
                        contact_offset=1e-6, rest_offset=0
                    ),
                ),
                link_attrs=attrs,
                filtered_link_pairs=sorted(
                    {
                        tuple(
                            sorted(
                                (
                                    self.bindings[first]["link"],
                                    self.bindings[second]["link"],
                                )
                            )
                        )
                        for first, second in self.contract["excluded_body_pairs"]
                        if self.bindings[first]["link"] != self.bindings[second]["link"]
                    }
                    | {
                        tuple(sorted(pair))
                        for pair in self.asset.get("implicit_collision_exclusions", [])
                    }
                ),
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True,
                    self_collision_enabled=True,
                    sleep_threshold=0,
                    min_position_iters=100,
                    min_velocity_iters=1,
                ),
                joint_drive_props=JointDrivePropertiesCfg(
                    drive_type="force",
                    target_mode="effort",
                    stiffness=0,
                    damping=0,
                    max_effort=efforts,
                    friction=0,
                    armature=dict(zip(self._names, self.contract["dof_armature"][:14])),
                ),
            )
        )

    def _prepare_scene(self, **kwargs: Any) -> None:
        def physics(mass: float | None = None):
            return RigidBodyPhysicsCfg(
                mass_props=MassPropertiesCfg(mass=mass) if mass is not None else None,
                rigid_props=DefaultRigidBodyPropertiesCfg(
                    linear_damping=0,
                    angular_damping=0,
                    sleep_threshold=0 if mass is not None else None,
                    min_position_iters=100,
                    min_velocity_iters=1,
                ),
                collision_props=CollisionPropertiesCfg(
                    contact_offset=1e-6, rest_offset=0
                ),
                material_props=RigidBodyMaterialCfg(
                    static_friction=1, dynamic_friction=1, restitution=0
                ),
            )

        self.cube = self.sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube",
                init_pos=(0.425, 0.1, 0.02),
                shape=CubeCfg(
                    size=[0.04] * 3,
                    visual_material=VisualMaterialCfg(
                        base_color=[0.96, 0.26, 0.33, 1], roughness=1
                    ),
                ),
                attrs=physics(0.07936),
            )
        )
        self.sim.add_rigid_object(
            RigidObjectCfg(
                uid="ogbench_floor",
                body_type="static",
                init_pos=(0, 0, -0.01),
                shape=CubeCfg(
                    size=[2, 2, 0.02],
                    visual_material=VisualMaterialCfg(
                        base_color=[1, 1, 1, 1],
                        roughness=1,
                        base_color_texture=str(self.root / "textures/grid.png"),
                    ),
                ),
                attrs=physics(),
            )
        )
        self.walls = []
        for geom in self.contract["geoms"]:
            if geom["name"].startswith("wall_"):
                self.walls.append(
                    self.sim.add_rigid_object(
                        RigidObjectCfg(
                            uid=geom["name"],
                            body_type="static",
                            init_pos=geom["pos"],
                            collision_filter_data=[0, 1, 2, 1],
                            shape=CubeCfg(size=(np.array(geom["size"]) * 2).tolist()),
                            attrs=physics(),
                        )
                    )
                )
        if self.cfg.enable_images:
            self.sim.add_light(
                LightCfg(
                    uid="ogbench_light",
                    light_type="direction",
                    intensity=5,
                    direction=(0, 0, -1),
                )
            )

    def _frame_pose(self, name: str) -> torch.Tensor:
        binding = self.bindings[name]
        local = torch.as_tensor(
            binding["transform"], dtype=torch.float32, device=self.device
        )
        return self.robot.get_link_pose(binding["link"], to_matrix=True) @ local

    def _setup_robot(self, **kwargs: Any):
        self._joint_ids = [self.robot.joint_names.index(name) for name in self._names]
        self.single_action_space = gym.spaces.Box(-1, 1, (5,), dtype=np.float32)
        robot_filter = torch.zeros(
            (self.num_envs, 4), dtype=torch.int32, device=self.device
        )
        robot_filter[:, 0] = torch.arange(self.num_envs, device=self.device)
        robot_filter[:, 1], robot_filter[:, 2] = 1, 1
        self.robot.set_collision_filter(robot_filter)
        for wall in self.walls:
            wall.set_visible(False)
        equality = self.contract["equality"]
        for index, kind in enumerate(equality["type"]):
            if kind != 0:
                continue
            body_a = self.contract["body_names"][equality["obj1id"][index]]
            body_b = self.contract["body_names"][equality["obj2id"][index]]
            first, second = self.bindings[body_a], self.bindings[body_b]
            frame_a, frame_b = np.array(first["transform"]), np.array(
                second["transform"]
            )
            anchor_a, anchor_b = np.eye(4), np.eye(4)
            anchor_a[:3, 3] = equality["data"][index][:3]
            anchor_b[:3, 3] = equality["data"][index][3:6]
            reference = equality["solref"][index]
            impedance = equality["solimp"][index]
            inverse_weight = sum(
                self.contract["body_invweight0"][equality[key][index]][0]
                for key in ("obj1id", "obj2id")
            )
            stiffness, damping = _base_point_compliance(
                reference, impedance, inverse_weight
            )
            self.sim.create_rigid_constraint(
                RigidConstraintCfg(
                    name=f"ogbench_connect_{index}",
                    rigid_object_a_uid=self.robot.uid,
                    rigid_object_b_uid=self.robot.uid,
                    articulation_link_a=first["link"],
                    articulation_link_b=second["link"],
                    local_frame_a=frame_a @ anchor_a,
                    local_frame_b=frame_b @ anchor_b,
                    constraint_type="spherical",
                    linear_drive_stiffness=stiffness,
                    linear_drive_damping=damping,
                    linear_drive_acceleration=False,
                )
            )
        if not self.asset.get("mimics"):
            raise ValueError(
                "official Cube requires the compiled driver joint equality"
            )
        mimic = self.asset["mimics"][0]
        self.robot.set_mimic_compliance(
            *_base_mimic_compliance(mimic["solref"], mimic["solimp"])
        )
        self._source_dynamics = _SourceDynamics(self)
        self._source_control_pose: torch.Tensor | None = None
        self._source_step_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self.cfg.enable_source_state:
            self._setup_source_state()
        return self.robot

    def _setup_source_state(self) -> None:
        names = self.robot.link_names
        bodies = {body["name"]: body for body in self.asset["bodies"]}
        self._obs_mass = torch.tensor(
            [bodies[name]["mass"] for name in names],
            dtype=torch.float32,
            device=self.device,
        )
        self._obs_com = torch.tensor(
            [bodies[name]["ipos"] for name in names],
            dtype=torch.float32,
            device=self.device,
        )
        pad_name = "ur5e/robotiq/right_pad"
        self._obs_pad = names.index(pad_name)
        self._obs_children = tuple(
            i for i, name in enumerate(names) if bodies[name]["parent_link"] == pad_name
        )
        self._obs_inertia = torch.tensor(
            bodies[pad_name]["inertia_matrix"], dtype=torch.float32, device=self.device
        )
        self._obs_joint_frames = self.robot.body_data.body_link_incoming_joint_frames
        self._obs_gravity = torch.as_tensor(
            self.cfg.sim_cfg.physics_cfg.gravity,
            dtype=torch.float32,
            device=self.device,
        )
        self._state_wrench_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._source_wrench_cache = torch.zeros(
            (self.num_envs, 6), dtype=torch.float32, device=self.device
        )
        self._has_source_dynamics = False

    def _setup_sensors(self, **kwargs: Any):
        if not self.cfg.enable_images:
            return {}
        camera = self.contract["camera"]
        rotation = Rotation.from_quat(np.roll(camera["quat_wxyz"], -1)).as_matrix()
        optical = Rotation.from_matrix(rotation @ np.diag([1, -1, -1])).as_quat()
        focal = 112 / np.tan(np.radians(camera["fovy_degrees"]) / 2)
        sensor = self.sim.add_sensor(
            CameraCfg(
                uid="main_cam",
                width=224,
                height=224,
                near=camera["znear"],
                far=camera["zfar"],
                intrinsics=(focal, focal, 112, 112),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    pos=tuple(camera["pos"]), quat=tuple(optical)
                ),
            )
        )
        return {"main_cam": sensor}

    def _initialize_episode(self, env_ids=None, **kwargs: Any) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        count = len(ids)
        default = np.r_[
            self.contract["home_qpos"], np.zeros(8), [0.425, 0.1, 0.02], [1, 0, 0, 0]
        ]
        qpos = np.broadcast_to(
            np.asarray(kwargs.get("qpos", default)), (count, 21)
        ).copy()
        qvel = np.broadcast_to(
            np.asarray(kwargs.get("qvel", np.zeros(20))), (count, 20)
        ).copy()
        self.robot.set_state(
            env_ids=ids,
            joint_ids=self._joint_ids,
            qpos=torch.as_tensor(qpos[:, :14], device=self.device),
            qvel=torch.as_tensor(qvel[:, :14], device=self.device),
            clear_dynamics=True,
            clamp_qpos=False,
        )
        pose = np.concatenate(
            (qpos[:, 14:17], np.roll(qpos[:, 17:21], -1, axis=1)), axis=1
        )
        self.cube.set_local_pose(
            torch.as_tensor(pose, device=self.device, dtype=torch.float32), env_ids=ids
        )
        angular_world = Rotation.from_quat(pose[:, 3:]).apply(qvel[:, 17:20])
        self.cube.set_velocity(
            lin_vel=torch.as_tensor(
                qvel[:, 14:17], dtype=torch.float32, device=self.device
            ),
            ang_vel=torch.as_tensor(
                angular_world, dtype=torch.float32, device=self.device
            ),
            env_ids=ids,
        )
        if "goal_pos" in kwargs:
            self._goal[ids.cpu().numpy()] = np.broadcast_to(
                kwargs["goal_pos"], (count, 3)
            )
        self._control[ids.cpu().numpy(), :6] = qpos[:, :6]
        self._control[ids.cpu().numpy(), 6] = (
            qpos[:, self._names.index("ur5e/robotiq/right_driver_joint")] / 0.8 * 255
        )
        self._source_step_valid[ids] = False
        if self.cfg.enable_source_state:
            self._state_wrench_valid[ids] = False
            if len(ids) == self.num_envs:
                self._has_source_dynamics = False

    def _preprocess_action(self, action):
        return torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(
            self.num_envs, 5
        )

    def _step_action(self, action):
        raw = action.detach().cpu().numpy() * np.array(self.contract["action_range"])
        pinch = self._control_frame_pose("ur5e/robotiq/pinch").cpu().numpy()
        qpos = self.robot.get_qpos()[:, self._joint_ids].cpu().numpy()
        for index in range(self.num_envs):
            # The down-facing pinch frame's yaw uses its world X axis.
            yaw = np.arctan2(pinch[index, 1, 0], pinch[index, 0, 0]) + raw[index, 3]
            target = np.eye(4)
            target[:3, 3] = np.clip(
                pinch[index, :3, 3] + raw[index, :3], *self.contract["workspace_bounds"]
            )
            target[:3, :3] = Rotation.from_euler("z", yaw).as_matrix() @ np.diag(
                [1, -1, -1]
            )
            attach = target @ np.array(self.contract["pinch_to_attachment"])
            self._control[index, :6] = self.controller.solve(attach, qpos[index, :6])
            driver = qpos[index, self._names.index("ur5e/robotiq/right_driver_joint")]
            self._control[index, 6] = 255 * np.clip(
                np.clip(driver / 0.8, 0, 1) + raw[index, 4], 0, 1
            )
        limits = np.array([a["ctrlrange"] for a in self.contract["actuators"][:6]])
        targets = np.clip(self._control[:, :6], limits[:, 0], limits[:, 1])
        self.robot.set_qpos(
            torch.as_tensor(targets, device=self.device, dtype=torch.float32),
            joint_ids=self._joint_ids[:6],
            target=True,
        )
        return action

    def _control_frame_pose(self, name: str) -> torch.Tensor:
        """Match source site FK cached before its final physics substep."""
        if self._source_control_pose is None:
            return self._frame_pose(name)
        binding = self.bindings[name]
        index = self.robot.link_names.index(binding["link"])
        current = self.robot.body_data.body_link_pose[:, index]
        pose = torch.where(
            self._source_step_valid[:, None],
            self._source_control_pose[:, index],
            current,
        )
        return xyz_quat_to_4x4_matrix(pose) @ torch.as_tensor(
            binding["transform"], dtype=pose.dtype, device=pose.device
        )

    def _capture_source_wrench(self, before_pose: torch.Tensor) -> None:
        """Recover measured force in its native frame, then shift the moment."""
        data = self.robot.body_data
        raw_pose = data.body_link_pose
        wrench = _source_body_external_wrench(
            raw_pose,
            data.body_link_vel,
            data.body_link_acceleration,
            data.body_link_incoming_joint_force,
            self._obs_mass,
            self._obs_com,
            self._obs_inertia,
            self._obs_joint_frames,
            self._obs_pad,
            self._obs_children,
            self._obs_gravity,
        )

        def subtree_com(pose: torch.Tensor) -> torch.Tensor:
            centers = pose[..., :3] + (
                matrix_from_quat(pose[..., 3:]) @ self._obs_com[None, ..., None]
            ).squeeze(-1)
            return (centers * self._obs_mass[None, :, None]).sum(
                1
            ) / self._obs_mass.sum()

        # Native measurements use the completed explicit step's frame. The
        # source reports moments about its cached pre-integration subtree COM.
        delta = subtree_com(raw_pose) - subtree_com(before_pose)
        wrench[:, :3] += torch.linalg.cross(delta, wrench[:, 3:], dim=-1)
        self._source_wrench_cache = wrench

    @torch.no_grad()
    def _advance_physics(self) -> None:
        control = torch.as_tensor(
            self._control, dtype=torch.float64, device=self.device
        )
        for substep in range(25):
            qpos = self.robot.get_qpos()[:, self._joint_ids].to(torch.float64)
            qvel = self.robot.get_qvel()[:, self._joint_ids].to(torch.float64)
            pose = self.robot.body_data.body_link_pose
            mass = self._source_dynamics.mass_matrix(pose)
            force, derivative = self._source_dynamics.forces_and_derivative(
                qpos, qvel, control
            )
            if substep == 24:
                self._source_control_pose = pose.clone()
            self.robot.set_qf(force.to(torch.float32), joint_ids=self._joint_ids)
            self.sim.update(0.002, step=1, render_final_step=False)
            physics_velocity = self.robot.get_qvel()[:, self._joint_ids].to(
                torch.float64
            )
            if substep == 24 and self.cfg.enable_source_state:
                self._capture_source_wrench(self._source_control_pose)
            position, velocity = _implicit_velocity_step(
                qpos, qvel, physics_velocity, mass, derivative, 0.002
            )
            self.robot.set_state(
                qpos=position,
                qvel=velocity,
                joint_ids=self._joint_ids,
                clear_dynamics=False,
                clamp_qpos=False,
            )
        self._source_step_valid.fill_(True)
        if self.cfg.enable_source_state:
            self._state_wrench_valid.fill_(True)
            self._has_source_dynamics = True

    def evaluate(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        position = self.cube.get_local_pose()[:, :3]
        goal = torch.as_tensor(self._goal, dtype=torch.float32, device=self.device)
        distance = torch.linalg.vector_norm(position - goal, dim=-1)
        return {
            "success": distance <= 0.04,
            "distance_to_goal": distance,
            "fail": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def get_reward(
        self, obs: TensorDict, action: torch.Tensor, info: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return the reference binary position-success reward."""
        return info["success"].float()

    def _extend_obs(self, obs: TensorDict, **kwargs: Any) -> TensorDict:
        cube = self.cube.get_local_pose()
        obs["cube_pose"] = cube
        if self.cfg.enable_source_state:
            if self._has_source_dynamics:
                wrench = self._source_wrench_cache
                wrench = torch.where(self._state_wrench_valid[:, None], wrench, 0)
            else:
                wrench = torch.zeros(
                    (self.num_envs, 6), dtype=torch.float32, device=self.device
                )
            pinch = self._control_frame_pose("ur5e/robotiq/pinch")
            obs["state"] = _source_state_observation(
                self.robot.get_qpos()[:, self._joint_ids],
                self.robot.get_qvel()[:, self._joint_ids],
                pinch,
                cube,
                wrench,
            )
            obs["right_pad_external_wrench"] = wrench
            obs["state_wrench_valid"] = self._state_wrench_valid.clone()
        return obs

    def close(self) -> None:
        """Release the owned simulator without terminating the host process."""
        self.sim.destroy(exit_process=False)
