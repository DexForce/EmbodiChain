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

"""Official OGBench single-Cube control and scene on DexUni.

Compiled reference assets are produced by the LeWM ``official-reference`` tool.
This module does not import MuJoCo or the upstream environment at runtime.
"""

from __future__ import annotations

from dataclasses import MISSING
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any

import gymnasium as gym
import numpy as np
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs import BaseEnv, EnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
    LightCfg,
    LinkPhysicsOverrideCfg,
    MassPropertiesCfg,
    NewtonPhysicsCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    RobotCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg, PlaneCfg
from embodichain.utils import configclass
from embodichain.utils.math import matrix_from_quat

__all__ = ["OGBenchCubeCfg", "OGBenchCubeEnv"]


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


# The compiled scene retains 21 robot links and a free Cube joint.
_CUBE_RIGID_JOINTS = tuple(range(22))


@configclass
class OGBenchCubeCfg(EnvCfg):
    """Configure the compiled OGBench single-Cube task on DexUni CPU."""

    reference_root: str = MISSING
    """Directory containing the audited physical contract and robot assets."""
    enable_images: bool = True
    """Acquire the calibrated 224-pixel reference camera."""
    sim_steps_per_control: int = 25
    sim_cfg: SimulationManagerCfg = SimulationManagerCfg(
        physics_cfg=NewtonPhysicsCfg(
            device="cpu",
            physics_dt=0.002,
            num_substeps=1,
            use_cuda_graph=False,
            solver_cfg={
                "solver_type": "dexuni",
                "mujoco_joints": _CUBE_RIGID_JOINTS,
                "mujoco_options": {
                    "use_mujoco_cpu": True,
                    "use_mujoco_contacts": True,
                    "separate_worlds": False,
                    "solver": "newton",
                    "integrator": "implicitfast",
                    "cone": "elliptic",
                },
            },
        ),
        headless=True,
        enable_ground_plane=False,
    )
    max_episode_steps: int = 100
    seed: int = 42


@register_env(
    "OGBenchCubeDexUni-v0", max_episode_steps=100, supports_handwritten_demo=False
)
class OGBenchCubeEnv(BaseEnv):
    """Single cube with reference XYZ, yaw and gripper control on DexUni."""

    def __init__(self, cfg: OGBenchCubeCfg, **kwargs: Any) -> None:
        cfg.validate()
        self._validate_physics(cfg)
        self.root = Path(cfg.reference_root)
        self.contract = json.loads((self.root / "physical-contract.json").read_text())
        if "sleep_enabled" not in self.contract:
            raise ValueError(
                "Source sleep setting is missing. Re-export official-reference."
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
        if (
            hashlib.sha256((self.root / "robot/robot.urdf").read_bytes()).hexdigest()
            != self.asset["sha256"]
        ):
            raise ValueError("official Cube robot checksum mismatch")
        if (
            cfg.sim_steps_per_control != 25
            or abs(cfg.sim_cfg.physics_dt - 0.002) > 1e-12
        ):
            raise ValueError("official Cube requires 0.002 s physics and 25 substeps")
        if cfg.sim_cfg.enable_ground_plane:
            raise ValueError(
                "official Cube owns its ground; set enable_ground_plane=False"
            )
        self.bindings = {frame["name"]: frame for frame in self.asset["source_frames"]}
        self.controller = _CubeController(self.root, self.asset)
        self._replay_no_auto_reset = True
        self._names = self.asset["joint_names"]
        self._goal = np.tile([0.425, -0.1, 0.02], (cfg.num_envs, 1))
        self._control = np.zeros((cfg.num_envs, 7))
        self._control[:, :6] = self.contract["home_qpos"]
        super().__init__(cfg, **kwargs)

    def _validate_physics(self, cfg: OGBenchCubeCfg) -> None:
        if cfg.num_envs != 1:
            raise ValueError("DexUni CPU requires one Cube environment per World")
        physics = cfg.sim_cfg.physics_cfg
        if not isinstance(physics, NewtonPhysicsCfg):
            raise ValueError("OGBenchCubeDexUni requires NewtonPhysicsCfg")
        solver = physics.to_dexsim_cfg(gpu_id=cfg.sim_cfg.gpu_id).solver_cfg
        if (
            solver.solver_type != "dexuni"
            or tuple(solver.mujoco_joints or ()) != _CUBE_RIGID_JOINTS
            or not solver.mujoco_options.get("use_mujoco_cpu")
            or torch.device(physics.device).type != "cpu"
            or physics.num_substeps != 1
        ):
            raise ValueError(
                "OGBenchCubeDexUni needs the DexUni CPU rigid-body path "
                "with all joints in MuJoCo ownership and one solver step per physics step"
            )
        if (
            cfg.enable_images
            and not (
                Path(cfg.reference_root) / "dexuni-render-materials.json"
            ).is_file()
        ):
            raise ValueError(
                "Prepare assets with 'embodichain run-example learning/lewm/cube prepare'"
            )

    def _requires_native_renderer(self) -> bool:
        return self.cfg.enable_images or super()._requires_native_renderer()

    def _declare_robot(self, **kwargs: Any):
        return self.sim.add_robot(self._robot_cfg())

    def _robot_cfg(self) -> RobotCfg:
        attrs = {}
        actuators = self.contract["actuators"]
        stiffness = {
            name: actuators[index]["gainprm"][0] if index < 6 else 0.0
            for index, name in enumerate(self._names)
        }
        damping = {
            name: -actuators[index]["biasprm"][2] if index < 6 else 0.0
            for index, name in enumerate(self._names)
        }
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
        return RobotCfg(
            uid="ogbench_robot",
            fpath=str((self.root / "robot/robot.urdf").resolve()),
            init_local_pose=np.array(self.asset["root_pose"]),
            control_parts={"arm": self._names[:6], "eef": self._names[6:]},
            init_qpos=self.contract["home_qpos"] + [0.0] * 8,
            solver_cfg=None,
            build_pk_chain=False,
            asset_physics_mode="overlay",
            link_attrs=attrs,
            root_props=ArticulationRootPropertiesCfg(
                fixed_base=True,
                self_collision_enabled=True,
            ),
            joint_drive_props=JointDrivePropertiesCfg(
                drive_type="force",
                target_mode={
                    name: "position_velocity" if index < 6 else "none"
                    for index, name in enumerate(self._names)
                },
                stiffness=stiffness,
                damping=damping,
                max_effort=efforts,
                friction=0,
                armature=dict(zip(self._names, self.contract["dof_armature"][:14])),
            ),
        )

    def _prepare_scene(self, **kwargs: Any) -> None:
        from dexsim.spawn import NewtonEqualityDesc, NewtonSceneDesc

        from ._ogbench_cube_newton import _CubeNewtonSpawnAdapter

        geoms = self.contract["geoms"]
        static = {"floor", *(g["name"] for g in geoms if g["name"].startswith("wall_"))}
        self.walls = []
        for geom in geoms:
            name = geom["name"]
            if name not in static and name != "object_0":
                continue
            shape = (
                PlaneCfg(
                    size=(2.0, 2.0),
                    visual_material=VisualMaterialCfg(
                        base_color=[1, 1, 1, 1],
                        roughness=1,
                        base_color_texture=str(self.root / "textures/grid.png"),
                    ),
                )
                if name == "floor"
                else CubeCfg(
                    size=(2 * np.asarray(geom["size"])).tolist(),
                    visual_material=VisualMaterialCfg(
                        base_color=geom["rgba"], roughness=1
                    ),
                )
            )
            if name in static:
                obj = self.sim.add_rigid_object(
                    RigidObjectCfg(
                        uid=name, shape=shape, body_type="static", init_pos=geom["pos"]
                    )
                )
                if name != "floor":
                    self.walls.append(obj)
            else:
                self.cube = self.sim.add_rigid_object(
                    RigidObjectCfg(
                        uid="cube",
                        shape=shape,
                        init_pos=(0.425, 0.1, 0.02),
                        attrs=RigidBodyPhysicsCfg(
                            mass_props=MassPropertiesCfg(mass=0.07936)
                        ),
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

        # Resolve the ordinary EmbodiChain declarations before authoring the
        # source-specific Newton scene metadata. Spawn materializes them once.
        spawn = self.sim._spawn_scene
        spawn.resolve_sources()
        scene = spawn.builder.scene_desc
        robot = scene.articulations[0]
        robot.filtered_link_pairs = sorted(
            {
                tuple(
                    sorted(
                        (self.bindings[first]["link"], self.bindings[second]["link"])
                    )
                )
                for first, second in self.contract["excluded_body_pairs"]
                if self.bindings[first]["link"] != self.bindings[second]["link"]
            }
            | {
                tuple(sorted(pair))
                for pair in self.asset.get("implicit_collision_exclusions", [])
            }
        )
        by_name = {g["name"]: g for g in geoms}
        collision_sources = {}
        for obj in scene.mesh_objects:
            source = by_name["object_0" if obj.name == "cube" else obj.name]
            collision = obj.collisions[0]
            collision.segment_name = source["name"]
            collision.newton = self._contact(source)
            collision_sources[collision.segment_name] = source

        authored = {
            link.get("name"): link.findall("collision")
            for link in ET.parse(self.root / "robot/robot.urdf")
            .getroot()
            .findall("link")
        }
        for link in robot.links:
            link.collisions = []
            for node in authored[link.name]:
                name = node.get("name")
                source = geoms[int(name.removeprefix("geom_"))]
                link.collisions.append(self._link_collision(node, source))
                collision_sources[name] = source
        for joint in robot.joints:
            # The source's compliant joint equality supplies the mimic relation.
            joint.mimic_parent = None

        equalities = []
        data = self.contract["equality"]
        for i, kind in enumerate(data["type"]):
            a, b = data["obj1id"][i], data["obj2id"][i]
            values = data["data"][i]
            if kind == 0:
                left, right = (
                    self.bindings[self.contract["body_names"][j]] for j in (a, b)
                )
                a, b = left["link"], right["link"]
                anchor = np.asarray(left["transform"]) @ np.r_[values[:3], 1]
                values = (*anchor[:3], *values[3:])
            else:
                a, b = self._names[a], self._names[b]
            equalities.append(
                NewtonEqualityDesc(
                    name=f"cube_eq_{i}",
                    equality_type=kind,
                    object1=a,
                    object2=b,
                    data=tuple(values),
                    solref=tuple(data["solref"][i]),
                    solimp=tuple(data["solimp"][i]),
                )
            )
        sources = list(collision_sources.items())
        filtered = [
            (name, peer)
            for i, (name, source) in enumerate(sources)
            for peer, other in sources[i + 1 :]
            if not (
                source["contype"] & other["conaffinity"]
                or other["contype"] & source["conaffinity"]
            )
        ]
        scene.newton = NewtonSceneDesc(
            options={
                key: self.contract[key]
                for key in ("iterations", "tolerance", "impratio")
            },
            filtered_geom_pairs=filtered,
        )
        spawn.builder._adapter = _CubeNewtonSpawnAdapter(tuple(equalities))

    def _link_collision(self, node: ET.Element, source: dict[str, Any]):
        from dexsim.spawn import CollisionDesc

        origin = node.find("origin")
        frame = np.eye(4)
        if origin is not None:
            frame[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            frame[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        options = {"segment_name": node.get("name"), "local_transform": frame}
        size = source["size"]
        kind = source["type"]
        if kind == 2:
            collision = CollisionDesc.sphere(size[0], **options)
        elif kind == 3:
            collision = CollisionDesc.capsule(size[0], size[1], **options)
        elif kind == 5:
            collision = CollisionDesc.cylinder(size[0], size[1], **options)
        elif kind == 6:
            collision = CollisionDesc.cube(tuple(2 * np.asarray(size)), **options)
        elif kind == 7:
            import trimesh

            filename = node.find("./geometry/mesh").get("filename")
            mesh = trimesh.load_mesh(self.root / "robot" / filename, process=False)
            collision = CollisionDesc.mesh(
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                triangles=np.asarray(mesh.faces, dtype=np.int32),
                **options,
            )
        else:
            raise ValueError(f"Unsupported Cube collider type {kind}")
        collision.newton = self._contact(source)
        return collision

    @staticmethod
    def _contact(source):
        from dexsim.spawn import NewtonCollisionDesc

        return NewtonCollisionDesc(
            solref=tuple(source["solref"]),
            solimp=tuple(source["solimp"]),
            mu=source["friction"][0],
            mu_torsional=source["friction"][1],
            mu_rolling=source["friction"][2],
            condim=source["condim"],
            priority=source["priority"],
            margin=source.get("margin", 0.0),
            gap=source.get("gap", 0.0),
            solmix=source.get("solmix", 1.0),
        )

    def _setup_robot(self, **kwargs: Any):
        solver = self.sim.spawn_result.get_solver_diagnostics()
        if solver["model"]["joint_count"] != len(_CUBE_RIGID_JOINTS):
            raise RuntimeError("Compiled DexUni Cube rigid-joint layout changed")
        self._joint_ids = [self.robot.joint_names.index(name) for name in self._names]
        for wall in self.walls:
            wall.set_visible(False)
        self.single_action_space = gym.spaces.Box(-1, 1, (5,), dtype=np.float32)
        return self.robot

    def _frame_pose(self, name: str) -> torch.Tensor:
        binding = self.bindings[name]
        local = torch.as_tensor(
            binding["transform"], dtype=torch.float32, device=self.device
        )
        return self.robot.get_link_pose(binding["link"], to_matrix=True) @ local

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

    def _preprocess_action(self, action):
        return torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(
            self.num_envs, 5
        )

    def _step_action(self, action):
        raw = action.detach().cpu().numpy() * np.array(self.contract["action_range"])
        pinch = self._frame_pose("ur5e/robotiq/pinch").cpu().numpy()
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

    def _passive_and_tendon_forces(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> np.ndarray:
        """Evaluate the reference generalized passive and split-tendon forces."""
        springs = np.array(self.contract["jnt_stiffness"][:14])
        resting = np.array(self.contract["qpos_spring"][:14])
        damping = np.array(self.contract["dof_damping"][:14])
        right, left = [
            self._names.index(f"ur5e/robotiq/{side}_driver_joint")
            for side in ("right", "left")
        ]
        force = -springs * (qpos - resting) - damping * qvel
        tendon_force = np.clip(
            0.3137255 * self._control[:, 6]
            - 50 * (qpos[:, right] + qpos[:, left])
            - 5 * (qvel[:, right] + qvel[:, left]),
            -5,
            5,
        )
        force[:, right] += tendon_force * 0.5
        force[:, left] += tendon_force * 0.5
        return force

    def _advance_physics(self) -> None:
        for _ in range(25):
            qpos = self.robot.get_qpos()[:, self._joint_ids].cpu().numpy()
            qvel = self.robot.get_qvel()[:, self._joint_ids].cpu().numpy()
            force = self._passive_and_tendon_forces(qpos, qvel)
            self.robot.set_qf(
                torch.as_tensor(force, dtype=torch.float32, device=self.device),
                joint_ids=self._joint_ids,
            )
            self.sim.update(0.002, step=1, render_final_step=False)

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
        obs["cube_pose"] = self.cube.get_local_pose()
        return obs

    def close(self) -> None:
        """Release the owned simulator without terminating the host process."""
        self.sim.destroy(exit_process=False)
