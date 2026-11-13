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

"""USD-native dexterous manipulation environments built on :class:`EmbodiedEnv`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from tensordict import TensorDict
import torch

from dexsim.engine.newton_physics.runtime_control import GraphRuntimeControl
from embodichain.data import get_data_path
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    UsdImportCfg,
    UsdSourceCfg,
)
from embodichain.lab.sim.objects import Articulation, RigidObject, Robot
from embodichain.lab.sim.spawn.usd_import import _parse_usd_source, _source_path
from embodichain.lab.sim.types import EnvAction, EnvObs
from embodichain.learning.rl.policy_evaluation import PolicyViewerCameraCfg

from ._leap_tasks import LeapTask
from ._shadow_task import ShadowState, ShadowTask

__all__ = ["DexterousManipulationEnv", "ENV_NAMES", "TASK_SPECS", "rotate_wxyz"]

ENV_NAMES: dict[str, str] = {
    "M1": "ShadowPenFixedRL-v1",
    "M2": "ShadowPenContinuousRL-v1",
    "M3": "LeapCubeFaceTurnRL-v1",
    "M4": "LeapCubeReorientRL-v1",
    "M5": "LeapCubeRotateZAxisRL-v1",
}


@dataclass(frozen=True)
class _TaskSpec:
    task_id: str
    env_id: str
    usd_asset: str
    initial_asset: str
    ctrl_dt: float
    sim_dt: float
    max_episode_steps: int
    action_dim: int
    policy_dim: int
    critic_dim: int
    has_episode_success: bool
    viewer_eye_offset: tuple[float, float, float]
    viewer_target_height: float


TASK_SPECS: dict[str, _TaskSpec] = {
    "M1": _TaskSpec(
        task_id="M1",
        env_id="ShadowPenFixedRL-v1",
        usd_asset="ShadowHandPen/usd/shadow_pen.usdc",
        initial_asset="ShadowHandPen/ShadowPenSpin-initial.npz",
        ctrl_dt=0.04,
        sim_dt=0.002,
        max_episode_steps=500,
        action_dim=18,
        policy_dim=87,
        critic_dim=87,
        has_episode_success=True,
        viewer_eye_offset=(0.35, -0.35, 0.22),
        viewer_target_height=0.20,
    ),
    "M2": _TaskSpec(
        task_id="M2",
        env_id="ShadowPenContinuousRL-v1",
        usd_asset="ShadowHandPen/usd/shadow_pen.usdc",
        initial_asset="ShadowHandPen/ShadowPenSpin-initial.npz",
        ctrl_dt=0.04,
        sim_dt=0.002,
        max_episode_steps=500,
        action_dim=18,
        policy_dim=87,
        critic_dim=87,
        has_episode_success=True,
        viewer_eye_offset=(0.35, -0.35, 0.22),
        viewer_target_height=0.20,
    ),
    "M3": _TaskSpec(
        task_id="M3",
        env_id="LeapCubeFaceTurnRL-v1",
        usd_asset="LeapHandCubeFaceTurn/usd/leap_cube_top_layer.usdc",
        initial_asset="LeapHandCubeFaceTurn/LeapCubeFaceTurn-initial.npz",
        ctrl_dt=0.05,
        sim_dt=0.005,
        max_episode_steps=500,
        action_dim=16,
        policy_dim=65,
        critic_dim=65,
        has_episode_success=True,
        viewer_eye_offset=(0.32, -0.32, 0.20),
        viewer_target_height=0.05,
    ),
    "M4": _TaskSpec(
        task_id="M4",
        env_id="LeapCubeReorientRL-v1",
        usd_asset="LeapHandCube/usd/M4.usdc",
        initial_asset="LeapHandCube/M4-initial.npz",
        ctrl_dt=0.05,
        sim_dt=0.01,
        max_episode_steps=1000,
        action_dim=16,
        policy_dim=57,
        critic_dim=128,
        has_episode_success=True,
        viewer_eye_offset=(0.34, -0.34, 0.22),
        viewer_target_height=0.08,
    ),
    "M5": _TaskSpec(
        task_id="M5",
        env_id="LeapCubeRotateZAxisRL-v1",
        usd_asset="LeapHandCube/usd/M5.usdc",
        initial_asset="LeapHandCube/M5-initial.npz",
        ctrl_dt=0.05,
        sim_dt=0.01,
        max_episode_steps=500,
        action_dim=16,
        policy_dim=32,
        critic_dim=105,
        has_episode_success=False,
        viewer_eye_offset=(0.32, -0.32, 0.20),
        viewer_target_height=0.08,
    ),
}


class _DexUniSceneJointOwnership(GraphRuntimeControl):
    """Assign all rigid and kinematic scene joints to DexUni's MuJoCo branch."""

    def __init__(self) -> None:
        self._assigned: tuple[int, ...] | None = None

    def initialize_model(self, model: Any, result: Any) -> None:
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        backend = get_newton_backend(result.world)
        if backend is None or backend.solver_type != "dexuni":
            return
        cfg = backend.cfg.solver_cfg
        if (
            cfg.mujoco_articulations is None
            and cfg.mujoco_articulation_paths is None
            and (cfg.mujoco_joints is None or cfg.mujoco_joints == self._assigned)
        ):
            self._assigned = tuple(range(model.joint_count))
            cfg.mujoco_joints = self._assigned

    def __call__(
        self, context: Any, substep_index: int, substep_count: int, substep_dt: float
    ) -> None:
        del context, substep_index, substep_count, substep_dt

    def prepare_episode_reset(
        self, arena_indices: tuple[int, ...]
    ) -> Callable[[], None]:
        del arena_indices
        return lambda: None


def rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vectors ``v`` by normalized ``wxyz`` quaternions ``q``."""
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def _xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat_xyzw[..., 3:4], quat_xyzw[..., :3]), dim=-1)


def _wxyz_to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat_wxyz[..., 1:4], quat_wxyz[..., :1]), dim=-1)


def _invert_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat_wxyz[..., :1], -quat_wxyz[..., 1:]), dim=-1)


class DexterousManipulationEnv(EmbodiedEnv):
    """Base USD-native :class:`EmbodiedEnv` for Shadow Hand and LEAP Hand RL tasks."""

    task_id: str = "M1"

    def __init__(self, cfg: Any = None, **kwargs: Any) -> None:
        if self.task_id not in TASK_SPECS:
            raise ValueError(
                f"Unknown dexterous manipulation task_id: {self.task_id!r}"
            )
        self._spec = TASK_SPECS[self.task_id]
        if cfg is not None:
            if not getattr(cfg, "control_parts", None) and not getattr(
                cfg, "active_joint_ids", None
            ):
                cfg.control_parts = ["hand"]
            for field_name in ("events", "observations", "rewards", "actions"):
                val = getattr(cfg, field_name, None)
                if (
                    val is not None
                    and isinstance(val, object)
                    and hasattr(val, "__dict__")
                    and not vars(val)
                ):
                    setattr(cfg, field_name, None)
        self.policy_viewer_camera_cfg = PolicyViewerCameraCfg(
            eye_offset=self._spec.viewer_eye_offset,
            target_height=self._spec.viewer_target_height,
        )
        super().__init__(cfg, **kwargs)

    @property
    def is_shadow(self) -> bool:
        """Return whether this task uses the Shadow Hand pen-spinning scene."""
        return self.task_id in ("M1", "M2")

    def _declare_robot(self, **kwargs: Any) -> Robot:
        del kwargs
        self.sim.remove_ground_plane()
        usd_path = Path(get_data_path(self._spec.usd_asset))
        if not usd_path.is_file():
            raise FileNotFoundError(
                f"Missing dexterous manipulation USD asset: {usd_path}"
            )
        self._usd_path = usd_path

        scene_desc = _parse_usd_source(
            UsdSourceCfg(file_path=str(usd_path)), name="scene", namespace=None
        )
        hand_art = next(
            (art for art in scene_desc.articulations if art.fixed_base), None
        )
        if hand_art is None:
            raise ValueError(f"No fixed-base hand articulation found in {usd_path}")

        dof_joints = [j for j in hand_art.joints if j.joint_type.name != "FIXED"]
        actuated_joints = [
            j for j in dof_joints if j.newton is not None and j.newton.target_mode == 1
        ]
        self._actuated_joint_names = [j.name for j in actuated_joints]
        policy_joint_names = (
            self._actuated_joint_names[2:]
            if self.is_shadow
            else self._actuated_joint_names
        )
        self._usd_ctrlrange = np.asarray(
            [j.newton.actuator_ctrlrange for j in actuated_joints], dtype=np.float32
        )
        self._usd_kp = np.asarray(
            [float(j.newton.target_ke or 0.0) for j in actuated_joints],
            dtype=np.float32,
        )
        self._usd_kd = np.asarray(
            [float(j.newton.target_kd or 0.0) for j in actuated_joints],
            dtype=np.float32,
        )
        self._usd_sites: dict[str, tuple[str, tuple[float, float, float]]] = {}
        for site in getattr(hand_art, "sites", ()):
            pos = tuple(
                float(v) for v in np.asarray(site.local_pose, dtype=np.float32)[:3, 3]
            )
            self._usd_sites[site.name] = (site.link_name, pos)

        default_uid = "shadow_hand" if self.is_shadow else "leap_hand"
        cfg_robot = getattr(self.cfg, "robot", None)
        robot_uid = (
            cfg_robot.uid
            if cfg_robot is not None and getattr(cfg_robot, "uid", None)
            else default_uid
        )
        robot_cfg = RobotCfg(
            uid=robot_uid,
            build_pk_chain=False,
            asset_physics_mode="preserve",
            source_pose_mode="preserve",
            enable_sensor=self.cfg.enable_sensor,
            control_parts={"hand": policy_joint_names},
            joint_drive_props=JointDrivePropertiesCfg(
                stiffness=None,
                damping=None,
                max_effort=None,
                max_velocity=None,
                friction=None,
                drive_type=None,
            ),
        )
        self._robot_sensor_cfgs = []
        if (
            self.sim.is_newton_backend
            and self.sim._active_newton_solver_type == "dexuni"
        ):
            self.sim._spawn_scene.builder.add_runtime_control(
                _DexUniSceneJointOwnership()
            )
        self.sim.add_usd(
            UsdImportCfg(
                name="scene",
                source=UsdSourceCfg(file_path=str(usd_path)),
                asset_cfgs={_source_path(hand_art): robot_cfg},
            )
        )
        return self.sim.get_robot(robot_uid)

    def _init_sim_state(self, **kwargs: Any) -> None:
        super()._init_sim_state(**kwargs)
        spec = self._spec
        expected_decimation = round(spec.ctrl_dt / spec.sim_dt)
        if not math.isclose(self.physics_dt, spec.sim_dt, abs_tol=1e-9):
            raise ValueError(
                f"{spec.env_id} requires physics_dt={spec.sim_dt}, got {self.physics_dt}."
            )
        if self.cfg.sim_steps_per_control != expected_decimation:
            raise ValueError(
                f"{spec.env_id} requires sim_steps_per_control={expected_decimation}, "
                f"got {self.cfg.sim_steps_per_control}."
            )

        self.action_dim = spec.action_dim
        self.policy_dim = spec.policy_dim
        self.critic_dim = spec.critic_dim
        self.single_action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self._actuated_joint_ids = [
            self.robot.joint_names.index(name) for name in self._actuated_joint_names
        ]
        self._actuated_joint_ids_tensor = torch.tensor(
            self._actuated_joint_ids, dtype=torch.long, device=self.device
        )
        self._kp = torch.as_tensor(
            self._usd_kp, dtype=torch.float32, device=self.device
        )
        self._kd = torch.as_tensor(
            self._usd_kd, dtype=torch.float32, device=self.device
        )

        detached_uids = [self.robot.uid]
        self._object: RigidObject | None = None
        self._goal_obj: RigidObject | None = None
        self._cube_art: Articulation | None = None
        self._face_joint_id: int = 0

        if self.is_shadow:
            self._object = self.sim.get_rigid_object("object")
            detached_uids.append("object")
            self._palm_link_idx = self.robot.link_names.index("robot0:palm")
        elif self.task_id == "M3":
            self._cube_art = self.sim.get_articulation("cube")
            self._face_joint_id = self._cube_art.joint_names.index("face_turn")
            detached_uids.append("cube")
        else:
            self._object = self.sim.get_rigid_object("cube")
            detached_uids.append("cube")
            if "goal" in self.sim._rigid_objects:
                self._goal_obj = self.sim.get_rigid_object("goal")
                self._goal_pos = self._goal_obj.get_local_pose()[:1, :3].clone()
                detached_uids.append("goal")

        self.add_detached_uids_for_reset(detached_uids)

        if self.task_id in ("M4", "M5"):
            site_names = ("grasp_site", "th_tip", "if_tip", "mf_tip", "rf_tip")
            self._site_link_indices = torch.tensor(
                [
                    self.robot.link_names.index(self._usd_sites[name][0])
                    for name in site_names
                ],
                dtype=torch.long,
                device=self.device,
            )
            self._site_local_pos = torch.tensor(
                [self._usd_sites[name][1] for name in site_names],
                dtype=torch.float32,
                device=self.device,
            )

        seed = 42 if self.cfg.seed is None else int(self.cfg.seed)
        initial_path = Path(get_data_path(spec.initial_asset))
        if self.is_shadow:
            self.task = ShadowTask.from_assets(
                initial_path.parent,
                self.num_envs,
                variant=self.task_id,
                device=self.device,
                lower=self._usd_ctrlrange[:, 0],
                upper=self._usd_ctrlrange[:, 1],
            )
            self._reset_generator = torch.Generator(device=self.device).manual_seed(
                seed
            )
        else:
            with np.load(initial_path) as initial:
                qpos0 = initial["qpos"].copy()
                ctrl0 = initial["ctrl"].copy()
            leap_task_name = {
                "M3": "LeapCubeFaceTurn",
                "M4": "LeapCubeReorient",
                "M5": "LeapCubeRotateZAxis",
            }[self.task_id]
            self.task = LeapTask(
                leap_task_name,
                self.num_envs,
                qpos0,
                ctrl0,
                self._usd_ctrlrange[:, 0],
                self._usd_ctrlrange[:, 1],
                device=self.device,
                seed=seed,
            )
            self._reset_generator = self.task.generator

        self._ctrl = torch.zeros(
            (self.num_envs, len(self._actuated_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        self._actuator_force = torch.zeros(
            (self.num_envs, len(self._actuated_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        self._raw_action = torch.zeros(
            (self.num_envs, self.action_dim), dtype=torch.float32, device=self.device
        )
        self._policy_obs = torch.zeros(
            (self.num_envs, self.policy_dim), dtype=torch.float32, device=self.device
        )
        self._critic_obs = torch.zeros(
            (self.num_envs, self.critic_dim), dtype=torch.float32, device=self.device
        )
        self._step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._step_terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._step_truncated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._episode_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._step_metrics: dict[str, torch.Tensor] = {}

    def _set_seed(self, seed: int) -> int:
        effective = int(seed)
        self.cfg.seed = effective
        if hasattr(self, "_reset_generator"):
            self._reset_generator.manual_seed(effective)
        for name, generator in getattr(self, "_component_generators", {}).items():
            generator.manual_seed(self._component_seed(name, effective))
        event_manager = getattr(self, "event_manager", None)
        if event_manager is not None:
            event_manager.set_seed(effective)
        return effective

    def get_policy_viewer_target_pose(self) -> tuple[float, ...]:
        """Return environment 0's manipulated object world pose ``(x, y, z, qx, qy, qz, qw)``."""
        if self._object is not None:
            pose = self._object.get_local_pose()[0].detach().clone()
        elif self._cube_art is not None:
            pose = self._cube_art.get_local_pose()[0].detach().clone()
        else:
            pose = self.robot.get_local_pose()[0].detach().clone()
        pose[:3] = pose[:3] + self.sim.arena_offsets[0].to(
            pose.device, dtype=pose.dtype
        )
        return tuple(float(v) for v in pose.cpu().tolist())

    def _local_link_poses(self) -> torch.Tensor:
        """Return robot link poses in local arena coordinates ``(N, L, 7)`` (xyzw)."""
        link_poses = self.robot.body_data.body_link_pose
        root_local = self.robot.get_local_pose()[:, :3]
        offset = link_poses[:, 0, :3] - root_local
        if bool((offset.abs() > 1e-6).any()):
            link_poses = link_poses.clone()
            link_poses[:, :, :3] = link_poses[:, :, :3] - offset[:, None, :]
        return link_poses

    def _read_physical_state(self) -> Any:
        """Read batched simulation state via public Robot/Articulation/RigidObject facades."""
        hand_qpos = self.robot.get_qpos()
        hand_qvel = self.robot.get_qvel()

        if self.is_shadow:
            assert self._object is not None
            obj_pose_xyzw = self._object.get_local_pose()
            obj_pos = obj_pose_xyzw[:, :3]
            obj_quat_wxyz = _xyzw_to_wxyz(obj_pose_xyzw[:, 3:7])
            obj_linvel = self._object.body_data.lin_vel
            obj_world_angvel = self._object.body_data.ang_vel
            obj_local_angvel = rotate_wxyz(
                _invert_wxyz(obj_quat_wxyz), obj_world_angvel
            )
            qpos = torch.cat((hand_qpos, obj_pos, obj_quat_wxyz), dim=-1)
            qvel = torch.cat((hand_qvel, obj_linvel, obj_local_angvel), dim=-1)
            palm_pos = self._local_link_poses()[:, self._palm_link_idx, :3]
            return ShadowState(qpos, qvel, palm_pos)

        if self.task_id == "M3":
            assert self._cube_art is not None
            cube_pose_xyzw = self._cube_art.get_local_pose()
            cube_pos = cube_pose_xyzw[:, :3]
            cube_quat_wxyz = _xyzw_to_wxyz(cube_pose_xyzw[:, 3:7])
            cube_linvel = self._cube_art.body_data.root_lin_vel
            cube_world_angvel = self._cube_art.body_data.root_ang_vel
            cube_local_angvel = rotate_wxyz(
                _invert_wxyz(cube_quat_wxyz), cube_world_angvel
            )
            face_qpos = self._cube_art.get_qpos()[
                :, self._face_joint_id : self._face_joint_id + 1
            ]
            face_qvel = self._cube_art.get_qvel()[
                :, self._face_joint_id : self._face_joint_id + 1
            ]
            return {
                "qpos": torch.cat(
                    (hand_qpos, cube_pos, cube_quat_wxyz, face_qpos), dim=-1
                ),
                "qvel": torch.cat(
                    (hand_qvel, cube_linvel, cube_local_angvel, face_qvel), dim=-1
                ),
            }

        assert self._object is not None
        cube_pose_xyzw = self._object.get_local_pose()
        cube_pos = cube_pose_xyzw[:, :3]
        cube_quat_wxyz = _xyzw_to_wxyz(cube_pose_xyzw[:, 3:7])
        cube_linvel = self._object.body_data.lin_vel
        cube_world_angvel = self._object.body_data.ang_vel
        cube_local_angvel = rotate_wxyz(_invert_wxyz(cube_quat_wxyz), cube_world_angvel)

        link_poses = self._local_link_poses()[:, self._site_link_indices, :]
        site_link_pos = link_poses[:, :, :3]
        site_link_quat_wxyz = _xyzw_to_wxyz(link_poses[:, :, 3:7])
        site_world_pos = site_link_pos + rotate_wxyz(
            site_link_quat_wxyz, self._site_local_pos.unsqueeze(0)
        )
        grasp_pos = site_world_pos[:, 0, :]
        palm_quat_inv = _invert_wxyz(site_link_quat_wxyz[:, 0, :])
        tip_offsets = site_world_pos[:, 1:, :] - grasp_pos[:, None, :]
        tips_in_grasp = rotate_wxyz(palm_quat_inv[:, None, :], tip_offsets).reshape(
            self.num_envs, 12
        )

        return {
            "qpos": torch.cat((hand_qpos, cube_pos, cube_quat_wxyz), dim=-1),
            "qvel": torch.cat((hand_qvel, cube_linvel, cube_local_angvel), dim=-1),
            "hand_qpos": hand_qpos,
            "hand_qvel": hand_qvel,
            "actuator_force": self._actuator_force.clone(),
            "cube_pos": cube_pos,
            "cube_quat": cube_quat_wxyz,
            "cube_linvel": cube_linvel,
            "cube_angvel": cube_world_angvel,
            "palm_pos": grasp_pos,
            "fingertip_pos": tips_in_grasp,
        }

    def _sync_goal_object(self, env_ids: torch.Tensor | None = None) -> None:
        if self._goal_obj is None or self.task_id != "M4":
            return
        if env_ids is None:
            goal_quat_xyzw = _wxyz_to_xyzw(self.task.goal_quat)
            goal_pose = torch.cat(
                (self._goal_pos.expand(self.num_envs, -1), goal_quat_xyzw), dim=-1
            )
            self._goal_obj.set_local_pose(goal_pose)
        else:
            goal_quat_xyzw = _wxyz_to_xyzw(self.task.goal_quat[env_ids])
            goal_pose = torch.cat(
                (self._goal_pos.expand(len(env_ids), -1), goal_quat_xyzw), dim=-1
            )
            self._goal_obj.set_local_pose(goal_pose, env_ids=env_ids)

    def _initialize_episode(
        self, env_ids: Sequence[int] | torch.Tensor | None = None, **kwargs: Any
    ) -> None:
        super()._initialize_episode(env_ids=env_ids, **kwargs)
        ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        if not ids.numel():
            return

        if self.is_shadow:
            qpos, _, ctrl = self.task.reset(ids, generator=self._reset_generator)
            hand_qpos = qpos[:, :24]
            obj_pos = qpos[:, 24:27]
            obj_quat_xyzw = _wxyz_to_xyzw(qpos[:, 27:31])
            assert self._object is not None
            zeros_3 = torch.zeros(
                (len(ids), 3), dtype=torch.float32, device=self.device
            )
            self._object.set_local_pose(
                torch.cat((obj_pos, obj_quat_xyzw), dim=-1), env_ids=ids
            )
            self._object.set_velocity(lin_vel=zeros_3, ang_vel=zeros_3, env_ids=ids)
        else:
            initial = self.task.reset(ids)
            qpos = initial["qpos"]
            ctrl = initial["ctrl"]
            hand_qpos = qpos[:, :16]
            cube_pos = qpos[:, 16:19]
            cube_quat_xyzw = _wxyz_to_xyzw(qpos[:, 19:23])
            if self.task_id == "M3":
                assert self._cube_art is not None
                face_qpos = qpos[:, 23:24]
                self._cube_art.set_local_pose(
                    torch.cat((cube_pos, cube_quat_xyzw), dim=-1), env_ids=ids
                )
                self._cube_art.set_root_velocity(
                    torch.zeros((len(ids), 6), dtype=torch.float32, device=self.device),
                    env_ids=ids,
                )
                self._cube_art.set_qpos(
                    face_qpos,
                    joint_ids=[self._face_joint_id],
                    env_ids=ids,
                    target=False,
                )
                self._cube_art.set_qvel(
                    torch.zeros_like(face_qpos),
                    joint_ids=[self._face_joint_id],
                    env_ids=ids,
                    target=False,
                )
                self._cube_art.set_qf(
                    torch.zeros_like(face_qpos),
                    joint_ids=[self._face_joint_id],
                    env_ids=ids,
                )
            else:
                assert self._object is not None
                zeros_3 = torch.zeros(
                    (len(ids), 3), dtype=torch.float32, device=self.device
                )
                self._object.set_local_pose(
                    torch.cat((cube_pos, cube_quat_xyzw), dim=-1), env_ids=ids
                )
                self._object.set_velocity(lin_vel=zeros_3, ang_vel=zeros_3, env_ids=ids)
                self._sync_goal_object(ids)

        zero_hand_qvel = torch.zeros_like(hand_qpos)
        self.robot.set_qpos(hand_qpos, env_ids=ids, target=False)
        for target in (False, True):
            self.robot.set_qvel(zero_hand_qvel, env_ids=ids, target=target)
        self.robot.set_qf(zero_hand_qvel, env_ids=ids)
        # clear_articulation_dynamics syncs Newton state_0 (including all world
        # articulations and standalone rigid objects) into the MuJoCo solver
        # data and refreshes forward kinematics; restore actuator position
        # targets right after clearing solver state.
        self.sim.clear_articulation_dynamics(env_ids=ids)
        self._ctrl[ids] = ctrl
        self._actuator_force[ids] = 0.0
        self.robot.set_qpos(
            ctrl, joint_ids=self._actuated_joint_ids, env_ids=ids, target=True
        )

        self._raw_action[ids] = 0.0
        self._episode_success[ids] = False
        self._step_terminated[ids] = False
        self._step_truncated[ids] = False

        state = self._read_physical_state()
        if self.is_shadow:
            obs_all = self.task.observe(state)
            self._policy_obs[ids] = obs_all[ids]
            self._critic_obs[ids] = obs_all[ids]
        else:
            raw = self.task.observe(state, env_ids=ids)
            self._policy_obs[ids] = raw["state"]
            self._critic_obs[ids] = raw["privileged_state"]

    def _preprocess_action(self, action: EnvAction) -> EnvAction:
        return action

    def _step_action(self, action: EnvAction) -> EnvAction:
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"Expected Tensor action, got {type(action).__name__}")
        if action.ndim != 2 or action.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Expected action shape ({self.num_envs}, {self.action_dim}), got {tuple(action.shape)}"
            )
        if not bool(torch.isfinite(action).all()):
            raise FloatingPointError("Non-finite policy action")
        clipped = action.clamp(-1.0, 1.0)
        self._raw_action.copy_(clipped)
        if self.is_shadow:
            targets = self.task.action_targets(clipped)
        else:
            targets = self.task.control_targets(clipped)
        self._ctrl.copy_(targets)
        self.robot.set_qpos(targets, joint_ids=self._actuated_joint_ids, target=True)
        return action

    def _advance_physics(self) -> None:
        if self.task_id == "M3":
            assert self._cube_art is not None
            for _ in range(self.cfg.sim_steps_per_control):
                face_angle = self._cube_art.get_qpos()[:, self._face_joint_id]
                face_force = self.task.face_detent(face_angle)
                self._cube_art.set_qf(
                    face_force[:, None], joint_ids=[self._face_joint_id]
                )
                self.sim.update(self.physics_dt, 1, render_final_step=False)
            return
        if self.task_id in ("M4", "M5"):
            substeps = self.cfg.sim_steps_per_control
            if substeps > 1:
                self.sim.update(self.physics_dt, substeps - 1, render_final_step=False)
            qpos_pre = self.robot.get_qpos()[:, self._actuated_joint_ids_tensor]
            qvel_pre = self.robot.get_qvel()[:, self._actuated_joint_ids_tensor]
            self._actuator_force.copy_(
                self._kp * (self._ctrl - qpos_pre) - self._kd * qvel_pre
            )
            self.sim.update(self.physics_dt, 1, render_final_step=False)
            return
        super()._advance_physics()

    def _update_sim_state(self, **kwargs: Any) -> None:
        super()._update_sim_state(**kwargs)
        state = self._read_physical_state()
        qpos_check = state.qpos if isinstance(state, ShadowState) else state["qpos"]
        if not bool(torch.isfinite(qpos_check).all()):
            raise FloatingPointError("Non-finite physical state")

        next_elapsed = self._elapsed_steps + 1
        if self.is_shadow:
            obs, reward, terminated, metrics = self.task.step(state, self._raw_action)
            self._policy_obs.copy_(obs)
            self._critic_obs.copy_(obs)
            truncated = (next_elapsed >= self.max_episode_steps) & ~terminated
        else:
            raw_obs, reward, terminated, _, metrics = self.task.after_step(
                state, self._raw_action
            )
            self._policy_obs.copy_(raw_obs["state"])
            self._critic_obs.copy_(raw_obs["privileged_state"])
            truncated = (next_elapsed >= self.max_episode_steps) & ~terminated
            if self.task_id == "M4":
                self._sync_goal_object()

        self._step_reward.copy_(reward)
        self._step_terminated.copy_(terminated.bool())
        self._step_truncated.copy_(truncated.bool())
        self._step_metrics = {key: value.clone() for key, value in metrics.items()}
        if "success" in metrics:
            self._episode_success |= metrics["success"].bool()

    def get_obs(self, **kwargs: Any) -> EnvObs:
        obs = TensorDict(
            {"policy": self._policy_obs.clone(), "critic": self._critic_obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )
        if self.sensors:
            sensor_obs = self._get_sensor_obs(**kwargs)
            if len(sensor_obs.keys()) > 0:
                obs["sensor"] = sensor_obs
        if self.cfg.observations and self.observation_manager is not None:
            obs = self.observation_manager.compute(obs)
        return obs

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del kwargs
        return (
            self._episode_success.clone(),
            self._step_terminated.clone(),
            {key: value.clone() for key, value in self._step_metrics.items()},
        )

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return self._episode_success.clone()

    def get_info(self, **kwargs: Any) -> dict[str, Any]:
        info = super().get_info(**kwargs)
        if not self._spec.has_episode_success:
            info.pop("success", None)
        return info

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        info = self.get_info(**kwargs)
        eval_dict: dict[str, Any] = {}
        if "success" in info:
            eval_dict["success"] = info["success"][0].item()
        for key, value in info.get("metrics", {}).items():
            eval_dict[key] = value
        return eval_dict

    def get_reward(
        self, obs: EnvObs, action: EnvAction, info: dict[str, Any]
    ) -> torch.Tensor:
        del obs, action, info
        return self._step_reward.clone()

    def _get_termination_flags(
        self, obs: EnvObs, info: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del obs, info
        terminated = self._step_terminated.clone()
        truncated = self._step_truncated.clone()
        if self.cfg.ignore_terminations:
            terminated[:] = False
        return terminated, truncated
