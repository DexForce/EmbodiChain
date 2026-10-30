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

"""Built-in independently applied policy-action terms."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

from embodichain.utils.math import (
    axis_angle_to_rotation_matrix,
    matrix_from_euler,
    matrix_from_quat,
)
from embodichain.utils.string import resolve_matching_names_values

from .action_manager import ActionTerm
from .action_types import ActionTermDescriptor
from .cfg import ActionTermCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

__all__ = [
    "DefaultJointPositionAction",
    "EefPoseAction",
    "JointEffortAction",
    "JointPositionAction",
    "JointPositionToLimitsAction",
    "JointVelocityAction",
    "ParallelGripperAction",
    "RelativeEefPoseAction",
    "RelativeJointPositionAction",
    "register_action_contract",
    "resolve_action_contract",
]


def _parameter_tensor(
    value: float | list[float] | tuple[float, ...] | torch.Tensor,
    *,
    width: int,
    device: torch.device | str,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        return tensor
    if tuple(tensor.shape) != (width,):
        raise ValueError(f"{name} must be scalar or contain {width} values.")
    return tensor


class _JointAction(ActionTerm):
    """Base class for terms that own an ordered non-mimic joint selection."""

    command_type: str
    representation: str
    units: str
    normalization: str | None = None

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        part_name = params.get("part_name")
        joint_names = params.get("joint_names")
        if (part_name is None) == (joint_names is None):
            raise ValueError(
                f"{type(self).__name__} requires exactly one of part_name or joint_names."
            )

        if part_name is not None:
            joint_ids = env.robot.get_joint_ids(str(part_name), remove_mimic=True)
        else:
            requested_names = [str(name) for name in joint_names]
            joint_ids = [env.robot.joint_names.index(name) for name in requested_names]
            mimic_ids = set(getattr(env.robot, "mimic_ids", ()))
            if any(joint_id in mimic_ids for joint_id in joint_ids):
                raise ValueError(
                    f"{type(self).__name__} joint_names must not select mimic followers."
                )
            if not bool(params.get("preserve_order", False)):
                joint_ids = sorted(joint_ids)

        self._joint_ids = tuple(int(joint_id) for joint_id in joint_ids)
        if not self._joint_ids:
            raise ValueError(f"{type(self).__name__} resolved no controlled joints.")
        if len(set(self._joint_ids)) != len(self._joint_ids):
            raise ValueError(f"{type(self).__name__} resolved duplicate joint IDs.")
        self._joint_names = tuple(
            env.robot.joint_names[index] for index in self._joint_ids
        )
        self._raw_actions = torch.zeros(
            env.num_envs, self.action_dim, dtype=torch.float32, device=env.device
        )
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Box: ...

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def previous_raw_actions(self) -> torch.Tensor:
        """Return raw actions from the previous control step."""
        return self._previous_raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def controlled_joint_ids(self) -> tuple[int, ...]:
        return self._joint_ids

    @property
    def descriptor(self) -> ActionTermDescriptor:
        return ActionTermDescriptor(
            representation=self.representation,
            action_dim=self.action_dim,
            feature_names=self._joint_names,
            units=(self.units,) * self.action_dim,
            normalization=self.normalization,
            joint_names=self._joint_names,
            metadata={},
            contract=self.resolved_contract_id(),
        )

    def _store_raw(self, actions: torch.Tensor) -> None:
        expected = (self.num_envs, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(
                f"{type(self).__name__} expected action shape {expected}, "
                f"got {tuple(actions.shape)}."
            )
        self._previous_raw_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(actions)

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0
        self._previous_raw_actions[ids] = 0
        self._processed_actions[ids] = 0


class JointPositionAction(_JointAction):
    """Apply absolute position actions to selected joints."""

    contract_id: ClassVar[str] = "joint_position.absolute@1"
    command_type = "qpos"
    representation = "joint_position"
    units = "rad_or_m"

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._scale = _parameter_tensor(
            cfg.params.get("scale", 1.0),
            width=self.action_dim,
            device=env.device,
            name="scale",
        )
        self._offset = _parameter_tensor(
            cfg.params.get("offset", 0.0),
            width=self.action_dim,
            device=env.device,
            name="offset",
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        limits = self._env.robot.body_data.qpos_limits[0, list(self._joint_ids)]
        return gym.spaces.Box(
            low=limits[:, 0].detach().cpu().numpy(),
            high=limits[:, 1].detach().cpu().numpy(),
            dtype=np.float32,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._store_raw(actions)
        self._processed_actions.copy_(self._raw_actions * self._scale + self._offset)

    def apply_actions(self) -> None:
        self._env.robot.set_qpos(
            qpos=self._processed_actions,
            joint_ids=list(self._joint_ids),
        )


class JointPositionToLimitsAction(_JointAction):
    """Map normalized selected-joint actions to position limits."""

    contract_id: ClassVar[str] = "joint_position.limits@1"
    command_type = "qpos"
    representation = "joint_position"
    units = "rad_or_m"
    normalization = "minus_one_to_one"

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._scale = _parameter_tensor(
            cfg.params.get("scale", 1.0),
            width=self.action_dim,
            device=env.device,
            name="scale",
        )
        limits = env.robot.body_data.qpos_limits[0, list(self._joint_ids)]
        self._lower = limits[:, 0]
        self._upper = limits[:, 1]

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.ones(self.action_dim, dtype=np.float32),
            high=np.ones(self.action_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._store_raw(actions)
        normalized = (self._raw_actions * self._scale).clamp(-1.0, 1.0)
        self._processed_actions.copy_(
            self._lower + (normalized + 1.0) * 0.5 * (self._upper - self._lower)
        )

    def apply_actions(self) -> None:
        self._env.robot.set_qpos(
            qpos=self._processed_actions,
            joint_ids=list(self._joint_ids),
        )


class RelativeJointPositionAction(_JointAction):
    """Add scaled relative actions to current selected positions.

    Optional ``target_limits`` contains a lower/upper pair per selected joint.
    It limits the resulting position command without changing the raw policy
    action used by observation histories.
    """

    contract_id: ClassVar[str] = "joint_position.relative@1"
    command_type = "qpos"
    representation = "relative_joint_position"
    units = "rad_or_m"

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._scale = _parameter_tensor(
            cfg.params.get("scale", 1.0),
            width=self.action_dim,
            device=env.device,
            name="scale",
        )
        clip = cfg.params.get("clip")
        self._clip = None if clip is None else float(clip)
        limits = cfg.params.get("target_limits")
        self._target_limits = None
        if limits is not None:
            limits = torch.as_tensor(limits, dtype=torch.float32, device=env.device)
            if (
                tuple(limits.shape) != (self.action_dim, 2)
                or not torch.isfinite(limits).all()
                or not (limits[:, 0] < limits[:, 1]).all()
            ):
                raise ValueError(
                    "target_limits must contain finite ordered lower/upper pairs "
                    "for each selected joint."
                )
            self._target_limits = limits

    @property
    def action_space(self) -> gym.spaces.Box:
        bound = np.inf if self._clip is None else self._clip
        return gym.spaces.Box(
            low=-np.full(self.action_dim, bound, dtype=np.float32),
            high=np.full(self.action_dim, bound, dtype=np.float32),
            dtype=np.float32,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._store_raw(actions)
        if self._clip is not None:
            self._raw_actions.clamp_(-self._clip, self._clip)
        current = self._env.robot.get_qpos()[:, list(self._joint_ids)]
        self._processed_actions.copy_(current + self._raw_actions * self._scale)
        if self._target_limits is not None:
            self._processed_actions.clamp_(
                min=self._target_limits[:, 0], max=self._target_limits[:, 1]
            )

    def apply_actions(self) -> None:
        self._env.robot.set_qpos(
            qpos=self._processed_actions,
            joint_ids=list(self._joint_ids),
        )


class JointVelocityAction(_JointAction):
    """Apply scaled velocity actions to selected joints."""

    contract_id: ClassVar[str] = "joint_velocity@1"
    command_type = "qvel"
    representation = "joint_velocity"
    units = "rad_or_m_per_s"

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._scale = _parameter_tensor(
            cfg.params.get("scale", 1.0),
            width=self.action_dim,
            device=env.device,
            name="scale",
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        limits = self._env.robot.body_data.qvel_limits[0, list(self._joint_ids)]
        return gym.spaces.Box(
            low=(-limits).detach().cpu().numpy(),
            high=limits.detach().cpu().numpy(),
            dtype=np.float32,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._store_raw(actions)
        self._processed_actions.copy_(self._raw_actions * self._scale)

    def apply_actions(self) -> None:
        self._env.robot.set_qvel(
            qvel=self._processed_actions,
            joint_ids=list(self._joint_ids),
        )


class JointEffortAction(_JointAction):
    """Apply scaled effort actions to selected joints."""

    contract_id: ClassVar[str] = "joint_effort@1"
    command_type = "qf"
    representation = "joint_effort"
    units = "N_or_Nm"

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._scale = _parameter_tensor(
            cfg.params.get("scale", 1.0),
            width=self.action_dim,
            device=env.device,
            name="scale",
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        limits = self._env.robot.body_data.qf_limits[0, list(self._joint_ids)]
        return gym.spaces.Box(
            low=(-limits).detach().cpu().numpy(),
            high=limits.detach().cpu().numpy(),
            dtype=np.float32,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._store_raw(actions)
        self._processed_actions.copy_(self._raw_actions * self._scale)

    def apply_actions(self) -> None:
        self._env.robot.set_qf(
            qf=self._processed_actions,
            joint_ids=list(self._joint_ids),
        )


class DefaultJointPositionAction(_JointAction):
    """Map normalized actions around a configured default joint pose.

    ``clip`` defaults to ``1.0``; ``None`` preserves the unbounded behavior of
    legacy ``DefaultJointPositionTerm`` snapshots.
    """

    contract_id: ClassVar[str] = "joint_position.default_offset@1"
    command_type = "qpos"
    representation = "default_joint_position"
    units = "normalized"
    normalization = "minus_one_to_one"

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        robot_default = torch.as_tensor(
            env.robot.cfg.init_qpos,
            dtype=torch.float32,
            device=env.device,
        )[list(self._joint_ids)]
        self._offset = _parameter_tensor(
            cfg.params.get("offset", robot_default),
            width=self.action_dim,
            device=env.device,
            name="offset",
        )
        self._scale = _parameter_tensor(
            cfg.params.get("scale", 1.0),
            width=self.action_dim,
            device=env.device,
            name="scale",
        )
        clip = cfg.params.get("clip", 1.0)
        self._clip = None if clip is None else float(clip)
        self.position_bias = torch.zeros_like(self._raw_actions)

    @property
    def action_space(self) -> gym.spaces.Box:
        if self._clip is None:
            bounds = np.full(self.action_dim, np.inf, dtype=np.float32)
        else:
            bounds = np.full(self.action_dim, self._clip, dtype=np.float32)
        return gym.spaces.Box(
            low=-bounds,
            high=bounds,
            dtype=np.float32,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._store_raw(actions)
        if self._clip is not None:
            self._raw_actions.clamp_(-self._clip, self._clip)
        self._processed_actions.copy_(
            self._offset + self._scale * self._raw_actions - self.position_bias
        )

    def apply_actions(self) -> None:
        self._env.robot.set_qpos(
            qpos=self._processed_actions,
            joint_ids=list(self._joint_ids),
        )

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        """Clear selected action history and commands without changing bias.

        Args:
            env_ids: Rows to reset. None selects all rows.
        """
        buffers = (
            self._raw_actions,
            self._previous_raw_actions,
            self._processed_actions,
        )
        if env_ids is None:
            for buffer in buffers:
                buffer.zero_()
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long).to(
                device=self.device, non_blocking=True
            )
            for buffer in buffers:
                buffer.index_fill_(0, ids, 0)


class EefPoseAction(ActionTerm):
    """Convert an absolute EEF pose into selected-arm joint targets."""

    contract_id: ClassVar[str] = "eef_pose.absolute@1"
    command_type = "qpos"

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        self._part_name = str(params["part_name"])
        self._joint_ids = tuple(
            int(value)
            for value in env.robot.get_joint_ids(
                self._part_name,
                remove_mimic=True,
            )
        )
        if not self._joint_ids:
            raise ValueError("EefPoseAction resolved no arm joints.")
        self._joint_names = tuple(
            env.robot.joint_names[index] for index in self._joint_ids
        )
        self._pose_representation = str(params.get("pose_representation", "xyz_rpy"))
        if self._pose_representation == "xyz_rpy":
            self._action_dim = 6
            self._feature_names = ("x", "y", "z", "roll", "pitch", "yaw")
            self._units = ("m", "m", "m", "rad", "rad", "rad")
            rotation = "rpy_radians"
        elif self._pose_representation == "xyz_quat_xyzw":
            self._action_dim = 7
            self._feature_names = ("x", "y", "z", "qx", "qy", "qz", "qw")
            self._units = (
                "m",
                "m",
                "m",
                "unitless",
                "unitless",
                "unitless",
                "unitless",
            )
            rotation = "quaternion_xyzw"
        else:
            raise ValueError(
                "pose_representation must be 'xyz_rpy' or 'xyz_quat_xyzw'."
            )
        self._scale = _parameter_tensor(
            params.get("scale", 1.0),
            width=self.action_dim,
            device=env.device,
            name="scale",
        )
        self._space = gym.spaces.Box(
            low=-np.full(self.action_dim, np.inf, dtype=np.float32),
            high=np.full(self.action_dim, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self._raw_actions = torch.zeros(
            env.num_envs, self.action_dim, dtype=torch.float32, device=env.device
        )
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros(
            env.num_envs,
            len(self._joint_ids),
            dtype=torch.float32,
            device=env.device,
        )
        self._ik_success = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        self._descriptor = ActionTermDescriptor(
            representation="eef_pose",
            action_dim=self.action_dim,
            feature_names=self._feature_names,
            units=self._units,
            normalization=None,
            joint_names=self._joint_names,
            metadata={"frame": "arena", "rotation": rotation},
            contract=self.resolved_contract_id(),
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._space

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def previous_raw_actions(self) -> torch.Tensor:
        """Return pose requests from the previous control step."""
        return self._previous_raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def ik_success(self) -> torch.Tensor:
        """Return per-environment IK success for the latest request."""
        return self._ik_success

    @property
    def controlled_joint_ids(self) -> tuple[int, ...]:
        return self._joint_ids

    @property
    def descriptor(self) -> ActionTermDescriptor:
        return self._descriptor

    @property
    def part_name(self) -> str:
        """Return the robot control part used for IK and FK."""
        return self._part_name

    def process_actions(self, actions: torch.Tensor) -> None:
        expected = (self.num_envs, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(
                f"EefPoseAction expected action shape {expected}, "
                f"got {tuple(actions.shape)}."
            )
        self._previous_raw_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(actions)
        scaled = self._raw_actions * self._scale
        target_pose = torch.eye(4, dtype=torch.float32, device=self.device).repeat(
            self.num_envs,
            1,
            1,
        )
        target_pose[:, :3, 3] = scaled[:, :3]
        if self._pose_representation == "xyz_rpy":
            target_pose[:, :3, :3] = matrix_from_euler(scaled[:, 3:6])
        else:
            target_pose[:, :3, :3] = matrix_from_quat(scaled[:, 3:7])

        current = self._env.robot.get_qpos()[:, list(self._joint_ids)]
        success, qpos = self._env.robot.compute_ik(
            pose=target_pose,
            joint_seed=current,
            name=self._part_name,
        )
        self._ik_success.copy_(success)
        self._processed_actions.copy_(torch.where(success[:, None], qpos, current))

    def apply_actions(self) -> None:
        self._env.robot.set_qpos(
            qpos=self._processed_actions,
            joint_ids=list(self._joint_ids),
        )

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0
        self._previous_raw_actions[ids] = 0
        self._processed_actions[ids] = 0
        self._ik_success[ids] = False


class RelativeEefPoseAction(EefPoseAction):
    """Convert normalized arena-frame pose deltas into selected-arm targets.

    Each request starts from FK of the measured joint positions. Translation
    adds ``translation_scale * clamp(u[:3], -1, 1)`` in meters; rotation
    left-multiplies the measured rotation by the exponential map of
    ``rotation_scale * clamp(u[3:], -1, 1)`` in radians. There is no accumulated
    pose target. Failed or nonfinite IK solutions hold the measured arm joints.
    Optional ``ik_method="dls"`` applies one damped, bounded local Jacobian
    increment. Its success flag means a finite controller step, not convergence
    to the requested pose.
    """

    contract_id: ClassVar[str] = "eef_pose.relative@1"

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        """Initialize the measured-pose delta controller.

        Args:
            cfg: Term config with ``part_name``, optional positive finite
                ``translation_scale`` (default 0.02 m), and ``rotation_scale``
                (default 0.10 rad), applied per component per control step.
                ``ik_method`` selects the existing ``"solver"`` backend or one
                ``"dls"`` step with positive finite ``dls_damping`` (0.05) and
                ``max_joint_delta`` (0.10, in each joint's position units).
            env: Environment supplying measured joints, FK, and IK.
        """
        if "pose_representation" in cfg.params or "scale" in cfg.params:
            raise ValueError(
                "RelativeEefPoseAction has fixed xyz/rotation-vector requests; "
                "configure translation_scale and rotation_scale."
            )
        self._translation_scale = float(cfg.params.get("translation_scale", 0.02))
        self._rotation_scale = float(cfg.params.get("rotation_scale", 0.10))
        self._ik_method = str(cfg.params.get("ik_method", "solver"))
        if self._ik_method not in {"solver", "dls"}:
            raise ValueError("ik_method must be 'solver' or 'dls'.")
        self._dls_damping = float(cfg.params.get("dls_damping", 0.05))
        self._max_joint_delta = float(cfg.params.get("max_joint_delta", 0.10))
        for name, value in (
            ("translation_scale", self._translation_scale),
            ("rotation_scale", self._rotation_scale),
            ("dls_damping", self._dls_damping),
            ("max_joint_delta", self._max_joint_delta),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        super().__init__(cfg, env)
        self._space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self._current_pose = torch.eye(4, device=env.device).repeat(env.num_envs, 1, 1)
        self._target_pose = self._current_pose.clone()
        self._clipped_mask = torch.zeros_like(self._raw_actions, dtype=torch.bool)
        self._joint_delta = torch.zeros_like(self._processed_actions)
        self._joint_step_limited = torch.zeros_like(self._ik_success)
        self._joint_limit_clipped_mask = torch.zeros_like(
            self._processed_actions, dtype=torch.bool
        )
        self._prepare_solver()
        dls_metadata = {}
        if self._ik_method == "dls":
            dls_metadata = {
                "ik_method": "dls",
                "dls_damping": self._dls_damping,
                "max_joint_delta": self._max_joint_delta,
                "controller_step": "bounded_local_jacobian_approximation",
                "ik_success_semantics": "finite_controller_step_not_pose_convergence",
            }
        self._descriptor = ActionTermDescriptor(
            representation="relative_eef_pose",
            action_dim=6,
            feature_names=(
                "delta_x",
                "delta_y",
                "delta_z",
                "delta_rx",
                "delta_ry",
                "delta_rz",
            ),
            units=("normalized",) * 6,
            normalization="minus_one_to_one",
            joint_names=self._joint_names,
            metadata={
                "frame": "arena",
                "rotation": "rotation_vector",
                "rotation_composition": "left",
                "target_reference": "measured_pose",
                "translation_scale": self._translation_scale,
                "rotation_scale": self._rotation_scale,
                "translation_units": "m",
                "rotation_units": "rad",
                "clip": [-1.0, 1.0],
                "ik_failure": "measured_joint_hold",
                **dls_metadata,
            },
            contract=self.resolved_contract_id(),
        )

    @property
    def current_pose(self) -> torch.Tensor:
        """Return latest measured arena-frame transforms, or identity after reset."""
        return self._current_pose

    @property
    def target_pose(self) -> torch.Tensor:
        """Return latest requested arena-frame transforms, or identity after reset."""
        return self._target_pose

    @property
    def clipped_mask(self) -> torch.Tensor:
        """Return which components exceeded normalized bounds on the latest step."""
        return self._clipped_mask

    @property
    def joint_delta(self) -> torch.Tensor:
        """Return the latest commanded increment in selected joint order."""
        return self._joint_delta

    @property
    def joint_step_limited(self) -> torch.Tensor:
        """Return DLS rows rescaled to respect the maximum joint increment."""
        return self._joint_step_limited

    @property
    def joint_limit_clipped_mask(self) -> torch.Tensor:
        """Return DLS joint components clipped to position limits."""
        return self._joint_limit_clipped_mask

    def _prepare_solver(self) -> None:
        self._solver = self._env.robot.get_solver(self._part_name)
        solver_names = tuple(getattr(self._solver, "joint_names", ()) or ())
        if (
            len(solver_names) != len(self._joint_names)
            or len(set(solver_names)) != len(solver_names)
            or set(solver_names) != set(self._joint_names)
        ):
            raise ValueError(
                "Relative EEF actions require solver joint_names matching selected joints."
            )
        chain = getattr(self._solver, "pk_serial_chain", None)
        chain_names = getattr(chain, "get_joint_parameter_names", None)
        if callable(chain_names) and tuple(chain_names()) != solver_names:
            raise ValueError("Solver joint_names must match its kinematic chain order.")
        self._solver_order = [self._joint_names.index(name) for name in solver_names]
        self._selection_order = [solver_names.index(name) for name in self._joint_names]
        if self._ik_method != "dls":
            return
        tcp = torch.as_tensor(
            self._solver.tcp_xpos, dtype=torch.float32, device=self.device
        )
        if tcp.shape != (4, 4) or not bool(torch.isfinite(tcp).all()):
            raise ValueError("DLS requires a finite 4-by-4 solver TCP transform.")
        self._tcp_offset = tcp[:3, 3]
        self._tcp_rotation = tcp[:3, :3]
        self._has_tcp_offset = bool(self._tcp_offset.any())

    def _dls_step(
        self, current: torch.Tensor, error: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ask for the end-link Jacobian without PK's locations adapter, whose
        # Transform3d construction can allocate CPU tensors for CUDA offsets.
        jacobian = self._solver.get_jacobian(current[:, self._solver_order])
        if tuple(jacobian.shape) != (self.num_envs, 6, len(self._joint_ids)):
            raise ValueError("DLS requires a batched full 6-by-DOF Jacobian.")
        rotation = self._env.robot.get_link_pose(
            link_name=self._solver.root_link_name, to_matrix=True
        )[:, :3, :3]
        jacobian = torch.cat(
            (rotation @ jacobian[:, :3], rotation @ jacobian[:, 3:]), dim=1
        )
        if self._has_tcp_offset:
            link_rotation = self._current_pose[:, :3, :3] @ self._tcp_rotation.T
            offset = (link_rotation @ self._tcp_offset).unsqueeze(1)
            shift = torch.cross(jacobian[:, 3:].transpose(1, 2), offset, dim=-1)
            jacobian[:, :3] += shift.transpose(1, 2)
        finite = torch.isfinite(jacobian).all(dim=(1, 2))
        jacobian = torch.where(finite[:, None, None], jacobian, 0.0)
        transpose = jacobian.transpose(-1, -2)
        identity = torch.eye(6, device=self.device, dtype=current.dtype)
        system = jacobian @ transpose + self._dls_damping**2 * identity
        finite &= torch.isfinite(system).all(dim=(1, 2))
        system = torch.where(finite[:, None, None], system, identity)
        solution, info = torch.linalg.solve_ex(system, error.unsqueeze(-1))
        delta = (transpose @ solution).squeeze(-1)[:, self._selection_order]
        finite &= (info == 0) & torch.isfinite(delta).all(dim=-1)
        delta = torch.where(finite[:, None], delta, 0.0)
        largest = delta.abs().amax(dim=-1, keepdim=True)
        self._joint_step_limited.copy_(finite & (largest[:, 0] > self._max_joint_delta))
        delta *= self._max_joint_delta / largest.clamp_min(self._max_joint_delta)
        requested = current + delta
        limits = self._env.robot.body_data.qpos_limits[:, list(self._joint_ids)]
        bounded = requested.clamp(limits[..., 0], limits[..., 1])
        self._joint_limit_clipped_mask.copy_(finite[:, None] & (bounded != requested))
        finite &= torch.isfinite(bounded).all(dim=-1) & (
            (bounded - current).abs().amax(dim=-1) <= self._max_joint_delta + 1e-6
        )
        return finite, bounded

    def process_actions(self, actions: torch.Tensor) -> None:
        """Resolve one finite normalized request from current measured joints.

        Args:
            actions: Floating requests with shape ``(num_envs, 6)``.

        Raises:
            TypeError: If requests are not floating tensors.
            ValueError: If their shape/device is invalid or any value is nonfinite.
        """
        if not isinstance(actions, torch.Tensor) or not actions.is_floating_point():
            raise TypeError("RelativeEefPoseAction requires a floating tensor.")
        if (
            tuple(actions.shape) != (self.num_envs, 6)
            or actions.device != self._raw_actions.device
        ):
            raise ValueError(
                f"RelativeEefPoseAction expects shape {(self.num_envs, 6)} "
                f"on {self._raw_actions.device}."
            )
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("RelativeEefPoseAction requests must be finite.")
        self._previous_raw_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(actions)
        self._clipped_mask.copy_(actions.abs() > 1.0)
        normalized = self._raw_actions.clamp(-1.0, 1.0)
        current = self._env.robot.get_qpos()[:, list(self._joint_ids)]
        self._current_pose.copy_(
            self._env.robot.compute_fk(
                qpos=current[:, self._solver_order],
                name=self._part_name,
                to_matrix=True,
            )
        )
        self._target_pose.copy_(self._current_pose)
        self._target_pose[:, :3, 3] += self._translation_scale * normalized[:, :3]
        self._target_pose[:, :3, :3] = (
            axis_angle_to_rotation_matrix(self._rotation_scale * normalized[:, 3:])
            @ self._current_pose[:, :3, :3]
        )
        if self._ik_method == "dls":
            error = torch.cat(
                (
                    self._translation_scale * normalized[:, :3],
                    self._rotation_scale * normalized[:, 3:],
                ),
                dim=-1,
            )
            success, qpos = self._dls_step(current, error)
        else:
            success, qpos = self._env.robot.compute_ik(
                pose=self._target_pose,
                joint_seed=current[:, self._solver_order],
                name=self._part_name,
            )
            qpos = qpos[:, self._selection_order]
        self._ik_success.copy_(success & torch.isfinite(qpos).all(dim=-1))
        self._processed_actions.copy_(
            torch.where(self._ik_success[:, None], qpos, current)
        )
        self._joint_delta.copy_(self._processed_actions - current)

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        """Clear selected action history and diagnostics.

        Args:
            env_ids: Environment rows to reset; None selects every row.
        """
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._current_pose[ids] = torch.eye(4, device=self.device)
        self._target_pose[ids] = torch.eye(4, device=self.device)
        self._clipped_mask[ids] = False
        self._joint_delta[ids] = 0
        self._joint_step_limited[ids] = False
        self._joint_limit_clipped_mask[ids] = False


class ParallelGripperAction(ActionTerm):
    """Map one scalar policy action to explicit parallel-gripper joints."""

    contract_id: ClassVar[str] = "gripper.parallel@1"
    command_type = "qpos"

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        self._part_name = str(params["part_name"])
        self._joint_ids = tuple(
            int(value)
            for value in env.robot.get_joint_ids(self._part_name, remove_mimic=True)
        )
        if not self._joint_ids:
            raise ValueError("ParallelGripperAction resolved no independent joints.")
        self._joint_names = tuple(
            env.robot.joint_names[index] for index in self._joint_ids
        )
        self._command_mode = str(params.get("command_mode", "continuous"))
        if self._command_mode not in {"continuous", "binary"}:
            raise ValueError("command_mode must be 'continuous' or 'binary'.")

        limits = env.robot.body_data.qpos_limits[0, list(self._joint_ids)]
        if self._command_mode == "continuous":
            if bool(params.get("use_joint_limits", False)):
                self._lower_command = limits[:, 0].clone()
                self._upper_command = limits[:, 1].clone()
            else:
                self._lower_command = self._resolve_command(
                    params.get("lower_command"),
                    field_name="lower_command",
                )
                self._upper_command = self._resolve_command(
                    params.get("upper_command"),
                    field_name="upper_command",
                )
            self._open_command = self._upper_command
            self._close_command = self._lower_command
        else:
            self._open_command = self._resolve_command(
                params.get("open_command"),
                field_name="open_command",
            )
            self._close_command = self._resolve_command(
                params.get("close_command"),
                field_name="close_command",
            )
            self._lower_command = self._close_command
            self._upper_command = self._open_command

        self._raw_actions = torch.zeros(
            env.num_envs, 1, dtype=torch.float32, device=env.device
        )
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros(
            env.num_envs,
            len(self._joint_ids),
            dtype=torch.float32,
            device=env.device,
        )
        self._space = gym.spaces.Box(
            low=-np.ones(1, dtype=np.float32),
            high=np.ones(1, dtype=np.float32),
            dtype=np.float32,
        )
        self._descriptor = ActionTermDescriptor(
            representation="parallel_gripper",
            action_dim=1,
            feature_names=("gripper",),
            units=("normalized",),
            normalization="minus_one_to_one",
            joint_names=self._joint_names,
            metadata={"command_mode": self._command_mode},
            contract=self.resolved_contract_id(),
        )

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._space

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def previous_raw_actions(self) -> torch.Tensor:
        """Return scalar requests from the previous control step."""
        return self._previous_raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def controlled_joint_ids(self) -> tuple[int, ...]:
        return self._joint_ids

    @property
    def descriptor(self) -> ActionTermDescriptor:
        return self._descriptor

    @property
    def part_name(self) -> str:
        """Return the robot control part owning the gripper joints."""
        return self._part_name

    @property
    def command_mode(self) -> str:
        """Return the continuous or binary scalar command mode."""
        return self._command_mode

    @property
    def lower_command(self) -> torch.Tensor:
        """Return the continuous command at normalized value ``-1``."""
        return self._lower_command

    @property
    def upper_command(self) -> torch.Tensor:
        """Return the continuous command at normalized value ``1``."""
        return self._upper_command

    @property
    def open_command(self) -> torch.Tensor:
        """Return the resolved open joint command."""
        return self._open_command

    @property
    def close_command(self) -> torch.Tensor:
        """Return the resolved close joint command."""
        return self._close_command

    def _resolve_command(self, value: Any, *, field_name: str) -> torch.Tensor:
        if not isinstance(value, dict):
            raise TypeError(f"{field_name} must be a mapping of joint expressions.")
        indices, _, values = resolve_matching_names_values(
            value,
            self._joint_names,
            preserve_order=True,
        )
        if sorted(indices) != list(range(len(self._joint_names))):
            raise ValueError(
                f"{field_name} must cover every gripper joint exactly once."
            )
        command = torch.empty(
            len(self._joint_names),
            dtype=torch.float32,
            device=self.device,
        )
        command[indices] = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )
        return command

    def process_actions(self, actions: torch.Tensor) -> None:
        expected = (self.num_envs, 1)
        if tuple(actions.shape) != expected:
            raise ValueError(
                f"ParallelGripperAction expected action shape {expected}, "
                f"got {tuple(actions.shape)}."
            )
        self._previous_raw_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(actions)
        if self._command_mode == "continuous":
            weight = (self._raw_actions.clamp(-1.0, 1.0) + 1.0) * 0.5
            self._processed_actions.copy_(
                torch.lerp(self._lower_command, self._upper_command, weight)
            )
        else:
            self._processed_actions.copy_(
                torch.where(
                    self._raw_actions >= 0.0,
                    self._open_command,
                    self._close_command,
                )
            )

    def apply_actions(self) -> None:
        self._env.robot.set_qpos(
            qpos=self._processed_actions,
            joint_ids=list(self._joint_ids),
        )

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0
        self._previous_raw_actions[ids] = 0
        self._processed_actions[ids] = 0


_ACTION_CONTRACTS: dict[str, type[ActionTerm]] = {}


def register_action_contract(contract_id: str, action_class: type[ActionTerm]) -> None:
    """Register an implementation for a stable action contract.

    Args:
        contract_id: Versioned, implementation-independent contract ID.
        action_class: Current :class:`ActionTerm` implementation.

    Raises:
        TypeError: If ``action_class`` is not an action term class.
        ValueError: If the contract is already registered to another class.
    """
    if not isinstance(contract_id, str) or not contract_id.strip():
        raise ValueError("contract_id must be a non-empty string.")
    if not isinstance(action_class, type) or not issubclass(action_class, ActionTerm):
        raise TypeError("action_class must inherit from ActionTerm.")
    previous = _ACTION_CONTRACTS.get(contract_id)
    if previous is not None and previous is not action_class:
        raise ValueError(
            f"Action contract {contract_id!r} is already registered to "
            f"{previous.__name__}."
        )
    _ACTION_CONTRACTS[contract_id] = action_class


def resolve_action_contract(contract_id: str) -> type[ActionTerm]:
    """Resolve a stable action contract to the current implementation.

    Args:
        contract_id: Versioned, implementation-independent contract ID.

    Returns:
        The current action-term implementation class.

    Raises:
        KeyError: If the contract is not registered.
    """
    try:
        return _ACTION_CONTRACTS[contract_id]
    except KeyError as error:
        available = ", ".join(sorted(_ACTION_CONTRACTS))
        raise KeyError(
            f"Unknown action contract {contract_id!r}. Available: {available}"
        ) from error


for _contract_id, _action_class in {
    "joint_position.absolute@1": JointPositionAction,
    "joint_position.default_offset@1": DefaultJointPositionAction,
    "joint_position.limits@1": JointPositionToLimitsAction,
    "joint_position.relative@1": RelativeJointPositionAction,
    "joint_velocity@1": JointVelocityAction,
    "joint_effort@1": JointEffortAction,
    "eef_pose.absolute@1": EefPoseAction,
    "eef_pose.relative@1": RelativeEefPoseAction,
    "gripper.parallel@1": ParallelGripperAction,
}.items():
    register_action_contract(_contract_id, _action_class)
