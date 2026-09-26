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

from embodichain.utils.math import matrix_from_euler, matrix_from_quat
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
    """Add scaled relative actions to current selected positions."""

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
    """Map normalized actions around a configured default joint pose."""

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
        self._clip = float(cfg.params.get("clip", 1.0))
        self.position_bias = torch.zeros_like(self._raw_actions)

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.full(self.action_dim, self._clip, dtype=np.float32),
            high=np.full(self.action_dim, self._clip, dtype=np.float32),
            dtype=np.float32,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._store_raw(actions)
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
        super().reset(env_ids)


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
    "gripper.parallel@1": ParallelGripperAction,
}.items():
    register_action_contract(_contract_id, _action_class)
