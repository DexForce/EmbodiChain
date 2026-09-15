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

"""Strict physical initial states for an exclusively owned simulation batch."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.motion.expansion import (
    ValidationCheck,
    ValidationResult,
)

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot

__all__ = ["SimInitialState", "SimInitialStateAdapter"]

_ROBOT_FIELDS = (
    "root_pose",
    "root_linear_velocity",
    "root_angular_velocity",
    "qpos",
    "qvel",
    "target_qpos",
    "target_qvel",
    "qf",
)
_RIGID_FIELDS = ("pose", "linear_velocity", "angular_velocity")


def _copy_fields(values: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """Own finite floating-point state independently of backend tensor buffers."""
    copied = {}
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("State fields must map names to tensors.")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"State field {key!r} must contain finite floating values."
            )
        copied[key] = value.detach().clone()
    return MappingProxyType(copied)


@dataclass(frozen=True)
class SimInitialState:
    """Owned physical state of every row in one fixed-base robot scene.

    Obtain this value from :meth:`SimInitialStateAdapter.capture`. Pose fields
    are local homogeneous matrices; joint fields include mimic and gripper
    joints in ``joint_names`` order. Nested mappings are read-only and tensors
    are cloned on construction. The adapter validates them again before use.
    This value contains no contact solver checkpoint, pending forces, manager
    state, or physical/visual property snapshot.

    Args:
        batch_size: Number of simulation rows.
        robot_uid: Registered robot identity.
        joint_names: Complete movable-joint order, including mimic joints.
        signature: Structural signature supplied by the capturing adapter.
        robot: Root, joint, drive-target, and joint-effort state tensors.
        rigid_objects: Pose and velocity tensors keyed by every rigid UID.
    """

    batch_size: int
    robot_uid: str
    joint_names: tuple[str, ...]
    signature: str
    robot: Mapping[str, torch.Tensor]
    rigid_objects: Mapping[str, Mapping[str, torch.Tensor]]

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(self.robot_uid, str) or not self.robot_uid:
            raise ValueError("robot_uid must be a nonempty string.")
        if not isinstance(self.signature, str) or not self.signature:
            raise ValueError("signature must be a nonempty string.")
        names = tuple(self.joint_names)
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("joint_names must contain unique nonempty names.")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "robot", _copy_fields(self.robot))
        rigid = {}
        for uid, fields in self.rigid_objects.items():
            if not isinstance(uid, str) or not uid:
                raise ValueError("Rigid object UIDs must be nonempty strings.")
            rigid[uid] = _copy_fields(fields)
        object.__setattr__(self, "rigid_objects", MappingProxyType(rigid))


class SimInitialStateAdapter:
    """Capture, restore, and verify one entire simulation batch.

    Supports one config-declared fixed-base robot and all ordinary rigid
    objects. Additional articulations, object groups, deformables, and rigid
    constraints are rejected. Asset-owned USD articulation properties are
    rejected because ``cfg.fix_base`` is not authoritative for those assets.
    The host owns exclusive batch access, deterministic task initialization,
    fixed scene properties, settling, observation refresh, and runtime epochs.
    Initial states must have no pending external forces or contact constraints.

    .. attention::
        Restoring the robot root calls the existing articulation pose setter,
        which can advance the whole world by 1 ms. Root restoration therefore
        precedes all remaining writes. No Gym step, reset event, or recording
        callback is invoked. The host must settle and verify before execution.

    Args:
        sim: Simulation manager whose complete batch is exclusively owned.
        robot: The only robot registered in ``sim``.
        atol: Absolute element-wise tolerance for physical verification and
            zero root/non-dynamic velocities. Relative tolerance is zero.
        tolerances_profile_id: Trusted integration ID naming this tolerance policy.
    """

    def __init__(
        self,
        sim: SimulationManager,
        robot: Robot,
        *,
        atol: float = 1e-5,
        tolerances_profile_id: str = "fixed_scene_tolerances",
    ) -> None:
        if isinstance(atol, bool) or not isinstance(atol, (int, float)):
            raise TypeError("atol must be a finite nonnegative number.")
        if not math.isfinite(atol) or atol < 0:
            raise ValueError("atol must be a finite nonnegative number.")
        self.sim = sim
        self.robot = robot
        self.atol = float(atol)
        if (
            not isinstance(tolerances_profile_id, str)
            or not tolerances_profile_id.strip()
        ):
            raise ValueError("tolerances_profile_id must be a nonempty string.")
        self.tolerances_profile_id = tolerances_profile_id
        self._check_scene()

    def _check_scene(self) -> None:
        """Reject unsupported or incompletely represented scenes before writes."""
        sim, robot = self.sim, self.robot
        if sim._robots != {robot.uid: robot}:
            raise ValueError(
                "Initial states require exactly the supplied registered robot."
            )
        if not robot.cfg.fix_base or getattr(robot.cfg, "use_usd_properties", False):
            raise ValueError("Initial states require a config-owned fixed-base robot.")
        for name in (
            "_articulations",
            "_rigid_object_groups",
            "_soft_objects",
            "_cloth_objects",
            "_constraints",
        ):
            if getattr(sim, name):
                raise ValueError(f"Initial state restoration does not support {name}.")
        if robot.num_instances != sim.num_envs:
            raise ValueError("The robot must cover the entire simulation batch.")
        names = tuple(robot.joint_names)
        if len(names) != robot.dof or len(set(names)) != len(names) or not names:
            raise ValueError(
                "Robot joint_names must identify every movable joint once."
            )
        for uid, obj in sim._rigid_objects.items():
            if uid != obj.uid or obj.num_instances != sim.num_envs:
                raise ValueError(
                    f"Rigid object {uid!r} does not match its full-batch registration."
                )

    def signature(self) -> str:
        """Return a stable topology and control-part signature of the batch.

        Includes row count, robot identity, complete joint order, named control
        parts, and rigid object identities/body modes. Current poses and fixed
        physical/visual properties are excluded; the host's preparation profile
        must separately identify and verify those fixed case conditions.
        """
        self._check_scene()
        payload = {
            "batch_size": self.sim.num_envs,
            "robot_uid": self.robot.uid,
            "joint_names": list(self.robot.joint_names),
            "control_parts": self.robot.control_parts or {},
            "rigid_objects": [
                (uid, bool(obj.is_static), bool(obj.is_non_dynamic))
                for uid, obj in sorted(self.sim._rigid_objects.items())
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _read(self) -> SimInitialState:
        """Read every supported field without invoking simulation updates."""
        signature = self.signature()
        robot = self.robot
        root_index = list(robot.body_data.link_names).index(robot.root_link_name)
        root_velocity = robot.body_data.body_link_vel[:, root_index]
        fields = {
            "root_pose": robot.get_local_pose(to_matrix=True),
            "root_linear_velocity": root_velocity[:, :3],
            "root_angular_velocity": root_velocity[:, 3:],
            "qpos": robot.get_qpos(),
            "qvel": robot.get_qvel(),
            "target_qpos": robot.get_qpos(target=True),
            "target_qvel": robot.get_qvel(target=True),
            "qf": robot.get_qf(),
        }
        rigid = {}
        for uid, obj in self.sim._rigid_objects.items():
            body_state = obj.body_state
            rigid[uid] = {
                "pose": obj.get_local_pose(to_matrix=True),
                "linear_velocity": body_state[:, 7:10],
                "angular_velocity": body_state[:, 10:13],
            }
        return SimInitialState(
            batch_size=self.sim.num_envs,
            robot_uid=robot.uid,
            joint_names=tuple(robot.joint_names),
            signature=signature,
            robot=fields,
            rigid_objects=rigid,
        )

    def _check_fields(
        self,
        fields: Mapping[str, torch.Tensor],
        shapes: Mapping[str, tuple[int, ...]],
        label: str,
    ) -> None:
        if set(fields) != set(shapes):
            raise ValueError(
                f"{label} fields do not match the complete initial state schema."
            )
        for name, shape in shapes.items():
            value = fields[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"{label}.{name} requires finite floating values of shape {shape}."
                )
            if name.endswith("pose"):
                pose = value.detach().to(dtype=torch.float64, device="cpu")
                rotation = pose[:, :3, :3]
                if (
                    not torch.allclose(
                        pose[:, 3],
                        torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=pose.dtype).expand(
                            shape[0], -1
                        ),
                        atol=1e-5,
                        rtol=0,
                    )
                    or not torch.allclose(
                        rotation.mT @ rotation,
                        torch.eye(3, dtype=pose.dtype).expand(shape[0], -1, -1),
                        atol=1e-5,
                        rtol=0,
                    )
                    or not torch.allclose(
                        torch.linalg.det(rotation),
                        torch.ones(shape[0], dtype=pose.dtype),
                        atol=1e-5,
                        rtol=0,
                    )
                ):
                    raise ValueError(f"{label}.{name} must contain rigid SE(3) poses.")

    def _check_state(self, state: SimInitialState) -> None:
        """Validate the full snapshot before any physical mutation."""
        if not isinstance(state, SimInitialState):
            raise TypeError("state must be a SimInitialState.")
        if (
            state.signature != self.signature()
            or state.batch_size != self.sim.num_envs
            or state.robot_uid != self.robot.uid
            or state.joint_names != tuple(self.robot.joint_names)
            or set(state.rigid_objects) != set(self.sim._rigid_objects)
        ):
            raise ValueError(
                "Initial state topology, joint order, or controller signature changed."
            )
        batch, dof = state.batch_size, self.robot.dof
        shapes = {key: (batch, dof) for key in _ROBOT_FIELDS}
        shapes.update(
            root_pose=(batch, 4, 4),
            root_linear_velocity=(batch, 3),
            root_angular_velocity=(batch, 3),
        )
        self._check_fields(state.robot, shapes, "robot")
        for key in ("root_linear_velocity", "root_angular_velocity"):
            if bool((state.robot[key].abs() > self.atol).any()):
                raise ValueError(
                    "A fixed-base initial state must have zero root velocity."
                )
        limits = (
            self.robot.get_qpos_limits().detach().to(device="cpu", dtype=torch.float64)
        )
        if (
            limits.shape != (batch, dof, 2)
            or bool(torch.isnan(limits).any())
            or bool((limits[..., 0] > limits[..., 1]).any())
        ):
            raise ValueError("Robot joint limits do not cover the full ordered batch.")
        for key in ("qpos", "target_qpos"):
            value = state.robot[key].detach().to(device="cpu", dtype=torch.float64)
            if bool(((value < limits[..., 0]) | (value > limits[..., 1])).any()):
                raise ValueError(
                    f"robot.{key} is outside joint limits and would be clamped."
                )
        for uid, fields in state.rigid_objects.items():
            shapes = {
                "pose": (batch, 4, 4),
                "linear_velocity": (batch, 3),
                "angular_velocity": (batch, 3),
            }
            self._check_fields(fields, shapes, f"rigid_objects.{uid}")
            if self.sim._rigid_objects[uid].is_non_dynamic and any(
                bool((fields[key].abs() > self.atol).any()) for key in _RIGID_FIELDS[1:]
            ):
                raise ValueError(
                    f"Non-dynamic object {uid!r} must have zero initial velocity."
                )

    def capture(self) -> SimInitialState:
        """Own a validated physical snapshot after host preparation and settling."""
        state = self._read()
        self._check_state(state)
        return state

    def restore(self, state: SimInitialState) -> None:
        """Restore all rows after complete preflight, without layout randomization.

        Resets pending rigid-body forces/torques, then restores captured
        velocities. Fixed-base root velocities must be zero. This method does
        not restore arbitrary contact states or hide write/backend failures.
        The caller must invalidate its epoch before entry and settle/verify
        after completion; an exception can leave partially restored state.
        """
        self._check_state(state)
        robot = self.robot
        fields = {
            key: value.detach().to(device=robot.device, dtype=torch.float32).clone()
            for key, value in state.robot.items()
        }
        robot.set_local_pose(fields["root_pose"])
        robot.set_qpos(fields["qpos"], target=False)
        robot.set_qvel(fields["qvel"], target=False)
        robot.set_qpos(fields["target_qpos"], target=True)
        robot.set_qvel(fields["target_qvel"], target=True)
        robot.set_qf(fields["qf"])
        for uid, source in state.rigid_objects.items():
            obj = self.sim._rigid_objects[uid]
            fields = {
                key: value.detach().to(device=obj.device, dtype=torch.float32).clone()
                for key, value in source.items()
            }
            obj.set_local_pose(fields["pose"])
            if not obj.is_non_dynamic:
                obj.clear_dynamics()
                obj.set_velocity(
                    lin_vel=fields["linear_velocity"],
                    ang_vel=fields["angular_velocity"],
                )

    def verify(self, state: SimInitialState) -> ValidationResult:
        """Check every physical field after settling, with zero relative tolerance.

        Structural incompatibility or a malformed snapshot raises before
        reading physical state. Live nonfinite state and numerical drift
        return a failed ``initial_state`` check, never an accepted result.
        """
        self._check_state(state)
        try:
            actual = self._read()
        except ValueError as error:
            return ValidationResult(
                (ValidationCheck("initial_state", "failed", str(error)),)
            )
        pairs = [("robot", state.robot, actual.robot)]
        pairs.extend(
            (f"rigid_objects.{uid}", fields, actual.rigid_objects[uid])
            for uid, fields in state.rigid_objects.items()
        )
        for label, expected, observed in pairs:
            for key, value in expected.items():
                current = observed[key].to(device=value.device, dtype=value.dtype)
                if current.shape != value.shape or not torch.allclose(
                    current, value, atol=self.atol, rtol=0
                ):
                    return ValidationResult(
                        (
                            ValidationCheck(
                                "initial_state",
                                "failed",
                                f"{label}.{key} differs from the specified initial state.",
                            ),
                        )
                    )
        return ValidationResult((ValidationCheck("initial_state", "passed"),))
