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

"""Closed-loop inference with an official RP1 PartyMimic sparse5 export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation
import yaml

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

__all__ = ["PartyMimicPolicy", "PartyMimicState", "sparse5_window"]

_HISTORY = 10
_JOINTS = 28
_PROPRIO = 3 * _JOINTS + 6
_WINDOW = 11
_KEYPOINTS = (
    "pelvis",
    "head",
    "left_wrist",
    "right_wrist",
    "left_ankle",
    "right_ankle",
)
_REFERENCE_SHAPE = (_WINDOW, len(_KEYPOINTS), 7)
_INPUTS = {
    "proprio": (_HISTORY * _PROPRIO,),
    "root_quat_hist": (_HISTORY * 8,),
    "ref_kp": (_WINDOW * len(_KEYPOINTS) * 7,),
}


@dataclass(frozen=True)
class PartyMimicState:
    """One RP1 sample in policy joint order and the reference world frame.

    Joint positions and velocities are in radians and radians per second.
    Root orientation is xyzw. Root angular velocity is in world axes.
    """

    joint_position: ArrayLike
    joint_velocity: ArrayLike
    root_quaternion: ArrayLike
    root_angular_velocity: ArrayLike


def sparse5_window(reference: NDArray, frame: int) -> NDArray:
    """Select the current and ten future poses, holding the final pose at clip end.

    Args:
        reference: Pose array with shape ``(frames, 6, 7)``.
        frame: Index of the current reference frame.

    Returns:
        Reference window with shape ``(11, 6, 7)``.
    """
    if reference.ndim != 3 or reference.shape[1:] != _REFERENCE_SHAPE[1:]:
        raise ValueError("The sparse5 clip must have shape (frames, 6, 7).")
    if not 0 <= frame < len(reference):
        raise IndexError(f"Sparse5 frame {frame} is outside the clip.")
    return reference[np.minimum(frame + np.arange(_WINDOW), len(reference) - 1)]


def _finite(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float32]:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return result


class PartyMimicPolicy:
    """Stateful adapter for one RP1 instance with the exported sparse5 actor.

    Construct with :meth:`from_onnx`. Reference poses are world-frame xyz
    followed by xyzw quaternion in :attr:`keypoint_names` order, sampled at
    50 Hz. The returned targets are radians in :attr:`joint_names` order.
    """

    def __init__(
        self, session: InferenceSession, deployment: Mapping[str, Any]
    ) -> None:
        """Bind the actor to its deployment metadata.

        Args:
            session: ONNX Runtime session for the exported sparse5 actor.
            deployment: Parsed YAML from the official exporter.
        """
        inputs = {
            entry.name: (tuple(entry.shape), entry.type)
            for entry in session.get_inputs()
        }
        outputs = {
            entry.name: (tuple(entry.shape), entry.type)
            for entry in session.get_outputs()
        }
        if inputs != {
            name: (shape, "tensor(float)") for name, shape in _INPUTS.items()
        }:
            raise ValueError(f"Unexpected RP1 sparse5 ONNX inputs: {inputs}.")
        if outputs != {"action": ((_JOINTS,), "tensor(float)")}:
            raise ValueError(f"Unexpected RP1 sparse5 ONNX outputs: {outputs}.")

        interface = deployment.get("interface", {})
        if interface.get("name") != "sparse5" or interface.get("goal_frames") != list(
            range(1, _WINDOW)
        ):
            raise ValueError("Expected an official ten-goal sparse5 deployment.")
        self.joint_names = tuple(deployment["policy_joint_names"])
        if len(set(self.joint_names)) != _JOINTS or len(self.joint_names) != _JOINTS:
            raise ValueError("RP1 sparse5 requires 28 unique policy joints.")
        self.keypoint_names = tuple(interface["keypoints"][key] for key in _KEYPOINTS)
        if self.keypoint_names != tuple(deployment["motion"]["body_names"]):
            raise ValueError("Sparse5 keypoints must match the deployment body order.")

        self.default_joint_position = self._joint_values(
            deployment, "default_joint_pos"
        )
        self.action_scale = self._joint_values(deployment, "action_scale")
        self.joint_stiffness = self._joint_values(deployment, "joint_kp")
        self.joint_damping = self._joint_values(deployment, "joint_kd")
        if np.any(self.action_scale <= 0):
            raise ValueError("RP1 action scales must be positive.")

        self._session = session
        self._proprio: NDArray[np.float32] | None = None
        self._root_history: NDArray[np.float32] | None = None
        self._last_action = np.zeros(_JOINTS, dtype=np.float32)

    @classmethod
    def from_onnx(cls, path: str | Path) -> PartyMimicPolicy:
        """Load a frozen actor and its adjacent official deployment YAML.

        Args:
            path: Path to the exported ONNX file.

        Returns:
            A policy backed by ONNX Runtime's CPU execution provider.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("PartyMimic requires the onnxruntime package.") from exc

        path = Path(path)
        with path.with_suffix(".yaml").open(encoding="utf-8") as stream:
            deployment = yaml.safe_load(stream)
        if not isinstance(deployment, Mapping):
            raise ValueError("Expected a PartyMimic deployment YAML mapping.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        return cls(session, deployment)

    def joint_indices(self, simulation_joint_names: Sequence[str]) -> NDArray[np.intp]:
        """Return simulator indices in policy order, validating the RP1 joint set.

        Args:
            simulation_joint_names: Joint names in simulator order.

        Returns:
            Simulator indices in policy order.
        """
        names = tuple(simulation_joint_names)
        if len(names) != _JOINTS or set(names) != set(self.joint_names):
            raise ValueError("Simulator joints must match the RP1 policy joints.")
        return np.asarray(
            [names.index(name) for name in self.joint_names], dtype=np.intp
        )

    def reset(self, state: PartyMimicState, reference: ArrayLike) -> None:
        """Fill both histories with the initial observation and zero prior action.

        Args:
            state: Measured RP1 state at reset.
            reference: Initial reference window of shape ``(11, 6, 7)``.
        """
        poses = self._reference(reference)
        previous = np.zeros(_JOINTS, dtype=np.float32)
        proprio, rotations = self._observation(state, poses, previous)
        self._proprio = np.repeat(proprio[None], _HISTORY, axis=0)
        self._root_history = np.repeat(rotations[None], _HISTORY, axis=0)
        self._last_action = previous

    def step(self, state: PartyMimicState, reference: ArrayLike) -> NDArray[np.float32]:
        """Advance one control tick.

        Args:
            state: Measured RP1 state at the current tick.
            reference: Current reference window of shape ``(11, 6, 7)``.

        Returns:
            Joint position targets in radians and policy order.
        """
        if self._proprio is None or self._root_history is None:
            raise RuntimeError("Reset the PartyMimic policy before stepping it.")
        poses = self._reference(reference)
        current, rotations = self._observation(state, poses, self._last_action)
        proprio = np.concatenate((self._proprio[1:], current[None]))
        root_history = np.concatenate((self._root_history[1:], rotations[None]))
        (action,) = self._session.run(
            ["action"],
            {
                "proprio": proprio.ravel(),
                "root_quat_hist": root_history.ravel(),
                "ref_kp": poses.ravel(),
            },
        )
        action = _finite(action, (_JOINTS,), "PartyMimic action")
        self._proprio = proprio
        self._root_history = root_history
        self._last_action = action
        return self.default_joint_position + self.action_scale * action

    def _joint_values(
        self, deployment: Mapping[str, Any], field: str
    ) -> NDArray[np.float32]:
        values = deployment[field]
        return _finite([values[name] for name in self.joint_names], (_JOINTS,), field)

    @staticmethod
    def _reference(reference: ArrayLike) -> NDArray[np.float32]:
        poses = _finite(reference, _REFERENCE_SHAPE, "Sparse5 reference")
        norm = np.linalg.norm(poses[..., 3:], axis=-1, keepdims=True)
        if np.any(norm < 1e-6):
            raise ValueError("Sparse5 quaternion norm must be nonzero.")
        # The exported graph expects MuJoCo's wxyz convention.
        return np.concatenate(
            (poses[..., :3], poses[..., [6, 3, 4, 5]] / norm), axis=-1
        )

    def _observation(
        self,
        state: PartyMimicState,
        reference: NDArray[np.float32],
        previous: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        q = _finite(state.joint_position, (_JOINTS,), "Joint position")
        qd = _finite(state.joint_velocity, (_JOINTS,), "Joint velocity")
        orientation = Rotation.from_quat(
            _finite(state.root_quaternion, (4,), "Root quaternion")
        )
        vectors = np.stack(
            (
                _finite(state.root_angular_velocity, (3,), "Root angular velocity"),
                [0, 0, -1],
            )
        )
        body_omega, body_gravity = orientation.apply(vectors, inverse=True)
        proprio = np.concatenate(
            (q - self.default_joint_position, qd, body_omega, body_gravity, previous),
            dtype=np.float32,
        )
        root_wxyz = orientation.as_quat()[[3, 0, 1, 2]]
        root_pair = np.concatenate((root_wxyz, reference[0, 0, 3:]), dtype=np.float32)
        return proprio, root_pair
