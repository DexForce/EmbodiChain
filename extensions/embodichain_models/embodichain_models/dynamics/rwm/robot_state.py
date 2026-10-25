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

"""Frame-local robot-state increments for learned transition models.

States use ``[position(3), rotation_matrix(9), linear_velocity(3),
angular_velocity(3), joint_position(J), joint_velocity(J)]``. Rotations map
body vectors into world coordinates. Increments use the current body frame;
rotation increments are right-multiplied exponential coordinates in radians.
The logarithm uses the shortest rotation and has the usual branch cut at pi.
Only first-order derivatives away from that cut are part of the contract.
"""

from __future__ import annotations

import torch

__all__ = [
    "rotation_exp",
    "rotation_log",
    "robot_state_features",
    "robot_state_increment",
    "integrate_robot_state",
    "robot_state_error",
]


def _skew(value: torch.Tensor) -> torch.Tensor:
    x, y, z = value.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        value.shape[:-1] + (3, 3)
    )


def rotation_exp(vector: torch.Tensor) -> torch.Tensor:
    """Map rotation vectors to matrices with finite derivatives at zero.

    Args:
        vector: Floating tensor with final dimension three, in radians.

    Returns:
        Rotation matrices with shape ``vector.shape[:-1] + (3, 3)``.
    """
    if vector.shape[-1:] != (3,) or not vector.is_floating_point():
        raise ValueError("rotation vectors must be floating tensors ending in 3")
    squared = vector.square().sum(-1, keepdim=True)
    safe_squared = squared.clamp_min(1e-8)
    angle = safe_squared.sqrt()
    small = squared < 1e-8
    a = torch.where(
        small,
        1 - squared / 6 + squared.square() / 120,
        angle.sin() / angle,
    )
    b = torch.where(
        small,
        0.5 - squared / 24 + squared.square() / 720,
        (1 - angle.cos()) / safe_squared,
    )
    skew = _skew(vector)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return eye + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * (skew @ skew)


def rotation_log(matrix: torch.Tensor) -> torch.Tensor:
    """Map proper rotation matrices to shortest rotation vectors.

    Args:
        matrix: Floating tensor ending in ``(3, 3)``. Inputs must be proper
            rotations; this function does not project arbitrary matrices.

    Returns:
        Rotation vectors. At the pi branch the selected axis is deterministic
        but its derivative is not a supported contract.
    """
    if matrix.shape[-2:] != (3, 3) or not matrix.is_floating_point():
        raise ValueError("rotations must be floating matrices ending in (3, 3)")
    vector = (
        torch.stack(
            (
                matrix[..., 2, 1] - matrix[..., 1, 2],
                matrix[..., 0, 2] - matrix[..., 2, 0],
                matrix[..., 1, 0] - matrix[..., 0, 1],
            ),
            -1,
        )
        * 0.5
    )
    squared = vector.square().sum(-1, keepdim=True)
    sine = vector.norm(dim=-1, keepdim=True)
    cosine = ((matrix.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) - 1) / 2).clamp(
        -1, 1
    )
    angle = torch.atan2(sine, cosine)
    scale = torch.where(
        (squared < 1e-8) & (cosine > 0),
        1 + squared / 6 + 3 * squared.square() / 40,
        angle / sine.clamp_min(1e-8),
    )
    ordinary = vector * scale
    # Near pi the antisymmetric part vanishes. For a proper rotation,
    # (R+R.T)/2-cos(theta)I equals (1-cos(theta))*axis*axis.T.
    eye = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    symmetric = (matrix + matrix.transpose(-1, -2)) / 2 - cosine.unsqueeze(-1) * eye
    index = symmetric.square().sum(-2).argmax(-1)
    axis = symmetric.gather(
        -1, index[..., None, None].expand(index.shape + (3, 1))
    ).squeeze(-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sign = torch.where((axis * vector).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    return torch.where(cosine < -0.9999, axis * sign * angle, ordinary)


def _parts(state: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if state.ndim < 1 or state.shape[-1] < 18 or (state.shape[-1] - 18) % 2:
        raise ValueError("robot state width must be 18 + 2 * num_joints")
    joints = (state.shape[-1] - 18) // 2
    return (
        state[..., :3],
        state[..., 3:12].reshape(state.shape[:-1] + (3, 3)),
        state[..., 12:15],
        state[..., 15:18],
        state[..., 18 : 18 + joints],
        state[..., 18 + joints :],
    )


def _body(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (rotation.transpose(-1, -2) @ vector.unsqueeze(-1)).squeeze(-1)


def robot_state_features(state: torch.Tensor) -> torch.Tensor:
    """Encode height, body-frame gravity and velocities, and joint state.

    Args:
        state: Robot states in the module's matrix-based layout. The world
            frame is Z-up and the gravity direction is ``(0, 0, -1)``.

    Returns:
        Features of width ``10 + 2 * num_joints``. Global XY position and yaw
        do not enter these features.
    """
    position, rotation, linear, angular, joint, velocity = _parts(state)
    gravity = -rotation[..., 2, :]
    return torch.cat(
        (
            position[..., 2:3],
            gravity,
            _body(rotation, linear),
            _body(rotation, angular),
            joint,
            velocity,
        ),
        -1,
    )


def robot_state_increment(
    state: torch.Tensor, next_state: torch.Tensor
) -> torch.Tensor:
    """Encode the transition to ``next_state`` in the current body frame.

    Args:
        state: Current states in the module's canonical layout.
        next_state: Aligned next states with exactly the same shape.

    Returns:
        Increments of width ``12 + 2 * num_joints``.
    """
    if state.shape != next_state.shape:
        raise ValueError("current and next robot state shapes must match")
    p, r, v, w, q, qd = _parts(state)
    p1, r1, v1, w1, q1, qd1 = _parts(next_state)
    return torch.cat(
        (
            _body(r, p1 - p),
            rotation_log(r.transpose(-1, -2) @ r1),
            _body(r, v1 - v),
            _body(r, w1 - w),
            q1 - q,
            qd1 - qd,
        ),
        -1,
    )


def integrate_robot_state(state: torch.Tensor, increment: torch.Tensor) -> torch.Tensor:
    """Apply frame-local increments while preserving proper rotations.

    Args:
        state: Current states in the module's canonical layout.
        increment: Aligned increments with final width ``state.shape[-1]-6``.

    Returns:
        Newly allocated next states with gradients to both arguments.
    """
    if increment.shape != state.shape[:-1] + (state.shape[-1] - 6,):
        raise ValueError("increment shape does not match the robot state")
    p, r, v, w, q, qd = _parts(state)
    joints = q.shape[-1]
    dp, dr, dv, dw, dq, dqd = increment.split((3, 3, 3, 3, joints, joints), -1)
    world = lambda value: (r @ value.unsqueeze(-1)).squeeze(-1)
    return torch.cat(
        (
            p + world(dp),
            (r @ rotation_exp(dr)).flatten(-2),
            v + world(dv),
            w + world(dw),
            q + dq,
            qd + dqd,
        ),
        -1,
    )


def robot_state_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Measure prediction error in the target state's body frame.

    Args:
        predicted: Predicted canonical robot states.
        target: Aligned target states.

    Returns:
        Position, angular, velocity and joint errors in increment layout.
    """
    return robot_state_increment(target, predicted)
