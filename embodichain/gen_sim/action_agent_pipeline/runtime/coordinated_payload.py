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

"""Track and validate payloads carried by coordinated dual-arm actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.runtime.action_runtime_types import (
    CoordinatedPayloadRuntimeState,
    ExecutedAtomicAction,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atomic_action_spec import (
    AtomicActionSpec,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.pose_utils import (
    _ensure_batched_pose_tensor,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.success_evaluator import (
    evaluate_configured_success,
)
from embodichain.lab.sim.atomic_actions import ObjectSemantics, WorldState
from embodichain.utils.math import pose_inv

__all__ = [
    "_record_coordinated_payload_runtime_state",
    "_coordinated_transport_failure_mask",
    "_has_coordinated_held_object",
]


def _record_coordinated_payload_runtime_state(
    env,
    spec: AtomicActionSpec,
    semantics: ObjectSemantics,
    carrier_pose: torch.Tensor,
) -> None:
    if "payloads" not in spec.target_object:
        return
    payload_uids = tuple(str(uid) for uid in spec.target_object.get("payloads", []))
    carrier_inverse = pose_inv(carrier_pose)
    carrier_to_payload = []
    for payload_uid in payload_uids:
        payload = env.sim.get_rigid_object(payload_uid)
        if payload is None:
            raise ValueError(f"Unknown coordinated payload uid: {payload_uid!r}.")
        payload_pose = _ensure_batched_pose_tensor(
            payload.get_local_pose(to_matrix=True), env.robot.device
        )
        carrier_to_payload.append(torch.bmm(carrier_inverse, payload_pose))
    metadata = getattr(env, "agent_coordinated_transport", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    vertices = torch.as_tensor(
        semantics.geometry.get("mesh_vertices"),
        dtype=torch.float32,
        device=env.robot.device,
    )
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
        raise ValueError("Coordinated payload guard requires carrier mesh vertices.")
    extents = vertices[:, :2].max(dim=0).values - vertices[:, :2].min(dim=0).values
    half_extents = metadata.get(
        "support_half_extents",
        [
            max(0.01, float(extents[0]) * 0.5 - 0.02),
            max(0.01, float(extents[1]) * 0.5 - 0.02),
        ],
    )
    half_x, half_y = float(half_extents[0]), float(half_extents[1])
    for payload_uid, relative_pose in zip(payload_uids, carrier_to_payload):
        relative_position = relative_pose[:, :3, 3]
        supported = (
            (relative_position[:, 0].abs() <= half_x)
            & (relative_position[:, 1].abs() <= half_y)
            & (relative_position[:, 2] >= -0.03)
            & (relative_position[:, 2] <= 0.35)
        )
        if not bool(supported.all()):
            raise ValueError(
                f"Declared coordinated payload {payload_uid!r} is not on the "
                "carrier support area before grasp."
            )
    setattr(
        env,
        "_action_agent_coordinated_payload_state",
        CoordinatedPayloadRuntimeState(
            carrier_uid=str(semantics.label),
            payload_uids=payload_uids,
            initial_carrier_pose=carrier_pose.clone(),
            carrier_to_payload=tuple(carrier_to_payload),
            support_half_extents=(half_x, half_y),
            max_payload_drift=float(metadata.get("max_payload_drift", 0.04)),
            max_carrier_tilt=float(metadata.get("max_carrier_tilt", np.deg2rad(10.0))),
        ),
    )


def _coordinated_transport_failure_mask(
    env,
    world_states: Mapping[str, WorldState],
    arm_actions: Mapping[str, Any],
) -> torch.Tensor:
    num_envs = int(getattr(env, "num_envs", 1))
    runtime_state = getattr(env, "_action_agent_coordinated_payload_state", None)
    if not isinstance(runtime_state, CoordinatedPayloadRuntimeState):
        return torch.zeros(num_envs, dtype=torch.bool)
    carrier = env.sim.get_rigid_object(runtime_state.carrier_uid)
    if carrier is None:
        return torch.ones(num_envs, dtype=torch.bool)
    carrier_pose = _ensure_batched_pose_tensor(
        carrier.get_local_pose(to_matrix=True), env.robot.device
    )
    initial_pose = runtime_state.initial_carrier_pose.to(
        device=carrier_pose.device, dtype=carrier_pose.dtype
    )
    relative_rotation = torch.bmm(
        initial_pose[:, :3, :3].transpose(1, 2), carrier_pose[:, :3, :3]
    )
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rotation_angle = torch.arccos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    failed = rotation_angle > runtime_state.max_carrier_tilt

    coordinated_active = _has_coordinated_held_object(world_states)
    action_classes = {
        action.atomic_action_class
        for action in arm_actions.values()
        if isinstance(action, ExecutedAtomicAction)
    }
    if coordinated_active:
        held = evaluate_configured_success(
            env,
            {
                "type": "object_held_by_both_grippers",
                "object": runtime_state.carrier_uid,
                "max_distance": 0.10,
            },
        ).to(device=failed.device)
        failed |= ~held
    if "CoordinatedPickment" in action_classes:
        failed |= (carrier_pose[:, 2, 3] - initial_pose[:, 2, 3]) < 0.08

    carrier_inverse = pose_inv(carrier_pose)
    for payload_uid, initial_relative in zip(
        runtime_state.payload_uids, runtime_state.carrier_to_payload
    ):
        payload = env.sim.get_rigid_object(payload_uid)
        if payload is None:
            failed |= torch.ones_like(failed)
            continue
        payload_pose = _ensure_batched_pose_tensor(
            payload.get_local_pose(to_matrix=True), env.robot.device
        )
        current_relative = torch.bmm(carrier_inverse, payload_pose)
        initial_relative = initial_relative.to(
            device=current_relative.device, dtype=current_relative.dtype
        )
        relative_position = current_relative[:, :3, 3]
        drift = torch.linalg.norm(
            relative_position - initial_relative[:, :3, 3], dim=-1
        )
        half_x, half_y = runtime_state.support_half_extents
        supported = (
            (relative_position[:, 0].abs() <= half_x)
            & (relative_position[:, 1].abs() <= half_y)
            & (relative_position[:, 2] >= -0.03)
            & (relative_position[:, 2] <= 0.35)
        )
        failed |= (drift > runtime_state.max_payload_drift) | ~supported
    if not coordinated_active:
        delattr(env, "_action_agent_coordinated_payload_state")
    return failed.detach().cpu()


def _has_coordinated_held_object(world_states: Mapping[str, WorldState]) -> bool:
    return any(
        state.coordinated_held_object is not None
        for state in world_states.values()
        if isinstance(state, WorldState)
    )
