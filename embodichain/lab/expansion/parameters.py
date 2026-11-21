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

"""Prepare measured CoM and drive changes from the native host's owned baseline."""

from __future__ import annotations
from collections.abc import Mapping
import torch
from embodichain.lab.sim.scene import SceneView

__all__: list[str] = []


def _apply_auxiliary_physics(
    view: SceneView,
    baseline: Mapping[str, torch.Tensor],
    *,
    robot_uid: str,
    com_offsets: Mapping[str, torch.Tensor],
    stiffness_scale: Mapping[str, torch.Tensor],
    damping_scale: Mapping[str, torch.Tensor],
) -> None:
    view.check_topology()
    bindings = {uid: (kind, entity) for kind, uid, entity in view.entities()}
    ids = list(view.env_ids)
    count = len(ids)
    com_writes = []
    for uid, offset in com_offsets.items():
        kind, body = bindings.get(uid, (None, None))
        if kind != "rigid_object" or body.body_type != "dynamic":
            raise ValueError(f"CoM change requires a dynamic rigid object: {uid}")
        offset = torch.as_tensor(offset, dtype=torch.float32, device=view.sim.device)
        if tuple(offset.shape) != (count, 3) or not bool(torch.isfinite(offset).all()):
            raise ValueError("CoM offsets must be finite (B,3) vectors")
        target = baseline[f"{uid}.com_pose"].clone()
        if tuple(target.shape) != (count, 7):
            raise ValueError("CoM baseline selection differs")
        target[:, :3] += offset
        com_writes.append((body, target))
    drive_writes = {}
    robot = bindings[robot_uid][1] if stiffness_scale or damping_scale else None
    for field, scales in (("stiffness", stiffness_scale), ("damping", damping_scale)):
        if not scales:
            continue
        target = baseline[f"{robot_uid}.joint_{field}"].clone()
        for name, scale in scales.items():
            if name not in robot.joint_names:
                raise ValueError(f"Unknown drive joint: {name}")
            scale = torch.as_tensor(scale, dtype=torch.float32, device=target.device)
            if (
                tuple(scale.shape) != (count,)
                or not bool(torch.isfinite(scale).all())
                or bool((scale < 0).any())
            ):
                raise ValueError(
                    "drive scales must be finite nonnegative per-row values"
                )
            target[:, robot.joint_names.index(name)] *= scale
        if not bool(torch.isfinite(target).all()):
            raise ValueError("drive baseline is not finite")
        drive_writes[field] = target
    # All values and bindings have been validated before the first write.
    for body, target in com_writes:
        body.set_com_pose(target, env_ids=ids)
    if drive_writes:
        robot.set_joint_drive(env_ids=ids, **drive_writes)
    for body, target in com_writes:
        actual = body.body_data.com_pose[ids]
        if not torch.allclose(actual, target, atol=1e-7, rtol=1e-5):
            raise RuntimeError(f"CoM write did not take effect for {body.uid}")
    for field, target in drive_writes.items():
        actual = getattr(robot.body_data, f"joint_{field}")[ids]
        if not torch.allclose(actual, target, atol=1e-6, rtol=1e-5):
            raise RuntimeError(f"Joint {field} write did not take effect")
