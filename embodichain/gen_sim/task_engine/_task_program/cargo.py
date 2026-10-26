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

"""Necessary terminal cargo-envelope checks, not an inner-cavity certificate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__: list[str] = []


def inside_envelope(
    vertices: torch.Tensor,
    object_pose: torch.Tensor,
    carrier_pose: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    """Check the full horizontal mesh extent and the cargo's lowest point."""
    relative = torch.linalg.solve(carrier_pose, object_pose)
    local = vertices @ relative[:, :3, :3].transpose(-1, -2) + relative[:, None, :3, 3]
    minimum, maximum = local.amin(1), local.amax(1)
    margin = 0.002
    return (
        torch.isfinite(local).all(dim=-1).all(dim=-1)
        & (minimum[:, :2] >= bounds[0, :2] - margin).all(-1)
        & (maximum[:, :2] <= bounds[1, :2] + margin).all(-1)
        & (minimum[:, 2] >= bounds[0, 2] - margin)
        & (minimum[:, 2] <= bounds[1, 2] + margin)
    )


@dataclass(frozen=True, slots=True)
class CargoEnvelope:
    carrier_id: str
    object_id: str
    carrier: Any
    cargo: Any
    vertices: torch.Tensor
    bounds: torch.Tensor
    initial_mask: torch.Tensor

    def check(self) -> torch.Tensor:
        contained = inside_envelope(
            self.vertices,
            self.cargo.get_local_pose(to_matrix=True).to(self.vertices),
            self.carrier.get_local_pose(to_matrix=True).to(self.vertices),
            self.bounds,
        )
        return ~self.initial_mask | contained


def capture_cargo(
    env: Any, scene: dict[str, Any], graph: dict[str, Any]
) -> list[CargoEnvelope]:
    """Capture initial contents, excluding objects intentionally manipulated."""
    from ..task_program_bundle import _mesh_vertices

    carriers, manipulated = set(), set()
    for node in graph["nodes"]:
        call = node["call"]
        obj = call.get("object", call.get("arguments", {}).get("object"))
        if isinstance(obj, str):
            (carriers if node["task_type"] == "E5" else manipulated).add(obj)
    if not carriers:
        return []
    simulation = getattr(env, "unwrapped", env).sim
    objects = {
        item["uid"]: item for item in scene["simulation"].get("rigid_object", [])
    }
    guards = []
    for carrier_id in sorted(carriers):
        carrier = simulation.get_rigid_object(carrier_id)
        carrier_pose = carrier.get_local_pose(to_matrix=True)
        mesh = torch.as_tensor(
            _mesh_vertices(objects[carrier_id]),
            device=carrier_pose.device,
            dtype=carrier_pose.dtype,
        )
        bounds = torch.stack((mesh.amin(0), mesh.amax(0)))
        for object_id, cfg in objects.items():
            if object_id in carriers or object_id in manipulated:
                continue
            cargo = simulation.get_rigid_object(object_id)
            vertices = torch.as_tensor(
                _mesh_vertices(cfg),
                device=carrier_pose.device,
                dtype=carrier_pose.dtype,
            )
            initial = inside_envelope(
                vertices,
                cargo.get_local_pose(to_matrix=True).to(vertices),
                carrier_pose,
                bounds,
            )
            if initial.any():
                guards.append(
                    CargoEnvelope(
                        carrier_id,
                        object_id,
                        carrier,
                        cargo,
                        vertices,
                        bounds,
                        initial.clone(),
                    )
                )
    return guards


def check_cargo(
    guards: list[CargoEnvelope],
    batch_size: int,
    *,
    trajectory: Any = None,
    step_counts: Any = None,
) -> dict[str, Any]:
    """Publish row-local necessary conditions without replacing skill results."""
    accepted = [True] * batch_size
    entries = []
    for guard in guards:
        mask = guard.check().detach().cpu().tolist()
        if len(mask) != batch_size:
            raise ValueError("Cargo envelope batch differs from execution batch.")
        first_failed_frames: list[int | None] = [None] * batch_size
        if trajectory is not None:
            from embodichain.utils.math import xyz_quat_to_4x4_matrix

            if step_counts is None or len(step_counts) != batch_size:
                raise ValueError(
                    "Cargo trajectory requires per-environment step counts."
                )
            objects = trajectory["rigid_objects"]
            for row in range(batch_size):
                if not bool(guard.initial_mask[row]):
                    continue
                count = int(step_counts[row])
                carrier = objects[guard.carrier_id]["pose"][row]
                cargo = objects[guard.object_id]["pose"][row]
                if not 0 < count <= min(len(carrier), len(cargo)):
                    raise ValueError(
                        "Cargo trajectory has no complete valid frame range."
                    )
                for start in range(0, count, 64):
                    stop = min(start + 64, count)
                    valid = inside_envelope(
                        guard.vertices,
                        xyz_quat_to_4x4_matrix(cargo[start:stop].to(guard.vertices)),
                        xyz_quat_to_4x4_matrix(carrier[start:stop].to(guard.vertices)),
                        guard.bounds,
                    )
                    failed = (~valid).nonzero()
                    if len(failed):
                        first_failed_frames[row] = start + int(failed[0, 0])
                        mask[row] = False
                        break
        accepted = [
            previous and current
            for previous, current in zip(accepted, mask, strict=True)
        ]
        entries.append(
            {
                "carrier": guard.carrier_id,
                "object": guard.object_id,
                "initial_mask": guard.initial_mask.detach().cpu().tolist(),
                "accepted_mask": mask,
                "first_failed_frames": first_failed_frames,
            }
        )
    return {
        "schema_version": "gen_sim.cargo-envelope/v1",
        "scope": (
            "recorded_trajectory_and_terminal" if trajectory is not None else "terminal"
        ),
        "accepted_mask": accepted,
        "contents": entries,
    }
