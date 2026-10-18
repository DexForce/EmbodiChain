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

"""Invocation-owned native hand/rigid-scene clearance before source contact."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import trimesh

__all__: list[str] = []


class NativeRigidRestScene:
    """Read the original cooked shapes and pair rest offsets, never contact halos."""

    def __init__(
        self, parts: tuple[Any, ...], targets: list[tuple[str, float, Any]]
    ) -> None:
        if not parts or any(
            not math.isfinite(value) or value < 0
            for value in [
                *(part.rest_offset for part in parts),
                *(rest for _, rest, _ in targets),
            ]
        ):
            raise ValueError(
                "Native scene requires complete hand geometry and finite rest offsets."
            )
        self.parts, self.targets = parts, tuple(targets)
        self.managers = []
        for part in parts:
            manager = trimesh.collision.CollisionManager()
            for index, mesh in enumerate(part.meshes):
                manager.add_object(str(index), mesh)
            self.managers.append(manager)
        self.cache: dict[tuple[bytes, int], dict[str, Any]] = {}

    def inspect(self, tcp_world: np.ndarray, count: int) -> dict[str, Any]:
        """Qualify exactly the source screen's nominal precontact grid prefix."""
        pose = np.asarray(tcp_world, dtype=np.float64)
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], (0, 0, 0, 1))
            or type(count) is not int
            or count < 1
            or any(count > len(part.tcp_poses) for part in self.parts)
        ):
            raise ValueError(
                "Native scene requires a finite TCP and its bounded precontact grid."
            )
        key = (pose.tobytes(), count)
        if key in self.cache:
            return dict(self.cache[key])
        minimum = math.inf
        result: dict[str, Any] = {
            "valid": True,
            "sample_count": count,
            "scope": "native_rigid_pair_rest_precontact_not_physical_grip",
        }
        for step in range(count):
            for part, manager in zip(self.parts, self.managers, strict=True):
                for index in range(len(part.meshes)):
                    manager.set_transform(str(index), pose @ part.tcp_poses[step])
                for uid, rest, target in self.targets:
                    distance = float(manager.min_distance_other(target))
                    if not math.isfinite(distance):
                        raise ValueError(
                            "Native hand/scene mesh distance is unavailable."
                        )
                    pair_rest = part.rest_offset + rest
                    margin = distance - pair_rest
                    minimum = min(minimum, margin)
                    if margin <= 0:
                        result.update(
                            valid=False,
                            kind="native_rigid_pair_rest",
                            link=part.name,
                            scene_uid=uid,
                            sample_index=step,
                            mesh_distance_m=distance,
                            native_pair_rest_offset_m=pair_rest,
                            minimum_rest_clearance_m=minimum,
                        )
                        self.cache[key] = result
                        return dict(result)
        result["minimum_rest_clearance_m"] = minimum if math.isfinite(minimum) else None
        self.cache[key] = result
        return dict(result)


def build_native_rigid_scene(simulation: Any, screen: Any) -> NativeRigidRestScene:
    """Capture one source-owned arena without stepping or mutating the world."""
    from .pick_contact import _native_meshes

    source_uid = screen.identity["simulation_uid"]
    if source_uid not in simulation.get_rigid_object_uid_list():
        raise ValueError("Native scene requires the exact bound rigid source.")
    scene = simulation.spawn_result
    handles = [scene.get(path) for path in scene.get_paths()]
    source = [
        handle
        for handle in handles
        if handle.get_name() == source_uid and callable(getattr(handle, "native", None))
    ]
    if len(source) != 1:
        raise ValueError("Native scene source arena is unavailable or ambiguous.")
    targets = []
    for uid in simulation.get_rigid_object_uid_list():
        if uid == source_uid:
            continue
        matches = [
            handle
            for handle in handles
            if handle.get_name() == uid
            and handle.arena_name == source[0].arena_name
            and callable(getattr(handle, "native", None))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Native scene rigid identity is unavailable or ambiguous: {uid}"
            )
        body = matches[0].native().get_physical_body()
        meshes = _native_meshes(body)
        if not meshes:
            continue
        rest = float(body.get_rest_offset())
        pose = (
            simulation.get_rigid_object(uid)
            .get_local_pose(to_matrix=True)[0]
            .detach()
            .cpu()
            .double()
            .numpy()
        )
        if (
            not math.isfinite(rest)
            or rest < 0
            or pose.shape != (4, 4)
            or not np.isfinite(pose).all()
        ):
            raise ValueError("Native scene rest offset or rigid pose is unavailable.")
        manager = trimesh.collision.CollisionManager()
        for index, mesh in enumerate(meshes):
            manager.add_object(f"{uid}:{index}", mesh, transform=pose)
        targets.append((uid, rest, manager))
    return NativeRigidRestScene(screen.parts, targets)
