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

"""Native geometry adapters for explicit control release observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import math
import numpy as np
import torch

from embodichain.compute.geometry.aabb import aabb_separation
from embodichain.lab.sim.utility.urdf_collision import load_urdf_collision_inputs
from .press_inputs import PressBinding
from .revolute_geometry import TwistGeometry

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot, Articulation

__all__ = ["NativeControlGeometry"]


class NativeControlGeometry:
    """Read-only B=1 release geometry with explicit finger and motion bindings.

    Press uses the original native link-mesh inputs. Twist uses complete authored
    finger/articulation collision vertices. Neither route certifies native
    cooking or continuous collision freedom, and the two sources are not mixed.
    """

    def __init__(
        self,
        robot: Robot,
        articulation: Articulation,
        *,
        finger_link_names: Sequence[str],
        motion_part: str,
        press_binding: PressBinding | None = None,
        twist_geometry: TwistGeometry | None = None,
    ) -> None:
        if robot.num_instances != 1 or articulation.num_instances != 1:
            raise ValueError("Control geometry supports native B=1 only.")
        if (press_binding is None) == (twist_geometry is None):
            raise ValueError(
                "Control geometry requires one explicit physical source kind."
            )
        if isinstance(finger_link_names, str):
            raise ValueError("Finger geometry requires an explicit link sequence.")
        names = tuple(finger_link_names)
        if (
            not names
            or len(set(names)) != len(names)
            or not set(names) <= set(robot.link_names)
        ):
            raise ValueError(
                "Control geometry requires an explicit complete finger roster."
            )
        self.robot, self.articulation = robot, articulation
        self._names, self._part, self._press, self._twist = (
            names,
            motion_part,
            press_binding,
            twist_geometry,
        )
        self._finger_vertices = None
        self._finger_inputs = None
        if twist_geometry is not None:
            meshes = load_urdf_collision_inputs(
                robot.cfg.fpath,
                names,
                body_scale=robot.cfg.body_scale,
                vertices_only=True,
            )
            self._finger_inputs = meshes
            self._finger_vertices = {
                name: np.concatenate([mesh.vertices for mesh in values])
                for name, values in meshes.items()
            }

    @property
    def finger_link_names(self) -> tuple[str, ...]:
        """Return the exact declared complete finger collision roster."""
        return self._names

    @property
    def motion_part(self) -> str:
        """Return the explicitly bound control part defining the tool frame."""
        return self._part

    def source_evidence(self, mass_evidence: dict[str, object]) -> dict[str, object]:
        """Describe authored Twist inputs without certifying native cooked shapes."""
        if self._twist is None:
            return {
                "frame": "native link mesh",
                "units": "m",
                "backend_cooked_shapes_verified": False,
            }

        def record(mesh):
            return {
                "path": mesh.path,
                "owner": mesh.owner_path,
                "approximation": mesh.approximation,
                "bounds": [
                    mesh.vertices.min(0).tolist(),
                    mesh.vertices.max(0).tolist(),
                ],
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.faces),
            }

        geometry = self._twist
        return {
            "schema": "scaled_owning_link_collision_inputs/v1",
            "mass_properties": mass_evidence,
            "frame": "scaled owning rigid-link local",
            "units": "m",
            "asset_scale_applied": geometry.scale,
            "native_pose_scaled_again": False,
            "source_sha256": geometry.source_sha256,
            "grip": record(geometry.grip),
            "target_collisions": [record(mesh) for mesh in geometry.target_collisions],
            "parent_collisions": [record(mesh) for mesh in geometry.parent_collisions],
            "articulation_collisions": [
                record(mesh) for mesh in geometry.articulation_collisions
            ],
            "fingers": {
                name: [record(mesh) for mesh in meshes]
                for name, meshes in self._finger_inputs.items()
            },
            "backend_cooked_shapes_verified": False,
            "continuous_collision_certificate": False,
        }

    def _fingers(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        clouds, bounds = [], []
        for name in self._names:
            pose = self.robot.get_link_pose(name, to_matrix=True)[0]
            vertices = (
                self.robot.get_link_vert_face(name)[0].to(pose)
                if self._finger_vertices is None
                else pose.new_tensor(self._finger_vertices[name])
            )
            world = vertices @ pose[:3, :3].T + pose[:3, 3]
            if not world.numel() or not torch.isfinite(world).all():
                raise ValueError(
                    "Control release requires finite complete finger geometry."
                )
            clouds.append(world)
            bounds.append(torch.stack((world.amin(0), world.amax(0))))
        return clouds, bounds

    def park_clearance(self) -> float:
        """Measure source world-mesh AABB separation without changing native state."""
        return self._park_from_bounds(self._fingers()[1])

    def _park_from_bounds(self, finger_bounds: list[torch.Tensor]) -> float:
        distances = []
        if self._twist is None:

            def native_inputs():
                for name in self.articulation.link_names:
                    vertices, _ = self.articulation.get_link_vert_face(name)
                    if vertices.numel():
                        yield name, vertices

            inputs = native_inputs()
        else:
            inputs = []
            for mesh in self._twist.articulation_collisions:
                name = mesh.owner_path.rsplit("/", 1)[-1]
                if name not in self.articulation.link_names:
                    raise ValueError(
                        "Control collider ownership differs from native links."
                    )
                inputs.append((name, mesh.vertices.copy()))
        for name, vertices in inputs:
            pose = self.articulation.get_link_pose(name, to_matrix=True)[0]
            points = (
                vertices.to(pose)
                if isinstance(vertices, torch.Tensor)
                else pose.new_tensor(vertices)
            )
            world = points @ pose[:3, :3].T + pose[:3, 3]
            if not torch.isfinite(world).all():
                raise ValueError("Control park requires finite target geometry.")
            bounds = torch.stack((world.amin(0), world.amax(0)))
            distances.extend(
                float(aabb_separation(bounds, finger)) for finger in finger_bounds
            )
        if not distances:
            raise ValueError(
                "Control park requires complete finger and target geometry."
            )
        return min(distances)

    def press_clearances(self, *, terminal: bool) -> dict[str, float | None]:
        """Read axial finger/TCP clearance, plus the distinct terminal park metric."""
        if self._press is None:
            raise ValueError("Axial press clearance requires a Press binding.")
        binding = self._press
        pose = self.articulation.get_link_pose(binding.link, to_matrix=True)[0]
        point = pose[:3, :3] @ pose.new_tensor(binding.press_position) + pose[:3, 3]
        inward = pose[:3, :3] @ pose.new_tensor(binding.axis)
        tcp = self.robot.compute_fk(
            qpos=self.robot.get_qpos(name=self._part), name=self._part, to_matrix=True
        )[0, :3, 3]
        clouds, bounds = self._fingers()
        clearance = min(float(((point - world) @ inward).min()) for world in clouds)
        return {
            "retreat_clearance": clearance,
            "tcp_clearance": float(torch.dot(point - tcp, inward)),
            "terminal_clearance": self._park_from_bounds(bounds) if terminal else None,
        }
