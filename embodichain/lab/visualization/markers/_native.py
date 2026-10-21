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

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING
from uuid import uuid4
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import json

from ..protocol import MeshMarkerOverlay

if TYPE_CHECKING:
    from dexsim.scene import Scene
    from dexsim.scene.objects import SpawnedRigidBody


@dataclass
class _Entry:
    handle: SpawnedRigidBody
    snapshot: MeshMarkerOverlay


def _pose(marker: MeshMarkerOverlay) -> np.ndarray:
    w, x, y, z = marker.wxyz
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    pose[:3, 3] = marker.position
    return pose


def _transforms(markers: tuple[MeshMarkerOverlay, ...]) -> np.ndarray:
    """Pack one group as scaled world matrices for the Scene visual batch."""
    from scipy.spatial.transform import Rotation

    matrices = np.broadcast_to(np.eye(4, dtype=np.float32), (len(markers), 4, 4)).copy()
    if markers:
        quaternions = np.asarray([marker.wxyz for marker in markers])
        matrices[:, :3, :3] = Rotation.from_quat(
            quaternions[:, [1, 2, 3, 0]]
        ).as_matrix()
        matrices[:, :3, :3] *= np.asarray([marker.scale for marker in markers])[
            :, None, :
        ]
        matrices[:, :3, 3] = [marker.position for marker in markers]
    return matrices


class NativeMarkerRenderer:
    """Own Scene render-only mesh handles on the simulation thread.

    Removal invalidates the selected handles. Materials remain independent
    between instances, and this consumer retains no material wrappers
    between updates.

    Args:
        scene: Prepared DexSim Scene owning render-only mesh objects.
            Creating markers does not prepare or rebuild physics.
    """

    def __init__(self, scene: Scene) -> None:
        try:
            spawn = import_module("dexsim.spawn")
            required = (
                (getattr(spawn, "RenderDesc", None), "render_mode"),
                (getattr(spawn, "MaterialDesc", None), "unlit"),
                (getattr(spawn, "MaterialDesc", None), "alpha_mode"),
                (getattr(spawn, "MaterialDesc", None), "depth_write"),
            )
            if (
                not hasattr(spawn, "MeshObjectDesc")
                or any(not hasattr(owner, name) for owner, name in required)
                or not callable(getattr(scene, "add_mesh_object", None))
                or not callable(getattr(scene, "remove_mesh_object", None))
            ):
                raise ImportError("Missing Scene mesh-object or overlay properties")
        except ImportError as exc:
            raise RuntimeError(
                "Native marker groups require Scene mesh-object overlay support. "
                "Install a DexSim build with render-only overlay support or use Viser."
            ) from exc
        self._spawn = spawn
        self._scene = scene
        self._groups: dict[str, dict[str, _Entry]] = {}
        self._texture_directory = TemporaryDirectory(
            prefix="embodichain-marker-textures-"
        )
        self._texture_paths: dict[int, tuple[object, str]] = {}
        self._batches: dict[str, tuple[tuple, object]] = {}

    def _create(
        self, marker: MeshMarkerOverlay, template: _Entry | None = None
    ) -> SpawnedRigidBody:
        """Declare one render-only object through its lifetime-owning Scene."""
        name = f"__embodichain_marker_{uuid4().hex}"
        if template is not None and hasattr(self._scene, "clone_mesh_object"):
            return self._scene.clone_mesh_object(template.handle.path, name)
        scene_lighting = self._scene_lit(marker)
        parameters = json.loads(marker.material) if marker.material is not None else {}
        # Source descriptors preserve lobes, maps, samplers and color space.
        # The marker's display color uses sRGB regardless of the source space.
        parameters.pop("name", None)
        parameters.pop("base_color", None)
        if parameters.get("color_space") == "linear":
            color = np.asarray(marker.color)
            color = color.copy()
            color[:3] = np.where(
                color[:3] <= 0.04045,
                color[:3] / 12.92,
                ((color[:3] + 0.055) / 1.055) ** 2.4,
            )
        else:
            color = marker.color
        texture_options = {}
        if marker.texture is not None and not (
            scene_lighting and parameters.get("base_color_map")
        ):
            from PIL import Image

            key = id(marker.texture)
            if key not in self._texture_paths:
                path = str(
                    Path(self._texture_directory.name)
                    / f"{len(self._texture_paths)}.png"
                )
                Image.fromarray(marker.texture).save(path)
                self._texture_paths[key] = (marker.texture, path)
            texture_options["base_color_map"] = self._texture_paths[key][1]
            texture_options["opacity_map"] = self._texture_paths[key][1]
            texture_options["texture_settings"] = {
                "base_color_map": self._spawn.MaterialTextureDesc(
                    color_space="sRGB",
                    wrap_s=marker.texture_wrap[0],
                    wrap_t=marker.texture_wrap[1],
                ),
                "opacity_map": self._spawn.MaterialTextureDesc(
                    channel="a",
                    color_space="raw",
                    wrap_s=marker.texture_wrap[0],
                    wrap_t=marker.texture_wrap[1],
                ),
            }
        if not scene_lighting:
            parameters = {}
            color = marker.color
        parameters.update(texture_options)
        if "texture_settings" in parameters:
            parameters["texture_settings"] = {
                key: (
                    self._spawn.MaterialTextureDesc(**value)
                    if isinstance(value, dict)
                    else value
                )
                for key, value in parameters["texture_settings"].items()
            }
        parameters.update(unlit=marker.unlit)
        if not scene_lighting:
            parameters.update(alpha_mode="blend", depth_write=False, double_sided=True)
        material = self._spawn.MaterialDesc(
            name=f"{name}_material",
            base_color=tuple(color),
            **parameters,
        )
        render = self._spawn.RenderDesc.from_geometry(
            self._spawn.GeometryDesc.mesh(
                vertices=np.ascontiguousarray(marker.vertices, dtype=np.float32),
                triangles=np.ascontiguousarray(marker.faces, dtype=np.int32),
                normals=marker.normals,
                **(
                    {
                        "uv_coords": (
                            marker.material_uv_coords
                            if scene_lighting and marker.material_uv_coords is not None
                            else marker.uv_coords
                        )
                    }
                    if marker.uv_coords is not None
                    else {}
                ),
            ),
            render_mode="visual" if scene_lighting else "overlay",
            cast_shadow=False,
            pickable=False,
            material=material,
        )
        desc = self._spawn.MeshObjectDesc(
            name=name,
            renders=[render],
            physics=None,
            per_env=False,
        )
        return self._scene.add_mesh_object(desc, arena_name="default")

    def _discard(self, handle: SpawnedRigidBody) -> None:
        # Scene.close() invalidates the selected handles.
        # Never inspect a borrowed native actor to decide how to remove it.
        if handle.is_valid:
            self._scene.remove_mesh_object(handle.path)

    @staticmethod
    def _scene_lit(marker: MeshMarkerOverlay) -> bool:
        return marker.scene_lighting and not marker.unlit and marker.color[3] == 1.0

    @staticmethod
    def _same_geometry(left: MeshMarkerOverlay, right: MeshMarkerOverlay) -> bool:
        return (
            left.unlit == right.unlit
            and NativeMarkerRenderer._scene_lit(left)
            == NativeMarkerRenderer._scene_lit(right)
            and left.material == right.material
            and all(
                getattr(left, key) is getattr(right, key)
                or np.array_equal(getattr(left, key), getattr(right, key))
                for key in (
                    "vertices",
                    "faces",
                    "uv_coords",
                    "texture",
                    "texture_wrap",
                    "normals",
                    "material_uv_coords",
                )
            )
        )

    @staticmethod
    def _apply(
        handle: SpawnedRigidBody,
        marker: MeshMarkerOverlay,
        previous: MeshMarkerOverlay | None = None,
        *,
        transform: bool = True,
    ) -> None:
        if transform:
            handle.set_world_pose(_pose(marker))
            handle.set_scale(*(float(value) for value in marker.scale))
        if previous is None or previous.color != marker.color:
            handle.get_material().set_base_color(marker.color)
        if previous is None or previous.visible != marker.visible:
            handle.set_visible(marker.visible)

    def publish(self, name: str, markers: tuple[MeshMarkerOverlay, ...]) -> None:
        """Apply a complete group, staging replacements before releasing old meshes.

        Args:
            name: Stable group name.
            markers: Complete detached group snapshots in world coordinates.
        """
        previous = self._groups.get(name, {})
        pending: dict[str, _Entry] = {}
        created = []
        touched = []
        batch_enabled = hasattr(self._scene, "create_visual_object_batch")
        templates = {}
        for entry in previous.values():
            templates.setdefault(id(entry.snapshot.vertices), entry)
        try:
            for marker in markers:
                entry = previous.get(marker.overlay_id)
                if entry is not None and self._same_geometry(entry.snapshot, marker):
                    handle = entry.handle
                    touched.append(entry)
                else:
                    template = templates.get(id(marker.vertices))
                    if template is not None and not self._same_geometry(
                        template.snapshot, marker
                    ):
                        template = None
                    handle = self._create(marker, template)
                    created.append(handle)
                self._apply(
                    handle,
                    marker,
                    (
                        entry.snapshot
                        if entry is not None and entry.handle is handle
                        else None
                    ),
                    transform=not batch_enabled,
                )
                pending[marker.overlay_id] = _Entry(handle, marker)
                templates.setdefault(id(marker.vertices), pending[marker.overlay_id])
            if batch_enabled and pending:
                transforms = _transforms(markers)
                handles = tuple(entry.handle for entry in pending.values())
                cached = self._batches.get(name)
                if (
                    cached is None
                    or len(cached[0]) != len(handles)
                    or any(a is not b for a, b in zip(cached[0], handles))
                ):
                    cached = (handles, self._scene.create_visual_object_batch(handles))
                cached[1].set_world_transforms(transforms)
                self._batches[name] = cached
            elif not pending:
                self._batches.pop(name, None)
        except Exception:
            for handle in created:
                self._discard(handle)
            for entry in touched:
                self._apply(entry.handle, entry.snapshot)
            raise
        for key, entry in previous.items():
            if key not in pending or pending[key].handle is not entry.handle:
                self._discard(entry.handle)
        self._groups[name] = pending
        if created or len(previous) != len(pending):
            self._prune_textures()

    def _prune_textures(self) -> None:
        """Release host image copies and staging files after native upload."""
        active = {
            id(entry.snapshot.texture)
            for group in self._groups.values()
            for entry in group.values()
            if entry.snapshot.texture is not None
        }
        for key in self._texture_paths.keys() - active:
            _, path = self._texture_paths.pop(key)
            Path(path).unlink(missing_ok=True)

    def remove(self, name: str) -> None:
        """Release all native objects in one group.

        Args:
            name: Group to release; absent groups are ignored.
        """
        self._batches.pop(name, None)
        for entry in self._groups.pop(name, {}).values():
            self._discard(entry.handle)
        self._prune_textures()

    def close(self) -> None:
        """Release marker handles; also safe after their Scene has closed."""
        for name in tuple(self._groups):
            self.remove(name)
        self._texture_paths.clear()
        self._texture_directory.cleanup()
