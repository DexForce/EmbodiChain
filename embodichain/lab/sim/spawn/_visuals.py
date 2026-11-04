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

"""Owner-local visual composition through the existing Spawn/Scene boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

from ..cfg.visuals import GaussianSplatCfg, VisualElementCfg, _visual_elements


def _semantic_mesh_arrays(shape: Any) -> tuple[np.ndarray, np.ndarray]:
    """Decode explicit semantic geometry without allocating renderer resources."""
    from ..cfg import VisualObjectCfg
    from .descriptors import visual_desc_from_cfg
    from dexsim.spawn.adapters.common import generated_mesh
    from embodichain.data import get_data_path

    descriptor, _ = visual_desc_from_cfg(VisualObjectCfg(uid="semantic", shape=shape))
    geometry = descriptor.renders[0]
    if geometry.file_path:
        import trimesh

        mesh = trimesh.load(
            get_data_path(geometry.file_path), force="mesh", process=False
        )
        return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(
            mesh.faces, dtype=np.int32
        )
    vertices, faces, _, _ = generated_mesh(geometry)
    return vertices, faces


@dataclass(frozen=True)
class _Element:
    config: VisualElementCfg
    link: str | None
    name: str
    descriptor: Any
    gaussian: bool


class _VisualBinding:
    """Borrow visual handles; the owner remains the only pose authority."""

    def __init__(self, facade: Any, spawn: Any) -> None:
        from ..cfg import VisualObjectCfg
        from .descriptors import visual_desc_from_cfg
        from embodichain.data import get_data_path

        self.owner = facade
        self.spawn = spawn
        self.elements: list[_Element] = []
        self.revision = -1
        self.handles: list[tuple[Any, ...]] = []
        self.gaussian_batch = None
        self.gaussian_rows: list[tuple[int, int]] = []
        self.semantic: tuple[torch.Tensor, torch.Tensor] | None = None
        self.visibility = np.full(
            facade.num_instances, getattr(facade.cfg, "visible", True), dtype=bool
        )
        self.default_materials: list[list[list[Any]]] = []
        self.materials: list[list[list[Any]]] = []
        cfg = facade.cfg
        self.body_scale = np.asarray(
            getattr(cfg, "body_scale", (1, 1, 1)), dtype=np.float32
        )
        if (
            self.body_scale.shape != (3,)
            or not np.isfinite(self.body_scale).all()
            or np.any(self.body_scale <= 0)
        ):
            raise ValueError(
                "Visual owner body_scale must contain three positive finite values."
            )
        groups = getattr(cfg, "link_visuals", {})
        if hasattr(cfg, "visuals"):
            groups = {None: cfg.visuals} if cfg.visuals is not None else {}
        self.groups = {
            link: _visual_elements(elements) for link, elements in groups.items()
        }
        if any(elements is None for elements in self.groups.values()):
            raise ValueError("link_visuals values must be lists, not None.")
        for link, elements in self.groups.items():
            for element in elements:
                name = f"{facade.uid}__visual_{len(self.elements)}_{element.id}"
                representation = deepcopy(element.representation)
                gaussian = isinstance(representation, GaussianSplatCfg)
                if gaussian:
                    if not np.allclose(self.body_scale, self.body_scale[0]):
                        raise ValueError("Gaussian owner body_scale must be uniform.")
                    from dexsim.spawn import GaussianSplatDesc

                    path = get_data_path(representation.file_path)
                    lod = (
                        get_data_path(representation.lod_path)
                        if representation.lod_path
                        else None
                    )
                    transform = element.local_pose.copy()
                    transform[:3, :3] *= element.scale * self.body_scale[0]
                    descriptor = GaussianSplatDesc(
                        name,
                        path,
                        lod_path=lod,
                        lod_enabled=representation.lod_enabled,
                        lod_detail_scale=representation.lod_detail_scale,
                        pose=transform,
                        visible=element.visible,
                        per_env=True,
                        color_gain=representation.color_gain,
                        opacity_gain=representation.opacity_gain,
                        point_opacities=representation.point_opacities,
                        sh_coefficients=representation.sh_coefficients,
                    )
                else:
                    if hasattr(representation, "fpath"):
                        representation.fpath = get_data_path(representation.fpath)
                    descriptor, materials = visual_desc_from_cfg(
                        VisualObjectCfg(
                            uid=name,
                            shape=representation,
                            body_scale=(1.0, 1.0, 1.0),
                        )
                    )
                    transform = np.diag([*self.body_scale, 1.0]) @ element.local_pose
                    transform[:3, :3] *= element.scale
                    for render in descriptor.renders:
                        render.local_transform = transform @ render.local_transform
                    spawn.builder.materials.update(spawn._resolve_materials(materials))
                    spawn._bind_resource_materials(descriptor)
                self.elements.append(
                    _Element(element, link, name, descriptor, gaussian)
                )

        # Host-authoritative per-element state; no Gaussian point arrays here.
        # Copies keep the initial reset baseline independent of live randomization.
        count = facade.num_instances
        self.gaussian_state = {}
        for i, e in enumerate(self.elements):
            if e.gaussian:
                rep = e.config.representation
                self.gaussian_state[i] = {
                    "color_gain": np.tile(rep.color_gain, (count, 1)).astype(
                        np.float32
                    ),
                    "opacity_gain": np.full(count, rep.opacity_gain, dtype=np.float32),
                    "geometry_scales": np.full(
                        count, e.config.scale * self.body_scale[0], dtype=np.float32
                    ),
                    "visible": np.full(count, e.config.visible, dtype=bool),
                    "lod_enabled": np.full(count, rep.lod_enabled, dtype=bool),
                    "lod_detail_scale": np.full(
                        count, rep.lod_detail_scale, dtype=np.float32
                    ),
                }
        self.initial_gaussian_state = deepcopy(self.gaussian_state)

    @property
    def has_gaussians(self) -> bool:
        return any(element.gaussian for element in self.elements)

    def declare(self) -> None:
        handles = self.spawn.handles(self.owner.uid)
        if handles and any(link is not None for link in self.groups):
            available = set(handles[0].get_link_names())
            missing = set(self.groups) - available
            if missing:
                raise ValueError(
                    f"Unknown visual links on {self.owner.uid!r}: {sorted(missing)}"
                )
        existing = {desc.name for desc in self.spawn.builder.scene_desc.iter_entities()}
        conflicts = existing.intersection(element.name for element in self.elements)
        if conflicts:
            raise ValueError(
                f"Visual resource names already exist: {sorted(conflicts)}"
            )
        added = []
        try:
            for element in self.elements:
                method = (
                    self.spawn.builder.add_gaussian_splat
                    if element.gaussian
                    else self.spawn.builder.add_object
                )
                added.append(element)
                method(element.descriptor)
        except Exception:
            for element in reversed(added):
                method = (
                    self.spawn.builder.remove_gaussian_splat
                    if element.gaussian
                    else self.spawn.builder.remove_object
                )
                method(element.name)
            raise

    def remove(self) -> None:
        self.gaussian_batch = None
        self.handles.clear()
        for element in reversed(self.elements):
            method = (
                self.spawn.builder.remove_gaussian_splat
                if element.gaussian
                else self.spawn.builder.remove_object
            )
            method(element.name)

    def bind(self, scene: Any) -> None:
        from ..objects._mesh_utils import get_combined_vertices, get_combined_triangles
        from ..material import _capture_render_materials, _restore_render_materials

        if self.revision == scene.topology_revision:
            return
        owner = self.owner
        for link in self.groups:
            if link is not None and link not in owner.link_names:
                raise ValueError(f"Unknown visual link {link!r} on {owner.uid!r}.")
        # Preserve the original semantic surface once, before replacing appearance.
        if (
            None in self.groups
            and self.semantic is None
            and hasattr(owner.cfg, "semantic_shape")
        ):
            shape = owner.cfg.semantic_shape
            if shape is None:
                vertices, faces = [], []
                for handle in owner._entities:
                    body = handle.get_render_body()
                    if body is None or body.get_mesh_count() == 0:
                        vertices.append(np.empty((0, 3), dtype=np.float32))
                        faces.append(np.empty((0, 3), dtype=np.int32))
                    else:
                        vertices.append(get_combined_vertices(handle))
                        faces.append(get_combined_triangles(handle))
            else:
                v, f = _semantic_mesh_arrays(shape)
                vertices, faces = [v] * owner.num_instances, [f] * owner.num_instances
            self.semantic = (
                torch.as_tensor(
                    np.asarray(vertices), dtype=torch.float32, device=owner.device
                ),
                torch.as_tensor(
                    np.asarray(faces), dtype=torch.int32, device=owner.device
                ),
            )
        for handle in owner._entities:
            for link in self.groups:
                scene.clear_visual_meshes(handle.path, link_name=link)
        self.handles = [
            tuple(
                scene.handles[f"{arena}/{e.name}"] for arena in self.spawn.arena_names
            )
            for e in self.elements
        ]
        self.gaussian_rows = [
            (i, env)
            for i, e in enumerate(self.elements)
            if e.gaussian
            for env in range(owner.num_instances)
        ]
        self.gaussian_batch = (
            scene.create_gaussian_splat_batch(
                [self.handles[i][env] for i, env in self.gaussian_rows]
            )
            if self.gaussian_rows
            else None
        )
        if owner.num_instances > 1:
            for e, handles in zip(self.elements, self.handles):
                if e.gaussian and len({h.get_resource_id() for h in handles}) != 1:
                    raise RuntimeError(
                        "Gaussian replicas do not share a resource; upgrade DexSim Spawn clone support."
                    )
        captured = [
            (
                [_capture_render_materials(h.get_render_body()) for h in handles]
                if not e.gaussian
                else [[] for _ in handles]
            )
            for e, handles in zip(self.elements, self.handles)
        ]
        if not self.default_materials:
            self.default_materials = captured
            self.materials = [[list(m) for m in group] for group in captured]
        else:
            for e, handles, materials in zip(
                self.elements, self.handles, self.materials
            ):
                if not e.gaussian:
                    for handle, values in zip(handles, materials):
                        _restore_render_materials(handle.get_render_body(), values)
        # Source render bodies are now empty. Their material reset must not
        # resurrect removed source appearances; companion materials live above.
        if hasattr(owner, "_original_visual_material"):
            if None in self.groups:
                owner._original_visual_material = [[] for _ in owner._entities]
                owner._original_visual_material_inst = [None] * owner.num_instances
            else:
                for row in owner._original_visual_material:
                    if isinstance(row, dict):
                        for link in self.groups:
                            row[link] = []
        if hasattr(owner, "_default_materials"):
            owner._default_materials = [[] for _ in owner._entities]
            owner._materials = [[] for _ in owner._entities]
        self.revision = scene.topology_revision

    def sync(self) -> None:
        scene = self.spawn.builder.result
        self.bind(scene)
        roots = np.stack(
            [
                scene.get_arena(arena).get_root_node().get_world_pose()
                for arena in self.spawn.arena_names
            ]
        )
        poses = {}
        for link in self.groups:
            value = (
                self.owner.get_local_pose(to_matrix=True)
                if link is None
                else self.owner.get_link_pose(link, to_matrix=True)
            )
            poses[link] = value.detach().cpu().numpy()
        transformed = []
        for element in self.elements:
            local = element.config.local_pose.copy()
            local[:3, 3] *= self.body_scale
            transformed.append(
                roots @ poses[element.link] @ local
                if element.gaussian
                else roots @ poses[element.link]
            )
        if self.gaussian_batch is not None:
            self.gaussian_batch.apply(
                world_poses=np.asarray(
                    [transformed[i][env] for i, env in self.gaussian_rows],
                    dtype=np.float32,
                ),
                visible=np.asarray(
                    [
                        self.visibility[env] and self.gaussian_state[i]["visible"][env]
                        for i, env in self.gaussian_rows
                    ]
                ),
                **{
                    key: np.asarray(
                        [
                            self.gaussian_state[i][key][env]
                            for i, env in self.gaussian_rows
                        ]
                    )
                    for key in (
                        "geometry_scales",
                        "color_gain",
                        "opacity_gain",
                        "lod_enabled",
                    )
                },
            )
            # Flat assets do not admit detail updates; select hierarchy rows only.
            lod_rows = [
                (i, env)
                for i, env in self.gaussian_rows
                if self.elements[i].config.representation.lod_path
            ]
            if lod_rows:
                scene.create_gaussian_splat_batch(
                    [self.handles[i][env] for i, env in lod_rows]
                ).apply(
                    lod_detail_scale=np.asarray(
                        [
                            self.gaussian_state[i]["lod_detail_scale"][env]
                            for i, env in lod_rows
                        ]
                    )
                )
        for i, (e, handles) in enumerate(zip(self.elements, self.handles)):
            if not e.gaussian:
                for env, handle in enumerate(handles):
                    handle.set_world_pose(transformed[i][env])
                    handle.set_visible(bool(self.visibility[env] and e.config.visible))

    def set_gaussian_state(
        self,
        element_id: str,
        env_ids: Any = None,
        *,
        link_name: str | None = None,
        **values: Any,
    ) -> None:
        ids = self.indices(env_ids)
        matches = [
            i
            for i, e in enumerate(self.elements)
            if e.config.id == element_id and e.link == link_name and e.gaussian
        ]
        if not matches:
            raise ValueError(
                f"No Gaussian visual {element_id!r} on link {link_name!r}."
            )
        i = matches[0]
        state = self.gaussian_state[i]
        updates = {}
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value, dtype=np.float64)
            shape = (len(ids), 3) if key == "color_gain" else (len(ids),)
            try:
                array = np.broadcast_to(array, shape)
            except ValueError as error:
                raise ValueError(
                    f"{key} requires {shape} or a broadcastable value."
                ) from error
            if key in {"visible", "lod_enabled"}:
                if not np.isin(array, [0, 1]).all():
                    raise ValueError(f"{key} requires boolean values.")
            elif (
                not np.isfinite(array).all()
                or np.any(array < 0)
                or np.any(array > np.finfo(np.float32).max)
                or (
                    key in {"geometry_scales", "lod_detail_scale"}
                    and np.any(array.astype(np.float32) <= 0)
                )
            ):
                raise ValueError(f"Invalid Gaussian {key} values.")
            if (
                key in {"lod_enabled", "lod_detail_scale"}
                and not self.elements[i].config.representation.lod_path
            ):
                if key == "lod_detail_scale" or array.any():
                    raise ValueError(
                        "Gaussian LoD changes require a hierarchy sidecar."
                    )
            updates[key] = array.astype(state[key].dtype)
        # Validate the complete request before changing any authoritative state.
        for key, array in updates.items():
            state[key][ids] = array

    def set_visible(self, visible: bool, env_ids: Any = None) -> None:
        ids = self.indices(env_ids)
        if not isinstance(visible, bool):
            raise TypeError("visible must be a bool.")
        self.visibility[ids] = visible

    def indices(self, env_ids: Any) -> list[int]:
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        ids = (
            list(range(self.owner.num_instances)) if env_ids is None else list(env_ids)
        )
        if any(
            isinstance(i, bool)
            or not isinstance(i, (int, np.integer))
            or i < 0
            or i >= self.owner.num_instances
            for i in ids
        ):
            raise IndexError("Visual environment indices are out of range.")
        if len(set(ids)) != len(ids):
            raise ValueError("Visual environment indices must be unique.")
        return ids

    def reset(self, env_ids: Any = None) -> None:
        ids = self.indices(env_ids)
        self.visibility[ids] = getattr(self.owner.cfg, "visible", True)
        for i, state in self.gaussian_state.items():
            for key, value in state.items():
                value[ids] = self.initial_gaussian_state[i][key][ids]
        self.restore_materials(ids)

    def restore_materials(self, env_ids: Any = None, link_names: Any = None) -> None:
        from ..material import _restore_render_materials

        if self.revision < 0:
            return
        for i, element in enumerate(self.elements):
            if element.gaussian or (
                link_names is not None and element.link not in link_names
            ):
                continue
            for env in self.indices(env_ids):
                values = self.default_materials[i][env]
                _restore_render_materials(
                    self.handles[i][env].get_render_body(), values
                )
                self.materials[i][env] = list(values)

    def apply_material_inst(
        self,
        env: int,
        material: Any,
        mesh_id: int | None = None,
        link_name: str | None = None,
    ) -> None:
        from ..material import _set_render_material, _capture_render_materials

        self.indices([env])
        segments = [
            (i, segment)
            for i, e in enumerate(self.elements)
            if not e.gaussian and e.link == link_name
            for segment in range(
                self.handles[i][env].get_render_body().get_mesh_count()
            )
        ]
        if not segments:
            raise TypeError(
                "Gaussian captured SH does not support mesh material assignment."
            )
        if mesh_id is not None and not 0 <= mesh_id < len(segments):
            raise IndexError("Visual mesh segment index is out of range.")
        for i, segment in (segments if mesh_id is None else [segments[mesh_id]]):
            body = self.handles[i][env].get_render_body()
            _set_render_material(body, segment, material)
            self.materials[i][env] = _capture_render_materials(body)

    def reuse_materials(self, env: int, link_name: str | None = None) -> list[Any]:
        from ..material import ReuseSegmentState, VisualMaterialInst

        originals = [
            material
            for i, e in enumerate(self.elements)
            if not e.gaussian and e.link == link_name
            for material in self.default_materials[i][env]
        ]
        if not originals:
            return []
        working = None
        segments = []
        for mesh_id, original in enumerate(originals):
            if original is None or original.get_template() is None:
                raise ValueError(
                    "Visual mesh segment has no reusable material template."
                )
            if working is None:
                name = f"{self.owner.uid}_reuse_{env}_{link_name or 'root'}"
                template = original.get_template()
                working = VisualMaterialInst(
                    name, template, mat_inst=template.create_inst(name)
                )
            segments.append(
                ReuseSegmentState(
                    mesh_id=mesh_id, original_inst=original, working_inst=working
                )
            )
        return segments

    def material_instances(
        self, env_ids: Any = None, link_name: str | None = None
    ) -> list[Any]:
        from ..material import _wrap_first_render_material

        meshes = [
            i
            for i, e in enumerate(self.elements)
            if not e.gaussian and e.link == link_name
        ]
        return [
            (
                _wrap_first_render_material(self.materials[meshes[0]][env])
                if meshes
                else None
            )
            for env in self.indices(env_ids)
        ]

    def mesh(
        self, env_ids: Any = None, scale: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ..objects._mesh_utils import get_combined_vertices, get_combined_triangles

        ids = self.indices(env_ids)
        if not ids:
            vertices, triangles = self.mesh([0], scale=scale)
            return vertices[:0], triangles[:0]
        vertices, triangles = [], []
        for env in ids:
            row_v, row_f, offset = [], [], 0
            for e, handles in zip(self.elements, self.handles):
                if e.gaussian:
                    continue
                v = get_combined_vertices(handles[env])
                if not scale:
                    v = v / self.body_scale
                f = get_combined_triangles(handles[env])
                row_v.append(v)
                row_f.append(f + offset)
                offset += len(v)
            if not row_v:
                raise ValueError(
                    "This visual has no triangle mesh; request semantic geometry explicitly."
                )
            vertices.append(np.concatenate(row_v))
            triangles.append(np.concatenate(row_f))
        return (
            torch.as_tensor(
                np.asarray(vertices), dtype=torch.float32, device=self.owner.device
            ),
            torch.as_tensor(
                np.asarray(triangles), dtype=torch.int32, device=self.owner.device
            ),
        )

    def set_material(
        self,
        material: Any,
        env_ids: Any = None,
        *,
        link_names: Any = None,
        update_default: bool = False,
        shared: bool = False,
    ) -> None:
        from ..material import _capture_render_materials

        mesh_indices = [
            i
            for i, e in enumerate(self.elements)
            if not e.gaussian and (link_names is None or e.link in link_names)
        ]
        if not mesh_indices:
            raise TypeError(
                "Gaussian captured SH does not support mesh material assignment."
            )
        ids = self.indices(env_ids)
        if shared and len(ids) != self.owner.num_instances:
            raise ValueError("Shared materials require all environments.")
        for i in mesh_indices:
            common = (
                material.create_instance(f"{material.uid}_{self.elements[i].name}")
                if shared
                else None
            )
            for env in ids:
                instance = common or material.create_instance(
                    f"{material.uid}_{self.elements[i].name}_{env}"
                )
                handle = self.handles[i][env]
                handle.set_material(instance.mat)
                self.materials[i][env] = _capture_render_materials(
                    handle.get_render_body()
                )
                if update_default:
                    self.default_materials[i][env] = list(self.materials[i][env])

    def describe(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "id": e.config.id,
                "link_name": e.link,
                "representation": "gaussian_splat" if e.gaussian else "mesh",
                "file_path": getattr(
                    e.config.representation,
                    "file_path",
                    getattr(e.config.representation, "fpath", None),
                ),
                "local_pose": e.config.local_pose.copy(),
                "scale": e.config.scale,
                "instance_state": (
                    {key: value.copy() for key, value in self.gaussian_state[i].items()}
                    if e.gaussian
                    else {}
                ),
                "resource_ids": (
                    tuple(h.get_resource_id() for h in handles) if e.gaussian else ()
                ),
            }
            for i, (e, handles) in enumerate(
                zip(self.elements, self.handles or [()] * len(self.elements))
            )
        )
