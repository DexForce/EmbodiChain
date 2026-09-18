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

import copy
from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from .._utils import to_numpy_array
from ..protocol import MeshMarkerOverlay
from .cfg import MarkerGroupCfg
from ._geometry import geometry_parts

__all__ = ["MarkerGroup"]


class MarkerGroup:
    """Validated marker state shared by native and browser visualization.

    Args:
        cfg: Name, prototypes and coordinate-system selection.
        origin: World translation for a standalone single environment.
        origins: World translations shaped (num_envs, 3).
        num_envs: Number of environment batches; world scope always uses one.
        pose_resolver: Owner callback returning environment-local parent poses
            shaped (E, 7), ordered xyz then xyzw, without preparing physics.
        on_change: Owner callback for publishing a complete state after validation.
        on_remove: Owner callback for releasing render resources.
    """

    def __init__(
        self,
        cfg: MarkerGroupCfg,
        *,
        origin: object = (0, 0, 0),
        origins: object | None = None,
        num_envs: int = 1,
        pose_resolver: Callable[[str, str | None, list[int]], object] | None = None,
        on_change: Callable[[MarkerGroup], None] | None = None,
        on_remove: Callable[[MarkerGroup], None] | None = None,
    ) -> None:
        cfg.validate()
        self._cfg = copy.deepcopy(cfg)
        if not isinstance(cfg.name, str) or not cfg.name:
            raise ValueError("Marker group name must be a non-empty string.")
        if cfg.scope not in {"env", "world"}:
            raise ValueError("scope must be 'env' or 'world'.")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer.")
        if not isinstance(cfg.prototypes, dict) or not cfg.prototypes:
            raise ValueError("Marker prototypes must be a non-empty dictionary.")
        self._num_envs = num_envs if cfg.scope == "env" else 1
        if cfg.scope == "world":
            self._origins = np.zeros((1, 3), dtype=np.float64)
        else:
            self._origins = to_numpy_array(
                np.tile(origin, (num_envs, 1)) if origins is None else origins,
                np.float64,
            )
        if (
            self._origins.shape != (self.num_envs, 3)
            or not np.isfinite(self._origins).all()
        ):
            raise ValueError("origins must be finite with shape (num_envs, 3).")
        self._parts = []
        for key, prototype in self._cfg.prototypes.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Prototype names must be non-empty strings.")
            prototype.validate()
            if prototype.shape not in {
                "box",
                "sphere",
                "cylinder",
                "capsule",
                "cone",
                "arrow",
                "frame",
                "mesh",
            }:
                raise ValueError(f"Unsupported marker shape: {prototype.shape!r}")
            if prototype.shape != "mesh" and (
                prototype.vertices is not None or prototype.faces is not None
            ):
                raise ValueError(
                    "vertices and faces are only valid for mesh prototypes."
                )
            parts = geometry_parts(prototype)
            # Reuse the transport boundary's geometry/material validation.
            for vertices, faces, _ in parts:
                MeshMarkerOverlay(
                    overlay_id="validation",
                    vertices=vertices,
                    faces=faces,
                    position=np.zeros(3),
                    wxyz=np.array([1, 0, 0, 0]),
                    scale=prototype.scale,
                    color=prototype.color,
                )
            self._parts.append(parts)
        self._on_change = on_change
        self._on_remove = on_remove
        self._visible = np.ones(self.num_envs, dtype=bool)
        self._pose_resolver = pose_resolver
        self._attachments: dict[int, tuple[str, str | None]] = {}
        self._parent_poses: dict[int, np.ndarray] = {}
        self._removed = False
        self._states = [self._defaults(0) for _ in range(self.num_envs)]

    @property
    def name(self) -> str:
        """Stable owner registry key."""
        return self._cfg.name

    @property
    def count(self) -> int:
        """Total logical instances across all environments, including hidden ones."""
        return sum(self.counts)

    @property
    def counts(self) -> tuple[int, ...]:
        """Logical instance counts in environment order (one for world scope)."""
        return tuple(len(state["translations"]) for state in self._states)

    @property
    def scope(self) -> str:
        """Coordinate scope, either ``env`` or ``world``."""
        return self._cfg.scope

    @property
    def num_envs(self) -> int:
        """Number of independently updateable batches."""
        return self._num_envs

    def _env_ids(self, env_ids: object | None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        # Do not coerce strings, floats, or booleans into integer selections.
        if np.isscalar(env_ids) or getattr(env_ids, "ndim", None) == 0:
            raise ValueError("env_ids must be a one-dimensional integer selection.")
        if isinstance(env_ids, (list, tuple)) and any(
            isinstance(value, (bool, np.bool_)) for value in env_ids
        ):
            raise ValueError("env_ids cannot contain booleans.")
        values = to_numpy_array(env_ids, None)
        if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
            raise ValueError("env_ids must be a one-dimensional integer selection.")
        ids = values.astype(np.int64).tolist()
        if len(set(ids)) != len(ids) or any(i < 0 or i >= self.num_envs for i in ids):
            raise ValueError("env_ids must be unique and in bounds.")
        return ids

    def _defaults(self, count: int) -> dict[str, np.ndarray | None]:
        return dict(
            translations=np.zeros((count, 3), dtype=np.float32),
            orientations_xyzw=np.tile([0.0, 0.0, 0.0, 1.0], (count, 1)),
            scales=np.ones((count, 3), dtype=np.float32),
            prototype_indices=np.zeros(count, dtype=np.int64),
            colors=None,
            visible=np.ones(count, dtype=bool),
        )

    def _check_live(self) -> None:
        if self._removed:
            raise RuntimeError(f"Marker group {self.name!r} has been removed.")

    def update(
        self,
        *,
        translations: object | None = None,
        orientations_xyzw: object | None = None,
        scales: object | None = None,
        prototype_indices: object | None = None,
        colors: object | None = None,
        visible: object | None = None,
        env_ids: object | None = None,
    ) -> None:
        """Publish validated arrays without preparing or stepping physics.

        Arrays use (E, M, 3) translations/scales, (E, M, 4) xyzw rotations/RGBA
        colors, and (E, M) prototype indices/visibility. E is the selected
        environment count. Single-environment selections also accept (M, ...).
        Quaternions are normalized; scales must be positive and RGBA in [0, 1].
        Translations set each selected environment's count. A changed count
        resets omitted fields in that environment; unchanged counts retain them.
        Attached translations and orientations are offsets in the parent frame.
        Empty selections are validated no-ops; arrays are never broadcast.

        Args:
            translations: Finite environment-local positions, or attached offsets.
            orientations_xyzw: Nonzero quaternions in xyzw order.
            scales: Positive instance scales, multiplied by prototype scales.
            prototype_indices: Integer indices into configured prototypes.
            colors: Optional RGBA overrides in [0, 1].
            visible: Per-instance boolean visibility.
            env_ids: Unique environment indices, or None for every environment.
        """
        self._check_live()
        ids = self._env_ids(env_ids)
        values = dict(
            translations=translations,
            orientations_xyzw=orientations_xyzw,
            scales=scales,
            prototype_indices=prototype_indices,
            colors=colors,
            visible=visible,
        )
        if all(value is None for value in values.values()):
            raise ValueError("At least one marker update array is required.")
        batches = {}
        for key, value in values.items():
            if value is None:
                continue
            array = to_numpy_array(value, np.float64)
            width = {
                "translations": 3,
                "orientations_xyzw": 4,
                "scales": 3,
                "colors": 4,
            }.get(key)
            ndim = 3 if width else 2
            if len(ids) == 1 and array.ndim == ndim - 1:
                array = array[None]
            if (
                array.ndim != ndim
                or array.shape[0] != len(ids)
                or (width and array.shape[-1] != width)
            ):
                raise ValueError(
                    f"{key} must have shape (E, M{', ' + str(width) if width else ''})."
                )
            batches[key] = array
        states = self._states.copy()
        for row, env_id in enumerate(ids):
            states[env_id] = self._updated_state(
                env_id, {key: value[row] for key, value in batches.items()}
            )
        if ids:
            self._commit(states=states, parent_poses=self._resolve_attachments())

    def _updated_state(
        self, env_id: int, arrays: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray | None]:
        count = (
            len(arrays["translations"])
            if "translations" in arrays
            else self.counts[env_id]
        )
        state = (
            self._defaults(count)
            if count != self.counts[env_id]
            else self._states[env_id].copy()
        )
        for key, array in arrays.items():
            width = {
                "translations": 3,
                "orientations_xyzw": 4,
                "scales": 3,
                "colors": 4,
            }.get(key)
            shape = (count, width) if width else (count,)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{key} must be finite with shape {shape}.")
            if np.any(np.abs(array) > np.finfo(np.float32).max):
                raise ValueError(f"{key} must be representable as float32.")
            if key == "scales" and np.any(array <= 0):
                raise ValueError("scales must be positive.")
            if key == "colors" and (np.any(array < 0) or np.any(array > 1)):
                raise ValueError("colors must be RGBA in [0, 1].")
            if key == "prototype_indices":
                if (
                    np.any(array != np.floor(array))
                    or np.any(array < 0)
                    or np.any(array >= len(self._parts))
                ):
                    raise ValueError(
                        "prototype_indices must reference configured prototypes."
                    )
                array = array.astype(np.int64)
            elif key == "visible":
                if np.any((array != 0) & (array != 1)):
                    raise ValueError("visible must contain booleans.")
                array = array.astype(bool)
            elif key == "orientations_xyzw":
                norms = np.linalg.norm(array, axis=1, keepdims=True)
                if np.any(norms <= 1e-12):
                    raise ValueError(
                        "orientations_xyzw must contain nonzero quaternions."
                    )
                array = array / norms
            state[key] = array
        prototype_scales = np.asarray(
            [prototype.scale for prototype in self._cfg.prototypes.values()],
            dtype=np.float64,
        )
        world_positions = state["translations"] + self._origins[env_id]
        combined_scales = state["scales"] * prototype_scales[state["prototype_indices"]]
        if np.any(np.abs(world_positions) > np.finfo(np.float32).max):
            raise ValueError("World positions must be representable as float32.")
        if np.any(combined_scales > np.finfo(np.float32).max) or np.any(
            combined_scales.astype(np.float32) <= 0
        ):
            raise ValueError("Combined scales must be positive float32 values.")
        return state

    def _commit(self, **changes: object) -> None:
        previous = {key: getattr(self, "_" + key) for key in changes}
        changed_envs = set()
        if "states" in changes:
            changed_envs.update(
                env_id
                for env_id, (old, new) in enumerate(
                    zip(self._states, changes["states"])
                )
                if old is not new
            )
        if "parent_poses" in changes:
            old_poses, new_poses = self._parent_poses, changes["parent_poses"]
            changed_envs.update(
                env_id
                for env_id in old_poses.keys() | new_poses.keys()
                if env_id not in old_poses
                or env_id not in new_poses
                or not np.array_equal(old_poses[env_id], new_poses[env_id])
            )
        for key, value in changes.items():
            setattr(self, "_" + key, value)
        try:
            # Geometry and styles are already validated. Check only changed
            # transforms here; render snapshots belong to the actual consumer.
            for env_id in changed_envs:
                positions, _ = self._poses(env_id)
                if not np.isfinite(positions).all() or np.any(
                    np.abs(positions) > np.finfo(np.float32).max
                ):
                    raise ValueError("World positions must be finite float32 values.")
            if self._on_change is not None:
                self._on_change(self)
        except Exception:
            for key, value in previous.items():
                setattr(self, "_" + key, value)
            raise

    def _poses(self, env_id: int) -> tuple[np.ndarray, np.ndarray]:
        state = self._states[env_id]
        xyz = state["translations"]
        xyzw = state["orientations_xyzw"]
        if env_id in self._parent_poses and len(xyz):
            parent = self._parent_poses[env_id]
            rotation = Rotation.from_quat(parent[3:])
            xyz = rotation.apply(xyz) + parent[:3]
            xyzw = (rotation * Rotation.from_quat(xyzw)).as_quat()
        return xyz + self._origins[env_id], xyzw

    def snapshot(self) -> tuple[MeshMarkerOverlay, ...]:
        """Return detached world-space snapshots without querying parent assets.

        Returns:
            Mesh parts with stable environment/instance IDs and wxyz rotations.
        """
        if self._removed:
            return ()
        result = []
        prototypes = tuple(self._cfg.prototypes.values())
        for env_id, state in enumerate(self._states):
            positions, orientations = self._poses(env_id)
            if not np.isfinite(positions).all() or np.any(
                np.abs(positions) > np.finfo(np.float32).max
            ):
                raise ValueError("World positions must be finite float32 values.")
            for index, prototype_index in enumerate(state["prototype_indices"]):
                prototype = prototypes[prototype_index]
                rgba = (
                    prototype.color
                    if state["colors"] is None
                    else state["colors"][index]
                )
                for part_index, (vertices, faces, rgb) in enumerate(
                    self._parts[prototype_index]
                ):
                    color = (
                        tuple(float(v) for v in rgba)
                        if rgb is None
                        else (*rgb, float(rgba[3]))
                    )
                    result.append(
                        MeshMarkerOverlay(
                            overlay_id=f"group:{self.name}:{env_id}:{index}:{part_index}",
                            vertices=vertices,
                            faces=faces,
                            position=positions[index],
                            wxyz=orientations[index][[3, 0, 1, 2]],
                            scale=state["scales"][index] * prototype.scale,
                            color=color,
                            visible=bool(
                                self._visible[env_id] and state["visible"][index]
                            ),
                            env_id=env_id if self.scope == "env" else None,
                        )
                    )
        return tuple(result)

    def set_visibility(self, visible: bool, *, env_ids: object | None = None) -> None:
        """Show or hide selected environments, preserving instance visibility.

        Args:
            visible: Whether selected environment batches should be displayed.
            env_ids: Unique environment indices, or None for every environment.
        """
        self._check_live()
        ids = self._env_ids(env_ids)
        if not isinstance(visible, (bool, np.bool_)):
            raise ValueError("visible must be a boolean.")
        values = self._visible.copy()
        values[ids] = visible
        if ids:
            self._commit(visible=values)

    def clear(self, *, env_ids: object | None = None) -> None:
        """Remove selected instances, retaining prototypes and attachment state.

        Args:
            env_ids: Unique environment indices, or None for every environment.
        """
        self._check_live()
        ids = self._env_ids(env_ids)
        states = self._states.copy()
        for env_id in ids:
            states[env_id] = self._defaults(0)
        if ids:
            self._commit(states=states)

    def attach(
        self,
        parent: str,
        *,
        link_name: str | None = None,
        env_ids: object | None = None,
    ) -> None:
        """Interpret existing selected poses as offsets from a live registered asset.

        The owner resolves prepared roots or articulation links without preparing
        physics. World scope is unsupported without a source environment selector.

        Args:
            parent: Registered rigid-object, robot or articulation UID.
            link_name: Articulation/robot link name, or None for the asset root.
            env_ids: Unique environment indices, or None for every environment.
        """
        self._check_live()
        ids = self._env_ids(env_ids)
        if self.scope == "world":
            raise ValueError("Attachment is unsupported for world-scope groups.")
        if (
            not isinstance(parent, str)
            or not parent
            or (
                link_name is not None
                and (not isinstance(link_name, str) or not link_name)
            )
        ):
            raise ValueError("parent and optional link_name must be non-empty strings.")
        attachments = self._attachments.copy()
        for env_id in ids:
            attachments[env_id] = (parent, link_name)
        if ids:
            self._commit(
                attachments=attachments,
                parent_poses=self._resolve_attachments(attachments),
            )

    def _resolve_attachments(
        self, attachments: dict[int, tuple[str, str | None]] | None = None
    ) -> dict[int, np.ndarray]:
        attachments = self._attachments if attachments is None else attachments
        if attachments and self._pose_resolver is None:
            raise RuntimeError("Attachment requires an owner-supplied pose resolver.")
        result = {}
        for target in dict.fromkeys(attachments.values()):
            ids = [env_id for env_id, value in attachments.items() if value == target]
            poses = to_numpy_array(self._pose_resolver(*target, ids), np.float64)
            if poses.shape != (len(ids), 7) or not np.isfinite(poses).all():
                raise ValueError("Parent poses must be finite with shape (E, 7).")
            norms = np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
            if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
                raise ValueError(
                    "Parent orientations must be nonzero finite quaternions."
                )
            poses[:, 3:] /= norms
            result.update(zip(ids, poses))
        return result

    def _refresh_attachments(self) -> None:
        if self._attachments:
            self._commit(parent_poses=self._resolve_attachments())

    def detach(
        self, *, env_ids: object | None = None, keep_world_pose: bool = True
    ) -> None:
        """Detach selected environments, preserving their live world poses by default.

        With ``keep_world_pose=False``, stored offsets become environment-local
        poses. Neither option advances physics.

        Args:
            env_ids: Unique environment indices, or None for every environment.
            keep_world_pose: Preserve current live world poses when True.
        """
        self._check_live()
        ids = self._env_ids(env_ids)
        if not isinstance(keep_world_pose, (bool, np.bool_)):
            raise ValueError("keep_world_pose must be a boolean.")
        attachments = self._attachments.copy()
        poses = (
            self._resolve_attachments()
            if keep_world_pose and ids
            else self._parent_poses.copy()
        )
        states = self._states.copy()
        for env_id in ids:
            if env_id not in attachments:
                continue
            if keep_world_pose:
                state = states[env_id].copy()
                xyz, xyzw = state["translations"], state["orientations_xyzw"]
                if len(xyz):
                    rotation = Rotation.from_quat(poses[env_id][3:])
                    state["translations"] = rotation.apply(xyz) + poses[env_id][:3]
                    state["orientations_xyzw"] = (
                        rotation * Rotation.from_quat(xyzw)
                    ).as_quat()
                states[env_id] = state
            del attachments[env_id]
            poses.pop(env_id, None)
        if ids:
            self._commit(states=states, attachments=attachments, parent_poses=poses)

    def _detach_parent(self, parent: str) -> None:
        ids = [
            env_id
            for env_id, target in self._attachments.items()
            if target[0] == parent
        ]
        if ids:
            self.detach(env_ids=ids)

    def remove(self) -> None:
        """Release the entire group once; further updates raise RuntimeError."""
        if self._removed:
            return
        if self._on_remove is not None:
            self._on_remove(self)
        self._removed = True
        self._states = [self._defaults(0) for _ in range(self.num_envs)]
        self._attachments.clear()
        self._parent_poses.clear()
        self._pose_resolver = self._on_change = self._on_remove = None
