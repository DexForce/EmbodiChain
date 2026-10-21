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

"""Articulated reference visuals composed from externally supplied link poses."""

from __future__ import annotations

import copy
from dataclasses import MISSING
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.utils import configclass
from embodichain.lab.sim.cfg.source import UsdSourceCfg, _usd_source_from_value
from .._utils import to_numpy_array
from .cfg import MarkerGroupCfg
from ._robot_visual_assets import load_urdf_visuals, load_usd_visuals

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__ = ["RobotVisualAppearanceCfg", "RobotVisualCfg", "RobotVisual"]


@configclass
class RobotVisualAppearanceCfg:
    """Independent material appearance and opacity.

    ``override`` replaces the base-color factor while retaining source textures
    and material parameters. ``asset`` preserves the original factor. Opacity
    controls coverage. Opaque visuals use scene lighting; translucent appearance
    may differ across renderers. Offscreen cameras exclude visuals until
    visualization is enabled.
    """

    mode: Literal["override", "asset"] = "override"
    color: tuple[float, float, float] = (0.2, 0.85, 0.45)
    opacity: float = 0.4
    unlit: bool = False
    """Use uniform native color without surface lighting when True."""


@configclass
class RobotVisualCfg:
    """Robot visual source and reference coordinate convention."""

    name: str = MISSING
    asset_path: str | None = None
    source: UsdSourceCfg | None = None
    """USD source selector shared with VisualObject, RigidObject and Robot."""
    prim_path: str | None = None
    appearance: RobotVisualAppearanceCfg = RobotVisualAppearanceCfg()
    coordinate_space: Literal["world", "env"] = "world"


def _appearance(value: RobotVisualAppearanceCfg) -> RobotVisualAppearanceCfg:
    value.validate()
    if value.mode not in {"override", "asset"}:
        raise ValueError("Ghost mode must be 'override' or 'asset'.")
    color = np.asarray(value.color, dtype=float)
    if (
        color.shape != (3,)
        or not np.isfinite(color).all()
        or np.any((color < 0) | (color > 1))
    ):
        raise ValueError("Ghost color must be RGB in [0, 1].")
    if not np.isfinite(value.opacity) or not 0 <= value.opacity <= 1:
        raise ValueError("Ghost opacity must be in [0, 1].")
    if not isinstance(value.unlit, bool):
        raise ValueError("Ghost unlit must be a boolean.")
    return copy.deepcopy(value)


def _matrices(poses: object) -> np.ndarray:
    value = to_numpy_array(poses, np.float64)
    if value.shape[-1:] == (7,):
        if not np.isfinite(value).all():
            raise ValueError("Link poses must be finite.")
        quaternions = value[..., 3:]
        if np.any(np.linalg.norm(quaternions, axis=-1) <= 1e-12):
            raise ValueError("Link xyzw quaternions must be nonzero.")
        result = np.broadcast_to(np.eye(4), (*value.shape[:-1], 4, 4)).copy()
        if value.size == 0:
            return result
        result[..., :3, :3] = (
            Rotation.from_quat(quaternions.reshape(-1, 4))
            .as_matrix()
            .reshape(*value.shape[:-1], 3, 3)
        )
        result[..., :3, 3] = value[..., :3]
        return result
    if value.shape[-2:] != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Poses must be finite matrices or xyz + xyzw arrays.")
    rotations = value[..., :3, :3]
    if (
        not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(
            np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-5, rtol=0
        )
        or np.any(np.linalg.det(rotations) <= 0)
    ):
        raise ValueError(
            "Link poses must be rigid transforms without scale or reflection."
        )
    return value


class RobotVisual:
    """Render a robot reference without creating a physical articulation.

    Args:
        cfg: URDF or USD source, appearance and coordinate convention.
        sim: Owner of the marker resources. Prepare native scenes explicitly.
    """

    def __init__(self, cfg: RobotVisualCfg, sim: SimulationManager) -> None:
        cfg.validate()
        if cfg.coordinate_space not in {"world", "env"}:
            raise ValueError("coordinate_space must be 'world' or 'env'.")
        self.cfg = copy.deepcopy(cfg)
        self.cfg.appearance = _appearance(cfg.appearance)
        source = _usd_source_from_value(cfg.source)
        if source is not None:
            source.validate()
            if cfg.asset_path is not None or cfg.prim_path is not None:
                raise ValueError("Use source or asset_path/prim_path, not both.")
            asset_path, prim_path = source.file_path, source.source_prim_path
        else:
            asset_path, prim_path = cfg.asset_path, cfg.prim_path
        if asset_path is None:
            raise ValueError("Robot visual requires asset_path or source.")
        path = Path(asset_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self._urdf_path = path if path.suffix.lower() == ".urdf" else None
        self._texture_files = TemporaryDirectory(
            prefix="embodichain-robot-visual-materials-"
        )
        if self._urdf_path is not None:
            self._visuals = load_urdf_visuals(path, Path(self._texture_files.name))
        elif path.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
            self._visuals = load_usd_visuals(
                path, prim_path, Path(self._texture_files.name)
            )
        else:
            raise ValueError("Robot visual source must be URDF or USD.")
        self._link_names = tuple(
            dict.fromkeys(item.link_name for item in self._visuals)
        )
        self._link_indices = np.array(
            [self._link_names.index(item.link_name) for item in self._visuals]
        )
        self._local_poses = np.stack([item.local_pose for item in self._visuals])
        self._group = sim.add_marker_group(
            MarkerGroupCfg(
                name=cfg.name,
                prototypes=self._prototypes(self.cfg.appearance),
                scope="env",
            )
        )
        self._chain = None

    def _prototypes(self, appearance: RobotVisualAppearanceCfg) -> dict:
        prototypes = {}
        for index, visual in enumerate(self._visuals):
            prototype = copy.copy(visual.prototype)
            prototype.unlit = appearance.unlit
            prototype.scene_lighting = True
            if appearance.mode == "override":
                prototype.color = (*appearance.color, appearance.opacity)
            else:
                prototype.color = (
                    *prototype.color[:3],
                    prototype.color[3] * appearance.opacity,
                )
            prototypes[str(index)] = prototype
        return prototypes

    @property
    def link_names(self) -> tuple[str, ...]:
        """Ordered visual link names expected by visualize_link_poses()."""
        return self._link_names

    @property
    def num_visuals(self) -> int:
        """Number of material-separated visual parts per robot."""
        return len(self._visuals)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """URDF joint order for visualize_qpos(), including mimic joints."""
        return tuple(self._kinematic_chain().get_joint_parameter_names())

    def _kinematic_chain(self):
        if self._urdf_path is None:
            raise ValueError(
                "Joint-space visualization requires a URDF source; use link poses for USD."
            )
        if self._chain is None:
            import pytorch_kinematics as pk

            self._chain = pk.build_chain_from_urdf(self._urdf_path.read_bytes())
        return self._chain

    def visualize_link_poses(
        self, poses: object, *, env_ids: object | None = None
    ) -> None:
        """Compose reference link poses with each visual's local transform.

        Args:
            poses: Array (E, L, 7) in xyz + xyzw order or (E, L, 4, 4).
                Insert an instance dimension K after E to display multiple
                references per environment. L follows link_names. Task alignment
                and sampling remain with the caller; mesh scale applies once.
            env_ids: Unique environments to update; other environments retain state.
        """
        ids = self._group._env_ids(env_ids)
        matrices = _matrices(poses)
        if matrices.ndim == 4:
            matrices = matrices[:, None]
        if (
            matrices.ndim != 5
            or matrices.shape[0] != len(ids)
            or matrices.shape[2:] != (len(self.link_names), 4, 4)
        ):
            raise ValueError(
                "Link pose shape must match selected environments and link_names."
            )
        if not ids:
            return
        count = matrices.shape[1] * self.num_visuals
        if count == 0:
            self._group.clear(env_ids=ids)
            return
        visual_poses = matrices[:, :, self._link_indices] @ self._local_poses
        translations = visual_poses[..., :3, 3].reshape(len(ids), count, 3)
        if self.cfg.coordinate_space == "world":
            translations = translations - self._group._origins[ids, None, :]
        rotations = Rotation.from_matrix(
            visual_poses[..., :3, :3].reshape(-1, 3, 3)
        ).as_quat()
        self._group.update(
            translations=translations,
            orientations_xyzw=rotations.reshape(len(ids), count, 4),
            prototype_indices=np.tile(
                np.arange(self.num_visuals), (len(ids), matrices.shape[1])
            ),
            env_ids=ids,
        )

    def visualize_qpos(
        self,
        qpos: object,
        *,
        root_poses: object | None = None,
        env_ids: object | None = None,
    ) -> None:
        """Compute URDF FK and publish reference poses in one marker update.

        Args:
            qpos: Joint positions (E, J), or (E, K, J) for K references per
                environment, in joint_names order.
            root_poses: Optional root transforms with the same leading E / E,K
                dimensions as qpos, ending in 7 or (4,4). Identity when omitted.
            env_ids: Environments to update, retaining other environments.
        """
        import torch

        q = (
            qpos
            if isinstance(qpos, torch.Tensor)
            else torch.as_tensor(qpos, dtype=torch.float32)
        )
        if not q.is_floating_point():
            q = q.float()
        ids = self._group._env_ids(env_ids)
        chain = self._kinematic_chain()
        if (
            q.ndim not in (2, 3)
            or q.shape[0] != len(ids)
            or q.shape[-1] != len(self.joint_names)
            or not bool(torch.isfinite(q).all())
        ):
            raise ValueError("qpos must be finite with shape (E, J) or (E, K, J).")
        if not ids:
            return
        if q.ndim == 3 and q.shape[1] == 0:
            self._group.clear(env_ids=ids)
            return
        chain = chain.to(dtype=q.dtype, device=q.device)
        self._chain = chain
        with torch.no_grad():
            batch_size = int(np.prod(q.shape[:-1]))
            fk = chain.forward_kinematics(q.reshape(batch_size, q.shape[-1]))
            matrices = torch.stack(
                [fk[name].get_matrix() for name in self.link_names], dim=1
            )
            matrices = matrices.reshape(*q.shape[:-1], len(self.link_names), 4, 4)
        matrices = to_numpy_array(matrices, np.float64)
        if root_poses is not None:
            roots = _matrices(root_poses)
            if roots.shape != (*q.shape[:-1], 4, 4):
                raise ValueError("root_poses must match selected environments.")
            matrices = roots[..., None, :, :] @ matrices
        self.visualize_link_poses(matrices, env_ids=ids)

    def set_appearance(self, appearance: RobotVisualAppearanceCfg) -> None:
        """Change color mode or opacity without changing the source asset.

        Args:
            appearance: Validated appearance, applied atomically to all parts.
        """
        appearance = _appearance(appearance)
        self._group._check_live()
        cfg = copy.copy(self._group._cfg)
        cfg.prototypes = self._prototypes(appearance)
        self._group._commit(on_change=self._group._on_change, cfg=cfg)
        self.cfg.appearance = appearance

    def set_visibility(self, visible: bool, *, env_ids: object | None = None) -> None:
        """Set selected ghosts' visibility while retaining their reference poses.

        Args:
            visible: Whether to display the selected ghosts.
            env_ids: Optional environment selection.
        """
        self._group.set_visibility(visible, env_ids=env_ids)

    def remove(self) -> None:
        """Release the ghost's render resources; repeated removal is safe."""
        self._group.remove()
        self._texture_files.cleanup()
