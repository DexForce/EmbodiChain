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

"""Deterministic startup selection of a task's rigid asset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.envs.managers.cfg import EventCfg, SceneEntityCfg
from embodichain.lab.gym.envs.managers.manager_base import Functor
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

__all__ = ["select_rigid_object_asset"]


def _rigidize_usd(source: Path) -> Path:
    """Cache a centered, single-body copy of the authored USD rest pose."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

    digest = hashlib.sha256(
        b"rigidized-rest-pose-v1\0" + source.read_bytes()
    ).hexdigest()
    directory = Path(EMBODICHAIN_DEFAULT_DATA_ROOT) / "generated" / "rigid_assets"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}.usdc"
    if destination.is_file():
        return destination
    original = Usd.Stage.Open(str(source))
    if original is None or not original.GetDefaultPrim():
        raise ValueError("rigidize_usd requires a USD asset with a default prim.")
    stage = Usd.Stage.Open(original.Flatten())
    geometry = stage.GetDefaultPrim()
    body = geometry.GetParent()
    if body.IsPseudoRoot() or not body.IsA(UsdGeom.Xform):
        raise ValueError("rigidize_usd requires the default prim beneath an Xform.")
    if any(
        prim.IsA(UsdGeom.Boundable) and not prim.GetPath().HasPrefix(geometry.GetPath())
        for prim in stage.Traverse()
    ):
        raise ValueError("rigidize_usd requires a single geometry subtree.")
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        .ComputeRelativeBound(geometry, body)
        .ComputeAlignedRange()
    )
    if bounds.IsEmpty():
        raise ValueError("Cannot rigidize an empty USD asset.")
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim.GetPath())
        for api in (
            UsdPhysics.ArticulationRootAPI,
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.MassAPI,
        ):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
    for path in joints:
        stage.RemovePrim(path)
    transform = UsdGeom.Xformable(geometry)
    centered = transform.GetLocalTransformation() * Gf.Matrix4d(1).SetTranslate(
        -bounds.GetMidpoint()
    )
    transform.MakeMatrixXform().Set(centered)
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    stage.SetDefaultPrim(body)
    # Export atomically: simultaneous workers may prepare the same source.
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        exported = Path(temporary) / "asset.usdc"
        if not stage.Export(str(exported)):
            raise ValueError(f"Cannot export rigid asset from {source}.")
        exported.replace(destination)
    return destination


class select_rigid_object_asset(Functor):
    """Select a named rigid asset once, before task adapters are constructed.

    Configure this event with ``mode="startup"``. Each asset entry contains
    a human-readable ``label`` and rigid-object config fields (except ``uid``).
    Optional ``rigidize_usd: true`` freezes a USD asset at its authored rest
    pose, centers its geometry, and removes articulation DOFs. The source is
    retained and the derived single-body asset is cached separately.

    The stable scene UID and task semantics stay unchanged. Replacement applies
    to the entire vector environment; per-row topology selection is unsupported.
    Labels and source identity are published for instruction binding/recording.
    """

    def __init__(self, cfg: EventCfg, env: EmbodiedEnv) -> None:
        """Capture the original rigid configuration for independent selections.

        Args:
            cfg: Startup event configuration.
            env: Environment containing the existing rigid object.

        Raises:
            ValueError: If the event phase or target object is unsupported.
        """
        super().__init__(cfg, env)
        if cfg.mode != "startup":
            raise ValueError("select_rigid_object_asset requires mode='startup'.")
        entity = env.sim.get_rigid_object(cfg.params["entity_cfg"].uid)
        if entity is None:
            raise ValueError("Asset selection requires an existing rigid object.")
        self._base_cfg = deepcopy(entity.cfg.to_dict())

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: torch.Tensor | Sequence[int] | None,
        entity_cfg: SceneEntityCfg,
        asset: str,
        assets: Mapping[str, Mapping[str, object]],
    ) -> None:
        """Replace the object and publish its identity after successful creation.

        Args:
            env: Environment whose asset should be selected.
            env_ids: All environment rows, or ``None``; partial batches fail.
            entity_cfg: Existing rigid object's stable UID.
            asset: Key of the selected asset entry.
            assets: Named labels and rigid-object configuration overrides.

        Raises:
            ValueError: If selection, fields, or the selected rows are invalid.
            RuntimeError: If a Task Program adapter has already bound the scene.
        """
        if getattr(env, "_task_program_adapter", None) is not None:
            raise RuntimeError("Select assets before constructing the task adapter.")
        if env_ids is not None:
            rows = torch.as_tensor(env_ids, device="cpu")
            if rows.numel() == 0:
                return
            if rows.dtype == torch.bool or rows.is_floating_point() or rows.ndim != 1:
                raise ValueError("env_ids must contain all integer environment IDs.")
            if not torch.equal(rows.sort().values, torch.arange(env.num_envs)):
                raise ValueError("Asset replacement requires every environment row.")
        if (
            type(asset) is not str
            or not isinstance(assets, Mapping)
            or asset not in assets
            or not isinstance(assets[asset], Mapping)
        ):
            raise ValueError(f"Unknown rigid asset {asset!r}.")
        declaration = deepcopy(dict(assets[asset]))
        allowed = set(self._base_cfg) - {"uid"}
        unknown = set(declaration) - allowed - {"label", "rigidize_usd"}
        if unknown:
            raise ValueError(f"Unknown asset fields: {sorted(unknown)}.")
        label = declaration.pop("label", None)
        if type(label) is not str or not label.strip() or label != label.strip():
            raise ValueError("Each asset requires a non-empty label.")
        rigidize = declaration.pop("rigidize_usd", False)
        if type(rigidize) is not bool:
            raise ValueError("rigidize_usd must be a boolean.")
        data = deepcopy(self._base_cfg)
        data.update(declaration)
        data["uid"] = entity_cfg.uid
        replacement = RigidObjectCfg.from_dict(data)
        source = None
        if isinstance(replacement.shape, MeshCfg) and replacement.shape.fpath:
            source = Path(get_data_path(replacement.shape.fpath)).resolve()
            replacement.shape.fpath = str(source)
        if rigidize:
            if source is None or source.suffix.lower() not in {
                ".usd",
                ".usda",
                ".usdc",
            }:
                raise ValueError("rigidize_usd requires a USD mesh asset.")
            replacement.shape.fpath = str(_rigidize_usd(source))
            replacement.asset_physics_mode = "overlay"
        env.sim.replace_rigid_object(replacement)
        selections = dict(getattr(env, "_asset_selections", {}))
        selections[entity_cfg.uid] = {
            "asset": asset,
            "label": label,
            "source": None if source is None else str(source),
            "rigidized": rigidize,
        }
        env._asset_selections = selections
