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

"""Rigid object and rigid-object-group configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, field
import os
import warnings
import numpy as np
from typing import Any, Dict, Literal

from dexsim.types import ActorType

from embodichain.utils import configclass, is_configclass, logger

from ..shapes import ShapeCfg
from .asset import AssetPhysicsMode, ObjectBaseCfg, _resolve_asset_physics_mode
from .rigid import RigidBodyPhysicsCfg
from .visuals import GaussianSplatCfg, VisualElementCfg
from .source import UsdSourceCfg


@configclass
class RigidObjectCfg(ObjectBaseCfg):
    """Configuration for a rigid body asset in the simulation.

    This class extends the base asset configuration to include specific properties for rigid bodies,
    such as physical attributes and collision group.
    """

    visuals: list[VisualElementCfg | GaussianSplatCfg] | GaussianSplatCfg | None = (
        field(default=None, kw_only=True)
    )
    __annotations__["visuals"] = (
        list[VisualElementCfg | GaussianSplatCfg] | GaussianSplatCfg | None
    )
    """Render replacement; one Gaussian may omit its element wrapper.

    None preserves source appearance; [] removes it. Multiple elements require
    explicit VisualElementCfg IDs.
    """

    semantic_shape: ShapeCfg | None = field(default=None, kw_only=True)
    """Optional independent mesh/primitive for semantic geometry queries."""

    shape: ShapeCfg | None = None
    """Shape configuration for the rigid body. """

    source: UsdSourceCfg | None = field(default=None, kw_only=True)
    """USD source asset; mutually exclusive with :attr:`shape`."""

    # TODO: supoort basic primitive shapes, such as box, sphere, etc cfg and spawn method.

    visual_shape: ShapeCfg | None = None
    """Deprecated input alias normalized into :attr:`visuals`."""

    particle_collision_shape: ShapeCfg | None = None
    """Optional Newton SDF mesh for particles on this same rigid body.

    Rigid contacts and mass use :attr:`shape`. This boundary follows the body's
    live pose and twist and contributes no mass or rigid contact geometry.
    """

    render_enabled: bool = True
    """Whether to create native render geometry for this rigid object."""

    attrs: RigidBodyPhysicsCfg = RigidBodyPhysicsCfg()
    """Rigid-body physics.

    :class:`RigidBodyPhysicsCfg` groups portable and backend-native intent.
    """

    body_type: Literal["dynamic", "kinematic", "static"] | None = None
    """Actor type for procedural bodies or explicit USD ``overlay`` mode.

    USD ``preserve`` mode always retains the source actor, including when
    loading older configurations that serialized a default actor type."""

    body_scale: tuple | list | None = None
    """Geometry scale for procedural bodies or USD ``overlay`` mode.

    USD ``preserve`` mode retains its source scale."""

    asset_physics_mode: AssetPhysicsMode = "preserve"
    """How a file-backed asset's physical properties are handled.

    ``"preserve"`` keeps the USD-authored physics. ``"overlay"`` applies
    configured properties on top of the parsed asset. Procedural shapes always
    use config.
    """

    env_variants: dict[int, RigidObjectCfg] | None = None
    """Preparation-time procedural rigid replacements indexed by environment ID.

    Omitted environments use this configuration. Each replacement supplies its
    own shape, physics, scale, mesh visuals and initial pose; its UID identifies the source
    asset, while the resulting object retains the parent's logical UID. Nested
    variants are unsupported. Requires the DexSim assembly variants capability
    and the Newton backend; topology is fixed until a scene rebuild. USD source,
    per-variant semantic proxies and Gaussian visuals are rejected. An explicit
    parent semantic_shape is shared; otherwise each row uses its own shape.
    """

    def __post_init__(self) -> None:
        self._resolve_procedural_defaults()
        self._normalize_visuals()
        self._validate_env_variants()

    def _normalize_visuals(self) -> None:
        """Resolve legacy appearance inputs into the single visual-element list."""
        from ..shapes import CompoundShapeCfg

        if self.env_variants is not None and not isinstance(self.env_variants, Mapping):
            raise TypeError(
                "env_variants must map environment indices to rigid configs."
            )
        from .visuals import _visual_elements
        from scipy.spatial.transform import Rotation

        parts = (
            self.shape.visual_parts
            if isinstance(self.shape, CompoundShapeCfg)
            else None
        )
        parts_owner = self.shape
        sources = sum(
            value is not None for value in (self.visuals, self.visual_shape, parts)
        )
        if sources > 1:
            raise ValueError(
                "Specify only one appearance input: visuals, visual_shape or shape.visual_parts."
            )
        if self.visual_shape is not None:
            if (
                isinstance(self.visual_shape, CompoundShapeCfg)
                and self.visual_shape.visual_parts is not None
            ):
                parts_owner = self.visual_shape
                parts = self.visual_shape.visual_parts
            else:
                self.visuals = [VisualElementCfg(representation=self.visual_shape)]
            self.visual_shape = None
            warnings.warn(
                "visual_shape is deprecated; use visuals.",
                DeprecationWarning,
                stacklevel=2,
            )
        if parts is not None:
            self.visuals = []
            for index, part in enumerate(parts):
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = Rotation.from_quat(part.quaternion).as_matrix()
                pose[:3, 3] = part.position
                representation = part.shape.copy()
                if representation.visual_material is None:
                    representation.visual_material = parts_owner.visual_material
                self.visuals.append(
                    VisualElementCfg(
                        id=f"part_{index}",
                        representation=representation,
                        local_pose=pose,
                    )
                )
            if parts_owner is self.shape:
                self.shape = self.shape.copy()
                self.shape.visual_parts = None
            warnings.warn(
                "shape.visual_parts is deprecated; use visuals.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.visuals = _visual_elements(self.visuals)

    def _validate_env_variants(self) -> None:
        """Reject variant fields without a per-environment runtime owner."""
        from ..shapes import CompoundShapeCfg

        def requests_uv(shape: ShapeCfg | None) -> bool:
            if isinstance(shape, CompoundShapeCfg):
                if isinstance(shape.parts, type(MISSING)):
                    return False
                return any(requests_uv(part.shape) for part in shape.parts)
            return bool(getattr(shape, "compute_uv", False))

        for env_id, variant in (self.env_variants or {}).items():
            if isinstance(env_id, bool) or not isinstance(env_id, int) or env_id < 0:
                raise ValueError(
                    "env_variants keys must be non-negative environment indices."
                )
            if not isinstance(variant, RigidObjectCfg) or variant.env_variants:
                raise ValueError(
                    "env_variants requires flat RigidObjectCfg replacements."
                )
            unsupported = []
            if variant.source is not None or str(
                getattr(variant.shape, "fpath", "")
            ).lower().endswith((".usd", ".usda", ".usdc", ".usdz")):
                unsupported.append("source")
            if variant.semantic_shape is not None:
                unsupported.append("semantic_shape")
            if variant.source_pose_mode != "auto":
                unsupported.append("source_pose_mode")
            if variant.asset_physics_mode != "preserve":
                unsupported.append("asset_physics_mode")
            variant._normalize_visuals()
            if any(
                isinstance(element.representation, GaussianSplatCfg)
                for element in variant.visuals or ()
            ):
                unsupported.append("Gaussian visuals")
            if any(not element.visible for element in variant.visuals or ()):
                unsupported.append("visuals.visible=False")
            if requests_uv(variant.shape) or any(
                requests_uv(element.representation) for element in variant.visuals or ()
            ):
                unsupported.append("compute_uv")
            if unsupported:
                raise ValueError(
                    f"env_variants[{env_id}] does not support {', '.join(unsupported)}; configure shared semantics on the parent or declare separate objects."
                )

    def _resolve_procedural_defaults(self) -> None:
        """Resolve defaults only after ordinary geometry is assembled."""
        if self.shape is None or self.source is not None:
            return
        path = getattr(self.shape, "fpath", None)
        if path is None or not str(path).lower().endswith(
            (".usd", ".usda", ".usdc", ".usdz")
        ):
            if self.body_type is None:
                self.body_type = "dynamic"
            if self.body_scale is None:
                self.body_scale = (1.0, 1.0, 1.0)

    def resolve_asset_physics_mode(self) -> AssetPhysicsMode:
        """Return the effective file-backed physics policy."""
        return _resolve_asset_physics_mode(self.asset_physics_mode)

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> RigidObjectCfg:
        """Parse a rigid object and normalize legacy mesh collision ownership."""
        data = dict(init_dict)
        if data.get("env_variants") is not None:
            variants = data["env_variants"]
            if not isinstance(variants, Mapping):
                raise TypeError(
                    "env_variants must map environment indices to rigid configs."
                )
            normalized = {}
            for key, value in variants.items():
                if isinstance(key, str) and key.isdecimal():
                    key = int(key)
                if isinstance(key, bool) or not isinstance(key, int) or key < 0:
                    raise ValueError(
                        "env_variants keys must be non-negative environment indices."
                    )
                if isinstance(value, Mapping):
                    unknown = set(value) - cls.__dataclass_fields__.keys()
                    if unknown:
                        raise KeyError(
                            f"Unknown env_variants[{key}] fields: {sorted(unknown)}"
                        )
                normalized[key] = (
                    cls.from_dict(dict(value)) if isinstance(value, Mapping) else value
                )
            data["env_variants"] = normalized

        for key in ("visual_shape", "particle_collision_shape"):
            if isinstance(data.get(key), Mapping):
                data[key] = ShapeCfg.from_dict(data[key])
        attrs_value = data.get("attrs")
        if isinstance(attrs_value, Mapping) and "mesh_collision_props" in attrs_value:
            shape_value = data.get("shape")
            if not isinstance(shape_value, Mapping):
                raise ValueError(
                    "Legacy attrs.mesh_collision_props requires a mapping-valued "
                    "MeshCfg shape so it can migrate to shape.collision."
                )
            shape_data = dict(shape_value)
            if shape_data.get("shape_type") != "Mesh":
                raise ValueError(
                    "Legacy attrs.mesh_collision_props can migrate only to a "
                    "MeshCfg shape."
                )
            if shape_data.get("collision") is not None:
                raise ValueError(
                    "attrs.mesh_collision_props cannot be combined with "
                    "shape.collision."
                )
            attrs_data = dict(attrs_value)
            shape_data["collision"] = attrs_data.pop("mesh_collision_props")
            data["shape"] = shape_data
            data["attrs"] = attrs_data
            warnings.warn(
                "RigidBodyPhysicsCfg.mesh_collision_props is deprecated; use "
                "MeshCfg.collision.",
                DeprecationWarning,
                stacklevel=2,
            )
        cfg = super().from_dict(data)
        path = (
            cfg.source.file_path
            if cfg.source is not None
            else getattr(cfg.shape, "fpath", None)
        )
        if path is not None and str(path).lower().endswith(
            (".usd", ".usda", ".usdc", ".usdz")
        ):
            if "body_type" not in data:
                cfg.body_type = None
            if "body_scale" not in data:
                cfg.body_scale = None
            if not {"init_pos", "init_rot", "init_local_pose"}.intersection(data):
                cfg.init_local_pose = None
        cfg._resolve_procedural_defaults()
        cfg._normalize_visuals()
        cfg._validate_env_variants()
        return cfg

    def to_dexsim_body_type(self) -> ActorType:
        """Convert the body type to dexsim ActorType."""
        if self.body_type == "dynamic":
            return ActorType.DYNAMIC
        elif self.body_type == "kinematic":
            return ActorType.KINEMATIC
        elif self.body_type == "static":
            return ActorType.STATIC
        else:
            logger.log_error(
                f"Invalid body type '{self.body_type}' specified. Must be one of 'dynamic', 'kinematic', or 'static'."
            )


@configclass
class RigidObjectGroupCfg:
    """Configuration for a rigid object group asset in the simulation.

    Rigid object groups can be initialized from multiple rigid object configurations specified in a folder.
    If `folder_path` is specified, user should provide a RigidObjectCfg in `rigid_objects` as a template configuration for
    all objects in the group.

    For example:
    ```python
    rigid_object_group: RigidObjectGroupCfg(
        folder_path="path/to/folder",
        max_num=5,
        rigid_objects={
            "template_obj": RigidObjectCfg(
                shape=MeshCfg(
                    fpath="",  # fpath will be ignored when folder_path is specified
                ),
                body_type="dynamic",
            )
        }
    )
    """

    uid: str | None = None

    rigid_objects: Dict[str, RigidObjectCfg] = MISSING
    """Configuration for the rigid objects in the group."""

    body_type: Literal["dynamic", "kinematic"] = "dynamic"
    """Body type for all rigid objects in the group. """

    folder_path: str | None = None
    """Path to the folder containing the rigid object assets.

    This is used to initialize multiple rigid object configurations from a folder.
    """

    max_num: int = 1
    """Maximum number of rigid objects to initialize from the folder.

    This is only used when `folder_path` is specified.
    """

    ext: str = ".obj"
    """File extension for the rigid object assets.

    This is only used when `folder_path` is specified.
    """

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> RigidObjectGroupCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                elif key == "rigid_objects" and not init_dict.get("folder_path"):
                    rigid_objects_cfg = {}
                    for obj_name, obj_cfg in value.items():
                        rigid_objects_cfg[obj_name] = RigidObjectCfg.from_dict(obj_cfg)
                    setattr(cfg, key, rigid_objects_cfg)
                elif key == "rigid_objects" and init_dict.get("folder_path"):
                    folder_path = init_dict["folder_path"]
                    max_num = init_dict.get("max_num", 1)
                    rigid_objects_cfg = {}
                    if os.path.exists(folder_path) and os.path.isdir(folder_path):
                        files = os.listdir(folder_path)
                        files = [f for f in files if f.endswith(cfg.ext)]
                        # select files up to max_num
                        n_file = len(files)
                        select_files = []
                        for i in range(max_num):
                            select_files.append(files[i % n_file])

                        for i, file_name in enumerate(select_files):
                            file_path = os.path.join(folder_path, file_name)
                            rigid_obj_cfg: RigidObjectCfg = RigidObjectCfg.from_dict(
                                list(init_dict["rigid_objects"].values())[0]
                            )
                            rigid_obj_cfg.uid = f"{cfg.uid}_obj_{i}"
                            if rigid_obj_cfg.source is not None:
                                rigid_obj_cfg.source.file_path = file_path
                            else:
                                rigid_obj_cfg.shape.fpath = file_path
                            rigid_objects_cfg[rigid_obj_cfg.uid] = rigid_obj_cfg
                        setattr(cfg, "rigid_objects", rigid_objects_cfg)
                    else:
                        logger.log_error(
                            f"Folder '{folder_path}' does not exist or is not a directory."
                        )
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg
