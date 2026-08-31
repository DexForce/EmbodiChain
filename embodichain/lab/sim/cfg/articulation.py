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

"""Articulation-root, per-link, joint, and articulation configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, fields
import numbers
from typing import Any, Dict, List, Literal, Sequence

import numpy as np
import torch

from embodichain.utils import configclass, is_configclass, logger

from .._legacy_cfg import RigidBodyAttributesCfg, RigidBodyAttributesOverrideCfg
from .asset import AssetPhysicsMode, ObjectBaseCfg, _resolve_asset_physics_mode
from .rigid import (
    RigidBodyPhysicsCfg,
    _rigid_body_attrs_from_dict,
)


def _normalize_joint_target_mode(value: object) -> int:
    """Normalize a portable joint target mode to its backend integer value."""
    if isinstance(value, str):
        normalized = value.replace("-", "_").lower()
        modes = {
            "none": 0,
            "position": 1,
            "velocity": 2,
            "position_velocity": 3,
            "effort": 4,
        }
        if normalized not in modes:
            raise ValueError(
                f"Unsupported joint target mode {value!r}; expected one of "
                f"{tuple(modes)}."
            )
        return modes[normalized]
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        mode = int(value)
        if 0 <= mode <= 4:
            return mode
        raise ValueError("Joint target-mode integers must be in [0, 4].")
    raise TypeError("Joint target mode must be a string or an integer in [0, 4].")


@configclass
class ArticulationRootPropertiesCfg:
    """Articulation-root properties shared by robot definitions.

    ``fixed_base`` and ``self_collision_enabled`` are consumed by both
    backends. ``sleep_threshold`` and the solver-iteration fields are supported
    only by the Default backend and are ignored by Newton. ``None`` preserves
    the source value or backend/import default.
    """

    fixed_base: bool | None = None
    """Whether the articulation root is rigidly fixed to the world frame."""

    self_collision_enabled: bool | None = None
    """Whether non-filtered link pairs in the articulation may self-collide.

    Newton may still filter adjacent parent-child bodies through
    :attr:`NewtonCollisionPropertiesCfg.collision_filter_parent`.
    """

    sleep_threshold: float | None = None
    """Default-only articulation sleep threshold; Newton ignores this field."""

    min_position_iters: int | None = None
    """Default-only minimum root position-solver iterations (1 to 255)."""

    min_velocity_iters: int | None = None
    """Default-only minimum root velocity-solver iterations (0 to 255)."""

    def __post_init__(self) -> None:
        """Require the two values consumed by the atomic Default setter."""
        if (self.min_position_iters is None) != (self.min_velocity_iters is None):
            raise ValueError(
                "Articulation-root min_position_iters and min_velocity_iters "
                "must be configured together."
            )

    @classmethod
    def from_dict(
        cls,
        init_dict: Mapping[str, Any],
    ) -> ArticulationRootPropertiesCfg:
        """Parse articulation-root properties without a backend subtype."""
        return cls(**dict(init_dict))


_REMOVED_ARTICULATION_CFG_FIELDS = {
    "fix_base": "root_props.fixed_base",
    "disable_self_collision": (
        "root_props.self_collision_enabled (invert the old boolean)"
    ),
    "sleep_threshold": "root_props.sleep_threshold",
    "min_position_iters": "root_props.min_position_iters",
    "min_velocity_iters": "root_props.min_velocity_iters",
    "articulation_props": "root_props",
    "drive_pros": "joint_drive_props",
    "joint_props": "joint_drive_props",
}


def _raise_removed_articulation_cfg_fields(init_dict: Mapping[str, Any]) -> None:
    """Reject removed flat articulation fields with actionable replacements."""
    removed = _REMOVED_ARTICULATION_CFG_FIELDS.keys() & init_dict.keys()
    if not removed:
        return
    replacements = ", ".join(
        f"{name} -> {_REMOVED_ARTICULATION_CFG_FIELDS[name]}"
        for name in sorted(removed)
    )
    raise ValueError(f"Removed ArticulationCfg fields: {replacements}.")


@configclass
class LinkPhysicsOverrideCfg:
    """Partial physics overlay for a selected set of articulation links.

    Regex/control-group resolution happens before Spawn updates exact source
    link names.  A link may match only one override group.
    """

    link_names_expr: list[str] = MISSING
    """Regular expressions matched against complete source link names."""

    attrs: RigidBodyPhysicsCfg | RigidBodyAttributesOverrideCfg = RigidBodyPhysicsCfg()
    """Partial grouped overlay, or the deprecated Default-only flat form."""

    replace_inertial: bool = False
    """Whether a mass/density override discards source inertia for recomputation.

    An explicitly configured inertia remains authoritative.  With ``False``, a
    source-authored inertia is retained when only mass or density changes.
    """

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> LinkPhysicsOverrideCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()
        for key, value in init_dict.items():
            if key == "attrs" and isinstance(value, dict):
                setattr(cfg, key, _rigid_body_attrs_from_dict(value, override=True))
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg


def link_attrs_from_dict(
    value: dict[str, Any],
) -> dict[str, LinkPhysicsOverrideCfg]:
    """Parse a ``link_attrs`` mapping from YAML/JSON-style dicts."""
    link_attrs: dict[str, LinkPhysicsOverrideCfg] = {}
    for group_name, group_cfg in value.items():
        if isinstance(group_cfg, LinkPhysicsOverrideCfg):
            link_attrs[group_name] = group_cfg
        elif isinstance(group_cfg, dict):
            link_attrs[group_name] = LinkPhysicsOverrideCfg.from_dict(group_cfg)
        else:
            raise TypeError(
                f"link_attrs['{group_name}'] must be a dict or "
                f"LinkPhysicsOverrideCfg, got {type(group_cfg)}."
            )
    return link_attrs


@configclass
class JointDrivePropertiesCfg:
    """Portable joint-drive and joint-dynamics properties.

    A scalar applies to every resolved joint.  A dictionary maps exact joint
    names, full-match regular expressions, or robot control-part names to
    values; exact/regex rules override broader control-part rules.  ``None``
    preserves source/backend ownership of a field.

    ``drive_type`` retains the Default drive response (force, acceleration, or
    disabled), while ``target_mode`` selects the commanded target components.
    Spawn resolves the two concepts before lowering them to the Default drive
    descriptor and Newton ``JointDofConfig``.

    Effort and velocity limits, friction, and armature share the same matching
    rules and descriptor compilation boundary as the actuator target and gains.

    Newton stores all fields in the model, but individual solvers may ignore
    limits, friction, armature, or target modes; consult the `Newton solver
    feature matrix
    <https://newton-physics.github.io/newton/latest/solvers/index.html>`_.
    """

    drive_type: Literal["force", "acceleration", "none"] | None = None
    """Joint drive type to apply.

    On the Default backend, ``"force"`` applies a force/torque drive,
    ``"acceleration"`` applies a mass-independent acceleration drive, and
    ``"none"`` disables the drive. Newton has no acceleration-drive
    equivalent. Unless :attr:`target_mode` is explicit, ``"force"`` and
    ``"acceleration"`` select ``"position_velocity"`` while ``"none"``
    selects ``"none"``.
    """

    target_mode: (
        Literal[
            "none",
            "position",
            "velocity",
            "position_velocity",
            "effort",
        ]
        | Dict[
            str,
            Literal[
                "none",
                "position",
                "velocity",
                "position_velocity",
                "effort",
            ]
            | int,
        ]
        | int
        | None
    ) = None
    """Portable actuator target mode, as a scalar or joint-rule mapping.

    Accepted names and integer values are ``"none"``/``0`` (passive),
    ``"position"``/``1``, ``"velocity"``/``2``,
    ``"position_velocity"``/``3``, and ``"effort"``/``4``. Default emulates
    these modes through its drive mode and effective gains. Newton authors the
    corresponding ``JointTargetMode``; solvers without native target-mode
    support use deterministic gain-based fallbacks where possible.
    """

    stiffness: Dict[str, float] | float | None = None
    """Proportional position gain of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s^2 (N/m).
    * For angular joints, the unit is kg-m^2/s^2/rad (N-m/rad).
    """

    damping: Dict[str, float] | float | None = None
    """Derivative velocity gain of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s (N-s/m).
    * For angular joints, the unit is kg-m^2/s/rad (N-m-s/rad).
    """

    max_effort: Dict[str, float] | float | None = None
    """Maximum drive effort [N for prismatic, N*m for revolute joints].

    The value is authored for both backends, but the selected Newton solver may
    not enforce it.
    """

    max_velocity: Dict[str, float] | float | None = None
    """Maximum joint speed [m/s for prismatic, rad/s for revolute joints].

    The value is authored for both backends, but support is solver-dependent in
    Newton.
    """

    friction: Dict[str, float] | float | None = None
    """Passive friction value applied along the joint degree of freedom.

    Interpretation and enforcement are backend/solver-dependent.
    """

    armature: Dict[str, float] | float | None = None
    """Artificial inertia added to the joint-space diagonal.

    Units depend on the joint model:

    * For prismatic (linear) joints, the unit is mass [kg].
    * For revolute (angular) joints, the unit is mass * scene_length^2 [kg-m^2].

    Armature changes the physical model and should normally reflect actuator or
    gearbox inertia.  Newton solver support varies.
    """

    def _resolve_modes(self) -> tuple[object, str | None]:
        """Resolve the target default implied by the original drive type."""
        target_mode = self.target_mode
        drive_type = self.drive_type
        if drive_type not in {None, "force", "acceleration", "none"}:
            raise ValueError(f"Unsupported joint drive type {drive_type!r}.")
        if target_mode is None:
            target_mode = {
                None: None,
                "force": "position_velocity",
                "acceleration": "position_velocity",
                "none": "none",
            }[drive_type]
        return target_mode, drive_type

    @classmethod
    def from_dict(
        cls,
        init_dict: Dict[str, Any],
        *,
        defaults: JointDrivePropertiesCfg | None = None,
    ) -> JointDrivePropertiesCfg:
        """Initialize the configuration from a dictionary.

        Args:
            init_dict: Joint-drive properties to override.
            defaults: Optional base properties whose unspecified values are
                preserved. If omitted, the class defaults are used.

        Returns:
            Parsed joint-drive properties.
        """
        data = dict(init_dict)
        backend = str(data.pop("backend", "common")).replace("-", "_").lower()
        wants_newton = backend == "newton"
        if backend not in {"common", "default", "newton"}:
            raise ValueError(
                "joint_drive_props.backend must be 'common', 'default', or 'newton', "
                f"got {backend!r}."
            )
        if wants_newton and not isinstance(defaults, NewtonJointDrivePropertiesCfg):
            cfg = NewtonJointDrivePropertiesCfg()
            if defaults is not None:
                for item in fields(JointDrivePropertiesCfg):
                    setattr(cfg, item.name, getattr(defaults, item.name))
        else:
            cfg = defaults.copy() if defaults is not None else cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize joint properties with their backend subtype."""
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        if isinstance(self, NewtonJointDrivePropertiesCfg):
            data["backend"] = "newton"
        return data


@configclass
class NewtonJointDrivePropertiesCfg(JointDrivePropertiesCfg):
    """Compatibility subtype for serialized Newton joint-drive configs.

    ``target_mode`` is now portable and lives on
    :class:`JointDrivePropertiesCfg`. The subtype remains so existing
    ``backend="newton"`` dictionaries and round trips retain their type; new
    robot definitions should use the common class.
    """


@configclass
class ArticulationCfg(ObjectBaseCfg):
    """Configuration for an articulation asset in the simulation.

    This class extends the base asset configuration to include specific properties for articulations,
    such as joint drive properties, physical attributes.
    """

    fpath: str = None
    """Path to the articulation asset file."""

    body_scale: tuple | list = (1.0, 1.0, 1.0)
    """Scale of the articulation in the simulation world frame."""

    compute_uv: bool = False
    """Whether to compute the UV mapping for the articulation link.

    Currently, the uv mapping is computed for each link with projection uv mapping method.
    """

    asset_physics_mode: AssetPhysicsMode = "preserve"
    """How source-authored articulation physics is handled.

    ``"preserve"`` keeps link, joint-drive, and joint-limit properties from
    either USD or URDF. ``"overlay"`` applies only explicitly configured
    values after the source has been resolved.

    Import policy such as root fixation and body scale remains controlled by
    :attr:`root_props` and :attr:`body_scale`.
    """

    attrs: RigidBodyPhysicsCfg | RigidBodyAttributesCfg = RigidBodyPhysicsCfg()
    """Physical attributes for all links. We use default mass from the USD/URDF file if available.
    The mass and density in attrs will only be used if specified. Deprecated
    flat :class:`RigidBodyAttributesCfg` inputs are Default-backend-only.
    """

    link_attrs: dict[str, LinkPhysicsOverrideCfg] | None = None
    """Named per-link physics override groups keyed by regex on link names.

    Each group applies :attr:`LinkPhysicsOverrideCfg.attrs` on top of :attr:`attrs` for
    matched links only. A link must not match more than one group.
    """

    root_props: ArticulationRootPropertiesCfg = ArticulationRootPropertiesCfg()
    """Grouped articulation-root properties.

    Fixed-base and self-collision intent is portable. Root sleep and solver
    iterations are Default-only fields and are ignored by Newton. ``None``
    preserves an authored USD/backend value. For URDF imports, unset portable
    fields use the established fixed-base, self-collision-off defaults.
    """

    joint_drive_props: JointDrivePropertiesCfg | None = None
    """Optional joint-drive and joint-dynamics overrides.

    ``None`` preserves source drive properties. Individual ``None`` fields in
    a provided config also preserve the corresponding source values.
    """

    init_qpos: torch.Tensor | np.ndarray | Sequence[float] = None
    """Initial joint positions of the articulation.

    If None, the joint positions will be set to zero.
    If provided, it should be an array of shape ``(num_dofs,)``.
    """

    qpos_limits: (
        torch.Tensor
        | np.ndarray
        | Sequence[Sequence[float]]
        | Dict[str, List[float]]
        | None
    ) = None
    """Override joint position limits of the articulation.

    If None, the joint position limits from the asset file (URDF/USD) are used.
    If provided as a tensor/array of shape ``(num_dofs, 2)``, it is applied in
    flattened source-resolved DOF order before the backend model is built.
    If provided as a dictionary, keys are joint names or regular expressions and
    values are ``[min, max]`` limits.

    This field replaces the asset limits for the articulation and can be used to
    either tighten or expand the allowed range.
    """

    build_pk_chain: bool = True
    """Whether to build pytorch-kinematics chain for forward kinematics and jacobian computation."""

    def resolve_asset_physics_mode(self) -> AssetPhysicsMode:
        """Return the effective file-backed physics policy."""
        return _resolve_asset_physics_mode(self.asset_physics_mode)

    @classmethod
    def from_dict(
        cls, init_dict: Dict[str, str | float | tuple | dict]
    ) -> ArticulationCfg:
        """Initialize the configuration from a dictionary."""
        _raise_removed_articulation_cfg_fields(init_dict)
        cfg = cls()
        for key, value in init_dict.items():
            if key == "link_attrs" and isinstance(value, dict):
                cfg.link_attrs = link_attrs_from_dict(value)
            elif key == "attrs" and isinstance(value, Mapping):
                cfg.attrs = _rigid_body_attrs_from_dict(value)
            elif key == "joint_drive_props" and isinstance(value, Mapping):
                cfg.joint_drive_props = JointDrivePropertiesCfg.from_dict(
                    dict(value),
                    defaults=cfg.joint_drive_props,
                )
            elif hasattr(cfg, key):
                attr = getattr(cfg, key)
                if is_configclass(attr):
                    setattr(cfg, key, attr.from_dict(value))
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )

        if cfg.init_local_pose is None:
            from scipy.spatial.transform import Rotation as R

            T = np.eye(4)
            T[:3, 3] = np.array(cfg.init_pos)
            T[:3, :3] = R.from_euler("xyz", np.deg2rad(cfg.init_rot)).as_matrix()
            cfg.init_local_pose = T
        else:
            from scipy.spatial.transform import Rotation as R

            cfg.init_pos = tuple(cfg.init_local_pose[:3, 3])
            cfg.init_rot = tuple(
                R.from_matrix(cfg.init_local_pose[:3, :3]).as_euler("xyz", degrees=True)
            )

        return cfg
