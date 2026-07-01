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
"""Compose a dual-manipulator (two-arm) robot from any single-arm robot cfg.

The :func:`build_dual_arm_cfg` engine takes a constructed single-arm
:class:`~embodichain.lab.sim.cfg.RobotCfg` (e.g. a ``URRobotCfg``) that follows
the existing ``"arm"`` convention -- one ``"arm"`` URDF component, one
``control_parts["arm"]`` entry and one ``solver_cfg["arm"]`` entry -- and emits a
fully-populated two-arm cfg whose left/right arms are mounted on a shared
synthetic ``base_link`` via :class:`~embodichain.lab.sim.cfg.URDFCfg`
multi-component assembly.

The :class:`DualArmRobotCfg` wrapper adds dict/YAML construction (``from_dict``)
on top of the engine: a registry resolves ``base_robot`` (e.g. ``"ur5"``) to the
single-arm cfg class, the engine duplicates the parts/solver/drive properties,
and the result plugs straight into ``sim.add_robot(cfg=...)`` and round-trips via
``to_dict()`` / ``from_dict()``.

Adding a future two-arm robot (e.g. Franka) only requires writing Franka's
single-arm cfg and one registry line -- no dual-arm class, no mixin.

Example:

    cfg = DualArmRobotCfg.from_dict(
        {"base_robot": "ur5", "mount": {"preset": "side_by_side", "separation": 0.6}}
    )
    robot = sim.add_robot(cfg=cfg)
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np
import torch

from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.solvers import SolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.lab.sim.robots.ur_robot import URRobotCfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["DualArmRobotCfg", "build_dual_arm_cfg", "resolve_mounts"]

# Each side -> (component_type, name prefix) used by ``URDFAssemblyManager``.
_SIDES = (("left", "left_arm", "left_"), ("right", "right_arm", "right_"))

#: Supported mount presets for :func:`resolve_mounts`.
_PRESETS = ("side_by_side", "facing_inward", "mirrored_rz")

#: Default mount used when none is specified.
_DEFAULT_MOUNT: dict = {"preset": "side_by_side", "separation": 0.6}

#: Registry mapping a ``base_robot`` key to ``(single-arm cfg class, init dict)``.
#: Adding a new single-arm robot here is all that is needed to make it
#: dual-arm-able: e.g. ``"franka": (FrankaRobotCfg, {})``.
_BASE_ROBOT_REGISTRY: Dict[str, tuple] = {
    "ur3": (URRobotCfg, {"robot_type": "ur3"}),
    "ur3e": (URRobotCfg, {"robot_type": "ur3e"}),
    "ur5": (URRobotCfg, {"robot_type": "ur5"}),
    "ur5e": (URRobotCfg, {"robot_type": "ur5e"}),
    "ur10": (URRobotCfg, {"robot_type": "ur10"}),
    "ur10e": (URRobotCfg, {"robot_type": "ur10e"}),
}

#: Drive-property fields that may be scalar or a regex->value dict.
_DRIVE_PROPS = (
    "stiffness",
    "damping",
    "max_effort",
    "max_velocity",
    "friction",
    "armature",
)


# --------------------------------------------------------------------------- #
# Naming helper
# --------------------------------------------------------------------------- #


def _prefixed_name(
    name: str | None,
    prefix: str,
    kind: str,
    name_case: dict[str, str] | None = None,
) -> str | None:
    """Apply the same prefix + case convention as ``URDFAssemblyManager``.

    Mirrors ``URDFComponentManager._generate_unique_name`` followed by
    ``NameNormalizer``: prepend ``prefix`` unless the name already starts with
    it, then case-normalize according to ``name_case``. This keeps the names
    predicted by the engine identical to the ones written into the assembled
    URDF; an integration test asserts they match.

    Args:
        name: The original joint/link name from the single-arm URDF.
        prefix: The side prefix (``"left_"`` / ``"right_"``).
        kind: ``"joint"`` or ``"link"`` (selects the case policy).
        name_case: Optional joint/link case policy, matching
            :class:`~embodichain.lab.sim.cfg.URDFCfg`.

    Returns:
        The prefixed, case-normalized name.
    """
    if name is None:
        return None
    base = name
    if prefix and not name.lower().startswith(prefix.lower()):
        base = f"{prefix}{name}"
    mode = (name_case or {}).get(kind, "original")
    if mode == "upper":
        return base.upper()
    if mode == "lower":
        return base.lower()
    return base


# --------------------------------------------------------------------------- #
# Mount resolution
# --------------------------------------------------------------------------- #


def _transform_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    """Build a 4x4 homogeneous transform from xyz translation and xyz euler.

    Args:
        xyz: Translation ``[x, y, z]``.
        rpy: Intrinsic xyz euler angles in radians ``[r, p, y]``.

    Returns:
        The 4x4 transform.
    """
    from scipy.spatial.transform import Rotation as R

    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    T[:3, :3] = R.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()
    return T


def resolve_mounts(mount_cfg: dict | None) -> Dict[str, np.ndarray]:
    """Resolve a mount config into left/right 4x4 transforms.

    Supported presets (the common bimanual layouts):

    * ``side_by_side`` -- left at ``+separation/2`` in Y, right at
      ``-separation/2``, same orientation.
    * ``facing_inward`` -- same ±Y separation, yawed ±90° so the arms face each
      other (mirror-symmetric).
    * ``mirrored_rz`` -- same ±Y separation, with yaw ``+rz`` on the left arm
      and ``-rz`` on the right arm.

    A per-arm ``left`` / ``right`` override (``{"xyz": [...], "rpy": [...]}``)
    may replace either side; both must be given together or neither.

    Args:
        mount_cfg: Mount configuration dict. ``None`` uses the default preset.

    Returns:
        ``{"left": T_left, "right": T_right}`` with 4x4 transforms.

    Raises:
        ValueError: If the preset is unknown, or only one per-arm override is
            given.
    """
    cfg = dict(mount_cfg or _DEFAULT_MOUNT)
    preset = cfg.get("preset", "side_by_side")
    if preset not in _PRESETS:
        raise ValueError(
            f"Unknown mount preset {preset!r}. Expected one of {sorted(_PRESETS)}."
        )
    separation = float(cfg.get("separation", _DEFAULT_MOUNT["separation"]))
    half = separation / 2.0
    rz = float(cfg.get("rz", 0.0))

    left_override = cfg.get("left")
    right_override = cfg.get("right")
    if (left_override is None) != (right_override is None):
        raise ValueError(
            "mount: provide both 'left' and 'right' overrides, or neither."
        )

    if preset == "side_by_side":
        left = {"xyz": [0.0, half, 0.0], "rpy": [0.0, 0.0, 0.0]}
        right = {"xyz": [0.0, -half, 0.0], "rpy": [0.0, 0.0, 0.0]}
    elif preset == "facing_inward":  # mirror yaw (+90 / -90 about Z)
        left = {"xyz": [0.0, half, 0.0], "rpy": [0.0, 0.0, np.pi / 2]}
        right = {"xyz": [0.0, -half, 0.0], "rpy": [0.0, 0.0, -np.pi / 2]}
    else:  # mirrored_rz: mirror the user-specified base yaw (+rz / -rz)
        left = {"xyz": [0.0, half, 0.0], "rpy": [0.0, 0.0, rz]}
        right = {"xyz": [0.0, -half, 0.0], "rpy": [0.0, 0.0, -rz]}

    if left_override:
        left.update(left_override)
    if right_override:
        right.update(right_override)

    return {
        "left": _transform_from_xyz_rpy(left["xyz"], left["rpy"]),
        "right": _transform_from_xyz_rpy(right["xyz"], right["rpy"]),
    }


# --------------------------------------------------------------------------- #
# Base-robot resolution
# --------------------------------------------------------------------------- #


def _resolve_base_cfg(base_robot: str | dict) -> RobotCfg:
    """Resolve a ``base_robot`` spec to a constructed single-arm cfg.

    Args:
        base_robot: Either a registry key (e.g. ``"ur5"``) or an explicit
            ``{"type": "ur5", "init": {...}}`` mapping for passing extra
            base-robot init params.

    Returns:
        A constructed single-arm :class:`RobotCfg`.

    Raises:
        ValueError: If the key is not in :data:`_BASE_ROBOT_REGISTRY`.
    """
    if isinstance(base_robot, dict):
        key = base_robot.get("type")
        init = base_robot.get("init", {}) or {}
    else:
        key = base_robot
        init = {}
    if key not in _BASE_ROBOT_REGISTRY:
        raise ValueError(
            f"Unknown base_robot {key!r}. Registered base robots: "
            f"{sorted(_BASE_ROBOT_REGISTRY)}."
        )
    cls_, default_init = _BASE_ROBOT_REGISTRY[key]
    merged_init = {**default_init, **init}
    return cls_.from_dict(merged_init)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def _mirror_drive_pros(
    base_drive: JointDrivePropertiesCfg, name_case: dict[str, str] | None = None
) -> JointDrivePropertiesCfg:
    """Mirror a single-arm drive config across left/right arms.

    Scalar fields apply to all joints uniformly and are copied verbatim. A
    regex->value dict (per-joint-index drive) is mirrored by uppercasing each
    pattern and emitting ``LEFT_`` / ``RIGHT_`` variants, matching the
    assembled (prefixed) joint names.

    .. note::
        Regex-pattern mirroring is best-effort. It prepends the side prefix and
        then applies the configured joint name case, so it stays aligned with
        the assembled URDF naming policy.

    Args:
        base_drive: The single-arm :class:`JointDrivePropertiesCfg`.
        name_case: Optional joint/link case policy, matching
            :class:`~embodichain.lab.sim.cfg.URDFCfg`.

    Returns:
        A fresh :class:`JointDrivePropertiesCfg` for the dual arm.
    """
    new = JointDrivePropertiesCfg(drive_type=base_drive.drive_type)
    for prop in _DRIVE_PROPS:
        val = getattr(base_drive, prop, None)
        if val is None:
            continue
        if isinstance(val, dict):
            mirrored: Dict[str, float] = {}
            for pattern, v in val.items():
                mirrored[_prefixed_name(str(pattern), "left_", "joint", name_case)] = v
                mirrored[_prefixed_name(str(pattern), "right_", "joint", name_case)] = v
            setattr(new, prop, mirrored)
        else:
            setattr(new, prop, val)
    return new


def _populate_dual_cfg(
    cfg: RobotCfg,
    base_cfg: RobotCfg,
    mounts: Dict[str, np.ndarray],
    *,
    dual_part: bool,
    arm_part: str,
) -> None:
    """Populate ``cfg`` (in place) with the dual-arm derivation of ``base_cfg``.

    Args:
        cfg: The target :class:`RobotCfg` (typically a ``DualArmRobotCfg``).
        base_cfg: A constructed single-arm robot cfg.
        mounts: ``{"left": T, "right": T}`` 4x4 mount transforms.
        dual_part: Whether to emit a ``"dual_arm"`` composite control part.
        arm_part: The base cfg's manipulator part name (default ``"arm"``).

    Raises:
        ValueError: If ``base_cfg`` lacks the ``arm_part`` control part or URDF
            component.
    """
    base_control = base_cfg.control_parts or {}
    if arm_part not in base_control:
        raise ValueError(
            f"Base robot cfg has no control part {arm_part!r} (available: "
            f"{list(base_control.keys())}). Set `arm_part` to the base robot's "
            f"manipulator part name."
        )

    base_solver_map = base_cfg.solver_cfg
    if isinstance(base_solver_map, dict):
        if arm_part not in base_solver_map:
            raise ValueError(
                f"Base robot solver_cfg has no part {arm_part!r} (available: "
                f"{list(base_solver_map.keys())})."
            )
        base_solver: SolverCfg = base_solver_map[arm_part]
    else:
        base_solver = base_solver_map

    base_components = base_cfg.urdf_cfg.components if base_cfg.urdf_cfg else {}
    if arm_part not in base_components:
        raise ValueError(
            f"Base robot urdf_cfg has no component {arm_part!r} (available: "
            f"{list(base_components.keys())})."
        )
    arm_urdf = base_components[arm_part]["urdf_path"]

    # A solver whose ``urdf_path`` is None adopts the *assembled* dual URDF at
    # ``Robot.init_solver`` time (e.g. OPW/SRS pk-based solvers), so its link
    # names must be prefixed to match the assembled URDF. A solver whose
    # ``urdf_path`` is pinned to the single-arm URDF (e.g. ``URSolverCfg``, set
    # in ``__post_init__``) operates arm-local, so link names stay unprefixed
    # and must match the single-arm URDF. This keeps FK/IK consistent per arm.
    use_assembled = getattr(base_solver, "urdf_path", None) is None

    cfg.uid = f"DualArm_{base_cfg.uid}" if base_cfg.uid else "DualArmRobot"

    cfg.urdf_cfg = URDFCfg(
        components=[
            {
                "component_type": comp_type,
                "urdf_path": arm_urdf,
                "transform": mounts[side],
            }
            for side, comp_type, _ in _SIDES
        ]
    )
    name_case = cfg.urdf_cfg.name_case

    # control_parts: duplicate every base part into left_/right_ with prefixed
    # joint names (generic over all base parts, e.g. arm *and* eef).
    new_control: Dict[str, List[str]] = {}
    for part_name, joints in base_control.items():
        for side, _comp, prefix in _SIDES:
            new_control[f"{side}_{part_name}"] = [
                _prefixed_name(j, prefix, "joint", name_case) for j in joints
            ]
    if dual_part:
        new_control["dual_arm"] = (
            new_control[f"left_{arm_part}"] + new_control[f"right_{arm_part}"]
        )
    cfg.control_parts = new_control

    # solver_cfg: one solver per arm. ``replace`` makes a fresh instance per side
    # (init_solver mutates urdf_path/joint_names in place, so sharing one
    # instance across sides would corrupt them).
    new_solver: Dict[str, SolverCfg] = {}
    for side, _comp, prefix in _SIDES:
        if use_assembled:
            root = _prefixed_name(base_solver.root_link_name, prefix, "link", name_case)
            end = _prefixed_name(base_solver.end_link_name, prefix, "link", name_case)
        else:
            root = base_solver.root_link_name
            end = base_solver.end_link_name
        new_solver[f"{side}_{arm_part}"] = base_solver.replace(
            root_link_name=root, end_link_name=end
        )
    cfg.solver_cfg = new_solver

    cfg.drive_pros = _mirror_drive_pros(base_cfg.drive_pros, name_case)
    cfg.attrs = base_cfg.attrs.copy()
    cfg.min_position_iters = base_cfg.min_position_iters
    cfg.min_velocity_iters = base_cfg.min_velocity_iters
    cfg.fix_base = base_cfg.fix_base
    cfg.disable_self_collision = base_cfg.disable_self_collision
    cfg.sleep_threshold = base_cfg.sleep_threshold


def build_dual_arm_cfg(
    base_cfg: RobotCfg,
    mounts: Dict[str, np.ndarray],
    *,
    dual_part: bool = True,
    arm_part: str = "arm",
) -> "DualArmRobotCfg":
    """Build a dual-arm cfg from a single-arm robot cfg.

    Args:
        base_cfg: A constructed single-arm :class:`RobotCfg` following the
            ``"arm"`` convention.
        mounts: ``{"left": T, "right": T}`` 4x4 mount transforms from
            :func:`resolve_mounts`.
        dual_part: Whether to include a ``"dual_arm"`` composite control part.
        arm_part: The base cfg's manipulator part name.

    Returns:
        A populated :class:`DualArmRobotCfg`.

    Example:

        base = URRobotCfg.from_dict({"robot_type": "ur5"})
        mounts = resolve_mounts({"preset": "side_by_side", "separation": 0.6})
        cfg = build_dual_arm_cfg(base, mounts)
    """
    cfg = DualArmRobotCfg()
    _populate_dual_cfg(cfg, base_cfg, mounts, dual_part=dual_part, arm_part=arm_part)
    return cfg


# --------------------------------------------------------------------------- #
# Wrapper cfg -- dict/YAML entry point
# --------------------------------------------------------------------------- #


@configclass
class DualArmRobotCfg(RobotCfg):
    """Configuration for a dual-manipulator composed from a single-arm robot.

    Two identical arms (the ``base_robot``) are mounted on a shared synthetic
    ``base_link``. The left/right ``control_parts``, per-arm ``solver_cfg`` and
    mirrored ``drive_pros`` are derived automatically by
    :func:`build_dual_arm_cfg`.

    Example:

        cfg = DualArmRobotCfg.from_dict(
            {"base_robot": "ur5",
             "mount": {"preset": "side_by_side", "separation": 0.6}}
        )
        robot = sim.add_robot(cfg=cfg)
    """

    base_robot: Union[str, dict] = "ur10"
    """Registry key (e.g. ``"ur5"``) or ``{"type": ..., "init": {...}}``."""

    arm_part: str = "arm"
    """Name of the base robot's manipulator control part."""

    dual_part: bool = True
    """Whether to emit a ``"dual_arm"`` composite control part."""

    mount: dict = field(default_factory=lambda: dict(_DEFAULT_MOUNT))
    """Mount configuration consumed by :func:`resolve_mounts`."""

    @classmethod
    def from_dict(cls, init_dict: dict) -> "DualArmRobotCfg":
        """Initialize from a dictionary.

        Args:
            init_dict: Configuration dict. ``base_robot``, ``mount``,
                ``arm_part`` and ``dual_part`` drive the dual-arm derivation;
                all other recognized :class:`RobotCfg` keys are merged on top
                via :func:`merge_robot_cfg`.

        Returns:
            A ``DualArmRobotCfg`` instance.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Populate default urdf/control/solver/physics for the dual arm.

        Args:
            init_dict: The raw override dict passed to ``from_dict``.
        """
        init_dict = init_dict or {}
        base_robot = init_dict.get("base_robot", self.base_robot)
        arm_part = init_dict.get("arm_part", self.arm_part)
        dual_part = init_dict.get("dual_part", self.dual_part)
        mount_cfg = init_dict.get("mount", self.mount)

        base_cfg = _resolve_base_cfg(base_robot)
        mounts = resolve_mounts(mount_cfg)

        _populate_dual_cfg(
            self, base_cfg, mounts, dual_part=dual_part, arm_part=arm_part
        )

        # Store the user-facing inputs so to_dict()/from_dict() round-trips.
        self.base_robot = base_robot
        self.arm_part = arm_part
        self.dual_part = dual_part
        self.mount = mount_cfg

    @property
    def _pk_urdf_path(self) -> str | None:
        """Single-arm URDF used for the per-arm FK/IK serial chain.

        Both arms are kinematically identical, so the left arm's URDF suffices.
        Returns ``None`` before :meth:`_build_defaults` has populated
        ``urdf_cfg`` (the property is queried by ``@configclass`` post-init
        before defaults are applied).
        """
        if self.urdf_cfg is None or "left_arm" not in self.urdf_cfg.components:
            return None
        return self.urdf_cfg.components["left_arm"]["urdf_path"]

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        """Build the per-arm pytorch-kinematics serial chains.

        Each chain is built from the single-arm URDF with the (arm-local) root
        and end link names taken from the left-arm solver, mirroring the
        :class:`CobotMagicCfg` pattern. Both arms share one URDF; the chains are
        keyed ``"left_arm"`` / ``"right_arm"`` for API symmetry.

        Args:
            device: The device to move the chains to. Defaults to CPU.
            **kwargs: Additional arguments for building the serial chains.

        Returns:
            A ``{"left_arm": chain, "right_arm": chain}`` mapping.
        """
        from embodichain.lab.sim.utility.solver_utils import (
            create_pk_serial_chain,
        )

        urdf_path = self._pk_urdf_path
        solver = self.solver_cfg["left_arm"]
        return {
            "left_arm": create_pk_serial_chain(
                urdf_path=urdf_path,
                device=device,
                end_link_name=solver.end_link_name,
                root_link_name=solver.root_link_name,
            ),
            "right_arm": create_pk_serial_chain(
                urdf_path=urdf_path,
                device=device,
                end_link_name=solver.end_link_name,
                root_link_name=solver.root_link_name,
            ),
        }


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg

    config = SimulationManagerCfg(
        headless=True,
        device="cpu",
        num_envs=1,
        render_cfg=RenderCfg(renderer="fast-rt"),
    )
    sim = SimulationManager(config)

    cfg = DualArmRobotCfg.from_dict(
        {
            "base_robot": "ur5",
            "mount": {"preset": "side_by_side", "separation": 0.6},
            "init_qpos": [
                0.0,
                0.0,
                -1.57,
                -1.57,
                1.57,
                1.57,
                -1.57,
                -1.57,
                -1.57,
                -1.57,
                0.0,
                0.0,
            ],
        }
    )
    robot = sim.add_robot(cfg=cfg)
    sim.open_window()

    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    # Round-trip check: from_dict(to_dict()) reproduces the cfg.
    cfg2 = DualArmRobotCfg.from_dict(cfg.to_dict())
    assert cfg2.base_robot == cfg.base_robot
    assert cfg2.control_parts == cfg.control_parts
    print(f"DualArm robot added ({cfg.base_robot}); round-trip OK.")

    from IPython import embed

    embed()  # noqa: F401
