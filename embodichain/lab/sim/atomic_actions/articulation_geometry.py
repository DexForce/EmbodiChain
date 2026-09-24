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

"""Adapt deterministic articulation facts into sampled affordance geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import numbers
from typing import TYPE_CHECKING, Protocol

import torch

from .affordance import AntipodalAffordance

from ._articulation_geometry_keys import (
    _ARTICULATION_POINT_CLOUD_KEY,
    _NON_TARGET_ARTICULATION_POINT_CLOUD_KEY,
    _TARGET_LINK_POINT_CLOUD_KEY,
    _TARGET_LINK_PRISMATIC_JOINT_AXIS_KEY,
    _TARGET_LINK_REVOLUTE_AXIS_ORIGIN_KEY,
    _TARGET_LINK_REVOLUTE_JOINT_AXIS_KEY,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Articulation


_MAX_RIGIDIZED_JOINT_POSITION_TOLERANCE = 1.0e-3
_MAX_RIGIDIZED_LINK_TRANSFORM_TOLERANCE = 1.0e-5


def _rigidized_articulation_tolerance(
    value: object,
    *,
    field_name: str,
) -> float:
    """Return one bounded positive tolerance for rigidized geometry."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{field_name} must be a real number.")
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"{field_name} must be finite and greater than zero.")
    maximum = {
        "joint_position_tolerance": _MAX_RIGIDIZED_JOINT_POSITION_TOLERANCE,
        "link_transform_tolerance": _MAX_RIGIDIZED_LINK_TRANSFORM_TOLERANCE,
    }[field_name]
    if tolerance > maximum:
        raise ValueError(f"{field_name} must be at most {maximum:.1e}.")
    return tolerance


def _validated_locked_qpos(
    articulation: Articulation,
    locked_qpos: Mapping[str, float],
) -> dict[str, float]:
    """Validate and own the declared joint configuration."""
    if not isinstance(locked_qpos, Mapping):
        raise TypeError("locked_qpos must be a mapping from joint names to values.")
    normalized: dict[str, float] = {}
    for name, value in locked_qpos.items():
        if type(name) is not str or not name or name != name.strip():
            raise ValueError("locked_qpos keys must be non-empty joint names.")
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"locked_qpos[{name!r}] must be a real number.")
        position = float(value)
        if not math.isfinite(position):
            raise ValueError(f"locked_qpos[{name!r}] must be finite.")
        normalized[name] = position
    return normalized


def _joint_readings(
    values: torch.Tensor,
    joint_count: int,
    *,
    field_name: str,
) -> torch.Tensor:
    """Normalize one batched per-joint reading to finite CPU float64 rows."""
    readings = torch.as_tensor(values).detach().cpu().to(torch.float64)
    if joint_count <= 0 or readings.numel() == 0 or readings.numel() % joint_count:
        raise ValueError(
            f"{field_name} must contain complete readings for {joint_count} joints."
        )
    readings = readings.reshape(-1, joint_count)
    if not bool(torch.isfinite(readings).all().item()):
        raise ValueError(f"{field_name} must contain only finite values.")
    return readings


def _first_flagged(flags: torch.Tensor) -> tuple[int, int]:
    """Return the ``(arena, joint)`` index of the first flagged entry."""
    index = int(torch.argmax(flags.flatten().to(torch.uint8)))
    return divmod(index, flags.shape[1])


def _root_to_link_transforms(
    articulation: Articulation,
    grasp_link: str,
    locked_qpos: Mapping[str, float],
) -> torch.Tensor:
    """Read or compute batched articulation-root-to-link transforms."""
    joint_names = tuple(articulation.joint_names)
    if getattr(articulation, "pk_chain", None) is not None:
        qpos = torch.tensor(
            [[locked_qpos[name] for name in joint_names]],
            dtype=torch.float32,
        )
        poses = torch.as_tensor(
            articulation.compute_fk(
                qpos,
                link_names=(grasp_link,),
                qpos_joint_names=joint_names,
            )
        )
        if poses.dim() == 4 and poses.shape[1:] == (1, 4, 4):
            poses = poses[:, 0]
        if poses.dim() != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                "Articulation named FK must return shape (N, 4, 4) or "
                "(N, 1, 4, 4) for one selected link."
            )
        transforms = poses.detach().cpu().to(torch.float64)
    else:
        root = (
            torch.as_tensor(articulation.get_local_pose(to_matrix=True))
            .detach()
            .cpu()
            .to(torch.float64)
            .reshape(-1, 4, 4)
        )
        link = (
            torch.as_tensor(articulation.get_link_pose(grasp_link, to_matrix=True))
            .detach()
            .cpu()
            .to(torch.float64)
            .reshape(-1, 4, 4)
        )
        if root.shape[0] != link.shape[0]:
            raise ValueError(
                f"{articulation.uid!r} reports {root.shape[0]} root poses but "
                f"{link.shape[0]} poses for link {grasp_link!r}; the two readings "
                "must share one arena batch."
            )
        transforms = torch.linalg.inv(root) @ link
    if transforms.shape[0] == 0 or not bool(torch.isfinite(transforms).all().item()):
        raise ValueError("Articulation root-to-link transforms must be finite.")
    return transforms


def _articulation_root_to_link(
    articulation: Articulation,
    grasp_link: str,
    locked_qpos: Mapping[str, float],
    *,
    transform_tolerance: float,
) -> torch.Tensor:
    """Return the common root-to-link transform for every arena."""
    per_arena = _root_to_link_transforms(articulation, grasp_link, locked_qpos)
    if per_arena.shape[0] > 1:
        spread = (per_arena - per_arena[0]).abs().amax(dim=(1, 2))
        worst = int(torch.argmax(spread))
        if float(spread[worst]) > transform_tolerance:
            raise ValueError(
                f"Arena {worst} of {articulation.uid!r} places {grasp_link!r} "
                f"{float(spread[worst]):.6f} away from the arena-0 root-to-link "
                "transform, so one grasp mesh cannot describe every arena."
            )
    return per_arena[0]


def _assert_link_is_rigid_to_root(
    articulation: Articulation,
    locked_qpos: Mapping[str, float],
    *,
    position_tolerance: float,
) -> None:
    """Reject a configuration that is not held as one compound rigid body."""
    joint_names = tuple(articulation.joint_names)
    declared = set(locked_qpos)
    missing = sorted(set(joint_names) - declared)
    if missing:
        raise ValueError(
            f"locked_qpos must declare every joint of {articulation.uid!r} so the "
            f"articulation is a compound rigid body; missing {missing}."
        )
    unknown = sorted(declared - set(joint_names))
    if unknown:
        raise ValueError(
            f"locked_qpos declares joints {unknown} that {articulation.uid!r} "
            f"does not have; available joints are {list(joint_names)}."
        )

    joint_count = len(joint_names)
    expected = torch.tensor(
        [locked_qpos[name] for name in joint_names],
        dtype=torch.float64,
    )
    measured = _joint_readings(
        articulation.get_qpos(), joint_count, field_name="articulation qpos"
    )
    displaced = (measured - expected).abs() > position_tolerance
    if bool(displaced.any()):
        arena, joint = _first_flagged(displaced)
        raise ValueError(
            f"Joint {joint_names[joint]!r} of {articulation.uid!r} reads "
            f"{float(measured[arena, joint]):.6f} in arena {arena} but was declared "
            f"locked at {float(expected[joint]):.6f}."
        )

    target = _joint_readings(
        articulation.get_qpos(target=True),
        joint_count,
        field_name="articulation target qpos",
    )
    stiffness = _joint_readings(
        articulation.get_joint_drive()[0],
        joint_count,
        field_name="articulation drive stiffness",
    )
    driven = (stiffness > 0.0) & ((target - expected).abs() <= position_tolerance)

    limits = torch.as_tensor(articulation.get_qpos_limits()).detach().cpu()
    if limits.numel() != measured.shape[0] * joint_count * 2:
        raise ValueError(
            "articulation qpos limits must contain lower and upper values for "
            "every arena and joint."
        )
    limits = limits.to(torch.float64).reshape(-1, joint_count, 2)
    if not bool(torch.isfinite(limits).all().item()):
        raise ValueError("articulation qpos limits must contain only finite values.")
    lower, upper = limits[..., 0], limits[..., 1]
    pinned = (
        ((upper - lower).abs() <= position_tolerance)
        & (expected >= lower - position_tolerance)
        & (expected <= upper + position_tolerance)
    )
    unheld = ~(driven | pinned)
    if bool(unheld.any()):
        arena, joint = _first_flagged(unheld)
        raise ValueError(
            f"Joint {joint_names[joint]!r} of {articulation.uid!r} rests at its "
            f"declared position in arena {arena} but nothing holds it there: drive "
            f"stiffness is {float(stiffness[arena, joint]):.6f} with position target "
            f"{float(target[arena, joint]):.6f}, and the position limits span "
            f"[{float(lower[arena, joint]):.6f}, {float(upper[arena, joint]):.6f}]."
        )


def create_rigidized_articulation_antipodal_affordance(
    articulation: Articulation,
    *,
    grasp_link: str,
    locked_qpos: Mapping[str, float],
    joint_position_tolerance: float = 1.0e-3,
    link_transform_tolerance: float = 1.0e-5,
) -> AntipodalAffordance:
    """Build root-frame, link-scoped grasp geometry for a rigidized articulation.

    Args:
        articulation: Native articulation providing named joint, link, pose,
            drive, limit, FK, and link-mesh queries.
        grasp_link: Native link whose mesh supplies the grasp geometry.
        locked_qpos: Exact position required for every native joint.
        joint_position_tolerance: Positive tolerance, no greater than ``1e-3``,
            for measured positions, targets, and direct-Python lock checks.
        link_transform_tolerance: Positive tolerance, no greater than ``1e-5``,
            for root-to-link transform agreement across arenas.

    Returns:
        Owned link-scoped antipodal mesh in the articulation-root frame.

    Raises:
        TypeError: If a declared value or native reading has an invalid type.
        ValueError: If the articulation is not consistently rigidized, the
            selected link is absent, or its transformed mesh is invalid.
    """
    joint_tolerance = _rigidized_articulation_tolerance(
        joint_position_tolerance, field_name="joint_position_tolerance"
    )
    transform_tolerance = _rigidized_articulation_tolerance(
        link_transform_tolerance, field_name="link_transform_tolerance"
    )
    locked = _validated_locked_qpos(articulation, locked_qpos)
    link_names = tuple(articulation.link_names)
    if grasp_link not in link_names:
        raise ValueError(
            f"Link {grasp_link!r} is not part of {articulation.uid!r}; "
            f"available links are {list(link_names)}."
        )
    _assert_link_is_rigid_to_root(
        articulation, locked, position_tolerance=joint_tolerance
    )
    root_to_link = _articulation_root_to_link(
        articulation,
        grasp_link,
        locked,
        transform_tolerance=transform_tolerance,
    )
    vertices, triangles = articulation.get_link_vert_face(grasp_link)
    vertices = torch.as_tensor(vertices)
    triangles = torch.as_tensor(triangles)
    if (
        not vertices.is_floating_point()
        or vertices.dim() != 2
        or vertices.shape[1:] != (3,)
        or vertices.shape[0] == 0
        or not bool(torch.isfinite(vertices).all().item())
    ):
        raise ValueError(
            f"Link {grasp_link!r} of {articulation.uid!r} has no usable mesh "
            "vertices; expected non-empty finite floating mesh vertices with "
            f"shape (N, 3), got {tuple(vertices.shape)}."
        )
    transform = root_to_link.to(dtype=torch.float64, device=vertices.device)
    local = vertices.to(torch.float64)
    root_frame = (local @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]).to(
        vertices.dtype
    )
    return AntipodalAffordance(
        mesh_vertices=root_frame,
        mesh_triangles=triangles,
        mesh_scope="link",
    )


class ArticulationJointGeometry(Protocol):
    """Structural joint geometry consumed by the articulation adapter.

    Attributes:
        name: Stable joint name.
        joint_type: Normalized joint type such as ``fixed``, ``prismatic``, or
            ``revolute``.
        parent_link_name: Stable parent-link name.
        child_link_name: Stable child-link name.
        origin_pose: Joint-frame pose in the parent-link frame, shape ``(4, 4)``.
        axis: Joint axis in the joint frame, shape ``(3,)``.
    """

    name: str
    joint_type: str
    parent_link_name: str
    child_link_name: str
    origin_pose: torch.Tensor
    axis: torch.Tensor


class ArticulationGeometryProvider(Protocol):
    """Deterministic articulation facts required for geometry adaptation.

    Implementations provide raw link meshes, FK, and immediate-parent-first
    joint topology. Initial configuration and scale are explicit adapter inputs
    so this protocol does not depend on an ``ArticulationCfg`` or PK-chain API.

    Attributes:
        device: Device on which geometry tensors are assembled.
        link_names: Stable articulation link names.
    """

    device: torch.device
    link_names: Sequence[str]

    def get_link_vert_face(
        self,
        link_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one link-local triangle mesh.

        Args:
            link_name: Stable link name.

        Returns:
            Link-local vertices and triangle indices.

        .. attention::
            A non-empty link mesh must contain at least one non-degenerate
            triangle surface.
        """
        ...

    def compute_fk(
        self,
        qpos: torch.Tensor,
        *,
        link_names: Sequence[str],
        qpos_joint_names: Sequence[str],
    ) -> torch.Tensor:
        """Return root-frame link poses for the supplied named joint state.

        Args:
            qpos: Joint positions with shape ``(B, J)``.
            link_names: Links whose poses should be returned.
            qpos_joint_names: Names corresponding to the last ``qpos`` axis.

        Returns:
            Root-frame poses with shape ``(B, L, 4, 4)``.
        """
        ...

    def get_parent_joint_chain(
        self,
        link_name: str,
    ) -> tuple[ArticulationJointGeometry, ...]:
        """Return parent joints ordered from the link toward the root.

        Args:
            link_name: Link at which to begin traversal.

        Returns:
            Immediate-parent-first structural joint geometry.
        """
        ...


@dataclass(frozen=True, slots=True, eq=False)
class ArticulationAffordanceGeometry:
    """Owned sampled geometry for articulation-link affordances.

    The point clouds and optional joint data are expressed in the target link's
    initial local frame. Joint axes are normalized. Use
    :meth:`to_object_geometry` at the ``ObjectSemantics`` boundary so the
    Atomic Action-specific string-key protocol remains in this module.

    Args:
        target_link_point_cloud: Sampled target-link surface, shape ``(N, 3)``.
        articulation_point_cloud: Sampled merged-articulation surface, shape
            ``(M, 3)``.
        prismatic_joint_axis: Optional nearest prismatic ancestor axis.
        revolute_joint_axis: Optional nearest revolute ancestor axis.
        revolute_axis_origin: Optional matching revolute-joint origin.
        non_target_articulation_point_cloud: Sampled merged surface of every
            link except the target, shape ``(K, 3)``. An empty tensor records
            that the articulation has no non-target link surface. ``None`` is
            accepted only for geometry created without source-link provenance.
    """

    target_link_point_cloud: torch.Tensor
    articulation_point_cloud: torch.Tensor
    prismatic_joint_axis: torch.Tensor | None = None
    revolute_joint_axis: torch.Tensor | None = None
    revolute_axis_origin: torch.Tensor | None = None
    non_target_articulation_point_cloud: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate and own all tensor fields."""
        target_points = _owned_point_cloud(
            self.target_link_point_cloud,
            field_name="target_link_point_cloud",
        )
        articulation_points = _owned_point_cloud(
            self.articulation_point_cloud,
            field_name="articulation_point_cloud",
        )
        if target_points.device != articulation_points.device:
            raise ValueError(
                "Articulation affordance point clouds must share a device."
            )
        non_target_points = _owned_optional_point_cloud(
            self.non_target_articulation_point_cloud,
            field_name="non_target_articulation_point_cloud",
        )
        if (
            non_target_points is not None
            and non_target_points.device != target_points.device
        ):
            raise ValueError(
                "Articulation affordance point clouds must share a device."
            )

        prismatic_axis = _owned_axis(
            self.prismatic_joint_axis,
            field_name="prismatic_joint_axis",
        )
        revolute_axis = _owned_axis(
            self.revolute_joint_axis,
            field_name="revolute_joint_axis",
        )
        revolute_origin = _owned_point(
            self.revolute_axis_origin,
            field_name="revolute_axis_origin",
        )
        if (revolute_axis is None) != (revolute_origin is None):
            raise ValueError(
                "revolute_joint_axis and revolute_axis_origin must be provided "
                "together."
            )
        for value, field_name in (
            (prismatic_axis, "prismatic_joint_axis"),
            (revolute_axis, "revolute_joint_axis"),
            (revolute_origin, "revolute_axis_origin"),
        ):
            if value is not None and value.device != target_points.device:
                raise ValueError(
                    f"{field_name} and articulation point clouds must share a device."
                )

        object.__setattr__(self, "target_link_point_cloud", target_points)
        object.__setattr__(self, "articulation_point_cloud", articulation_points)
        object.__setattr__(
            self,
            "non_target_articulation_point_cloud",
            non_target_points,
        )
        object.__setattr__(self, "prismatic_joint_axis", prismatic_axis)
        object.__setattr__(self, "revolute_joint_axis", revolute_axis)
        object.__setattr__(self, "revolute_axis_origin", revolute_origin)

    def to_object_geometry(self) -> dict[str, torch.Tensor]:
        """Return an owned ``ObjectSemantics.geometry`` dictionary.

        Returns:
            A new real dictionary containing cloned point clouds and whichever
            optional joint entries are available.
        """
        geometry = {
            _TARGET_LINK_POINT_CLOUD_KEY: self.target_link_point_cloud.clone(),
            _ARTICULATION_POINT_CLOUD_KEY: self.articulation_point_cloud.clone(),
        }
        if self.non_target_articulation_point_cloud is not None:
            geometry[_NON_TARGET_ARTICULATION_POINT_CLOUD_KEY] = (
                self.non_target_articulation_point_cloud.clone()
            )
        if self.prismatic_joint_axis is not None:
            geometry[_TARGET_LINK_PRISMATIC_JOINT_AXIS_KEY] = (
                self.prismatic_joint_axis.clone()
            )
        if self.revolute_joint_axis is not None:
            geometry[_TARGET_LINK_REVOLUTE_JOINT_AXIS_KEY] = (
                self.revolute_joint_axis.clone()
            )
        if self.revolute_axis_origin is not None:
            geometry[_TARGET_LINK_REVOLUTE_AXIS_ORIGIN_KEY] = (
                self.revolute_axis_origin.clone()
            )
        return geometry


def sample_initial_articulation_geometry(
    provider: ArticulationGeometryProvider,
    target_link_name: str,
    *,
    initial_qpos: torch.Tensor | Sequence[float],
    initial_qpos_joint_names: Sequence[str],
    body_scale: torch.Tensor | Sequence[float],
    articulation_point_count: int = 100_000,
    target_point_count: int = 5_000,
) -> ArticulationAffordanceGeometry:
    """Sample initial articulation geometry for Atomic Action affordances.

    The adapter evaluates FK at an explicitly named initial joint state,
    transforms every raw link mesh into the target link's initial local frame,
    merges those meshes, and samples the target and merged surfaces with
    Open3D. It also transforms the nearest prismatic and revolute ancestor
    joint geometry into that frame.

    Args:
        provider: Structural source of deterministic link meshes, FK, and
            parent-joint topology.
        target_link_name: Link whose initial local frame defines the result.
        initial_qpos: Initial joint positions with shape ``(J,)``.
        initial_qpos_joint_names: Names corresponding to ``initial_qpos``.
        body_scale: Configured articulation scale. Only unit scale is currently
            supported because raw meshes and FK must share one metric frame.
        articulation_point_count: Merged-articulation surface sample count.
            The same count is used for the non-target merged surface when one
            exists.
        target_point_count: Target-link surface sample count.

    Returns:
        Owned typed affordance geometry in the target link's initial frame.

    Raises:
        TypeError: If a name, joint-name sequence, or point count has the wrong
            type.
        ValueError: If the target, initial state, scale, FK output, topology, or
            mesh geometry is invalid.
    """
    link_names = _validate_adapter_inputs(
        provider,
        target_link_name=target_link_name,
        articulation_point_count=articulation_point_count,
        target_point_count=target_point_count,
    )
    device = torch.device(provider.device)
    initial_qpos, qpos_joint_names = _validate_initial_state(
        initial_qpos,
        initial_qpos_joint_names,
        device=device,
    )
    _validate_body_scale(body_scale)

    initial_link_poses = provider.compute_fk(
        initial_qpos.unsqueeze(0),
        link_names=link_names,
        qpos_joint_names=qpos_joint_names,
    )
    if (
        not isinstance(initial_link_poses, torch.Tensor)
        or initial_link_poses.shape != (1, len(link_names), 4, 4)
        or not bool(torch.isfinite(initial_link_poses).all().item())
    ):
        raise ValueError(
            "Initial FK must return finite poses with shape "
            f"(1, {len(link_names)}, 4, 4)."
        )
    initial_link_poses = initial_link_poses[0].to(
        device=device,
        dtype=torch.float32,
    )
    target_index = link_names.index(target_link_name)
    target_from_root = torch.linalg.inv(initial_link_poses[target_index])

    prismatic_axis, revolute_axis, revolute_origin = _resolve_joint_geometry(
        provider,
        target_link_name=target_link_name,
        link_names=link_names,
        initial_link_poses=initial_link_poses,
        target_from_root=target_from_root,
        device=device,
    )
    (
        target_vertices,
        target_triangles,
        articulation_vertices,
        articulation_triangles,
        non_target_vertices,
        non_target_triangles,
    ) = _merge_initial_link_meshes(
        provider,
        target_link_name=target_link_name,
        link_names=link_names,
        initial_link_poses=initial_link_poses,
        target_from_root=target_from_root,
        device=device,
    )

    target_points = _sample_mesh_surface_points(
        target_vertices,
        target_triangles,
        target_point_count,
    )
    articulation_points = _sample_mesh_surface_points(
        articulation_vertices,
        articulation_triangles,
        articulation_point_count,
    )
    non_target_points = (
        _sample_mesh_surface_points(
            non_target_vertices,
            non_target_triangles,
            articulation_point_count,
        )
        if non_target_vertices.shape[0] > 0
        else torch.empty((0, 3), dtype=target_points.dtype, device=device)
    )
    return ArticulationAffordanceGeometry(
        target_link_point_cloud=target_points,
        articulation_point_cloud=articulation_points,
        prismatic_joint_axis=prismatic_axis,
        revolute_joint_axis=revolute_axis,
        revolute_axis_origin=revolute_origin,
        non_target_articulation_point_cloud=non_target_points,
    )


def _validate_adapter_inputs(
    provider: ArticulationGeometryProvider,
    *,
    target_link_name: str,
    articulation_point_count: int,
    target_point_count: int,
) -> list[str]:
    """Validate names and point counts shared by adapter stages."""
    if type(target_link_name) is not str:
        raise TypeError("target_link_name must be a string.")
    if not target_link_name or target_link_name != target_link_name.strip():
        raise ValueError("target_link_name must be non-empty.")
    link_names = list(provider.link_names)
    if target_link_name not in link_names:
        raise ValueError(
            f"Unknown articulation link {target_link_name!r}. Available links: "
            f"{link_names}."
        )
    for value, field_name in (
        (articulation_point_count, "articulation_point_count"),
        (target_point_count, "target_point_count"),
    ):
        if type(value) is not int:
            raise TypeError(f"{field_name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{field_name} must be positive.")
    return link_names


def _validate_initial_state(
    initial_qpos: torch.Tensor | Sequence[float],
    initial_qpos_joint_names: Sequence[str],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Validate one explicitly named initial joint state."""
    if isinstance(initial_qpos_joint_names, (str, bytes)) or not isinstance(
        initial_qpos_joint_names,
        Sequence,
    ):
        raise TypeError("initial_qpos_joint_names must be a sequence of names.")
    joint_names = tuple(initial_qpos_joint_names)
    if any(
        type(name) is not str or not name or name != name.strip()
        for name in joint_names
    ):
        raise ValueError(
            "initial_qpos_joint_names must contain non-empty string names."
        )
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("initial_qpos_joint_names must not contain duplicates.")
    qpos = torch.as_tensor(
        initial_qpos,
        dtype=torch.float32,
        device=device,
    )
    if qpos.shape != (len(joint_names),) or not bool(torch.isfinite(qpos).all().item()):
        raise ValueError(
            "initial_qpos must be a finite vector matching its joint-name sequence."
        )
    return qpos, joint_names


def _validate_body_scale(body_scale: torch.Tensor | Sequence[float]) -> None:
    """Require the raw mesh and FK frames to use unit body scale."""
    scale = torch.as_tensor(
        body_scale,
        dtype=torch.float32,
        device="cpu",
    )
    if scale.shape != (3,) or not torch.allclose(
        scale,
        torch.ones(3, dtype=torch.float32),
    ):
        raise ValueError(
            "Initial articulation geometry sampling currently requires unit "
            "body_scale."
        )


def _resolve_joint_geometry(
    provider: ArticulationGeometryProvider,
    *,
    target_link_name: str,
    link_names: list[str],
    initial_link_poses: torch.Tensor,
    target_from_root: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Resolve nearest prismatic/revolute axes and the revolute origin."""
    resolved_axes: dict[str, torch.Tensor] = {}
    revolute_origin: torch.Tensor | None = None
    for joint in provider.get_parent_joint_chain(target_link_name):
        joint_type = joint.joint_type
        if joint_type not in ("prismatic", "revolute") or joint_type in resolved_axes:
            continue
        if joint.parent_link_name not in link_names:
            raise ValueError(
                f"{joint_type.capitalize()} joint {joint.name!r} parent link "
                f"{joint.parent_link_name!r} is not an articulation link."
            )
        joint_origin_pose = torch.as_tensor(
            joint.origin_pose,
            dtype=torch.float32,
            device=device,
        )
        if joint_origin_pose.shape != (4, 4) or not bool(
            torch.isfinite(joint_origin_pose).all().item()
        ):
            raise ValueError(
                f"{joint_type.capitalize()} joint {joint.name!r} origin pose "
                "must be finite with shape (4, 4)."
            )
        joint_axis = torch.as_tensor(
            joint.axis,
            dtype=torch.float32,
            device=device,
        )
        if (
            joint_axis.shape != (3,)
            or not bool(torch.isfinite(joint_axis).all().item())
            or float(torch.linalg.vector_norm(joint_axis).item())
            <= torch.finfo(joint_axis.dtype).eps
        ):
            raise ValueError(
                f"{joint_type.capitalize()} joint {joint.name!r} axis must be "
                "finite and nonzero with shape (3,)."
            )
        parent_index = link_names.index(joint.parent_link_name)
        target_from_joint = target_from_root @ initial_link_poses[parent_index]
        target_from_joint = target_from_joint @ joint_origin_pose
        target_axis = target_from_joint[:3, :3] @ joint_axis
        axis_norm = torch.linalg.vector_norm(target_axis)
        if (
            not bool(torch.isfinite(target_axis).all().item())
            or not bool(torch.isfinite(axis_norm).item())
            or float(axis_norm.item()) <= torch.finfo(target_axis.dtype).eps
        ):
            raise ValueError(
                f"{joint_type.capitalize()} joint {joint.name!r} axis must "
                "transform to a finite, nonzero target-link vector."
            )
        resolved_axes[joint_type] = target_axis / axis_norm
        if joint_type == "revolute":
            revolute_origin = target_from_joint[:3, 3].clone()
        if len(resolved_axes) == 2:
            break
    return (
        resolved_axes.get("prismatic"),
        resolved_axes.get("revolute"),
        revolute_origin,
    )


def _merge_initial_link_meshes(
    provider: ArticulationGeometryProvider,
    *,
    target_link_name: str,
    link_names: list[str],
    initial_link_poses: torch.Tensor,
    target_from_root: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Transform and merge raw link meshes in the target-link frame."""
    merged_vertices: list[torch.Tensor] = []
    merged_triangles: list[torch.Tensor] = []
    vertex_offset = 0
    non_target_vertices: list[torch.Tensor] = []
    non_target_triangles: list[torch.Tensor] = []
    non_target_vertex_offset = 0
    target_vertices: torch.Tensor | None = None
    target_triangles: torch.Tensor | None = None
    for link_index, link_name in enumerate(link_names):
        vertices, triangles = provider.get_link_vert_face(link_name)
        vertices = torch.as_tensor(
            vertices,
            dtype=torch.float32,
            device=device,
        )
        triangles = torch.as_tensor(
            triangles,
            device=device,
        )
        if (
            triangles.dtype == torch.bool
            or triangles.is_floating_point()
            or triangles.is_complex()
        ):
            raise ValueError(f"Link {link_name!r} triangles must use integer indices.")
        triangles = triangles.to(dtype=torch.long)
        _validate_point_cloud_mesh(vertices, triangles, link_name=link_name)
        if vertices.shape[0] == 0:
            continue

        target_from_link = target_from_root @ initial_link_poses[link_index]
        transformed_vertices = (
            vertices @ target_from_link[:3, :3].transpose(0, 1)
            + target_from_link[:3, 3]
        )
        merged_vertices.append(transformed_vertices)
        if triangles.shape[0] > 0:
            merged_triangles.append(triangles + vertex_offset)
        if link_name == target_link_name:
            target_vertices = transformed_vertices
            target_triangles = triangles
        else:
            non_target_vertices.append(transformed_vertices)
            if triangles.shape[0] > 0:
                non_target_triangles.append(triangles + non_target_vertex_offset)
            non_target_vertex_offset += vertices.shape[0]
        vertex_offset += vertices.shape[0]

    if target_vertices is None or target_vertices.shape[0] == 0:
        raise ValueError(
            f"Target link {target_link_name!r} has no point-cloud geometry."
        )
    if not merged_vertices:
        raise ValueError("Articulation has no point-cloud geometry.")
    articulation_vertices = torch.cat(merged_vertices, dim=0)
    articulation_triangles = (
        torch.cat(merged_triangles, dim=0)
        if merged_triangles
        else torch.empty((0, 3), dtype=torch.long, device=device)
    )
    non_target_articulation_vertices = (
        torch.cat(non_target_vertices, dim=0)
        if non_target_vertices
        else torch.empty((0, 3), dtype=articulation_vertices.dtype, device=device)
    )
    non_target_articulation_triangles = (
        torch.cat(non_target_triangles, dim=0)
        if non_target_triangles
        else torch.empty((0, 3), dtype=torch.long, device=device)
    )
    assert target_triangles is not None
    return (
        target_vertices,
        target_triangles,
        articulation_vertices,
        articulation_triangles,
        non_target_articulation_vertices,
        non_target_articulation_triangles,
    )


def _validate_point_cloud_mesh(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    *,
    link_name: str,
) -> None:
    """Validate one link-local mesh used for surface sampling."""
    if vertices.dim() != 2 or vertices.shape[1:] != (3,):
        raise ValueError(f"Link {link_name!r} vertices must have shape (N, 3).")
    if not bool(torch.isfinite(vertices).all().item()):
        raise ValueError(f"Link {link_name!r} vertices must be finite.")
    if triangles.dim() != 2 or triangles.shape[1:] != (3,):
        raise ValueError(f"Link {link_name!r} triangles must have shape (M, 3).")
    if triangles.shape[0] == 0:
        if vertices.shape[0] > 0:
            raise ValueError(
                f"Link {link_name!r} must contain at least one non-degenerate "
                "triangle surface."
            )
        return
    if vertices.shape[0] == 0:
        raise ValueError(
            f"Link {link_name!r} triangles cannot reference an empty mesh."
        )
    if bool((triangles < 0).any().item()) or int(triangles.max().item()) >= len(
        vertices
    ):
        raise ValueError(f"Link {link_name!r} triangles reference invalid vertices.")
    if not bool(_valid_triangle_mask(vertices, triangles).any().item()):
        raise ValueError(
            f"Link {link_name!r} must contain at least one non-degenerate "
            "triangle surface."
        )


def _valid_triangle_mask(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
) -> torch.Tensor:
    """Return the non-degenerate triangle mask used by validation and sampling."""
    face_vertices = vertices[triangles]
    face_areas = 0.5 * torch.linalg.vector_norm(
        torch.cross(
            face_vertices[:, 1] - face_vertices[:, 0],
            face_vertices[:, 2] - face_vertices[:, 0],
            dim=1,
        ),
        dim=1,
    )
    return face_areas > torch.finfo(vertices.dtype).eps


def _sample_mesh_surface_points(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    point_count: int,
) -> torch.Tensor:
    """Uniformly sample a triangle mesh with Open3D's CPU sampler."""
    if type(point_count) is not int:
        raise TypeError("point_count must be an integer.")
    if point_count <= 0:
        raise ValueError("point_count must be positive.")
    if vertices.shape[0] == 0:
        raise ValueError("Cannot sample an empty mesh.")
    if triangles.shape[0] == 0:
        indices = (
            torch.linspace(
                0,
                vertices.shape[0] - 1,
                point_count,
                device=vertices.device,
            )
            .round()
            .to(torch.long)
        )
        return vertices[indices].clone()

    valid_faces = _valid_triangle_mask(vertices, triangles)
    if not bool(valid_faces.any().item()):
        indices = (
            torch.linspace(
                0,
                vertices.shape[0] - 1,
                point_count,
                device=vertices.device,
            )
            .round()
            .to(torch.long)
        )
        return vertices[indices].clone()

    import numpy as np
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(
            vertices.detach().to(device="cpu", dtype=torch.float64).numpy()
        ),
        triangles=o3d.utility.Vector3iVector(
            triangles[valid_faces].detach().to(device="cpu", dtype=torch.int32).numpy()
        ),
    )
    point_cloud = mesh.sample_points_uniformly(number_of_points=point_count)
    sampled_points = np.asarray(point_cloud.points).copy()
    if sampled_points.shape != (point_count, 3):
        raise RuntimeError(
            "Open3D surface sampling returned an unexpected point-cloud shape "
            f"{sampled_points.shape}; expected ({point_count}, 3)."
        )
    return torch.tensor(
        sampled_points,
        device=vertices.device,
        dtype=vertices.dtype,
    )


def _owned_point_cloud(value: object, *, field_name: str) -> torch.Tensor:
    """Validate and clone one floating point cloud."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor.")
    if (
        not value.is_floating_point()
        or value.dim() != 2
        or value.shape[1:] != (3,)
        or value.shape[0] == 0
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(
            f"{field_name} must be a non-empty finite floating tensor with "
            "shape (N, 3)."
        )
    return value.to(dtype=torch.float32).clone()


def _owned_optional_point_cloud(
    value: object | None,
    *,
    field_name: str,
) -> torch.Tensor | None:
    """Validate and clone one optional point cloud that may be empty."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor.")
    if (
        not value.is_floating_point()
        or value.dim() != 2
        or value.shape[1:] != (3,)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(
            f"{field_name} must be a finite floating tensor with shape (N, 3)."
        )
    return value.to(dtype=torch.float32).clone()


def _owned_axis(
    value: object | None,
    *,
    field_name: str,
) -> torch.Tensor | None:
    """Validate, normalize, and clone one optional axis."""
    if value is None:
        return None
    axis = _owned_point(value, field_name=field_name)
    assert axis is not None
    axis_norm = torch.linalg.vector_norm(axis)
    if float(axis_norm.item()) <= 1.0e-6:
        raise ValueError(f"{field_name} must be non-zero.")
    return axis / axis_norm


def _owned_point(
    value: object | None,
    *,
    field_name: str,
) -> torch.Tensor | None:
    """Validate and clone one optional local point."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor.")
    if (
        not value.is_floating_point()
        or value.shape != (3,)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(
            f"{field_name} must be a finite floating tensor with shape (3,)."
        )
    return value.to(dtype=torch.float32).clone()


__all__ = [
    "ArticulationAffordanceGeometry",
    "ArticulationGeometryProvider",
    "ArticulationJointGeometry",
    "create_rigidized_articulation_antipodal_affordance",
    "sample_initial_articulation_geometry",
]
