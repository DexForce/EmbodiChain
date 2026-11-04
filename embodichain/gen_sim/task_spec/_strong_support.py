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

"""Revision-one support and full-footprint proof checks on owned snapshots."""

from __future__ import annotations

from collections.abc import Mapping
import math
from types import SimpleNamespace
from typing import Any

from ._support_contract import support_window_flags

__all__: list[str] = []


def _bool_rows(value: Any, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(type(v) is bool for v in value)
    )


def _condition_values(value: Mapping[str, Any]) -> SimpleNamespace:
    """Validate the source condition's scalar/vector declarations for re-evaluation."""
    defaults = dict(
        duration=3.0,
        timeout=12.0,
        reference=None,
        local_axis=None,
        reference_axis=None,
        displacement=None,
        target_position=None,
        reference_half_extents=None,
        object_bottom=0.0,
        reference_top=0.0,
        position_tolerance=0.04,
        minimum_alignment=0.8660254037844386,
        axis_direction_free=False,
        support_tolerance=0.01,
        translation_drift=0.02,
        rotation_drift=math.pi / 18.0,
        motion_parts=(),
        world_axis=None,
        support_contract_revision=None,
        com_tolerance=None,
    )
    if set(value) - (set(defaults) | {"entity", "kind"}):
        raise ValueError("Support condition has unknown fields.")
    data = {**defaults, **value}
    if (
        data.get("kind") not in {"stack", "supported_placement", "supported_region"}
        or type(data["support_contract_revision"]) is not int
        or data["support_contract_revision"] != 1
    ):
        raise ValueError(
            "Strong support requires its declared condition kind/revision."
        )
    for name in ("entity", "reference"):
        if (
            type(data.get(name)) is not str
            or not data[name]
            or data[name] != data[name].strip()
        ):
            raise ValueError("Strong support endpoints must be exact identifiers.")
    for name in (
        "duration",
        "timeout",
        "position_tolerance",
        "support_tolerance",
        "translation_drift",
        "rotation_drift",
    ):
        if (
            type(data[name]) not in (int, float)
            or not math.isfinite(data[name])
            or data[name] <= 0
        ):
            raise ValueError(f"Strong support {name} must be finite and positive.")
    if data["timeout"] < data["duration"]:
        raise ValueError("Support timeout cannot precede its duration.")
    for name in ("object_bottom", "reference_top", "minimum_alignment"):
        if type(data[name]) not in (int, float) or not math.isfinite(data[name]):
            raise ValueError(f"Strong support {name} must be finite.")
    if not 0 <= data["minimum_alignment"] <= 1:
        raise ValueError("Support minimum_alignment must be in [0,1].")
    if type(data["axis_direction_free"]) is not bool or (
        data["axis_direction_free"] and data["local_axis"] is None
    ):
        raise ValueError("Support axis freedom needs a declared axis.")
    if data["com_tolerance"] is not None and (
        type(data["com_tolerance"]) not in (int, float)
        or not math.isfinite(data["com_tolerance"])
        or data["com_tolerance"] <= 0
    ):
        raise ValueError("Support COM tolerance must be finite and positive.")
    for name in (
        "local_axis",
        "world_axis",
        "reference_axis",
        "displacement",
        "target_position",
        "reference_half_extents",
    ):
        vector = data[name]
        if vector is None:
            continue
        size = 2 if name == "reference_half_extents" else 3
        if (
            not isinstance(vector, (list, tuple))
            or len(vector) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
        ):
            raise ValueError(f"Support {name} has invalid dimensions or values.")
        vector = tuple(float(v) for v in vector)
        if name.endswith("axis"):
            norm = math.sqrt(sum(v * v for v in vector))
            if norm <= 1e-8 or not math.isfinite(norm):
                raise ValueError("Support axis must be finite and nonzero.")
            vector = tuple(v / norm for v in vector)
        if name == "reference_half_extents" and min(vector) <= 0:
            raise ValueError("Support extents must be positive.")
        data[name] = vector
    if data["world_axis"] is not None:
        raise ValueError(
            "The source strong support kinds do not declare a world-axis hold."
        )
    if data["target_position"] is not None and data["displacement"] is not None:
        raise ValueError("Support cannot combine absolute and relative targets.")
    if data["kind"] == "supported_placement" and data["displacement"] is None:
        raise ValueError("Supported placement needs its relative target.")
    if data["kind"] == "stack" and any(
        data[name] is None
        for name in ("reference_half_extents", "local_axis", "reference_axis")
    ):
        raise ValueError("Stack support needs both axes and support extents.")
    if not isinstance(data["motion_parts"], (list, tuple)) or any(
        type(part) is not str or not part or part != part.strip()
        for part in data["motion_parts"]
    ):
        raise ValueError(
            "Support motion_parts must contain exact control-part identifiers."
        )
    return SimpleNamespace(**data)


def _constraint_axis_alignment(cfg: Any, pose: torch.Tensor) -> torch.Tensor:
    world = pose.new_tensor(cfg.world_axis or (0.0, 0.0, 1.0))
    alignment = ((pose[:, :3, :3] @ pose.new_tensor(cfg.local_axis)) * world).sum(-1)
    return alignment.abs() if cfg.axis_direction_free else alignment


def _constraint_pose_flags(
    cfg: Any,
    pose: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Recheck the unchanged static targets independently of a past stable mask."""
    import torch

    valid = torch.isfinite(pose).all(dim=-1).all(dim=-1) & torch.isfinite(
        reference
    ).all(dim=-1).all(dim=-1)
    if cfg.local_axis is not None:
        valid &= _constraint_axis_alignment(cfg, pose) >= cfg.minimum_alignment
    target = None
    if cfg.target_position is not None:
        target = pose.new_tensor(cfg.target_position)
    elif cfg.displacement is not None:
        target = reference[:, :3, 3] + pose.new_tensor(cfg.displacement)
    if target is not None:
        valid &= (
            torch.linalg.vector_norm(pose[:, :3, 3] - target, dim=-1)
            <= cfg.position_tolerance
        )
    if cfg.kind == "stack":
        valid &= (reference[:, :3, :3] @ pose.new_tensor(cfg.reference_axis))[
            :, 2
        ] >= cfg.minimum_alignment
        gap = pose[:, 2, 3] + cfg.object_bottom - reference[:, 2, 3] - cfg.reference_top
        valid &= gap.abs() <= cfg.support_tolerance
        valid &= (
            (pose[:, :2, 3] - reference[:, :2, 3]).abs()
            <= pose.new_tensor(cfg.reference_half_extents)
        ).all(-1)
    return valid


def _region_poses_match(
    measurements: Mapping[str, Any],
    object_pose: torch.Tensor,
    reference_pose: torch.Tensor,
    num_envs: int,
) -> bool:
    import torch
    from embodichain.compute.geometry.rigid import (
        validate_rigid_pose_batch as _valid_pose,
    )

    try:
        observed = torch.tensor(measurements["object_pose"]).to(object_pose)
        reference = torch.tensor(measurements["reference_pose"]).to(reference_pose)
        _valid_pose(observed, num_envs)
        _valid_pose(reference, num_envs)
        return bool(
            torch.allclose(observed, object_pose, atol=1e-4, rtol=0)
            and torch.allclose(reference, reference_pose, atol=1e-4, rtol=0)
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False


def _strong_support_trace(
    boundary: Mapping[str, Any],
    cfg: Mapping[str, Any],
    preset: str,
    uid: str,
    reference: str,
    num_envs: int,
) -> Mapping[str, Any]:
    """Resolve the declared terminal policy without borrowing another occurrence."""
    prefix = boundary["metadata"]["program_segment_source_path"]
    traces = [
        p
        for p in boundary["metadata"].get("post_policies", [])
        if p.get("kind") == "wait_stable"
        and type(p.get("policy_index")) is int
        and p.get("source_path") == [*prefix, "post", p["policy_index"]]
        and p.get("result", {}).get("preset") == preset
    ]
    if len(traces) != 1:
        raise ValueError("Strong support lacks its unique terminal policy.")
    trace, result = traces[0], traces[0]["result"]
    native = result.get("measurements", {}).get("native_support", {})
    duration = cfg.get("duration", 3.0)
    seconds = result.get("stable_seconds", [])
    start, end = result.get("start_step"), result.get("end_step")
    started, ended, dt = (
        result.get("clock_started_at"),
        result.get("clock_ended_at"),
        native.get("step_dt"),
    )
    if (
        result.get("kind") != cfg["kind"]
        or result.get("support_contract_revision") != 1
        or result.get("entity") != uid
        or result.get("reference") != reference
        or result.get("required_seconds") != duration
        or any(
            not _bool_rows(v, num_envs) or not all(v)
            for v in (trace.get("result_mask"), result.get("accepted_mask"))
        )
        or not isinstance(seconds, list)
        or len(seconds) != num_envs
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < duration
            for v in seconds
        )
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start <= end
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in (started, ended, dt)
        )
        or dt == 0
        or not math.isclose(started, start * dt, rel_tol=0, abs_tol=1e-8)
        or not math.isclose(ended, end * dt, rel_tol=0, abs_tol=1e-8)
        or ended - started + 1e-8 < duration
        or any(v > ended - started + 1e-8 for v in seconds)
        or native.get("observed_step") != end
        or native.get("observed_at") != ended
        or boundary.get("end_steps") != [end] * num_envs
    ):
        raise ValueError("Strong support policy lacks its exact stable clock/boundary.")
    return result


def _strong_native_loads(native: Mapping[str, Any], num_envs: int) -> list[bool]:
    """Recheck the actual pair and load rather than trusting derived booleans."""
    import numpy as np
    from embodichain.compute.geometry.contact_support import support_from_contacts

    result = []
    for row, pair in zip(native["rows"], native["actor_pairs"], strict=True):
        raw = row.get("raw_contacts", {})
        pairs = np.asarray(raw.get("user_ids", []))
        points, normals = (
            np.asarray(raw.get(key, [])) for key in ("position", "normal")
        )
        impulses = np.asarray(raw.get("impulse", []))
        center = np.asarray(row.get("center_of_mass_world"))
        tolerance = row.get("com_tolerance_m")
        if (
            pairs.ndim != 2
            or pairs.shape[1] != 2
            or pairs.dtype.kind not in "iu"
            or points.shape != (len(pairs), 3)
            or normals.shape != points.shape
            or impulses.shape != (len(pairs),)
            or center.shape != (3,)
            or not np.isfinite(center).all()
            or type(tolerance) not in (int, float)
            or not math.isfinite(tolerance)
            or not 0 < tolerance <= 0.01
        ):
            raise ValueError(
                "Strong native support lacks its original contact payload."
            )
        accepted, recomputed = support_from_contacts(
            pairs,
            points,
            normals,
            impulses,
            center,
            object_actor=pair["object_actor"],
            reference_actor=pair["reference_actor"],
            tolerance=tolerance,
        )
        if row.get("contacts_world") != recomputed["contacts_world"] or any(
            type(row.get(key)) not in (int, float)
            or type(recomputed[key]) not in (int, float)
            or not math.isclose(row[key], recomputed[key], rel_tol=1e-6, abs_tol=1e-12)
            for key in ("com_patch_distance_m", "upward_impulse")
        ):
            raise ValueError(
                "Strong native support derived patch differs from raw contacts."
            )
        result.append(accepted)
    if len(result) != num_envs:
        raise ValueError("Strong native support differs from the environment batch.")
    return result


def _region_flags(
    native: Mapping[str, Any],
    landing: Mapping[str, Any],
    pose: torch.Tensor,
    reference_pose: torch.Tensor,
    uid: str,
    contacts: list[Any],
    num_envs: int,
    *,
    reference_uid: str,
) -> list[bool]:
    """Conjoin exact native load and the complete measured planar footprint."""
    import torch
    from embodichain.compute.geometry.contact_support import (
        contact_support_heights as _contact_support_heights,
    )

    SUPPORT_TOLERANCE_M, PENETRATION_TOLERANCE_M = 0.05, 0.01

    def _support_gap_valid(value: float) -> bool:
        return -PENETRATION_TOLERANCE_M <= value <= SUPPORT_TOLERANCE_M

    supported = native.get("supported", [])
    footprints = landing.get("full_footprint_inside", [])
    gaps, areas = landing.get("support_gap_m", []), landing.get("footprint_area_m2", [])
    if (
        native.get("scope") != "native_contact_support"
        or native.get("entity") != uid
        or native.get("reference") != reference_uid
        or native.get("env_ids") != list(range(num_envs))
        or any(type(v) is not int for v in native.get("env_ids", []))
        or not _region_poses_match(native, pose, reference_pose, num_envs)
        or not _region_poses_match(landing, pose, reference_pose, num_envs)
        or landing.get("scope") != "measured_planar_landing_region"
        or landing.get("support_tolerance_m") != SUPPORT_TOLERANCE_M
        or landing.get("penetration_tolerance_m") != PENETRATION_TOLERANCE_M
        or not all(_bool_rows(v, num_envs) for v in (supported, footprints))
        or any(
            not isinstance(values, list)
            or len(values) != num_envs
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            for values in (gaps, areas)
        )
        or any(v <= 0 for v in areas)
        or not isinstance(contacts, list)
        or len(contacts) != num_envs
    ):
        raise ValueError(
            "Region placement lacks exact contact/planar geometry evidence."
        )
    qualified = torch.isfinite(
        _contact_support_heights(
            torch.tensor(supported, dtype=torch.bool), native, pose[:, 2, 3]
        )
    ).tolist()
    return [
        bool(
            s
            and q
            and f
            and _support_gap_valid(g)
            and type(c) in (int, float)
            and math.isfinite(c)
            and abs(c) <= 0.01
        )
        for s, q, f, g, c in zip(
            supported, qualified, footprints, gaps, contacts, strict=True
        )
    ]


def _strong_support_flags(
    native: Mapping[str, Any],
    landing: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
    window: Mapping[str, Any],
    trace: Mapping[str, Any],
    uid: str,
    reference: str,
    num_envs: int,
    contacts: list[Any] | None = None,
) -> list[bool]:
    import torch
    from embodichain.compute.geometry.rigid import (
        validate_rigid_pose_batch as _valid_pose,
    )
    from embodichain.compute.geometry.contact_support import (
        contact_support_heights as _contact_support_heights,
    )

    pose, ref_pose = torch.tensor(native["object_pose"]), torch.tensor(
        native["reference_pose"]
    )
    _valid_pose(pose, num_envs)
    _valid_pose(ref_pose, num_envs)
    flags = support_window_flags(
        native,
        window,
        num_envs,
        entity=uid,
        reference=reference,
        native_entity=uid,
        native_reference=reference,
    )
    origin = trace["measurements"]["native_support"]
    original_pose, original_ref = torch.tensor(origin["object_pose"]), torch.tensor(
        origin["reference_pose"]
    )
    _valid_pose(original_pose, num_envs)
    _valid_pose(original_ref, num_envs)
    constraint = _condition_values(cfg)
    if not _constraint_pose_flags(constraint, original_pose, original_ref).all():
        raise ValueError(
            "Strong history violates its original placement target or axes."
        )
    static_flags = _constraint_pose_flags(constraint, pose, ref_pose).tolist()
    flags = [a and b for a, b in zip(flags, static_flags, strict=True)]
    if (
        native.get("episode_generations") != origin.get("episode_generations")
        or native.get("actor_pairs") != origin.get("actor_pairs")
        or trace["clock_started_at"] + 1e-8 < window["completed_at"]
    ):
        raise ValueError("Strong support differs from the release episode or actors.")
    source_flags = support_window_flags(
        origin,
        window,
        num_envs,
        entity=uid,
        reference=reference,
        native_entity=uid,
        native_reference=reference,
    )
    source_supported = origin.get("supported", [])
    if (
        not _bool_rows(source_supported, num_envs)
        or not all(source_supported)
        or not all(source_flags)
        or not all(_strong_native_loads(origin, num_envs))
    ):
        raise ValueError("Strong stability lacks its own qualified native sample.")
    if not torch.isfinite(
        _contact_support_heights(
            torch.tensor(source_supported, dtype=torch.bool), origin, pose[:, 2, 3]
        )
    ).all():
        raise ValueError("Strong stability lacks its own qualified load patch.")
    supported = native.get("supported", [])
    if not _bool_rows(supported, num_envs):
        raise ValueError("Strong support lacks measured load flags.")
    loads = torch.isfinite(
        _contact_support_heights(
            torch.tensor(supported, dtype=torch.bool), native, pose[:, 2, 3]
        )
    ).tolist()
    raw_loads = _strong_native_loads(native, num_envs)
    for row, item in enumerate(native["rows"]):
        sample = item.get("sample")
        flags[row] &= bool(
            supported[row] and loads[row] and raw_loads[row] and sample is not None
        )
    if cfg["kind"] == "supported_region":
        if not isinstance(landing, Mapping):
            raise ValueError(
                "Strong region support lacks its complete landing geometry."
            )
        original_pose, original_ref = torch.tensor(origin["object_pose"]), torch.tensor(
            origin["reference_pose"]
        )
        if not all(
            _region_flags(
                origin,
                trace["measurements"].get("landing_region", {}),
                original_pose,
                original_ref,
                uid,
                trace["measurements"].get("support_gap", []),
                num_envs,
                reference_uid=reference,
            )
        ):
            raise ValueError("Strong stability lacks its own complete planar support.")
        # The collector computes this from the same native object mesh and patch.
        physical = _region_flags(
            native,
            landing,
            pose,
            ref_pose,
            uid,
            contacts or [],
            num_envs,
            reference_uid=reference,
        )
        flags = [a and b for a, b in zip(flags, physical, strict=True)]
    return flags
