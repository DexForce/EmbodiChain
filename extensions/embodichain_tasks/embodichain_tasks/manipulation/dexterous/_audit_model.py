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

"""CPU comparison of source and solver-exported manipulation MJCF models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

import mujoco
import numpy as np

__all__ = ["audit_model"]


def _rotation(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def _contact_reason(model: Any, first: int, second: int) -> str:
    """Return the MuJoCo static geom-pair filter, or ``eligible``."""
    if any(
        {int(a), int(b)} == {first, second}
        for a, b in zip(model.pair_geom1, model.pair_geom2)
    ):
        return "explicit_pair"
    b1, b2 = int(model.geom_bodyid[first]), int(model.geom_bodyid[second])

    def has_dynamic_ancestor(body: int) -> bool:
        while body:
            if model.body_dofnum[body]:
                return True
            body = int(model.body_parentid[body])
        return False

    if not has_dynamic_ancestor(b1) and not has_dynamic_ancestor(b2):
        return "static_static"
    w1, w2 = int(model.body_weldid[b1]), int(model.body_weldid[b2])
    if w1 == w2:
        return "same_weld"
    if not (
        int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    ):
        p1 = int(model.body_weldid[model.body_parentid[w1]])
        p2 = int(model.body_weldid[model.body_parentid[w2]])
        if w1 and w2 and (w1 == p2 or w2 == p1):
            return "parent_child"
    if not (
        int(model.geom_contype[first]) & int(model.geom_conaffinity[second])
        or int(model.geom_contype[second]) & int(model.geom_conaffinity[first])
    ):
        return "collision_masks"
    lo, hi = sorted((b1, b2))
    if (lo << 16) + hi in model.exclude_signature:
        return "explicit_exclude"
    return "eligible"


def audit_model(
    source: Any,
    imported: Any,
    joint_map: Mapping[int, int],
    actuator_map: Sequence[int],
    body_map: Mapping[str, int],
    *,
    atol: float = 2e-6,
    rtol: float = 2e-5,
) -> dict[str, Any]:
    """Compare physical arrays after matching source entities to imported ones.

    Args:
        source: Source MuJoCo model with runtime option overrides applied.
        imported: MuJoCo model exported by the active Newton solver.
        joint_map: Source joint index to imported joint index.
        actuator_map: Imported actuator indices in source actuator order.
        body_map: Source body names to imported body indices.
        atol: Absolute numeric tolerance.
        rtol: Relative numeric tolerance.

    Returns:
        JSON-compatible checks, mismatches, collision mapping and option flags.
        This function reports differences without silently changing the model.
    """
    checks: list[dict[str, Any]] = []

    def compare(
        name: str,
        expected: object,
        actual: object,
        *,
        ignored_indices: Sequence[int] = (),
        reason: str | None = None,
    ) -> None:
        a, b = np.asarray(expected), np.asarray(actual)
        same_shape = a.shape == b.shape
        raw_matches = bool(same_shape and np.allclose(a, b, atol=atol, rtol=rtol))
        semantic_a, semantic_b = a.copy(), b.copy()
        if same_shape and ignored_indices:
            semantic_a.flat[list(ignored_indices)] = 0
            semantic_b.flat[list(ignored_indices)] = 0
        difference = (
            float(np.max(np.abs(a.astype(float) - b.astype(float))))
            if same_shape and a.size
            else 0.0
        )
        checks.append(
            {
                "field": name,
                "matches": bool(
                    same_shape
                    and np.allclose(semantic_a, semantic_b, atol=atol, rtol=rtol)
                ),
                "raw_matches": raw_matches,
                "equivalence_reason": (
                    reason if not raw_matches and ignored_indices else None
                ),
                "max_abs_error": difference if same_shape else None,
                "source_shape": list(a.shape),
                "imported_shape": list(b.shape),
                "source": a.tolist() if a.size <= 9 else None,
                "imported": b.tolist() if b.size <= 9 else None,
            }
        )

    for field in (
        "gravity",
        "timestep",
        "integrator",
        "solver",
        "iterations",
        "ls_iterations",
        "ccd_iterations",
        "cone",
        "impratio",
        "tolerance",
        "ls_tolerance",
        "ccd_tolerance",
        "enableflags",
        "disableflags",
    ):
        compare(
            f"option.{field}", getattr(source.opt, field), getattr(imported.opt, field)
        )

    bmap = {0: 0}
    for source_id in range(1, source.nbody):
        name = source.body(source_id).name
        if name not in body_map:
            checks.append(
                {"field": f"body[{name or source_id}].mapping", "matches": False}
            )
            continue
        imported_id = body_map[name]
        bmap[source_id] = imported_id
        for field in ("body_mass", "body_ipos", "body_pos"):
            ignored = (
                (0, 1, 2)
                if (
                    field == "body_ipos"
                    and source.body_mass[source_id] == 0
                    and imported.body_mass[imported_id] == 0
                )
                else ()
            )
            compare(
                f"body[{name}].{field}",
                getattr(source, field)[source_id],
                getattr(imported, field)[imported_id],
                ignored_indices=ignored,
                reason="Zero-mass body center of mass does not enter dynamics.",
            )
        compare(
            f"body[{name}].rotation",
            _rotation(source.body_quat[source_id]),
            _rotation(imported.body_quat[imported_id]),
        )
        # Principal-axis eigenvectors may change sign/order while the tensor agrees.
        ra, rb = _rotation(source.body_iquat[source_id]), _rotation(
            imported.body_iquat[imported_id]
        )
        compare(
            f"body[{name}].inertia_tensor",
            ra @ np.diag(source.body_inertia[source_id]) @ ra.T,
            rb @ np.diag(imported.body_inertia[imported_id]) @ rb.T,
        )

    dof_map = []
    for source_id in range(source.njnt):
        imported_id = joint_map[source_id]
        name = source.joint(source_id).name or str(source_id)
        for field in (
            "jnt_type",
            "jnt_pos",
            "jnt_axis",
            "jnt_limited",
            "jnt_range",
            "jnt_actfrclimited",
            "jnt_actfrcrange",
            "jnt_solref",
            "jnt_solimp",
            "jnt_stiffness",
        ):
            ignored = ()
            if (
                field == "jnt_range"
                and not source.jnt_limited[source_id]
                and not imported.jnt_limited[imported_id]
            ):
                ignored = (0, 1)
            if (
                field == "jnt_actfrcrange"
                and not source.jnt_actfrclimited[source_id]
                and not imported.jnt_actfrclimited[imported_id]
            ):
                ignored = (0, 1)
            compare(
                f"joint[{name}].{field}",
                getattr(source, field)[source_id],
                getattr(imported, field)[imported_id],
                ignored_indices=ignored,
                reason="Range is inactive because both limit flags are disabled.",
            )
        width = (6, 3, 1, 1)[int(source.jnt_type[source_id])]
        sa, ia = int(source.jnt_dofadr[source_id]), int(
            imported.jnt_dofadr[imported_id]
        )
        dof_map.extend(range(ia, ia + width))
        for field in ("dof_damping", "dof_armature", "dof_frictionloss"):
            compare(
                f"joint[{name}].{field}",
                getattr(source, field)[sa : sa + width],
                getattr(imported, field)[ia : ia + width],
            )
    amap = np.asarray(actuator_map, dtype=int)
    for field in (
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_gear",
        "actuator_ctrlrange",
        "actuator_ctrllimited",
        "actuator_forcerange",
        "actuator_forcelimited",
        "actuator_trntype",
        "actuator_dyntype",
        "actuator_gaintype",
        "actuator_biastype",
    ):
        compare(field, getattr(source, field), getattr(imported, field)[amap])

    # Verify collision geoms by name first; source unnamed geoms use body-local
    # type and position. Visual-only removals are expected from parse_visuals=False.
    collision_map: dict[int, int] = {}
    source_collision = np.flatnonzero(source.geom_contype | source.geom_conaffinity)
    imported_collision = np.flatnonzero(
        imported.geom_contype | imported.geom_conaffinity
    )
    used: set[int] = set()
    for sid in source_collision:
        name = source.geom(int(sid)).name
        candidates = [
            int(i)
            for i in imported_collision
            if int(i) not in used
            and int(imported.geom_bodyid[i])
            == bmap.get(int(source.geom_bodyid[sid]), -1)
        ]
        named = [
            i
            for i in candidates
            if name
            and (
                imported.geom(i).name == name
                or imported.geom(i).name.endswith("_" + name)
                or re.sub(r"_\d+$", "", imported.geom(i).name).split("/")[-1] == name
                or re.sub(r"_\d+$", "", imported.geom(i).name).endswith("_" + name)
            )
        ]
        if len(named) == 1:
            target = named[0]
        else:
            geometric = [
                i
                for i in candidates
                if imported.geom_type[i] == source.geom_type[sid]
                and np.allclose(
                    imported.geom_pos[i], source.geom_pos[sid], atol=atol, rtol=rtol
                )
            ]
            if len(geometric) != 1:
                checks.append(
                    {
                        "field": f"geom[{name or sid}].mapping",
                        "matches": False,
                        "candidates": candidates,
                    }
                )
                continue
            target = geometric[0]
        used.add(target)
        collision_map[int(sid)] = target
        is_mesh = int(source.geom_type[sid]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        for field in (
            "geom_type",
            "geom_pos",
            "geom_size",
            "geom_friction",
            "geom_condim",
            "geom_solref",
            "geom_solimp",
            "geom_margin",
            "geom_gap",
            "geom_priority",
            "geom_solmix",
        ):
            ignored = ()
            reason = None
            if is_mesh and field in ("geom_pos", "geom_size"):
                ignored = (0, 1, 2)
                reason = (
                    "Mesh principal frame may change; body-frame vertices are checked."
                )
            if field == "geom_size":
                kind = int(source.geom_type[sid])
                if kind == int(mujoco.mjtGeom.mjGEOM_PLANE):
                    ignored, reason = (
                        0,
                        1,
                        2,
                    ), "Plane size only controls visualization."
                elif kind in (
                    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                ):
                    ignored, reason = (
                        2,
                    ), "Capsule/cylinder collision uses radius and half-length."
                elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    ignored, reason = (1, 2), "Sphere collision uses only radius."
            compare(
                f"geom[{name or sid}].{field}",
                getattr(source, field)[sid],
                getattr(imported, field)[target],
                ignored_indices=ignored,
                reason=reason,
            )
        compare(
            f"geom[{name or sid}].rotation",
            _rotation(source.geom_quat[sid]),
            _rotation(imported.geom_quat[target]),
            ignored_indices=tuple(range(9)) if is_mesh else (),
            reason="Mesh principal frame may change; body-frame vertices are checked.",
        )
        if is_mesh:
            source_mesh, imported_mesh = int(source.geom_dataid[sid]), int(
                imported.geom_dataid[target]
            )
            source_start, imported_start = (
                source.mesh_vertadr[source_mesh],
                imported.mesh_vertadr[imported_mesh],
            )
            va = source.mesh_vert[
                source_start : source_start + source.mesh_vertnum[source_mesh]
            ]
            vb = imported.mesh_vert[
                imported_start : imported_start + imported.mesh_vertnum[imported_mesh]
            ]
            va = va @ _rotation(source.geom_quat[sid]).T + source.geom_pos[sid]
            vb = (
                vb @ _rotation(imported.geom_quat[target]).T + imported.geom_pos[target]
            )
            compare(
                f"geom[{name or sid}].body_frame_mesh_bounds",
                np.stack((va.min(0), va.max(0))),
                np.stack((vb.min(0), vb.max(0))),
            )
            # Newton can recenter/reorder the same convex mesh. Compare both
            # point sets in the body frame without requiring matching indices.
            distances = np.linalg.norm(va[:, None] - vb[None, :], axis=-1)
            vertex_error = max(
                float(distances.min(0).max()), float(distances.min(1).max())
            )
            checks.append(
                {
                    "field": f"geom[{name or sid}].body_frame_mesh_vertices",
                    "matches": vertex_error <= atol,
                    "max_abs_error": vertex_error,
                    "source_vertex_count": len(va),
                    "imported_vertex_count": len(vb),
                }
            )
    compare("collision_geom_count", len(source_collision), len(imported_collision))
    source_excludes = {
        tuple(
            sorted(
                (
                    bmap.get(int(signature) >> 16, -1),
                    bmap.get(int(signature) & 0xFFFF, -1),
                )
            )
        )
        for signature in source.exclude_signature
    }
    imported_excludes = {
        tuple(sorted((int(signature) >> 16, int(signature) & 0xFFFF)))
        for signature in imported.exclude_signature
    }
    pair_differences = []
    mapped = sorted(collision_map)
    for offset, first in enumerate(mapped):
        for second in mapped[offset + 1 :]:
            expected = _contact_reason(source, first, second)
            actual = _contact_reason(
                imported, collision_map[first], collision_map[second]
            )
            if (expected in ("eligible", "explicit_pair")) != (
                actual in ("eligible", "explicit_pair")
            ):
                pair_differences.append(
                    {
                        "source_geoms": [
                            source.geom(first).name or first,
                            source.geom(second).name or second,
                        ],
                        "source_filter": expected,
                        "imported_filter": actual,
                    }
                )
    inverse_bmap = {v: k for k, v in bmap.items()}
    extra_excludes = []
    for first, second in sorted(imported_excludes - source_excludes):
        sa, sb = inverse_bmap.get(first), inverse_bmap.get(second)
        reasons: dict[str, int] = {}
        if sa is not None and sb is not None:
            for ga in np.flatnonzero(source.geom_bodyid == sa):
                for gb in np.flatnonzero(source.geom_bodyid == sb):
                    why = _contact_reason(source, int(ga), int(gb))
                    reasons[why] = reasons.get(why, 0) + 1
        extra_excludes.append(
            {
                "imported_bodies": [first, second],
                "source_bodies": [
                    source.body(sa).name if sa is not None else None,
                    source.body(sb).name if sb is not None else None,
                ],
                "source_geom_pair_filters": reasons,
                "redundant": sa is not None
                and sb is not None
                and not reasons.get("eligible", 0),
            }
        )
    checks.append(
        {
            "field": "contact_excludes",
            "matches": not pair_differences
            and len(collision_map) == len(source_collision),
            "raw_matches": source_excludes == imported_excludes,
            "missing_source_pairs": sorted(source_excludes - imported_excludes),
            "additional_imported_pairs": sorted(imported_excludes - source_excludes),
            "additional_imported_pair_details": extra_excludes,
            "effective_geom_pair_differences": pair_differences,
            "equivalence_reason": "Static geom-pair eligibility includes explicit pairs, welds, parent filtering, masks and excludes.",
        }
    )
    mismatches = [item for item in checks if not item["matches"]]
    return {
        "status": "passed" if not mismatches else "differences",
        "atol": atol,
        "rtol": rtol,
        "check_count": len(checks),
        "mismatches": mismatches,
        "checks": checks,
        "equivalent_differences": [
            item
            for item in checks
            if item["matches"] and not item.get("raw_matches", True)
        ],
        "collision_geom_map": collision_map,
        "source_disableflags": int(source.opt.disableflags),
        "imported_disableflags": int(imported.opt.disableflags),
    }
