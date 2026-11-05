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
"""Offline P-03 pose metrics and exact convex-polyhedron SAT contact geometry."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

__all__ = ["relative_pose", "convex_gap", "metrics"]


def relative_pose(tcp: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """Return object pose in the TCP frame, using xyzw quaternions."""
    qt = tcp[..., 3:7].astype(np.float64)
    qo = obj[..., 3:7].astype(np.float64)
    qt /= np.linalg.norm(qt, axis=-1, keepdims=True)
    qo /= np.linalg.norm(qo, axis=-1, keepdims=True)
    v, w = -qt[..., :3], qt[..., 3:4]
    d = obj[..., :3] - tcp[..., :3]
    cross = 2 * np.cross(v, d)
    p = d + w * cross + np.cross(v, cross)
    r = np.concatenate(
        [
            w * qo[..., :3] + qo[..., 3:4] * v + np.cross(v, qo[..., :3]),
            w * qo[..., 3:4] - np.sum(v * qo[..., :3], axis=-1, keepdims=True),
        ],
        axis=-1,
    )
    return np.concatenate([p, r], axis=-1)


def convex_gap(
    a: np.ndarray, b: np.ndarray, pa: np.ndarray, pb: np.ndarray
) -> np.ndarray:
    """Signed SAT gap for convex polyhedra at paired poses.

    Positive means a separating projection exists (not Euclidean distance).
    Negative is the minimum translation distance to remove intersection.
    Face normals of both hulls and all nonparallel edge cross-products are used.
    """
    ha, hb = ConvexHull(a), ConvexHull(b)
    normals, edges = [], []
    for vertices, hull in [(a, ha), (b, hb)]:
        ns = hull.equations[:, :3].copy()
        edge_faces = {}
        for face_id, tri in enumerate(hull.simplices):
            for u, v in zip(tri, np.roll(tri, 1)):
                edge_faces.setdefault(tuple(sorted((u, v))), []).append(face_id)
        true_edges = [
            e
            for e, fs in edge_faces.items()
            if len(fs) == 2 and np.linalg.norm(np.cross(ns[fs[0]], ns[fs[1]])) > 1e-8
        ]
        es = np.array([vertices[u] - vertices[v] for u, v in true_edges])
        es /= np.linalg.norm(es, axis=-1, keepdims=True)
        for vectors in [ns, es]:
            dominant = np.argmax(abs(vectors), axis=1)
            vectors *= np.where(vectors[np.arange(len(vectors)), dominant] < 0, -1, 1)[
                :, None
            ]
        ns = np.unique(np.round(ns, 8), axis=0)
        es = np.unique(np.round(es, 8), axis=0)
        normals.append(ns)
        edges.append(es)
    ra = Rotation.from_quat(pa[:, 3:7]).as_matrix()
    rb = Rotation.from_quat(pb[:, 3:7]).as_matrix()
    va = np.einsum("tij,vj->tvi", ra, a) + pa[:, None, :3]
    vb = np.einsum("tij,vj->tvi", rb, b) + pb[:, None, :3]
    na = np.einsum("tij,vj->tvi", ra, normals[0])
    nb = np.einsum("tij,vj->tvi", rb, normals[1])
    ea = np.einsum("tij,vj->tvi", ra, edges[0])
    eb = np.einsum("tij,vj->tvi", rb, edges[1])
    result = np.full(len(pa), -np.inf)
    for start in range(0, len(pa), 64):
        s = slice(start, start + 64)
        cross = np.cross(ea[s, :, None], eb[s, None]).reshape(len(ra[s]), -1, 3)
        norm = np.linalg.norm(cross, axis=-1, keepdims=True)
        cross = np.divide(cross, norm, out=np.zeros_like(cross), where=norm > 1e-8)
        axes = np.concatenate([na[s], nb[s], cross], axis=1)
        aa = np.einsum("tvi,tai->tav", va[s], axes)
        bb = np.einsum("tvi,tai->tav", vb[s], axes)
        gap = np.maximum(aa.min(-1) - bb.max(-1), bb.min(-1) - aa.max(-1))
        gap[np.linalg.norm(axes, axis=-1) < 0.5] = -np.inf
        result[s] = gap.max(-1)
    return result


def metrics(
    x: np.ndarray, t: np.ndarray, shape: str, fingers: dict | None = None
) -> tuple[dict, dict]:
    """Measure one complete trajectory; preserve transient losses before recovery."""
    x = np.asarray(x, dtype=np.float64)
    obj, tcp = x[:, 39:46], x[:, 18:25]
    rel = relative_pose(tcp, obj)
    anchor = np.argmin(abs(t - 3))
    delta = relative_pose(np.broadcast_to(rel[anchor], rel.shape), rel)
    translation = np.linalg.norm(rel[:, :3] - rel[anchor, :3], axis=1)
    angle = 2 * np.arctan2(np.linalg.norm(delta[:, 3:6], axis=1), abs(delta[:, 6]))
    rotation = Rotation.from_quat(obj[:, 3:7]).as_matrix()
    if shape == "cube":
        support = 0.025 * np.abs(rotation[:, 2]).sum(1)
        vertices = np.array(
            [
                [i, j, k]
                for i in [-0.025, 0.025]
                for j in [-0.025, 0.025]
                for k in [-0.025, 0.025]
            ]
        )
    else:
        z = rotation[:, 2]
        theta = np.arctan2(z[:, 1], z[:, 0])
        nearest = np.round(theta / (2 * np.pi / 64)) * (2 * np.pi / 64)
        support = 0.025 * (
            np.linalg.norm(z[:, :2], axis=1) * np.cos(theta - nearest) + abs(z[:, 2])
        )
        angles = np.arange(64) * 2 * np.pi / 64
        vertices = np.array(
            [
                [0.025 * np.cos(a), 0.025 * np.sin(a), h]
                for a in angles
                for h in [-0.025, 0.025]
            ]
        )
    bottom_gap = obj[:, 2] - support - 0.75
    held = (t >= 3) & (t <= 7)
    hold = (t >= 3) & (t <= 5)
    perturb = (t >= 5) & (t <= 7)
    # 2 mm is the combined 1 mm contact-generation neighborhood, not a lift tolerance.
    clear = bottom_gap > 0.002
    out = dict(
        finite=bool(np.isfinite(x).all()),
        lift_at_3s_mm=float((obj[anchor, 2] - obj[0, 2]) * 1000),
        lift_error_at_3s_mm=float((obj[anchor, 2] - obj[0, 2] - 0.1) * 1000),
        min_bottom_clearance_3_7_mm=float(bottom_gap[held].min() * 1000),
        max_lift_mm=float((obj[:, 2].max() - obj[0, 2]) * 1000),
        airborne_at_3s=bool(clear[anchor]),
        continuous_airborne_3_7=bool(clear[held].all()),
        support_recontact_3_7=bool(clear[anchor] and (~clear[held]).any()),
        table_penetration_max_mm=float(np.maximum(-bottom_gap, 0).max() * 1000),
        return_xy_error_mm=float(np.linalg.norm(obj[-1, :2] - obj[0, :2]) * 1000),
        final_bottom_gap_mm=float(bottom_gap[-1] * 1000),
        final_linear_speed_mm_s=float(np.linalg.norm(x[-1, 46:49]) * 1000),
        final_angular_speed_deg_s=float(np.rad2deg(np.linalg.norm(x[-1, 49:52]))),
        perturb_increment_from_5s_max_mm=float(
            np.linalg.norm(
                rel[perturb, :3] - rel[np.argmin(abs(t - 5)), :3], axis=1
            ).max()
            * 1000
        ),
        mimic_error_hold_max_mm=float(abs(x[held, 7] - x[held, 8]).max() * 1000),
        joint_speed_peak=np.abs(x[:, 9:18]).max(0).tolist(),
        first_clear_s=float(t[np.flatnonzero(clear)[0]]) if clear.any() else None,
        first_support_recontact_3_7_s=(
            float(t[np.flatnonzero(held & ~clear)[0]])
            if clear[anchor] and (~clear[held]).any()
            else None
        ),
    )
    for name, mask in [("hold", hold), ("perturb", perturb)]:
        velocity = np.diff(rel[mask, :3], axis=0) / np.diff(t[mask])[:, None]
        out.update(
            {
                f"{name}_slip_max_mm": float(translation[mask].max() * 1000),
                f"{name}_rotation_max_deg": float(np.rad2deg(angle[mask].max())),
                f"{name}_relative_speed_rms_mm_s": float(
                    np.sqrt(np.mean(np.sum(velocity**2, axis=1))) * 1000
                ),
                f"{name}_relative_position_span_mm": float(
                    np.linalg.norm(np.ptp(rel[mask, :3], axis=0)) * 1000
                ),
            }
        )
    trace = dict(
        t=t,
        bottom_gap_m=bottom_gap,
        rel_pose=rel,
        slip_m=translation,
        rotation_rad=angle,
        lift_m=obj[:, 2] - obj[0, 2],
        finger_error_m=x[:, 7] - x[:, 8],
    )
    if fingers is not None:
        gaps = np.stack(
            [
                convex_gap(
                    fingers[name + "_vertices"], vertices, x[:, span : span + 7], obj
                )
                for name, span in [("panda_leftfinger", 25), ("panda_rightfinger", 32)]
            ],
            axis=1,
        )
        contact = (gaps <= 0.002).all(1)
        penetration = np.maximum(-gaps, 0)
        out.update(
            finger_penetration_max_mm=float(penetration.max() * 1000),
            finger_penetration_p95_hold_mm=float(
                np.percentile(penetration[held], 95) * 1000
            ),
            finger_penetration_duration_s=float(
                (penetration[1:].max(1) > 1e-5).sum() * (t[1] - t[0])
            ),
            bilateral_contact_3_7_fraction=float(contact[held].mean()),
            contact_loss_3_7=bool((~contact[held]).any()),
            first_contact_loss_3_7_s=(
                float(t[np.flatnonzero(held & ~contact)[0]])
                if (~contact[held]).any()
                else None
            ),
            released_final=bool((gaps[-1] > 0.002).all()),
            cycle_completed=bool(
                clear[held].all()
                and contact[held].all()
                and abs(bottom_gap[-1]) <= 0.002
                and (gaps[-1] > 0.002).all()
            ),
        )
        trace["finger_gap_m"] = gaps
    return out, trace
