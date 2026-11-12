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
"""Scene-conditioned joint residuals preserving the original demonstration clock.

This is the existing six-axis/paired-jaw, 30 Hz method, extracted from the
example. Object pose hypotheses associate later contacts; they never drive a
simulator. Use the original robot, not cross-robot retargeting or a task planner.
"""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d, label
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .inputs import ReconstructionInput, RegisteredScene, RefinedTrajectory
from ._kinematics import Robot
from ._contact_fit import fit_grasp, _support_height as _support_at_release

__all__ = ["refine_trajectory", "detect_gripper_events"]


def _clock_gaps(inputs: ReconstructionInput) -> list[int]:
    """Accept a nominal 30 Hz recording with isolated missing samples, without resampling."""
    demo = inputs.demonstration
    intervals = np.diff(demo.timestamps)
    tolerance = 5e-4 if inputs.robot.gripper_calibrations else 1e-6
    steps = np.rint(intervals * 30)
    if (
        demo.time_unit != "s"
        or not np.isclose(np.median(intervals), 1 / 30, atol=tolerance, rtol=0)
        or not np.isin(steps, [1, 2]).all()
        or not np.allclose(intervals, steps / 30, atol=tolerance, rtol=0)
        or np.any((steps[:-1] == 2) & (steps[1:] == 2))
    ):
        raise ValueError(
            "The event method requires nominal 30 Hz observations; only isolated "
            "one-sample gaps are supported"
        )
    return np.flatnonzero(steps == 2).tolist()


def _container_release_lift(
    object_vertices: np.ndarray, finger_vertices: np.ndarray, rim_z: float
) -> float:
    """Keep the held geometry above the opening instead of pushing to its floor."""
    lowest = min(float(object_vertices[:, 2].min()), float(finger_vertices[:, 2].min()))
    return max(0.0, rim_z + 0.003 - lowest)


def _inside_receiver(category: str, grounded: dict | None) -> bool:
    """Prefer observed placement semantics over generated asset category names."""
    if grounded is not None and "receiver_relation" in grounded:
        return grounded["receiver_relation"] == "inside"
    # Compatibility for callers with older, explicitly supplied grounding.
    return category in {
        "bowl",
        "cup",
        "basket",
        "pot",
        "saucepan",
        "steamer",
        "container",
        "food_container",
        "pen_holder",
        "pencil_holder",
        "desk_organizer",
        "bin",
    }


def _approach_opening_profile(
    original: np.ndarray,
    adjusted: np.ndarray,
    close_start: int,
    grasp: int,
    fitted_half: float,
    limit: float,
    earliest: int = 0,
) -> tuple[np.ndarray, dict]:
    """Open before approaching the fitted object, retaining the source closure shape."""
    # The grip offset can outlast a rapid source release. Bound that residual
    # here so saved joint references match what native playback can execute.
    result = np.minimum(adjusted, limit)
    clamped = np.flatnonzero(adjusted > limit).tolist()
    # The arm residual has reached the fitted approach two seconds before
    # source closure. Opening relative to close_start alone can be too late:
    # the fingers may already have pushed the reconstructed object away.
    ready = min(close_start, max(earliest + 1, close_start - 60))
    start = max(earliest, ready - 30, 0)
    ids = np.arange(start, close_start + 1)
    opening = min(
        limit, max(float(original[start : grasp + 1].max()), fitted_half + 0.01)
    )
    ramp = np.clip((ids - start) / (ready - start), 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)
    result[ids] = np.minimum(
        limit,
        np.maximum(result[ids], original[start] + ramp * (opening - original[start])),
    )
    ids = np.arange(close_start, grasp + 1)
    observed = original[ids]
    fraction = np.clip(
        (observed - observed[-1]) / max(float(observed.max() - observed[-1]), 1e-6),
        0,
        1,
    )
    result[ids] = fitted_half + fraction * (opening - fitted_half)
    return result, dict(
        start_frame=start,
        ready_frame=ready,
        half_open_m=opening,
        limit_clamped_frames=clamped,
    )


def detect_gripper_events(
    q: np.ndarray, *, require_open_transition: bool = True
) -> list[dict]:
    """Detect close/open episodes using observed jaw change, excluding final reset.

    Args:
        q: One/two eight-column arm/jaw groups in metres and radians.
        require_open_transition: Require a recorded opening before closure. Set
            false only for a source known to begin with already-open jaws.

    Returns:
        Ordered observed close/grasp/release/open sample intervals.
    """
    result = []
    if q.ndim != 2 or q.shape[1] not in {8, 16}:
        raise ValueError("Expected one or two six-joint arms with paired jaw joints")
    for side, col in enumerate(range(6, q.shape[1], 8)):
        x = gaussian_filter1d(q[:, col], 3)
        segments = {}
        for sign in [-1, 1]:
            active, n = label(np.gradient(x) * sign > 0.00008)
            found = []
            for j in range(1, n + 1):
                indices = np.flatnonzero(active == j)
                a, b = int(indices[0]), int(indices[-1])
                if abs(x[b] - x[a]) > 0.004:
                    found.append((a, b))
            segments[sign] = found
        for a, b in segments[-1]:
            following = [(c, d) for c, d in segments[1] if c > b]
            preceding = [(c, d) for c, d in segments[1] if d < a]
            if not following or (require_open_transition and not preceding):
                continue
            c, d = following[0]
            if c - b < 10:
                continue
            result.append(
                dict(
                    side=side,
                    close_start=a,
                    grasp=b - 4,
                    release=(c + d) // 2,
                    open_end=d,
                )
            )
    # Several tightening commands can precede one release. Only the final
    # closure establishes contact; earlier approach closures are not new grasps.
    holds = {}
    for event in result:
        key = (event["side"], event["release"])
        if key not in holds or event["grasp"] > holds[key]["grasp"]:
            holds[key] = event
    return sorted(holds.values(), key=lambda e: e["grasp"])


def _ik(
    robot: Robot, source: np.ndarray, side: int, target: np.ndarray
) -> tuple[np.ndarray, dict]:
    cols = robot.arm_columns[side]
    initial = source[cols]

    def residual(x: np.ndarray) -> np.ndarray:
        q = source.copy()
        q[cols] = x
        actual = robot.tcp(q[None], side)[0]
        return np.r_[
            (actual[:3, 3] - target[:3, 3]) / 0.003,
            Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec() / 0.035,
            (x - initial) / 0.5,
        ]

    fit = least_squares(
        residual, initial, bounds=(initial - 0.65, initial + 0.65), max_nfev=80
    )
    result = source.copy()
    result[cols] = fit.x
    actual = robot.tcp(result[None], side)[0]
    return result, dict(
        tcp_error_m=float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
        max_joint_residual_rad=float(np.max(abs(fit.x - initial))),
        orientation_error_deg=float(
            np.rad2deg(
                Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
            )
        ),
    )


def _warp(
    robot: Robot,
    q: np.ndarray,
    original: np.ndarray,
    side: int,
    grasp: int,
    release: int,
    start: int,
    end: int,
    grasp_q: np.ndarray,
    release_q: np.ndarray,
    approach: int | None = None,
) -> None:
    cols = robot.arm_columns[side]
    knots = np.array([max(0, start), grasp, release, min(len(q) - 1, end)])
    if not np.all(np.diff(knots) > 0):
        raise ValueError("Invalid event warp ordering")
    residual = np.stack(
        [
            np.zeros(len(cols)),
            grasp_q[cols] - original[grasp, cols],
            release_q[cols] - original[release, cols],
            np.zeros(len(cols)),
        ]
    )
    if approach is not None and knots[0] < approach < grasp:
        knots = np.insert(knots, 1, approach)
        residual = np.insert(residual, 1, residual[1], axis=0)
    ids = np.arange(knots[0], knots[-1] + 1)
    q[np.ix_(ids, cols)] = original[np.ix_(ids, cols)] + PchipInterpolator(
        knots, residual
    )(ids)


def _demonstration_events(
    inputs: ReconstructionInput, robot: Robot | None = None
) -> list[dict]:
    """Use the same calibrated aperture clock for grounding and refinement."""
    original = inputs.demonstration.qpos
    if not inputs.robot.gripper_calibrations:
        return detect_gripper_events(original)
    robot = robot if robot is not None else Robot(inputs.robot)
    event_q = np.zeros((len(original), len(robot.prefixes) * 8))
    for arm in range(len(robot.prefixes)):
        event_q[:, arm * 8 + 6 : arm * 8 + 8] = robot.half_open(original, arm)[:, None]
    return detect_gripper_events(event_q, require_open_transition=False)


def refine_trajectory(
    inputs: ReconstructionInput,
    registered: RegisteredScene,
    *,
    approach_lead_frames: int = 120,
    gripper_preload_m: float = 0.002,
    grounding: tuple[dict, ...] = (),
) -> RefinedTrajectory:
    """Fit local anchors and smooth residuals without modifying source observations.

    Args:
        inputs: Dataset-neutral source and matching robot.
        registered: Generated geometry in robot-base metres, with table plane.
        approach_lead_frames: Extra samples for introducing the grasp correction.
        gripper_preload_m: Per-jaw closing margin beyond geometric contact. Native
            force-limited drives supply holding force; object poses are not constrained.
        grounding: Optional source-observed object/receiver associations by event.

    Returns:
        Separate nominal joints, unchanged timestamps, and geometric diagnostics.
    """
    demo = inputs.demonstration
    clock_gaps = _clock_gaps(inputs)
    expected_units = tuple(
        "rad" if i % 8 < 6 else "m" for i in range(len(demo.joint_names))
    )
    if not inputs.robot.gripper_calibrations and demo.joint_units != expected_units:
        raise ValueError("Explicit radian arms and metre jaw positions are required")
    if type(approach_lead_frames) is not int or approach_lead_frames < 0:
        raise ValueError("approach_lead_frames must be nonnegative")
    if not np.isfinite(gripper_preload_m) or not 0 <= gripper_preload_m <= 0.005:
        raise ValueError("gripper_preload_m must be between zero and 0.005 metres")
    original = demo.qpos.copy()
    q = original.copy()
    n = len(q)
    robot = Robot(inputs.robot)
    scene = deepcopy(registered.description)
    objects = {x["id"]: x for x in scene["objects"] if x["id"] != "table"}
    vertices = {k: np.asarray(registered.vertices[k]) for k in objects}
    tracks = {k: np.tile(np.eye(4), (n, 1, 1)) for k in objects}
    plane = np.asarray(scene["source_table_plane"])
    phases = _demonstration_events(inputs, robot)
    grounding = {(e["side"], e["grasp"]): e for e in grounding}
    reports = []
    skipped = []
    for event in phases:
        side, g, r = event["side"], event["grasp"], event["release"]
        grounded = grounding.get((side, g))
        if grounded and grounded["object_id"] is None:
            skipped.append(
                dict(**event, reason="Source-video grounding: " + grounded["reason_zh"])
            )
            continue
        tcp = robot.tcp(q[g : g + 1], side)[0]
        # Source-defined TCP lies forward of the inner finger pads.
        pad_z = robot.config.get(
            "pad_center_tcp_z_m",
            -0.035 if robot.config["robot_type"] == "cobotmagic" else -0.02,
        )
        pad = tcp[:3, 3] + tcp[:3, 2] * pad_z
        candidates = {}
        for uid, v in vertices.items():
            current = v @ tracks[uid][g, :3, :3].T + tracks[uid][g, :3, 3]
            distance = float(cKDTree(current).query(pad)[0])
            if not objects[uid].get("articulated", False) or (
                grounded
                and grounded["object_id"] == uid
                and grounded.get("internal_articulation_motion_visible") is False
            ):
                candidates[uid] = distance
        if not candidates or min(candidates.values()) > 0.15:
            skipped.append(
                dict(
                    **event,
                    reason="No generated object near the recorded finger pads; retain source motion without attaching an object",
                    object_candidates_m=candidates,
                )
            )
            continue
        active = [
            e["object"]
            for e in reports
            if e["grasp"] <= g <= e["release"]
            and e["side"] != side
            and candidates.get(e["object"], 1) < 0.15
        ]
        uid = (
            min(active, key=candidates.get)
            if active
            else min(candidates, key=candidates.get)
        )
        if grounded:
            uid = grounded["object_id"]
            if candidates.get(uid, float("inf")) > 0.15:
                raise ValueError("Grounded object too far from recorded grasp: " + uid)
            event["source_video_grounding"] = grounded
        if objects[uid].get("articulated", False):
            event["articulation_scope"] = (
                "Carry the generated object's rigid root using its SAM3D observation proxy; internal joints remain fixed"
            )
        current = vertices[uid] @ tracks[uid][g, :3, :3].T + tracks[uid][g, :3, 3]
        if uid in active and candidates[uid] > 0.05:
            previous = next(
                e
                for e in reports
                if e["object"] == uid and e["grasp"] <= g <= e["release"]
            )
            donor = previous["side"]
            nearest = current[np.argmin(np.linalg.norm(current - pad, axis=1))]
            donor_shift = (pad - nearest) * 0.5
            donor_tcp = robot.tcp(q[g : g + 1], donor)[0]
            donor_target = donor_tcp.copy()
            donor_target[:3, 3] += donor_shift
            adjusted, donor_fit = _ik(robot, original[g], donor, donor_target)
            columns = robot.arm_columns[donor]
            addition = adjusted[columns] - q[g, columns]
            knots = [
                max(previous["grasp"] + 1, g - 90),
                g,
                min(n - 1, previous["open_end"] + 60),
            ]
            indices = np.arange(knots[0], knots[-1] + 1)
            q[np.ix_(indices, columns)] += PchipInterpolator(
                knots,
                np.stack([np.zeros(len(columns)), addition, np.zeros(len(columns))]),
            )(indices)
            donor_tcps = robot.tcp(q, donor)
            attachment = (
                np.linalg.inv(donor_tcps[previous["grasp"]])
                @ tracks[uid][previous["grasp"]]
            )
            tracks[uid][previous["grasp"] : previous["release"] + 1] = (
                donor_tcps[previous["grasp"] : previous["release"] + 1] @ attachment
            )
            tracks[uid][previous["release"] + 1 :] = tracks[uid][previous["release"]]
            current = vertices[uid] @ tracks[uid][g, :3, :3].T + tracks[uid][g, :3, 3]
            event["coupled_handoff"] = dict(
                donor_side=donor,
                donor_translation_m=donor_shift.tolist(),
                donor_fit=donor_fit,
                remaining_receiver_surface_distance_m=float(
                    cKDTree(current).query(pad)[0]
                ),
            )
        category = objects[uid]["category"]

        grasp_q, half, fit = fit_grasp(
            scene,
            inputs.robot.visuals,
            robot,
            original[g],
            side,
            objects[uid],
            tracks[uid][g],
            {key: value[g] for key, value in tracks.items()},
            closing_half_open_m=float(
                np.max(
                    robot.half_open(
                        original[max(0, event["close_start"] - 10) : g + 1], side
                    )
                )
            ),
        )
        shift = np.asarray(fit["tcp_shift_m"])
        contact_half = half
        half = max(0.001, contact_half - gripper_preload_m)
        event["gripper_drive"] = dict(
            geometric_contact_half_open_m=contact_half,
            commanded_half_open_m=half,
            preload_per_jaw_m=contact_half - half,
            purpose="holding force through native force-limited joint drives",
        )
        start = event["close_start"] - 60 - approach_lead_frames
        approach = event["close_start"] - 60 if approach_lead_frames else None
        end = event["open_end"] + 60
        previous = [e for e in phases if e["side"] == side and e["open_end"] < g]
        following = [e for e in phases if e["side"] == side and e["close_start"] > r]
        if previous:
            start = max(start, max(e["open_end"] for e in previous))
        if following:
            end = min(end, min(e["close_start"] for e in following))
        release_q = original[r].copy()
        columns = robot.arm_columns[side]
        release_q[columns] += grasp_q[columns] - original[g, columns]
        _warp(robot, q, original, side, g, r, start, end, grasp_q, release_q, approach)
        # Grip residual follows source closure/release, with no changes to untouched arms.
        ids = np.arange(
            max(0, event["close_start"] - 10), min(n, event["open_end"] + 1)
        )
        weights = np.interp(
            ids,
            [max(0, event["close_start"] - 10), g, r, event["open_end"]],
            [0, 1, 1, 0],
        )
        if robot.calibrations:
            opening = np.maximum(
                0.001,
                robot.half_open(original[ids], side)
                + weights * (half - float(robot.half_open(original[g], side))),
            )
            q[ids] = robot.set_half_open(q[ids], side, opening)
            q[g : r + 1] = robot.set_half_open(q[g : r + 1], side, half)
        else:
            for col in robot.jaw_columns[side]:
                q[ids, col] = np.maximum(
                    0.001, original[ids, col] + weights * (half - original[g, col])
                )
                q[g : r + 1, col] = half
        # The source closed-jaw approach can enter a larger reconstructed object
        # before the final grasp anchor. Open during the same approach phase,
        # then map the observed closure shape onto the fitted contact aperture.
        opening, event["approach_opening"] = _approach_opening_profile(
            robot.half_open(original, side),
            robot.half_open(q, side),
            event["close_start"],
            g,
            half,
            robot.config["max_half_open_m"],
            max((e["open_end"] for e in previous), default=0),
        )
        q = robot.set_half_open(q, side, opening)
        tcps = robot.tcp(q, side)
        captured = tracks[uid][g].copy()
        attachment = np.linalg.inv(tcps[g]) @ captured
        carried = tcps[g : r + 1] @ attachment
        final = vertices[uid] @ carried[-1, :3, :3].T + carried[-1, :3, 3]
        center = final.mean(0)
        receiver = None
        best = 0.13
        # Overlapping grasps can manipulate different objects. Require geometric
        # evidence that the receiving fingers approach this carried object.
        later = [e for e in phases if g < e["grasp"] <= r and e["side"] != side]
        handoff = False
        for other_event in later:
            t = other_event["grasp"]
            other_tcp = robot.tcp(original[t : t + 1], other_event["side"])[0]
            other_pad = other_tcp[:3, 3] + other_tcp[:3, 2] * pad_z
            held = vertices[uid] @ carried[t - g, :3, :3].T + carried[t - g, :3, 3]
            held_distance = float(cKDTree(held).query(other_pad)[0])
            alternatives = [
                float(
                    cKDTree(v @ tracks[k][t, :3, :3].T + tracks[k][t, :3, 3]).query(
                        other_pad
                    )[0]
                )
                for k, v in vertices.items()
                if k != uid
            ]
            if (
                held_distance < 0.15
                and held_distance <= min(alternatives, default=1) + 0.03
            ):
                handoff = True
        placement_shift = np.zeros(3)
        if not handoff:
            source_release = robot.tcp(original[r : r + 1], side)[0]
            source_pad = source_release[:3, 3] + source_release[:3, 2] * pad_z
            release_target = tcps[r].copy()
            demonstrated_release = release_target.copy()
            if category in {"cube", "bowl", "plate"}:
                initial_up = np.asarray(objects[uid]["transform"])[:3, 1]
                initial_up /= np.linalg.norm(initial_up)
                current_up = carried[-1, :3, :3] @ initial_up
                support_normal = np.array([-plane[0], -plane[1], 1.0])
                support_normal /= np.linalg.norm(support_normal)
                correction = Rotation.align_vectors([support_normal], [current_up])[
                    0
                ].as_matrix()
                pivot = (final.min(0) + final.max(0)) * 0.5
                final = (final - pivot) @ correction.T + pivot
                release_target[:3, :3] = correction @ release_target[:3, :3]
                release_target[:3, 3] = (
                    correction @ (release_target[:3, 3] - pivot) + pivot
                )
            for other, v in vertices.items():
                if other == uid or objects[other]["category"] not in {
                    "cube",
                    "bowl",
                    "plate",
                    "cup",
                    "basket",
                    "tray",
                    "pot",
                    "saucepan",
                    "steamer",
                    "container",
                    "food_container",
                    "pen_holder",
                    "pencil_holder",
                    "desk_organizer",
                    "bin",
                }:
                    continue
                v = v @ tracks[other][r, :3, :3].T + tracks[other][r, :3, 3]
                receiver_xy = (v.min(0) + v.max(0))[:2] * 0.5
                distance = min(
                    np.linalg.norm(center[:2] - receiver_xy),
                    np.linalg.norm(source_pad[:2] - receiver_xy),
                )
                if distance < best:
                    receiver = other
                    best = distance
            if grounded and grounded.get("receiver_object_id") in objects:
                receiver = grounded["receiver_object_id"]
                event["receiver_source"] = "Astra association from original video"
            if receiver:
                rv = (
                    vertices[receiver] @ tracks[receiver][r, :3, :3].T
                    + tracks[receiver][r, :3, 3]
                )
                receiver_center = (rv.min(0) + rv.max(0)) * 0.5
                final_center = (final.min(0) + final.max(0)) * 0.5
                if category in {"banana", "cube", "bowl"}:
                    target_xy = receiver_center[:2]
                else:
                    # Keep the demonstrated placement location inside a support
                    # instead of collapsing several objects onto its center.
                    margin = np.minimum(
                        np.ptp(final[:, :2], axis=0) * 0.5 + 0.004,
                        np.ptp(rv[:, :2], axis=0) * 0.45,
                    )
                    target_xy = np.clip(
                        final_center[:2],
                        rv[:, :2].min(0) + margin,
                        rv[:, :2].max(0) - margin,
                    )
                placement_shift[:2] = target_xy - final_center[:2]
                if category == "cup" and objects[receiver]["category"] == "cup":
                    # Nested cups retain the demonstrated insertion depth. A
                    # generic top-surface placement would stack them end to end.
                    support = float(final[:, 2].min()) - 0.001
                elif objects[receiver]["category"] == "cube":
                    support = float(rv[:, 2].max())
                else:
                    inner = rv[
                        np.linalg.norm(rv[:, :2] - receiver_center[:2], axis=1)
                        < np.min(np.ptp(rv[:, :2], axis=0)) * 0.2
                    ]
                    support = (
                        float(np.quantile(inner[:, 2], 0.8))
                        if len(inner)
                        else float(rv[:, 2].min())
                    )
                    if category == "bowl":
                        support += float(np.ptp(rv[:, 2])) * 0.28
            else:
                support = float(center[:2] @ plane[:2] + plane[2])
                low, high = final[:, :2].min(0), final[:, :2].max(0)
                footprint = np.array(
                    [center[:2], low, high, [low[0], high[1]], [high[0], low[1]]]
                )
                support, support_id = _support_at_release(
                    objects,
                    {key: value[r] for key, value in tracks.items()},
                    uid,
                    footprint,
                    support,
                )
                event["placement_support"] = support_id or "table"
            from ._contact_fit import _mesh

            finger_links, _ = robot.finger_surfaces(side)
            fk = robot.links(q[r : r + 1])
            release_delta = release_target @ np.linalg.inv(tcps[r])
            finger_points = []
            for visual in inputs.robot.visuals:
                if visual["link"] in finger_links:
                    mesh = _mesh(Path(visual["path"]))
                    mesh.apply_transform(
                        release_delta
                        @ fk[visual["link"]][0]
                        @ np.asarray(visual["origin"])
                    )
                    finger_points.append(np.asarray(mesh.vertices))
            if not finger_points:
                raise ValueError("Release clearance requires actual finger geometry")
            release_fingers = np.concatenate(finger_points)
            if (
                receiver
                and _inside_receiver(objects[receiver]["category"], grounded)
                and category not in {"cup", "bowl"}
            ):
                placement_shift[2] = _container_release_lift(
                    final, release_fingers, float(rv[:, 2].max())
                )
                event["release_goal"] = {
                    "relation": "inside_container_after_gravity",
                    "receiver": receiver,
                    "rim_z_m": float(rv[:, 2].max()),
                    "clearance_m": 0.003,
                    "vertical_lift_m": float(placement_shift[2]),
                }
            else:
                placement_shift[2] = _container_release_lift(
                    final, release_fingers, support
                )
                event["release_goal"] = {
                    "relation": "supported_after_gravity",
                    "support_z_m": support,
                    "clearance_m": 0.003,
                    "vertical_lift_m": float(placement_shift[2]),
                }
            release_target[:3, 3] += placement_shift
            release_q, release_fit = _ik(robot, original[r], side, release_target)
            _warp(
                robot, q, original, side, g, r, start, end, grasp_q, release_q, approach
            )
            tcps = robot.tcp(q, side)
            attachment = np.linalg.inv(tcps[g]) @ captured
            carried = tcps[g : r + 1] @ attachment
        else:
            release_fit = {}
        if not handoff and not active:
            from ._approach import clear_approach

            q, event["approach_clearance"] = clear_approach(
                robot,
                q,
                side,
                scene,
                {key: value[g] for key, value in tracks.items()},
                inputs.robot.visuals,
                start,
                event["close_start"],
                _ik,
                grasp=g,
            )
        tracks[uid][g : r + 1] = carried
        tracks[uid][r + 1 :] = carried[-1]
        # This carry hypothesis associates later grasps; it is not a simulator command.
        event.update(
            object=uid,
            object_candidates_m=candidates,
            receiver=receiver,
            handoff=handoff,
            grasp_fit=fit,
            release_fit=release_fit,
            gripper_half_open_m=half,
            requested_grasp_shift_m=shift.tolist(),
            requested_place_shift_m=placement_shift.tolist(),
            approach_residual_ready_frame=approach,
        )
        reports.append(event)
    if not reports:
        raise ValueError("No source-near grasp could be associated with this scene")
    from ._approach import _idle_windows, clear_approach

    idle_clearance = []
    for window in _idle_windows(phases, grounding, n, approach_lead_frames):
        q, clearance = clear_approach(
            robot,
            q,
            window["side"],
            scene,
            {key: value[window["start"]] for key, value in tracks.items()},
            inputs.robot.visuals,
            window["start"],
            window["checked_end"],
            _ik,
            grasp=window["fade_end"],
        )
        idle_clearance.append(dict(**window, **clearance))
    armcols = np.concatenate(robot.arm_columns).tolist()
    report = dict(
        method="mesh grasp anchors + source-near IK + coupled handover + PCHIP joint residuals",
        frames=n,
        full_source_frames_retained=True,
        missing_sample_after_indices=clock_gaps,
        source_timestamps_preserved=True,
        events=reports,
        skipped_source_closures=skipped,
        non_transport_clearance=idle_clearance,
        max_arm_joint_residual_rad=float(
            abs(q[:, armcols] - original[:, armcols]).max()
        ),
        max_adjacent_arm_step_rad=float(abs(np.diff(q[:, armcols], axis=0)).max()),
        physical_success_verified=False,
        limitations=[
            "Object pose hypotheses are contact-association estimates, not simulator commands",
            "One demonstration provides a local prior, not a learned distribution",
            "This method does not establish full-path collision freedom or physical grasp success",
        ],
    )
    return RefinedTrajectory(
        q,
        inputs.demonstration.timestamps.copy(),
        {name: robot.tcp(q, side) for side, name in enumerate(inputs.robot.side_names)},
        report,
        tracks,
    )
