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

"""Identify explicit motion goals and reconstruct them through native skills."""

from __future__ import annotations

from dataclasses import asdict, replace
import time
from typing import Any, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from embodichain.compute.trajectory.phase_alignment import align_joint_phase
from embodichain.compute.trajectory.splicing import (
    joint_path_diagnostics,
    splice_joint_segment,
)
from embodichain.data_pipeline.skill_fitting import JointRecording
from embodichain.utils import configclass
from .control import OPEN_COMMAND, GRASP_COMMAND
from .policies import MotionPolicy, RecoveryPolicy
from .primitives import (
    PickUpMotionGoal,
    PickUpOptions,
    SlideMotionGoal,
    SlideOptions,
    TwistMotionGoal,
    TwistOptions,
    HandOverMotionGoal,
    HandOverOptions,
)
from .state import RobotObservation

if TYPE_CHECKING:
    from .engine import AtomicSkillEngine

__all__ = ["SkillFitCfg", "fit_skill_segment"]


@configclass
class SkillFitCfg:
    """Finite fitting budgets and acceptance tolerances in physical units."""

    control_dt: float = 0.01
    blend_samples: int = 12
    blend_duration_s: float = 0.5
    """Minimum boundary-transition duration, capped independently by phase anchors.

    ``blend_samples`` additionally sets the minimum sample width. Both outer
    joins retain the original position and derivative stencils exactly.
    """
    max_samples: int = 20_000
    max_candidates: int = 3
    max_seconds: float = 120.0
    max_joint_rmse: float = 0.15
    max_tcp_rmse_m: float = 0.015
    max_rotation_rmse_rad: float = 0.08
    max_approach_rotation_rad: float = 0.03
    """Allowed pre-grasp orientation deviation when inferring an approach anchor.

    This inference tolerance accommodates measured orientation noise. Final
    native/contact reconstruction still uses its independent error tolerances.
    """
    max_contact_error_m: float = 0.005
    max_contact_rotation_rad: float = 0.03
    min_twist_angle_rad: float = 0.05

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"SkillFitCfg.{key} must be finite and positive.")
        for key in ("blend_samples", "max_samples", "max_candidates"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer.")
        if (
            self.max_candidates > 9
            or self.max_samples > 200_000
            or self.blend_samples < 4
        ):
            raise ValueError("Fitting budgets are outside supported bounds.")


def _transitions(
    q: np.ndarray, opened: np.ndarray, closed: np.ndarray
) -> tuple[list, list]:
    delta = closed - opened
    progress = ((q - opened) @ delta) / (delta @ delta)
    changes = np.diff(progress)
    result = []
    for sign in (1, -1):
        ids = np.flatnonzero(sign * changes > max(1e-5, np.ptp(progress) * 1e-3)) + 1
        groups = np.split(ids, np.flatnonzero(np.diff(ids) > 2) + 1) if len(ids) else []
        # A meaningful open/close event covers at least 10% of the profile travel.
        result.append(
            [
                (int(g[0] - 1), int(g[-1] + 1))
                for g in groups
                if len(g) and abs(progress[g[-1]] - progress[g[0] - 1]) >= 0.1
            ]
        )
    return result[0], result[1]


def _rot_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(
        left[..., :3, :3] @ right[..., :3, :3].swapaxes(-1, -2)
    ).magnitude()


def _approach(
    poses: np.ndarray,
    close_start: int,
    *,
    max_rotation_rad: float,
    pre_index: int | None = None,
) -> tuple[np.ndarray, float, int]:
    """Resolve an explicit reach anchor or infer one within rotation tolerance."""
    grasp = poses[close_start]
    if pre_index is None:
        rotation_error = _rot_error(poses[:close_start], grasp)
        candidates = np.flatnonzero(rotation_error <= max_rotation_rad)
        if len(candidates) < 2:
            nearest = (
                f"{rotation_error.min():.6g} rad"
                if len(rotation_error)
                else "no pre-grasp samples"
            )
            raise ValueError(
                "No stable TCP orientation before grasp within "
                f"max_approach_rotation_rad={max_rotation_rad:.6g} "
                f"(nearest deviation: {nearest}). Increase this inference "
                "tolerance or refine the selected core interval."
            )
        # The farthest point after attaining the grasp orientation is the pre-grasp.
        offsets = grasp[:3, 3] - poses[candidates, :3, 3]
        norms = np.linalg.norm(offsets, axis=1)
        pre_index = int(candidates[np.flatnonzero(norms >= np.max(norms) - 1e-8)[-1]])
    elif not 0 <= pre_index < close_start:
        raise ValueError("The hinted reach anchor must precede gripper closure.")
    offset = grasp[:3, 3] - poses[pre_index, :3, 3]
    distance = float(np.linalg.norm(offset))
    if distance < 1e-4:
        raise ValueError("Approach displacement is not observable.")
    return offset / distance, distance, pre_index


def _twist_line(
    poses: np.ndarray, minimum_angle: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    rotation = Rotation.from_matrix(poses[:, :3, :3] @ poses[0, :3, :3].T)
    vectors = rotation.as_rotvec()
    _, singular, vt = np.linalg.svd(vectors, full_matrices=False)
    axis = vt[0]
    angles = np.unwrap(vectors @ axis)
    angle = float(angles[-1])
    if abs(angle) < minimum_angle:
        raise ValueError("Twist angle is too small to identify a rotation line.")
    if angle < 0:
        axis, angles, angle = -axis, -angles, -angle
    rotations = Rotation.from_rotvec(angles[:, None] * axis).as_matrix()
    matrix = np.eye(3)[None] - rotations
    right = poses[:, :3, 3] - np.einsum("nij,j->ni", rotations, poses[0, :3, 3])
    # Resolve the axis-point gauge by choosing its closest point to the initial TCP.
    origin = np.linalg.lstsq(
        np.concatenate((matrix.reshape(-1, 3), axis[None]), axis=0),
        np.concatenate((right.reshape(-1), [axis @ poses[0, :3, 3]])),
        rcond=None,
    )[0]
    predicted = origin + np.einsum("nij,j->ni", rotations, poses[0, :3, 3] - origin)
    error = float(np.sqrt(np.mean(np.sum((predicted - poses[:, :3, 3]) ** 2, axis=1))))
    return axis, origin, angle, error


def _phase_map(
    q_native: np.ndarray,
    native_lengths: dict[str, int],
    phases: list[dict[str, Any]],
    observations: np.ndarray,
    approach_anchor: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if list(native_lengths) != [p["name"] for p in phases]:
        raise ValueError(
            f"Source phases must match native phases {list(native_lengths)}."
        )
    output = np.empty_like(observations, dtype=np.float64)
    timing = np.empty(len(observations))
    cursor = 0
    for phase in phases:
        count = native_lengths[phase["name"]]
        part = q_native[cursor : cursor + count]
        cursor += count
        start, stop = phase["start"], phase["stop"]
        if stop <= start:
            raise ValueError("Every source phase must contain a sample.")
        if phase["name"] == "approach" and approach_anchor is not None:
            source_anchor, native_anchor = approach_anchor
            if not start < source_anchor < stop or not 0 < native_anchor < count - 1:
                raise ValueError("Pre-grasp anchor must be inside its approach phase.")
            first, clock_a = align_joint_phase(
                part[: native_anchor + 1], observations[start : source_anchor + 1]
            )
            second, clock_b = align_joint_phase(
                part[native_anchor:], observations[source_anchor:stop]
            )
            output[start : source_anchor + 1], output[source_anchor:stop] = (
                first,
                second,
            )
            timing[start : source_anchor + 1] = (
                clock_a * native_anchor / max(1, count - 1)
            )
            timing[source_anchor:stop] = (
                native_anchor + clock_b * (count - 1 - native_anchor)
            ) / max(1, count - 1)
        else:
            output[start:stop], timing[start:stop] = align_joint_phase(
                part, observations[start:stop]
            )
    if cursor != len(q_native):
        raise ValueError("Native phase lengths do not cover the generated plan.")
    return output, timing


def fit_skill_segment(
    recording: JointRecording,
    engine: AtomicSkillEngine,
    *,
    skill_id: str,
    endpoints: dict[str, dict[str, str]],
    start: int,
    stop: int,
    acceleration_limits: np.ndarray,
    cfg: SkillFitCfg | None = None,
    phase_hints: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Identify and regenerate one segment without object pose or geometry.

    Only the native AtomicSkillEngine creates candidate joint motion. Source
    joints supply observations, initial seeds, phase clocks and outer boundary
    stencils. The returned reference is not a measured-state prediction.

    Args:
        recording: Explicitly timed, named source joints.
        engine: Engine bound to a one-row robot kinematics port.
        skill_id: pick_up, slide, twist or hand_over.
        endpoints: Existing explicit participant-to-control-part bindings.
        start: First included recording sample.
        stop: First excluded recording sample.
        acceleration_limits: Explicit per-joint acceleration caps.
        cfg: Numerical budgets and independent native/patched acceptance gates.
        phase_hints: Optional complete contiguous phase partition. Indices are
            relative to this segment, with half-open bounds.

    Returns:
        Full-length original/native/patched arrays and fit metadata. Rejected
        fits retain their native candidate and leave patched_qpos unchanged.
    """
    cfg = SkillFitCfg() if cfg is None else cfg
    cfg.validate()
    cfg.__post_init__()
    began = time.monotonic()
    if skill_id not in {"pick_up", "slide", "twist", "hand_over"}:
        raise ValueError("Unsupported reconstruction skill.")
    if tuple(engine.robot.joint_names) != recording.joint_names:
        raise ValueError("Recording must use the engine's exact named joint order.")
    if (
        not 0 <= start < stop <= len(recording.qpos)
        or not 12 <= stop - start <= cfg.max_samples
    ):
        raise ValueError("Invalid or oversized selected segment.")
    q = recording.qpos[start:stop]
    times = recording.timestamps[start:stop]
    robot = engine.robot
    device = engine.device
    hints = {} if phase_hints is None else {p["name"]: p for p in phase_hints}
    if phase_hints is not None and (
        not phase_hints
        or len(hints) != len(phase_hints)
        or phase_hints[0]["start"] != 0
        or phase_hints[-1]["stop"] != len(q)
        or any(
            type(p["start"]) is not int
            or type(p["stop"]) is not int
            or not 0 <= p["start"] < p["stop"] <= len(q)
            for p in phase_hints
        )
        or any(a["stop"] != b["start"] for a, b in zip(phase_hints, phase_hints[1:]))
    ):
        raise ValueError("Phase hints must be a complete contiguous segment partition.")
    participant_data = {}
    for role, ports in endpoints.items():
        arm, hand = ports["motion"], ports["grasp"]
        arm_ids, hand_ids = robot.get_joint_ids(arm), robot.get_joint_ids(hand)
        profile = engine.control_profiles[hand]
        opened = profile.commands[OPEN_COMMAND].positions
        closed = profile.commands[GRASP_COMMAND].positions
        opened = torch.as_tensor(opened).cpu().numpy().reshape(-1)
        closed = torch.as_tensor(closed).cpu().numpy().reshape(-1)
        with torch.no_grad():
            poses = (
                robot.compute_fk(
                    torch.as_tensor(q[:, arm_ids], device=device, dtype=torch.float32),
                    name=arm,
                    to_matrix=True,
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        closes, opens = _transitions(q[:, hand_ids], opened, closed)
        close_name = "pickup_close" if role == "source" else "close"
        open_name = (
            "release"
            if role == "source"
            else ("receive_release" if role == "destination" else "open")
        )
        if close_name in hints:
            closes = [(hints[close_name]["start"], hints[close_name]["stop"])]
        if open_name in hints:
            opens = [(hints[open_name]["start"], hints[open_name]["stop"])]
        participant_data[role] = (poses, closes, opens, arm_ids, hand_ids)

    params: dict[str, Any] = {"pose_origin": "fk_derived"}
    source_template_errors: dict[str, Any] = {}
    source_template_failures: list[str] = []
    contact_ranges: list[tuple[str, int, int]] = []

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.tensor(value, device=device, dtype=torch.float32)

    def partition(boundaries: list[tuple[str, int]]) -> list[dict[str, Any]]:
        return [
            {"name": name, "start": left, "stop": right}
            for (name, left), (_, right) in zip(boundaries, boundaries[1:])
        ]

    if skill_id != "hand_over":
        if set(endpoints) != {"primary"}:
            raise ValueError("Single-arm skills require the primary participant.")
        poses, closes, opens, _, _ = participant_data["primary"]
        if len(closes) != 1:
            raise ValueError(
                "Expected one observable closure; select a complete skill or provide matching command data."
            )
        close_start, lift_start = closes[0]
        if (
            np.max(
                np.linalg.norm(
                    poses[close_start:lift_start, :3, 3] - poses[close_start, :3, 3],
                    axis=1,
                )
            )
            > cfg.max_contact_error_m
            or np.max(_rot_error(poses[close_start:lift_start], poses[close_start]))
            > cfg.max_contact_rotation_rad
        ):
            raise ValueError(
                "The native skill requires a stationary arm during closure."
            )
        direction, pre_distance, pre_index = _approach(
            poses,
            close_start,
            max_rotation_rad=cfg.max_approach_rotation_rad,
            pre_index=hints["reach"]["start"] if "reach" in hints else None,
        )
        grasp = poses[close_start]
        close_count = lift_start - close_start
        if skill_id == "pick_up":
            lift = poses[-1, :3, 3] - grasp[:3, 3]
            if lift[2] <= 0:
                raise ValueError("Pickup requires an observable world-Z lift.")
            lateral = float(np.linalg.norm(lift[:2]))
            rotation = float(_rot_error(poses[-1], grasp))
            source_template_errors["pickup_lift"] = {
                "translation_m": lift.tolist(),
                "lateral_displacement_m": lateral,
                "rotation_change_rad": rotation,
            }
            # A measurable lift still supplies a native vertical candidate.
            # Keep incompatible source motion inspectable without accepting it
            # or replacing the original reference outside the acceptance gates.
            if lateral > cfg.max_contact_error_m:
                source_template_failures.append(
                    "Pickup lift deviates from world Z: "
                    f"lateral displacement {lateral:.6g} m exceeds "
                    f"max_contact_error_m={cfg.max_contact_error_m:.6g}"
                )
            if rotation > cfg.max_contact_rotation_rad:
                source_template_failures.append(
                    "Pickup lift changes TCP orientation: "
                    f"{rotation:.6g} rad exceeds "
                    f"max_contact_rotation_rad={cfg.max_contact_rotation_rad:.6g}"
                )
            goal = PickUpMotionGoal(tensor(grasp))
            options = PickUpOptions(
                lift_height=float(lift[2]),
                pre_grasp_distance=pre_distance,
                approach_direction=tensor(direction),
                hand_interp_steps=close_count,
            )
            phases = partition(
                [
                    ("approach", 0),
                    ("close", close_start),
                    ("lift", lift_start),
                    ("end", len(q)),
                ]
            )
            params.update(
                lift_height=float(lift[2]),
                pre_grasp_distance=pre_distance,
                approach_direction=direction.tolist(),
                grasp_xpos=grasp.tolist(),
            )
            contact_ranges = [("primary", close_start, len(q))]
        else:
            releases = [event for event in opens if event[0] >= lift_start]
            if len(releases) != 1:
                raise ValueError("Slide/Twist requires one observable release.")
            open_start, retract_start = releases[0]
            if skill_id == "slide":
                move = poses[open_start, :3, 3] - grasp[:3, 3]
                distance = float(np.linalg.norm(move))
                if distance < 1e-4:
                    raise ValueError("Slide displacement is not observable.")
                rail = move / distance
                retract_distance = np.linalg.norm(
                    poses[-1, :3, 3] - poses[open_start, :3, 3]
                )
                mode = "push" if retract_distance > cfg.max_contact_error_m else "pull"
                axis = rail if mode == "push" else -rail
                projected = (poses[lift_start:open_start, :3, 3] - grasp[:3, 3]) @ rail
                residual = (
                    poses[lift_start:open_start, :3, 3]
                    - grasp[:3, 3]
                    - projected[:, None] * rail
                )
                if np.max(np.linalg.norm(residual, axis=1)) > cfg.max_contact_error_m:
                    raise ValueError("Slide contact motion is not a straight rail.")
                goal = SlideMotionGoal(tensor(grasp), tensor(axis))
                options = SlideOptions(
                    direction=mode,
                    translation_distance=distance,
                    approach_distance=pre_distance,
                    hand_interp_steps=close_count,
                )
                boundaries = [
                    ("approach", 0),
                    ("reach", pre_index),
                    ("close", close_start),
                    (mode, lift_start),
                    ("open", open_start),
                ]
                if mode == "push":
                    boundaries.append(("return", retract_start))
                phases = partition(boundaries + [("end", len(q))])
                params.update(
                    translation_axis=axis.tolist(),
                    translation_distance=distance,
                    direction=mode,
                    approach_distance=pre_distance,
                    grasp_xpos=grasp.tolist(),
                )
            else:
                axis, origin, angle, error = _twist_line(
                    poses[lift_start : open_start + 1], cfg.min_twist_angle_rad
                )
                if error > cfg.max_contact_error_m:
                    raise ValueError("Twist does not fit a rigid rotation line.")
                goal = TwistMotionGoal(tensor(grasp), tensor(axis), tuple(origin))
                options = TwistOptions(
                    twist_angle=angle,
                    pre_grasp_distance=pre_distance,
                    hand_interp_steps=close_count,
                    twist_waypoint_count=2,
                )
                phases = partition(
                    [
                        ("approach", 0),
                        ("reach", pre_index),
                        ("close", close_start),
                        ("twist", lift_start),
                        ("open", open_start),
                        ("retract", retract_start),
                        ("end", len(q)),
                    ]
                )
                params.update(
                    twist_axis=axis.tolist(),
                    axis_origin=origin.tolist(),
                    twist_angle=angle,
                    pre_grasp_distance=pre_distance,
                    grasp_xpos=grasp.tolist(),
                    axis_fit_rmse_m=error,
                    axis_point_convention="closest_to_initial_tcp",
                )
            contact_ranges = [("primary", close_start, open_start)]
    else:
        if set(endpoints) != {"source", "destination"}:
            raise ValueError(
                "Handover requires explicit source/destination participants."
            )
        source, source_closes, source_opens, source_ids, _ = participant_data["source"]
        receive, receive_closes, receive_opens, _, _ = participant_data["destination"]
        if len(receive_closes) != 1 or len(source_opens) != 1 or len(source_closes) > 1:
            raise ValueError(
                "Select one complete handover with observable hand events."
            )
        rc0, rc1 = receive_closes[0]
        so0, so1 = source_opens[0]
        if rc1 >= so0:
            raise ValueError("Handover needs a receiving hold before source release.")
        for observed in (source, receive):
            hold = observed[rc1 - 1 : so0 + 1]
            if (
                np.max(np.linalg.norm(hold[:, :3, 3] - hold[0, :3, 3], axis=1))
                > cfg.max_contact_error_m
                or np.max(_rot_error(hold, hold[0])) > cfg.max_contact_rotation_rad
            ):
                raise ValueError(
                    "Handover exchange requires stationary overlap anchors."
                )
        moving = (
            np.flatnonzero(
                np.max(np.abs(np.diff(q[:, source_ids], axis=0)), axis=1) > 1e-5
            )
            + 1
        )
        transfer_end = (
            hints["approach"]["start"]
            if "approach" in hints
            else int(max((i for i in moving if i < rc0), default=0)) + 1
        )
        retreat_end = (
            hints["place"]["start"]
            if "place" in hints
            else int(max((i for i in moving if i >= so1), default=so1)) + 1
        )
        if not transfer_end < rc0 or retreat_end >= len(q):
            raise ValueError(
                "Handover phase split needs explicit phase hints or stationary boundaries."
            )
        pickup = source[source_closes[0][0]] if source_closes else None
        placement = receive_opens[0] if len(receive_opens) == 1 else None
        goal = HandOverMotionGoal(
            tensor(source[rc1 - 1]),
            tensor(receive[rc1 - 1]),
            tensor(source[retreat_end - 1]),
            None if pickup is None else tensor(pickup),
            None if placement is None else tensor(receive[placement[0]]),
            None if placement is None else tensor(receive[-1]),
        )
        options = HandOverOptions(
            arm_selection="bound",
            release_at_target=placement is not None,
            hand_interp_steps=rc1 - rc0,
            hold_steps=so0 - rc1,
            retreat_steps=max(3, retreat_end - so1),
        )
        boundaries = []
        if source_closes:
            sc0, sc1 = source_closes[0]
            boundaries.extend([("pickup_approach", 0), ("pickup_close", sc0)])
            transfer_start = sc1
        else:
            transfer_start = 0
        boundaries.extend(
            [
                ("transfer", transfer_start),
                ("approach", transfer_end),
                ("close", rc0),
                ("hold", rc1),
                ("release", so0),
                ("retreat", so1),
            ]
        )
        if placement:
            boundaries.extend(
                [
                    ("place", retreat_end),
                    ("receive_release", placement[0]),
                    ("receive_retreat", placement[1]),
                ]
            )
        phases = partition(boundaries + [("end", len(q))])
        contact_ranges = [
            ("source", source_closes[0][0] if source_closes else 0, so0),
            ("destination", rc0, placement[0] if placement else len(q)),
        ]
        params.update(
            source_exchange=source[rc1 - 1].tolist(),
            destination_exchange=receive[rc1 - 1].tolist(),
            relative_exchange=(
                np.linalg.inv(source[rc1 - 1]) @ receive[rc1 - 1]
            ).tolist(),
            source_retreat=source[retreat_end - 1].tolist(),
        )

    if phase_hints is not None:
        if [p["name"] for p in phase_hints] != [p["name"] for p in phases]:
            raise ValueError(
                "Phase hints must name the complete native phase sequence."
            )
        phases = [dict(p) for p in phase_hints]
    if (
        phases[0]["start"] != 0
        or phases[-1]["stop"] != len(q)
        or any(
            type(p["start"]) is not int
            or type(p["stop"]) is not int
            or p["stop"] <= p["start"]
            for p in phases
        )
        or any(a["stop"] != b["start"] for a, b in zip(phases, phases[1:]))
    ):
        raise ValueError("Phases must be a complete nonempty contiguous partition.")
    context = engine.initial_context(control_dt=cfg.control_dt)
    context = replace(
        context,
        robot=RobotObservation(
            timestamp=0,
            qpos=tensor(q[0])[None],
            qvel=torch.zeros_like(tensor(q[0])[None]),
        ),
    )
    if context.batch_size != 1:
        raise ValueError("Skill reconstruction currently requires B=1.")
    # Native timing remains inspectable; phasewise mapping has its own fixed
    # source clock and all final derivative checks occur after that mapping.
    policy = MotionPolicy(
        strategy="motion_gen" if skill_id == "pick_up" else "ik_interp",
        sample_count=max(len(q), 80),
        acceleration_limit=0.5,
    )
    candidates = []
    for candidate in range(cfg.max_candidates if skill_id == "pick_up" else 1):
        if time.monotonic() - began > cfg.max_seconds:
            break
        distance = (
            pre_distance
            + (0.0, -0.005, 0.005, -0.01, 0.01, -0.02, 0.02, -0.03, 0.03)[candidate]
            if skill_id == "pick_up"
            else None
        )
        trial = (
            replace(options, pre_grasp_distance=max(0.0001, distance))
            if distance is not None
            else options
        )
        invocation = engine.make_invocation(
            skill_id,
            goal,
            control_parts=endpoints,
            motion_policy=policy,
            skill_options=trial,
            recovery_policy=RecoveryPolicy(max_replans=0, max_skill_retries=0),
        )
        plan = engine.plan(invocation, context)
        if not bool(plan.plan_success.all()):
            continue
        native = plan.joint_trajectory.positions[0].detach().cpu().numpy()
        lengths = {p.name: p.stop - p.start for p in plan.segments}
        anchor = None
        if skill_id == "pick_up":
            with torch.no_grad():
                approach_fk = (
                    robot.compute_fk(
                        tensor(
                            native[
                                : lengths["approach"], participant_data["primary"][3]
                            ]
                        ),
                        name=endpoints["primary"]["motion"],
                        to_matrix=True,
                    )
                    .cpu()
                    .numpy()
                )
            target_pre = grasp[:3, 3] - direction * trial.pre_grasp_distance
            native_pre = int(
                np.argmin(np.linalg.norm(approach_fk[:, :3, 3] - target_pre, axis=1))
            )
            anchor = (
                (pre_index, native_pre) if pre_index > 0 and native_pre > 0 else None
            )
        mapped, timing = _phase_map(native, lengths, phases, q, anchor)
        owned = sorted(
            {j for data in participant_data.values() for ids in data[3:] for j in ids}
        )
        score = float(np.sqrt(np.mean((mapped[:, owned] - q[:, owned]) ** 2)))
        candidates.append((score, mapped, native, lengths, trial, timing))
    if not candidates:
        raise ValueError("Native skill generation failed within the fitting budget.")
    score, mapped, raw_native, lengths, selected_options, timing = min(
        candidates, key=lambda item: item[0]
    )
    if skill_id == "pick_up":
        params["pre_grasp_distance"] = selected_options.pre_grasp_distance
    # Every non-participant column stays observed, including passive/mimic state.
    editable = tuple(owned)
    mapped[:, [i for i in range(q.shape[1]) if i not in editable]] = q[
        :, [i for i in range(q.shape[1]) if i not in editable]
    ]
    candidate_full = recording.qpos.copy()
    candidate_full[start:stop] = mapped
    patched = splice_joint_segment(
        recording.qpos,
        mapped,
        recording.timestamps,
        start=start,
        stop=stop,
        joint_indices=editable,
        blend_samples=cfg.blend_samples,
        blend_duration_s=cfg.blend_duration_s,
        anchors=tuple(start + p["start"] for p in phases[1:]),
    )
    errors = {}
    failures = list(source_template_failures)
    for role, (observed, _, _, arm_ids, _) in participant_data.items():

        def fk(values: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return (
                    robot.compute_fk(
                        tensor(values[:, arm_ids]),
                        name=endpoints[role]["motion"],
                        to_matrix=True,
                    )
                    .cpu()
                    .numpy()
                )

        native_tcp, patched_tcp = fk(mapped), fk(patched[start:stop])
        position = np.linalg.norm(native_tcp[:, :3, 3] - observed[:, :3, 3], axis=1)
        rotation = _rot_error(native_tcp, observed)
        errors[role] = {
            "native_tcp_rmse_m": float(np.sqrt(np.mean(position**2))),
            "native_rotation_rmse_rad": float(np.sqrt(np.mean(rotation**2))),
        }
        if (
            errors[role]["native_tcp_rmse_m"] > cfg.max_tcp_rmse_m
            or errors[role]["native_rotation_rmse_rad"] > cfg.max_rotation_rmse_rad
        ):
            failures.append(f"{role}: native TCP reconstruction exceeds tolerance")
        for contact_role, a, b in contact_ranges:
            if contact_role == role:
                if (
                    position[a:b].max() > cfg.max_contact_error_m
                    or rotation[a:b].max() > cfg.max_contact_rotation_rad
                ):
                    failures.append(
                        f"{role}: native contact path exceeds observation tolerance"
                    )
                correction = np.linalg.norm(
                    patched_tcp[a:b, :3, 3] - native_tcp[a:b, :3, 3], axis=1
                )
                correction_rotation = _rot_error(patched_tcp[a:b], native_tcp[a:b])
                if (
                    correction.max() > cfg.max_contact_error_m
                    or correction_rotation.max() > cfg.max_contact_rotation_rad
                ):
                    failures.append(
                        f"{role}: boundary correction changes protected contact motion"
                    )
    if score > cfg.max_joint_rmse:
        failures.append("native joint RMSE exceeds tolerance")
    limits = robot.get_qpos_limits()[0].cpu().numpy()
    velocity = robot.get_qvel_limits()[0].cpu().numpy()
    acceleration = np.asarray(acceleration_limits, dtype=np.float64)
    diagnostics = joint_path_diagnostics(
        patched,
        recording.timestamps,
        lower=limits[:, 0],
        upper=limits[:, 1],
        velocity_limits=velocity,
        acceleration_limits=acceleration,
    )
    if not diagnostics["accepted"]:
        failures.append("patched path exceeds joint limits")
    accepted = not failures

    def serializable(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {k: serializable(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [serializable(v) for v in value]
        return value

    report = {
        "schema_version": 1,
        "skill_id": skill_id,
        "accepted": accepted,
        "failures": failures,
        "parameters": params,
        "phases": phases,
        "native_phase_lengths": lengths,
        "native_joint_rmse": score,
        "native_options": serializable(asdict(selected_options)),
        "native_goal": serializable(asdict(goal)),
        "initial_qpos": q[0].tolist(),
        "algorithm_revision": 3,
        "boundary_connection": {
            "method": "physical_time_quintic_envelope",
            "requested_duration_s": cfg.blend_duration_s,
            "minimum_samples": cfg.blend_samples,
            "fixed_samples_per_end": 3,
            "endpoint_position_error": float(
                np.max(
                    np.abs(
                        patched[[start, stop - 1]] - recording.qpos[[start, stop - 1]]
                    )
                )
            ),
        },
        "source_template_errors": source_template_errors,
        "per_joint_native_rmse": np.sqrt(np.mean((mapped - q) ** 2, axis=0)).tolist(),
        "time_mapping": "phasewise_monotone_native_path_projection",
        "native_motion_policy": {
            "strategy": policy.strategy,
            "sample_count": policy.sample_count,
            "acceleration_limit": policy.acceleration_limit,
            "control_dt": cfg.control_dt,
        },
        "patched_joint_rmse": float(
            np.sqrt(np.mean((patched[start:stop, owned] - q[:, owned]) ** 2))
        ),
        "tcp_errors": errors,
        "limits": diagnostics,
        "start": start,
        "stop": stop,
        "editable_joints": owned,
        "qpos_semantics": recording.semantics,
        "output_semantics": "design_reference",
        "physical_success": "unknown",
        "collision_evidence": "unverified",
        "fit_cfg": asdict(cfg),
        "candidate_count": len(candidates),
        "elapsed_seconds": time.monotonic() - began,
        "joint_names": list(recording.joint_names),
    }
    return {
        "timestamps": recording.timestamps.copy(),
        "original_qpos": recording.qpos.copy(),
        "native_qpos": candidate_full,
        "patched_qpos": patched if accepted else recording.qpos.copy(),
        "proposed_qpos": patched,
        "native_phase_qpos": raw_native,
        "phase_progress": timing,
        "modified_mask": (
            (patched != recording.qpos)
            if accepted
            else np.zeros_like(patched, dtype=bool)
        ),
    }, report
