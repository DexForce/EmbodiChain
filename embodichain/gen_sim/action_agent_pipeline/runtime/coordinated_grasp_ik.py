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

"""Select dual-arm grasp candidates through coordinated IK feasibility checks."""

from __future__ import annotations

import torch

from embodichain.gen_sim.action_agent_pipeline.runtime.action_runtime_types import (
    CoordinatedGraspPair,
)
from embodichain.utils.logger import log_warning
from embodichain.utils.math import matrix_from_quat, quat_from_matrix

__all__ = [
    "_select_ik_feasible_coordinated_grasp_pair",
    "_select_coordinated_grasp_pair_tcp_roll_variant",
    "_select_coordinated_arm_tcp_roll_variant",
    "_coordinated_grasp_ik_sequence",
    "_coordinated_pickment_segment_lengths",
    "_coordinated_motion_keyframe_indices",
    "_interpolate_coordinated_object_pose",
    "_coordinated_sequence_ik",
    "_has_coordinated_ik_api",
    "_current_coordinated_arm_qpos",
]


def _select_ik_feasible_coordinated_grasp_pair(
    candidates: list[CoordinatedGraspPair],
    *,
    object_initial_pose: torch.Tensor,
    object_target_pose: torch.Tensor | None,
    pre_grasp_distance: float,
    lift_height: float,
    sample_interval: int = 120,
    hand_interp_steps: int = 10,
    hold_steps: int = 4,
    object_motion_keyframes: int = 6,
    env,
    device,
    env_id: int = 0,
) -> CoordinatedGraspPair | None:
    if not _has_coordinated_ik_api(env):
        return candidates[0] if candidates else None
    for candidate in candidates:
        selected = _select_coordinated_grasp_pair_tcp_roll_variant(
            candidate,
            object_initial_pose=object_initial_pose,
            object_target_pose=object_target_pose,
            pre_grasp_distance=pre_grasp_distance,
            lift_height=lift_height,
            sample_interval=sample_interval,
            hand_interp_steps=hand_interp_steps,
            hold_steps=hold_steps,
            object_motion_keyframes=object_motion_keyframes,
            env=env,
            device=device,
            env_id=env_id,
        )
        if selected is not None:
            return selected
    return None


def _select_coordinated_grasp_pair_tcp_roll_variant(
    candidate: CoordinatedGraspPair,
    *,
    object_initial_pose: torch.Tensor,
    object_target_pose: torch.Tensor | None,
    pre_grasp_distance: float,
    lift_height: float,
    sample_interval: int = 120,
    hand_interp_steps: int = 10,
    hold_steps: int = 4,
    object_motion_keyframes: int = 6,
    env,
    device,
    env_id: int = 0,
) -> CoordinatedGraspPair | None:
    left_seed, right_seed = _current_coordinated_arm_qpos(env, device, env_id=env_id)
    left_object_to_eef = _select_coordinated_arm_tcp_roll_variant(
        candidate.left_object_to_eef,
        object_initial_pose=object_initial_pose,
        object_target_pose=object_target_pose,
        pre_grasp_distance=pre_grasp_distance,
        lift_height=lift_height,
        sample_interval=sample_interval,
        hand_interp_steps=hand_interp_steps,
        hold_steps=hold_steps,
        object_motion_keyframes=object_motion_keyframes,
        env=env,
        is_left=True,
        qpos_seed=left_seed,
        env_id=env_id,
    )
    if left_object_to_eef is None:
        return None
    right_object_to_eef = _select_coordinated_arm_tcp_roll_variant(
        candidate.right_object_to_eef,
        object_initial_pose=object_initial_pose,
        object_target_pose=object_target_pose,
        pre_grasp_distance=pre_grasp_distance,
        lift_height=lift_height,
        sample_interval=sample_interval,
        hand_interp_steps=hand_interp_steps,
        hold_steps=hold_steps,
        object_motion_keyframes=object_motion_keyframes,
        env=env,
        is_left=False,
        qpos_seed=right_seed,
        env_id=env_id,
    )
    if right_object_to_eef is None:
        return None
    if (
        left_object_to_eef is candidate.left_object_to_eef
        and right_object_to_eef is candidate.right_object_to_eef
    ):
        return candidate
    return CoordinatedGraspPair(
        left_object_to_eef=left_object_to_eef,
        right_object_to_eef=right_object_to_eef,
        priority=candidate.priority,
        score=candidate.score,
        axis_kind=candidate.axis_kind,
    )


def _select_coordinated_arm_tcp_roll_variant(
    object_to_eef: torch.Tensor,
    *,
    object_initial_pose: torch.Tensor,
    object_target_pose: torch.Tensor | None,
    pre_grasp_distance: float,
    lift_height: float,
    sample_interval: int,
    hand_interp_steps: int,
    hold_steps: int,
    object_motion_keyframes: int,
    env,
    is_left: bool,
    qpos_seed: torch.Tensor | None,
    env_id: int,
) -> torch.Tensor | None:
    mirrored_object_to_eef = object_to_eef.clone()
    mirrored_object_to_eef[:3, 0] = -mirrored_object_to_eef[:3, 0]
    mirrored_object_to_eef[:3, 1] = -mirrored_object_to_eef[:3, 1]
    for variant in (object_to_eef, mirrored_object_to_eef):
        sequence = _coordinated_grasp_ik_sequence(
            object_initial_pose=object_initial_pose,
            object_target_pose=object_target_pose,
            object_to_eef=variant,
            pre_grasp_distance=pre_grasp_distance,
            lift_height=lift_height,
            sample_interval=sample_interval,
            hand_interp_steps=hand_interp_steps,
            hold_steps=hold_steps,
            object_motion_keyframes=object_motion_keyframes,
        )
        ok, _ = _coordinated_sequence_ik(
            env,
            sequence,
            is_left=is_left,
            qpos_seed=qpos_seed,
            env_id=env_id,
        )
        if ok:
            return variant
    return None


def _coordinated_grasp_ik_sequence(
    *,
    object_initial_pose: torch.Tensor,
    object_target_pose: torch.Tensor | None,
    object_to_eef: torch.Tensor,
    pre_grasp_distance: float,
    lift_height: float,
    sample_interval: int = 120,
    hand_interp_steps: int = 10,
    hold_steps: int = 4,
    object_motion_keyframes: int = 6,
) -> list[torch.Tensor]:
    grasp = object_initial_pose @ object_to_eef
    pre_grasp = grasp.clone()
    pre_grasp[:3, 3] = grasp[:3, 3] - grasp[:3, 2] * float(pre_grasp_distance)
    lift_object_pose = object_initial_pose.clone()
    lift_object_pose[2, 3] += float(lift_height)
    segments = _coordinated_pickment_segment_lengths(
        sample_interval=sample_interval,
        hand_interp_steps=hand_interp_steps,
        hold_steps=hold_steps,
    )
    lift_object_poses = _interpolate_coordinated_object_pose(
        object_initial_pose,
        lift_object_pose,
        segments["lift"],
        include_orientation=False,
    )
    lift_keyframes = _coordinated_motion_keyframe_indices(
        segments["lift"], object_motion_keyframes, object_initial_pose.device
    )
    sequence = [pre_grasp, grasp]
    sequence.extend(
        lift_object_poses[waypoint_idx] @ object_to_eef
        for waypoint_idx in lift_keyframes.tolist()
    )
    if object_target_pose is not None:
        move_object_poses = _interpolate_coordinated_object_pose(
            lift_object_pose,
            object_target_pose,
            segments["move"],
            include_orientation=True,
        )
        move_keyframes = _coordinated_motion_keyframe_indices(
            segments["move"], object_motion_keyframes, object_initial_pose.device
        )
        sequence.extend(
            move_object_poses[waypoint_idx] @ object_to_eef
            for waypoint_idx in move_keyframes.tolist()
        )
    return sequence


def _coordinated_pickment_segment_lengths(
    *,
    sample_interval: int,
    hand_interp_steps: int,
    hold_steps: int,
) -> dict[str, int]:
    n_close = max(2, int(hand_interp_steps))
    n_hold = max(0, int(hold_steps))
    n_motion = int(sample_interval) - n_close - n_hold
    n_approach = n_motion // 3
    n_lift = n_motion // 3
    n_move = n_motion - n_approach - n_lift
    if min(n_approach, n_lift, n_move) < 2:
        raise ValueError("Not enough waypoints for CoordinatedPickment IK precheck.")
    return {
        "approach": n_approach,
        "close": n_close,
        "lift": n_lift,
        "move": n_move,
        "hold": n_hold,
    }


def _coordinated_motion_keyframe_indices(
    n_waypoints: int,
    object_motion_keyframes: int,
    device,
) -> torch.Tensor:
    n_keyframes = min(max(2, int(object_motion_keyframes)), int(n_waypoints))
    return (
        torch.linspace(0, n_waypoints - 1, steps=n_keyframes, device=device)
        .round()
        .to(dtype=torch.long)
    )


def _interpolate_coordinated_object_pose(
    start_pose: torch.Tensor,
    end_pose: torch.Tensor,
    n_waypoints: int,
    *,
    include_orientation: bool,
) -> torch.Tensor:
    weights = torch.linspace(
        0.0,
        1.0,
        steps=n_waypoints,
        device=start_pose.device,
        dtype=start_pose.dtype,
    )
    poses = start_pose.unsqueeze(0).repeat(n_waypoints, 1, 1)
    poses[:, :3, 3] = torch.lerp(
        start_pose[None, :3, 3],
        end_pose[None, :3, 3],
        weights[:, None],
    )
    if not include_orientation:
        return poses

    start_quat = quat_from_matrix(start_pose[:3, :3])
    end_quat = quat_from_matrix(end_pose[:3, :3])
    if float(torch.sum(start_quat * end_quat)) < 0.0:
        end_quat = -end_quat
    quat = torch.lerp(start_quat[None], end_quat[None], weights[:, None])
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
    poses[:, :3, :3] = matrix_from_quat(quat)
    return poses


def _coordinated_sequence_ik(
    env,
    poses: list[torch.Tensor],
    *,
    is_left: bool,
    qpos_seed: torch.Tensor | None,
    env_id: int = 0,
) -> tuple[bool, torch.Tensor | None]:
    seed = qpos_seed
    for pose in poses:
        try:
            ok, qpos = env.get_arm_ik(
                pose,
                is_left=is_left,
                qpos_seed=seed,
                env_ids=[env_id],
            )
        except Exception as exc:
            side = "left" if is_left else "right"
            log_warning(
                "CoordinatedPickment IK precheck failed for "
                f"{side} arm in env {env_id}: "
                f"{exc}"
            )
            return False, seed
        if not ok:
            return False, seed
        seed = torch.as_tensor(qpos, dtype=torch.float32, device=pose.device).reshape(
            1, -1
        )
    return True, seed


def _has_coordinated_ik_api(env) -> bool:
    return env is not None and callable(getattr(env, "get_arm_ik", None))


def _current_coordinated_arm_qpos(
    env,
    device,
    env_id: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if env is None or not hasattr(env, "get_current_qpos_agent"):
        return None, None
    try:
        left_qpos, right_qpos = env.get_current_qpos_agent()
    except Exception:
        return None, None
    left_qpos = torch.as_tensor(left_qpos, dtype=torch.float32, device=device)
    right_qpos = torch.as_tensor(right_qpos, dtype=torch.float32, device=device)
    if left_qpos.ndim == 1:
        left_qpos = left_qpos.unsqueeze(0)
    else:
        left_qpos = left_qpos[env_id : env_id + 1]
    if right_qpos.ndim == 1:
        right_qpos = right_qpos.unsqueeze(0)
    else:
        right_qpos = right_qpos[env_id : env_id + 1]
    return left_qpos, right_qpos
