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

"""Required Place endpoint qualification without changing goals or RNG state."""

from __future__ import annotations

from typing import Any
import torch

from embodichain.lab.sim.motion.search import PlanningRandomState
from embodichain.lab.sim.motion.solvers.tolerances import pose_validation_tolerances
from ..bindings import JointPositionTarget

__all__: list[str] = []


def qualify_required_waypoints(
    action: Any, request: Any, context: Any
) -> dict[str, Any] | None:
    """Screen ordinary required poses before expanding an expensive path.

    This is a finite IK qualification, not proof of global unreachability. It
    neither changes a target nor consumes the caller's random stream.
    """
    from embodichain.lab.sim.atomic_actions.primitives.place import Place, PlaceGoal
    from ._pick_path import _pose_path_valid
    from embodichain.lab.sim.motion.search import measure_search

    if (
        context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or not isinstance(request.goal, PlaceGoal)
        or request.goal.tcp_symmetry != "none"
        or not request.skill_options.preserve_current_object_orientation
        or request.skill_options.max_approach_retract_z is not None
        or request.motion_policy.strategy != "ik_interp"
        or request.motion_policy.dynamic_collision_mode != "off"
    ):
        raise ValueError(
            "Required Place waypoint qualification supports the explicit B1 ordinary preserved-orientation scope."
        )
    endpoint = request.binding.endpoint("primary", "motion")
    arm = endpoint.require_target(JointPositionTarget)
    target, _ = action._resolve_place_xpos(
        request.goal, context, endpoint.task_state_key, request=request
    )
    if target.ndim == 4:
        if target.shape[1] != 1:
            raise ValueError(
                "Required Place waypoint qualification needs one target pose."
            )
        target = target[:, 0]
    if target.shape != (1, 4, 4):
        raise ValueError(
            "Required Place waypoint qualification needs one B1 target pose."
        )
    start_qpos = context.robot.qpos[:, list(arm.joint_ids)]
    start = action.robot.compute_fk(
        qpos=start_qpos, name=arm.control_part, to_matrix=True
    )
    lifted = Place._lifted_pose(action, target, request.skill_options)
    poses = torch.cat(
        (
            Place._translation_keyframes(
                action, start, torch.stack((lifted, target), 1), request.skill_options
            ),
            Place._translation_keyframes(
                action, target, lifted[:, None], request.skill_options
            ),
        ),
        1,
    )
    state = PlanningRandomState.capture(
        (action.robot.device, context.robot.qpos.device)
    )
    record: dict[str, Any] = {
        "valid": False,
        "scope": "finite_required_waypoint_ik_not_global_unreachability",
        "physical_evidence": False,
        "required_keyframes": poses.shape[1],
        "full_plans": 0,
        "required_poses": poses.detach().cpu().tolist(),
    }
    try:
        with measure_search("place_keyframe_preflight"):
            ok, qpos = action.robot.compute_batch_ik(
                pose=poses,
                name=arm.control_part,
                joint_seed=start_qpos[:, None].expand(-1, poses.shape[1], -1).clone(),
            )
            bounds = action.robot.get_qpos_limits(name=arm.control_part).to(qpos)
            record["ik_valid"] = bool(ok.all())
            record["joint_limits_valid"] = bool(
                torch.isfinite(qpos).all()
                and (qpos >= bounds[:, None, :, 0]).all()
                and (qpos <= bounds[:, None, :, 1]).all()
            )
            if record["ik_valid"] and record["joint_limits_valid"]:
                fk = action.robot.compute_batch_fk(
                    qpos=qpos, name=arm.control_part, to_matrix=True
                )
                pos_eps, rot_eps = pose_validation_tolerances(
                    action.robot.get_solver(arm.control_part)
                )
                record["fk_valid"] = _pose_path_valid(fk, poses, pos_eps, rot_eps)
                record["valid"] = record["fk_valid"]
    finally:
        state.restore()
    return record
