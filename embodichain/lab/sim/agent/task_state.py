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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ObjectState:
    name: str
    position: list[float]
    initial_position: list[float] | None
    vertical_alignment: float
    height_drop: float
    xy_displacement: float | None
    is_upright: bool
    is_below_tabletop: bool


@dataclass
class HoldState:
    robot_name: str
    obj_name: str
    distance: float | None
    gripper_distance: float | None
    held_like: bool


@dataclass
class TaskStateSummary:
    semantic_success: bool
    failure_reasons: list[str] = field(default_factory=list)
    object_states: dict[str, ObjectState] = field(default_factory=dict)
    hold_states: list[HoldState] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_success": self.semantic_success,
            "failure_reasons": list(self.failure_reasons),
            "object_states": {
                name: {
                    "name": state.name,
                    "position": state.position,
                    "initial_position": state.initial_position,
                    "vertical_alignment": state.vertical_alignment,
                    "height_drop": state.height_drop,
                    "xy_displacement": state.xy_displacement,
                    "is_upright": state.is_upright,
                    "is_below_tabletop": state.is_below_tabletop,
                }
                for name, state in self.object_states.items()
            },
            "hold_states": [
                {
                    "robot_name": state.robot_name,
                    "obj_name": state.obj_name,
                    "distance": state.distance,
                    "gripper_distance": state.gripper_distance,
                    "held_like": state.held_like,
                }
                for state in self.hold_states
            ],
        }


def summarize_pour_water_state(
    env,
    *,
    object_names: list[str] | tuple[str, ...] = ("bottle", "cup"),
    arm_object_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (
        ("right_arm", "bottle"),
        ("left_arm", "cup"),
    ),
    upright_threshold: float = 0.65,
    max_height_drop: float = 0.04,
    max_final_hold_distance: float = 0.08,
    closed_gripper_threshold: float = 0.025,
) -> TaskStateSummary:
    """Summarize final pour-water state with simple geometry checks.

    The evaluator intentionally avoids camera perception. It checks simulation
    object poses for coarse task semantics so demo results do not treat an
    executor return code as task success.
    """

    env = _unwrap_env(env)
    _update_env_object_info(env)
    failure_reasons: list[str] = []
    object_states: dict[str, ObjectState] = {}

    for obj_name in object_names:
        try:
            state = _summarize_object_state(
                env,
                obj_name=obj_name,
                upright_threshold=upright_threshold,
                max_height_drop=max_height_drop,
            )
        except Exception as exc:
            failure_reasons.append(f"{obj_name}: unavailable ({type(exc).__name__}: {exc})")
            continue

        object_states[obj_name] = state
        if not state.is_upright:
            failure_reasons.append(
                f"{obj_name}: toppled_or_tilted vertical_alignment={state.vertical_alignment:.3f}"
            )
        if state.is_below_tabletop:
            failure_reasons.append(
                f"{obj_name}: below_initial_height height_drop={state.height_drop:.3f}"
            )

    hold_states: list[HoldState] = []
    for robot_name, obj_name in arm_object_pairs:
        if obj_name not in object_states:
            continue
        hold_state = _summarize_hold_state(
            env,
            robot_name=robot_name,
            obj_name=obj_name,
            max_final_hold_distance=max_final_hold_distance,
            closed_gripper_threshold=closed_gripper_threshold,
        )
        hold_states.append(hold_state)
        if hold_state.held_like:
            failure_reasons.append(
                f"{robot_name}/{obj_name}: still held at final state "
                f"distance={_fmt_optional(hold_state.distance)} "
                f"gripper={_fmt_optional(hold_state.gripper_distance)}"
            )

    return TaskStateSummary(
        semantic_success=len(failure_reasons) == 0,
        failure_reasons=failure_reasons,
        object_states=object_states,
        hold_states=hold_states,
    )


def _update_env_object_info(env) -> None:
    sim = getattr(env, "sim", None)
    if sim is not None and hasattr(sim, "update"):
        sim.update(step=20)
    if hasattr(env, "update_obj_info"):
        env.update_obj_info()


def _unwrap_env(env):
    return getattr(env, "unwrapped", env)


def _summarize_object_state(
    env,
    *,
    obj_name: str,
    upright_threshold: float,
    max_height_drop: float,
) -> ObjectState:
    obj_pose = _get_object_pose(env, obj_name)
    position = obj_pose[:3, 3]
    vertical_alignment = float(torch.abs(obj_pose[:3, 2][2]).item())

    initial_pose = _initial_object_pose(env, obj_name, obj_pose)
    initial_position = None
    xy_displacement = None
    initial_height = float(position[2].item())
    if initial_pose is not None:
        initial_position_tensor = initial_pose[:3, 3]
        initial_position = _tensor_to_list(initial_position_tensor)
        xy_displacement = float(
            torch.norm(position[:2] - initial_position_tensor[:2]).item()
        )
        initial_height = float(initial_position_tensor[2].item())

    height_drop = initial_height - float(position[2].item())
    return ObjectState(
        name=obj_name,
        position=_tensor_to_list(position),
        initial_position=initial_position,
        vertical_alignment=vertical_alignment,
        height_drop=height_drop,
        xy_displacement=xy_displacement,
        is_upright=vertical_alignment >= upright_threshold,
        is_below_tabletop=height_drop > max_height_drop,
    )


def _summarize_hold_state(
    env,
    *,
    robot_name: str,
    obj_name: str,
    max_final_hold_distance: float,
    closed_gripper_threshold: float,
) -> HoldState:
    distance = _safe_arm_object_distance(env, robot_name, obj_name)
    gripper_distance = _safe_gripper_distance(env, robot_name)
    held_like = (
        distance is not None
        and gripper_distance is not None
        and distance <= max_final_hold_distance
        and gripper_distance <= closed_gripper_threshold
    )
    return HoldState(
        robot_name=robot_name,
        obj_name=obj_name,
        distance=distance,
        gripper_distance=gripper_distance,
        held_like=held_like,
    )


def _get_object_pose(env, obj_name: str) -> torch.Tensor:
    obj = env.sim.get_rigid_object(obj_name)
    pose = obj.get_local_pose(to_matrix=True).squeeze(0)
    return torch.as_tensor(pose, dtype=torch.float32)


def _initial_object_pose(env, obj_name: str, current_pose: torch.Tensor) -> torch.Tensor | None:
    obj_info = getattr(env, "obj_info", {}).get(obj_name, {})
    pose = obj_info.get("initial_pose")
    if pose is None:
        pose = obj_info.get("pose")
    if pose is None:
        return None
    pose_tensor = torch.as_tensor(pose, dtype=torch.float32)
    if pose_tensor.ndim == 3 and pose_tensor.shape[0] == 1:
        pose_tensor = pose_tensor.squeeze(0)

    # ``update_obj_info`` in older envs mutates ``pose`` in-place as current
    # pose. When no immutable initial pose is available, fall back to the stored
    # initial height so height checks still work without inventing xy baselines.
    if torch.allclose(pose_tensor[:3, 3], current_pose[:3, 3], atol=1e-6):
        height = obj_info.get("height")
        if height is None:
            return pose_tensor
        initial_pose = pose_tensor.clone()
        initial_pose[2, 3] = torch.as_tensor(height, dtype=torch.float32)
        return initial_pose
    return pose_tensor


def _safe_arm_object_distance(env, robot_name: str, obj_name: str) -> float | None:
    try:
        from embodichain.lab.sim.agent.monitor_utils import get_arm_object_distance

        return float(get_arm_object_distance(env, robot_name, obj_name))
    except Exception:
        return None


def _safe_gripper_distance(env, robot_name: str) -> float | None:
    try:
        from embodichain.lab.sim.agent.monitor_utils import get_gripper_distance

        return float(get_gripper_distance(env, robot_name))
    except Exception:
        return None


def _tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().flatten().tolist()]


def _fmt_optional(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.3f}"
