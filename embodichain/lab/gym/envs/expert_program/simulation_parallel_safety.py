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

"""cuRobo-backed physical safety gate for synchronized simulation commands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    JointPositionPayload,
    JointPositionTarget,
    RuntimeCommandFrame,
)
from embodichain.lab.sim.planners import CuroboPlanner, MotionGenerator
from embodichain.lab.semantic_skills import (
    RegistrySceneProvider,
    SceneRegistry,
)
from embodichain.lab.expert_program._parallel_executor import ParallelSafetyError


def _identifier(value: object, *, field_name: str) -> str:
    """Validate one exact identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


@dataclass(frozen=True, slots=True)
class CuroboParallelSafetyValidatorFactory:
    """Create exact-sample collision gates for one aggregate control part.

    ``validation_control_part`` must contain every joint that any parallel
    branch can command. A common dual-arm example is ``"dual_arm"``. The
    cuRobo model for that part remains the authoritative bounds, self-collision,
    and world-collision model.

    Args:
        validation_control_part: Aggregate robot control part containing every
            joint that a parallel lane may command.
        max_joint_step: Maximum absolute joint displacement between collision
            samples in radians or the joint's native linear unit.
        max_interpolation_samples: Fail-closed upper bound on samples per frame.
    """

    validator_id: ClassVar[str] = "builtin.simulation.curobo_parallel_safety"
    revision: ClassVar[str] = "1"
    supported_transport_ids: ClassVar[frozenset[str]] = frozenset(
        {JointPositionTarget.TRANSPORT_ID}
    )

    validation_control_part: str
    max_joint_step: float = 0.025
    max_interpolation_samples: int = 256

    def __post_init__(self) -> None:
        _identifier(
            self.validation_control_part,
            field_name="validation_control_part",
        )
        if isinstance(self.max_joint_step, bool) or not isinstance(
            self.max_joint_step,
            (int, float),
        ):
            raise TypeError("max_joint_step must be a real number.")
        normalized_step = float(self.max_joint_step)
        if not math.isfinite(normalized_step) or normalized_step <= 0.0:
            raise ValueError("max_joint_step must be finite and positive.")
        object.__setattr__(self, "max_joint_step", normalized_step)
        if (
            type(self.max_interpolation_samples) is not int
            or self.max_interpolation_samples < 2
            or self.max_interpolation_samples > 4096
        ):
            raise ValueError(
                "max_interpolation_samples must be an integer in [2, 4096]."
            )

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> CuroboParallelCommandSafetyValidator:
        """Create one fresh validator bound to the assembled live runtime."""
        del simulation
        if type(scene_registry) is not SceneRegistry:
            raise TypeError("scene_registry must be exactly SceneRegistry.")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        if engine.robot is not robot:
            raise ValueError("engine and factory must reference the exact same robot.")
        return CuroboParallelCommandSafetyValidator(
            robot=robot,
            motion_generator=engine.motion_generator,
            scene_registry=scene_registry,
            validation_control_part=self.validation_control_part,
            max_joint_step=self.max_joint_step,
            max_interpolation_samples=self.max_interpolation_samples,
        )


class CuroboParallelCommandSafetyValidator:
    """Validate the exact synchronized joint segment before transport dispatch.

    Args:
        robot: Live robot supplying measured joint state and control-part IDs.
        motion_generator: Runtime motion generator backed by exact cuRobo.
        scene_registry: Authoritative live collision-scene registry.
        validation_control_part: Aggregate control part for merged commands.
        max_joint_step: Maximum displacement between collision samples.
        max_interpolation_samples: Fail-closed sample-count upper bound.
    """

    def __init__(
        self,
        *,
        robot: object,
        motion_generator: MotionGenerator,
        scene_registry: SceneRegistry,
        validation_control_part: str,
        max_joint_step: float,
        max_interpolation_samples: int,
    ) -> None:
        if type(scene_registry) is not SceneRegistry:
            raise TypeError("scene_registry must be exactly SceneRegistry.")
        if not isinstance(motion_generator, MotionGenerator):
            raise TypeError("motion_generator must be a MotionGenerator.")
        if type(motion_generator.planner) is not CuroboPlanner:
            raise TypeError(
                "CuroboParallelCommandSafetyValidator requires the active "
                "CuroboPlanner backend."
            )
        if not motion_generator.supports_joint_trajectory_validation:
            raise ValueError(
                "The active motion generator does not validate exact joint "
                "trajectories."
            )
        get_joint_ids = getattr(robot, "get_joint_ids", None)
        if not callable(get_joint_ids):
            raise TypeError("robot must provide get_joint_ids().")
        joint_ids = tuple(get_joint_ids(name=validation_control_part))
        if not joint_ids or not all(
            type(joint_id) is int and joint_id >= 0 for joint_id in joint_ids
        ):
            raise ValueError(
                "The validation control part must resolve non-negative joint IDs."
            )
        if len(set(joint_ids)) != len(joint_ids):
            raise ValueError("The validation control part joint IDs must be unique.")
        self._robot = robot
        self._motion_generator = motion_generator
        self._scene_registry = scene_registry
        self._validation_control_part = validation_control_part
        self._validation_joint_ids = joint_ids
        self._local_joint_columns = {
            joint_id: index for index, joint_id in enumerate(joint_ids)
        }
        self._max_joint_step = max_joint_step
        self._max_interpolation_samples = max_interpolation_samples
        self._scene_provider: RegistrySceneProvider | None = None
        self._scene_timestamp = 0.0

    def validate(
        self,
        *,
        branch_frames: Mapping[str, RuntimeCommandFrame],
        merged_frame: RuntimeCommandFrame,
    ) -> None:
        """Reject a merged command whose exact interpolated segment collides."""
        if not isinstance(branch_frames, Mapping) or len(branch_frames) < 2:
            raise TypeError("branch_frames must contain at least two branch frames.")
        if type(merged_frame) is not RuntimeCommandFrame:
            raise TypeError("merged_frame must be exactly RuntimeCommandFrame.")
        for branch_id, frame in branch_frames.items():
            _identifier(branch_id, field_name="parallel branch IDs")
            if type(frame) is not RuntimeCommandFrame:
                raise TypeError(
                    "branch_frames values must be exact RuntimeCommandFrame values."
                )
            if not torch.equal(frame.env_ids, merged_frame.env_ids):
                raise ValueError("Parallel branch and merged env_ids must match.")

        active = merged_frame.active_mask
        if not bool(active.any().item()):
            return
        current = self._current_control_part_qpos(merged_frame.env_ids)
        target = current.clone()
        commanded_joint_ids: set[int] = set()
        for command in merged_frame.commands:
            if (
                type(command.target) is not JointPositionTarget
                or type(command.payload) is not JointPositionPayload
            ):
                raise ParallelSafetyError(
                    "cuRobo parallel safety accepts only exact joint-position "
                    "targets and payloads."
                )
            missing = sorted(
                set(command.target.joint_ids).difference(self._local_joint_columns)
            )
            if missing:
                raise ParallelSafetyError(
                    f"Parallel target {command.target.target_id!r} commands joints "
                    f"{missing} outside validation control part "
                    f"{self._validation_control_part!r}."
                )
            for payload_column, joint_id in enumerate(command.target.joint_ids):
                if joint_id in commanded_joint_ids:
                    raise ParallelSafetyError(
                        f"Merged parallel commands overlap on joint {joint_id}."
                    )
                commanded_joint_ids.add(joint_id)
                target[:, self._local_joint_columns[joint_id]] = (
                    command.payload.positions[:, payload_column]
                )
        target = torch.where(active[:, None], target, current)
        trajectory = self._interpolate(current, target)
        obstacle_poses = self._obstacle_poses(
            env_ids=merged_frame.env_ids,
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        validity = self._motion_generator.validate_joint_trajectory(
            trajectory,
            control_part=self._validation_control_part,
            obstacle_poses=obstacle_poses,
        )
        row_valid = validity.all(dim=1)
        failed = active & ~row_valid
        if not bool(failed.any().item()):
            return
        failed_rows = failed.nonzero(as_tuple=False).flatten()
        failed_env_ids = merged_frame.env_ids.index_select(0, failed_rows)
        first_invalid_samples = tuple(
            int((~validity[row]).nonzero(as_tuple=False)[0, 0].item())
            for row in failed_rows.detach().cpu().tolist()
        )
        raise ParallelSafetyError(
            "Merged parallel joint segment is not collision-free for env IDs "
            f"{tuple(failed_env_ids.detach().cpu().tolist())}; first invalid "
            f"samples={first_invalid_samples}."
        )

    def _current_control_part_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Read current full robot state and select the validator joint order."""
        getter = getattr(self._robot, "get_qpos", None)
        if not callable(getter):
            raise TypeError("robot must provide get_qpos().")
        full = getter(target=False)
        if (
            not isinstance(full, torch.Tensor)
            or not full.is_floating_point()
            or full.dim() != 2
            or not bool(torch.isfinite(full).all().item())
        ):
            raise ValueError("robot.get_qpos() must return finite floating (B, D).")
        if env_ids.device != full.device:
            raise ValueError("Parallel env_ids and robot qpos must share a device.")
        if (
            bool((env_ids < 0).any().item())
            or int(env_ids.max().item()) >= full.shape[0]
        ):
            raise ValueError("Parallel env_ids do not address robot qpos rows.")
        if max(self._validation_joint_ids) >= full.shape[1]:
            raise ValueError(
                "Validation control-part joint IDs exceed robot qpos width."
            )
        rows = full.index_select(0, env_ids)
        columns = torch.tensor(
            self._validation_joint_ids,
            dtype=torch.long,
            device=full.device,
        )
        return rows.index_select(1, columns).clone()

    def _interpolate(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Densify the exact controller segment under a bounded joint step."""
        max_delta = float((target - current).abs().max().item())
        sample_count = max(2, math.ceil(max_delta / self._max_joint_step) + 1)
        if sample_count > self._max_interpolation_samples:
            raise ParallelSafetyError(
                "Merged parallel joint segment needs "
                f"{sample_count} collision samples at max_joint_step="
                f"{self._max_joint_step}, exceeding configured limit "
                f"{self._max_interpolation_samples}."
            )
        alpha = torch.linspace(
            0.0,
            1.0,
            sample_count,
            device=current.device,
            dtype=current.dtype,
        )
        return (
            current[:, None, :] + alpha[None, :, None] * (target - current)[:, None, :]
        )

    def _obstacle_poses(
        self,
        *,
        env_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mapping[str, torch.Tensor] | None:
        """Observe the exact dynamic collision world for this safety decision."""
        if not self._scene_registry.dynamic_collision_entity_ids:
            return None
        if self._scene_provider is None:
            self._scene_provider = self._scene_registry.make_scene_provider(
                batch_size=int(env_ids.numel())
            )
        snapshot = self._scene_provider.snapshot(
            timestamp=self._scene_timestamp,
            env_ids=env_ids,
        )
        self._scene_timestamp += 1.0
        return snapshot.collision_obstacle_poses(
            batch_size=int(env_ids.numel()),
            device=device,
            dtype=dtype,
        )


__all__ = [
    "CuroboParallelCommandSafetyValidator",
    "CuroboParallelSafetyValidatorFactory",
]
