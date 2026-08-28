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

"""Tests for the production cuRobo parallel-command safety gate."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.envs.expert_program import (
    CuroboParallelCommandSafetyValidator,
    CuroboParallelSafetyValidatorFactory,
)
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndpointCommand,
    JointPositionPayload,
    JointPositionTarget,
    RuntimeCommandFrame,
)
from embodichain.lab.sim.planners import CuroboPlanner, MotionGenerator
from embodichain.lab.semantic_skills import SceneRegistry
from embodichain.lab.expert_program._parallel_executor import ParallelSafetyError


class _Robot:
    """Expose current full joint state and one aggregate validation part."""

    def __init__(self) -> None:
        self.qpos = torch.zeros(2, 3)

    def get_qpos(self, *, target: bool = False) -> torch.Tensor:
        assert target is False
        return self.qpos.clone()

    def get_joint_ids(self, *, name: str) -> list[int]:
        assert name == "dual_arm"
        return [0, 1]


def _motion_generator(
    *,
    reject_env: int | None = None,
) -> tuple[MotionGenerator, list[torch.Tensor]]:
    """Build a no-CUDA shell around the exact CuroboPlanner type."""
    observed: list[torch.Tensor] = []
    planner = CuroboPlanner.__new__(CuroboPlanner)

    def validate_joint_trajectory(
        self: CuroboPlanner,
        trajectory: torch.Tensor,
        *,
        control_part: str,
        obstacle_poses: object,
    ) -> torch.Tensor:
        del self, obstacle_poses
        assert control_part == "dual_arm"
        observed.append(trajectory.clone())
        validity = torch.ones(trajectory.shape[:2], dtype=torch.bool)
        if reject_env is not None:
            validity[reject_env, trajectory.shape[1] // 2] = False
        return validity

    planner.validate_joint_trajectory = MethodType(  # type: ignore[method-assign]
        validate_joint_trajectory,
        planner,
    )
    generator = MotionGenerator.__new__(MotionGenerator)
    generator.planner = planner
    return generator, observed


def _frame(
    *commands: EndpointCommand,
    active: tuple[bool, bool] = (True, True),
) -> RuntimeCommandFrame:
    """Build one two-row runtime frame."""
    return RuntimeCommandFrame(
        commands=commands,
        active_mask=torch.tensor(active, dtype=torch.bool),
        env_ids=torch.tensor((0, 1), dtype=torch.long),
        hold_duration=torch.full((2,), 0.05),
    )


def _command(
    target_id: str,
    joint_id: int,
    positions: tuple[float, float],
) -> EndpointCommand:
    """Build one single-joint batched command."""
    return EndpointCommand(
        target=JointPositionTarget(target_id, (joint_id,)),
        payload=JointPositionPayload(
            positions=torch.tensor(positions, dtype=torch.float32).unsqueeze(1)
        ),
    )


def _validator(
    *,
    reject_env: int | None = None,
    max_joint_step: float = 0.05,
    max_interpolation_samples: int = 16,
) -> tuple[CuroboParallelCommandSafetyValidator, list[torch.Tensor]]:
    """Create one validator with a deterministic collision backend shell."""
    generator, observed = _motion_generator(reject_env=reject_env)
    return (
        CuroboParallelCommandSafetyValidator(
            robot=_Robot(),
            motion_generator=generator,
            scene_registry=SceneRegistry(),
            validation_control_part="dual_arm",
            max_joint_step=max_joint_step,
            max_interpolation_samples=max_interpolation_samples,
        ),
        observed,
    )


def test_curobo_parallel_safety_checks_exact_dense_merged_segment() -> None:
    """Disjoint lane targets become one densely sampled aggregate trajectory."""
    validator, observed = _validator()
    left = _command("left_arm", 0, (0.1, 0.0))
    right = _command("right_arm", 1, (0.2, -0.1))

    validator.validate(
        branch_frames={"left": _frame(left), "right": _frame(right)},
        merged_frame=_frame(left, right),
    )

    assert len(observed) == 1
    trajectory = observed[0]
    # float32 represents 0.2 just above the mathematical value, so the strict
    # 0.05 maximum step requires five intervals rather than rounding down.
    assert trajectory.shape == (2, 6, 2)
    torch.testing.assert_close(trajectory[:, 0], torch.zeros(2, 2))
    torch.testing.assert_close(
        trajectory[:, -1],
        torch.tensor(((0.1, 0.2), (0.0, -0.1))),
    )


def test_curobo_parallel_safety_reports_row_local_collision() -> None:
    """One invalid environment rejects dispatch with its stable env ID."""
    validator, _ = _validator(reject_env=1)
    left = _command("left_arm", 0, (0.1, 0.1))
    right = _command("right_arm", 1, (0.2, 0.2))

    with pytest.raises(ParallelSafetyError, match=r"env IDs \(1,\)"):
        validator.validate(
            branch_frames={"left": _frame(left), "right": _frame(right)},
            merged_frame=_frame(left, right),
        )


def test_curobo_parallel_safety_rejects_uncovered_joint() -> None:
    """Every outgoing joint must belong to the aggregate collision model."""
    validator, _ = _validator()
    left = _command("left_arm", 0, (0.1, 0.1))
    hand = _command("left_hand", 2, (0.2, 0.2))

    with pytest.raises(ParallelSafetyError, match="outside validation control part"):
        validator.validate(
            branch_frames={"left": _frame(left), "hand": _frame(hand)},
            merged_frame=_frame(left, hand),
        )


def test_curobo_parallel_safety_fails_instead_of_under_sampling() -> None:
    """The configured memory bound cannot silently enlarge the joint step."""
    validator, _ = _validator(
        max_joint_step=0.01,
        max_interpolation_samples=4,
    )
    left = _command("left_arm", 0, (0.1, 0.1))
    right = _command("right_arm", 1, (0.0, 0.0))

    with pytest.raises(ParallelSafetyError, match="exceeding configured limit"):
        validator.validate(
            branch_frames={"left": _frame(left), "right": _frame(right)},
            merged_frame=_frame(left, right),
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"validation_control_part": ""},
        {"validation_control_part": "dual_arm", "max_joint_step": 0.0},
        {
            "validation_control_part": "dual_arm",
            "max_interpolation_samples": 1,
        },
    ),
)
def test_curobo_parallel_safety_factory_validates_configuration(
    kwargs: dict[str, object],
) -> None:
    """Invalid safety declarations fail during task registration."""
    with pytest.raises((TypeError, ValueError)):
        CuroboParallelSafetyValidatorFactory(**kwargs)  # type: ignore[arg-type]


def test_curobo_parallel_safety_factory_binds_exact_runtime_components() -> None:
    """The production factory consumes the assembled engine and registry."""
    robot = _Robot()
    motion_generator, _ = _motion_generator()
    engine = AtomicActionEngine.__new__(AtomicActionEngine)
    engine._planning_services = SimpleNamespace(  # type: ignore[attr-defined]
        robot=robot,
        motion_generator=motion_generator,
    )
    factory = CuroboParallelSafetyValidatorFactory(validation_control_part="dual_arm")

    validator = factory.create(
        simulation=object(),
        robot=robot,
        scene_registry=SceneRegistry(),
        engine=engine,
    )

    assert type(validator) is CuroboParallelCommandSafetyValidator
