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

"""Motion timing must follow the executed skill and every controlled arm."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.atomic_skills.plans import CompiledTrajectory, TimedTrajectory
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain_benchmarks.motion_generation.config import ProtocolCfg, SuiteCfg
from embodichain_benchmarks.motion_generation.metrics.trajectory import (
    compute_case_outcomes,
)
from embodichain_benchmarks.motion_generation.models import BenchmarkCase
from embodichain_benchmarks.motion_generation.scenarios.atomic_skill import (
    AtomicSkillScenario,
)


class _Robot:
    device = torch.device("cpu")
    joint_names = ["left", "right"]
    mimic_ids = ()
    mimic_parents = ()
    mimic_multipliers = ()
    num_instances = 1

    def get_joint_ids(self, name: str) -> list[int]:
        return [0] if name == "left_arm" else [1]

    def get_qvel_limits(self, *, name=None, joint_ids=None) -> torch.Tensor:
        ids = joint_ids if joint_ids is not None else self.get_joint_ids(name)
        return torch.tensor([[2.0, 0.2]])[:, ids]

    def get_qpos_limits(self, *, name: str) -> torch.Tensor:
        return torch.tensor([[[-2.0, 2.0]]])

    def compute_batch_fk(self, *, qpos, name, to_matrix):
        poses = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        poses[..., 0, 3] = qpos[..., 0]
        return poses


def _case(skill_id: str = "hand_over") -> BenchmarkCase:
    target = torch.eye(4).reshape(1, 1, 4, 4)
    target[..., 0, 3] = 0.2
    return BenchmarkCase(
        suite_version="timing_v1",
        track="atomic",
        scenario_id=skill_id,
        case_id=skill_id,
        seed=11,
        batch_size=1,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="nominal",
        start_qpos=torch.zeros(1, 1),
        target_waypoints=target,
        reference_qpos=torch.tensor([[[0.2]]]),
        full_start_qpos=torch.zeros(1, 2),
        skill_id=skill_id,
    )


def _compiled() -> CompiledTrajectory:
    positions = torch.tensor([[[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]])
    return CompiledTrajectory(
        plan_success=torch.tensor([True]),
        trajectory=TimedTrajectory.from_uniform_step(
            positions, env_ids=torch.tensor([0]), step_dt=0.1
        ),
        skill_plans=(),
        projected_context=SimpleNamespace(),
    )


@pytest.mark.parametrize("timing_is_solved", [False, True])
def test_secondary_arm_uses_same_dynamic_policy(timing_is_solved: bool) -> None:
    robot, result, case = _Robot(), _compiled(), _case()
    suite = SuiteCfg(protocol=ProtocolCfg())
    primary = compute_case_outcomes(
        PlanResult(
            success=result.plan_success,
            positions=result.trajectory.positions[:, :, :1],
            dt=result.trajectory.dt,
        ),
        case,
        robot,
        "left_arm",
        validation_samples=32,
        position_threshold_m=0.01,
        rotation_threshold_rad=0.1,
        joint_limit_tolerance_rad=1.0e-5,
        timing_is_solved=timing_is_solved,
    )
    assert primary[0].motion_valid
    case = replace(
        case,
        case_parameters={
            "arm_waypoints": {"right_arm": case.target_waypoints.tolist()}
        },
    )
    combined = AtomicSkillScenario._with_other_arm_validity(
        result, case, robot, suite, primary, timing_is_solved=timing_is_solved
    )[0]
    assert combined.ordered_waypoints_reached
    assert combined.geometry_valid
    assert combined.motion_valid is (not timing_is_solved)
    assert combined.timed_motion_valid is (False if timing_is_solved else None)
    assert combined.velocity_limit_violation is True


def test_secondary_arm_preserves_unbounded_dynamic_evidence() -> None:
    """An invalid clock in the second arm survives JSON sanitization."""
    from embodichain_benchmarks.motion_generation.models import (
        AlgorithmRole,
        CaseOutcome,
        TrialPhase,
        TrialRecord,
    )

    robot, result, case = _Robot(), _compiled(), _case()
    primary = compute_case_outcomes(
        PlanResult(
            success=result.plan_success,
            positions=result.trajectory.positions[:, :, :1],
            dt=result.trajectory.dt,
        ),
        case,
        robot,
        "left_arm",
        validation_samples=32,
        position_threshold_m=0.01,
        rotation_threshold_rad=0.1,
        joint_limit_tolerance_rad=1.0e-5,
    )
    invalid = SimpleNamespace(
        plan_success=result.plan_success,
        trajectory=SimpleNamespace(
            positions=result.trajectory.positions, dt=torch.zeros(1, 3)
        ),
    )
    case = replace(
        case,
        case_parameters={
            "arm_waypoints": {"right_arm": case.target_waypoints.tolist()}
        },
    )
    combined = AtomicSkillScenario._with_other_arm_validity(
        invalid,
        case,
        robot,
        SuiteCfg(),
        primary,
    )[0]
    record = TrialRecord(
        suite_version=case.suite_version,
        track=case.track,
        scenario_id=case.scenario_id,
        case_id=case.case_id,
        algorithm_id="toppra",
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        model_revision="N/A",
        planner_config_hash="test",
        seed=case.seed,
        repeat=0,
        batch_size=1,
        waypoint_count=1,
        path_shape="direct",
        start_state_bin="nominal",
        phase=TrialPhase.MEASURED,
        outcomes=(combined,),
    )
    payload = record.to_dict()["outcomes"][0]
    payload.pop("end_to_end_success")
    restored = CaseOutcome(**payload)
    assert restored.dynamic_measurement_status == "unbounded"
    assert restored.velocity_limit_violation is True
    assert restored.timed_motion_valid is False


def test_ik_skill_timing_is_independent_of_unused_planner(monkeypatch) -> None:
    case = _case("press")
    scenario = AtomicSkillScenario()
    scenario._case_providers[case.case_id] = object()
    monkeypatch.setattr(
        scenario, "_replay_runtime", lambda: SimpleNamespace(execute=lambda *args: None)
    )
    suite = SuiteCfg(protocol=ProtocolCfg(joint_velocity_limit_rad_s=0.05))
    rows = [
        scenario.evaluate_case(
            _compiled(),
            case,
            _Robot(),
            "left_arm",
            suite,
            planning_time_ms=1.0,
            timing_is_solved=selected_backend_timing,
        ).outcomes[0]
        for selected_backend_timing in (False, True)
    ]
    assert rows[0].motion_valid is False
    assert rows[1].motion_valid is False
    assert rows[0].velocity_limit_violation is True
    assert rows[1].velocity_limit_violation is True


def test_coordinated_pickment_keeps_its_fixed_clock_out_of_planner_attribution() -> (
    None
):
    from embodichain_benchmarks.motion_generation.scenarios.atomic_policy import (
        benchmark_motion_policy_strategy,
    )

    metadata = AtomicSkillScenario().trial_metadata(_case("coordinated_pickment"))
    assert metadata["selected_planner_backend_used"] is False
    assert metadata["effective_motion_algorithm"] == "coordinated_ik_projection"
    assert metadata["timing_is_solved"] is False
    assert metadata["motion_policy_strategy"] == "motion_gen"
    assert benchmark_motion_policy_strategy("coordinated_pickment") == "motion_gen"


def test_coordinated_placement_keeps_the_selected_planner_attribution() -> None:
    metadata = AtomicSkillScenario().trial_metadata(_case("coordinated_placement"))
    assert metadata["selected_planner_backend_used"] is True
    assert "timing_is_solved" not in metadata


def _budget_invocation(skill_id: str, strategy: str):
    from embodichain.lab.sim.atomic_skills import SkillInvocation, MotionPolicy
    from embodichain.lab.sim.atomic_skills.bindings import SkillBinding

    return SkillInvocation(
        skill_id=skill_id,
        goal=None,
        binding=SkillBinding(owner_id="test", endpoints={}),
        motion_policy=MotionPolicy(strategy=strategy),
    )


@pytest.mark.parametrize(
    "skill,strategy", [("press", "ik_interp"), ("coordinated_pickment", "motion_gen")]
)
def test_scenario_owned_motion_consumes_shared_scalar_budget(
    skill: str, strategy: str
) -> None:
    from embodichain_benchmarks.motion_generation.scenarios.atomic_policy import (
        with_motion_budget,
    )

    invocation = _budget_invocation(skill, strategy)
    updated = with_motion_budget(
        invocation,
        ProtocolCfg(
            joint_velocity_limit_rad_s=0.2, joint_acceleration_limit_rad_s2=3.0
        ),
    )
    assert updated.motion_policy.velocity_limit == 0.2
    assert updated.motion_policy.acceleration_limit == 3.0
    assert updated.motion_policy.strategy == strategy
    assert invocation.motion_policy.velocity_limit is None


def test_vector_policy_budget_is_conservative_and_recorded() -> None:
    from embodichain_benchmarks.motion_generation.scenarios.atomic_policy import (
        with_motion_budget,
    )

    protocol = ProtocolCfg(
        joint_velocity_limit_rad_s=[0.2, 0.1],
        joint_acceleration_limit_rad_s2=[3.0, 2.0],
    )
    updated = with_motion_budget(_budget_invocation("press", "ik_interp"), protocol)
    scenario = AtomicSkillScenario()
    scenario.suite = SuiteCfg(protocol=protocol)
    recorded = scenario.trial_metadata(_case("press"))["policy_motion_limits"]
    assert updated.motion_policy.velocity_limit == recorded["velocity_limit"] == 0.1
    assert (
        updated.motion_policy.acceleration_limit
        == recorded["acceleration_limit"]
        == 2.0
    )
    assert recorded["per_joint_reduction"] == "minimum"


def test_policy_budget_keeps_unset_defaults_and_selected_backend_vectors() -> None:
    from embodichain_benchmarks.motion_generation.scenarios.atomic_policy import (
        with_motion_budget,
    )

    own_motion = _budget_invocation("press", "ik_interp")
    assert with_motion_budget(own_motion, ProtocolCfg()) is own_motion
    selected = _budget_invocation("move_end_effector", "motion_gen")
    protocol = ProtocolCfg(
        joint_velocity_limit_rad_s=[0.2, 0.1],
        joint_acceleration_limit_rad_s2=[3.0, 2.0],
    )
    assert with_motion_budget(selected, protocol) is selected
